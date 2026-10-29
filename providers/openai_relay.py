"""Shared base for plain OpenAI-compatible relays that need no API key.

A "relay" here is any endpoint that speaks the OpenAI chat-completions wire --
``POST {base_url}{chat_path}`` with ``{"model": ..., "messages": [...]}`` and
standard SSE -- but does not require a credential. A handful of those exist in
the free-AI world, and they are all short-lived and quota-throttled, so the
value of this module is that the six subclasses below are ~40 lines each.

What the base owns:

* payload assembly and the short-alias -> upstream-id table (:attr:`catalog`)
* one retryable JSON POST plus an incremental SSE reader
* catalog fetching that tolerates both ``{"data": [...]}`` and ``{"models": [...]}``
* status mapping, including the two shapes keyless services use for "you are
  out of free quota": ``429`` and ``402``

What the base assumes: the endpoint is not behind a Cloudflare managed
challenge. The mechanics it inherits from :class:`CloudflareBase` (injectable
transport, header assembly, retries) still apply; the challenge detection only
fires on the HTML "Just a moment" body, so it is inert here.

The upstream model ids below were read off each service's live ``/models``
endpoint on 2026-10-29; free tiers churn, so re-run
``python -m providers.stillworks`` or each subclass's ``verify()`` before
trusting a stale table.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from oinyx import (
    AuthenticationError,
    ChatMessage,
    ChatResponse,
    OinyxError,
    ProviderError,
    RateLimitError,
    StreamChunk,
    StreamInterruptedError,
    Usage,
    estimate_tokens,
    retry,
)

from .cloudflare_base import CloudflareBase, UpstreamError
from .http_utils import as_dict, as_list, describe_error, error_payload, sse_data

#: Query parameters forwarded to the upstream verbatim.
PASSTHROUGH_PARAMS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "tools",
    "tool_choice",
    "user",
)

#: Values that make a quota-style 4xx/5xx worth retrying rather than surfacing.
_QUOTA_HINTS = ("at its limit", "quota", "limit right now", "rate limit", "too many")


def catalog_items(body: Any) -> list[Any]:
    """Pull the row list out of ``{"data": [...]}``, ``{"models": [...]}``, or a bare list."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return as_list(body.get("data")) or as_list(body.get("models"))
    return []


class OpenAiRelay(CloudflareBase):
    """Base provider for unauthenticated OpenAI-shaped chat endpoints."""

    #: ``https://host/segment/v1`` -- no trailing ``/chat/completions`` here.
    base_url: str = ""
    chat_path: str = "/chat/completions"
    models_path: str = "/models"
    default_model: str = ""
    #: Short alias -> upstream model id. Bare ids outside the table pass through.
    catalog: dict[str, str] = {}
    #: Sent as ``Authorization: Bearer ...`` when no key is configured. Some
    #: relays (AI Horde) reject anonymous calls outright and accept this token.
    anonymous_key: str = ""
    #: Free tiers answer 429/402 per IP; retrying the *same* call rarely helps,
    #: but transient 5xx and queue jitter are common enough to be worth two.
    default_attempts: int = 2
    #: Non-zero only where the upstream is slow by design (queued workers).
    default_timeout: float = 0.0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        if "attempts" not in self.config:
            self.attempts = self.default_attempts
        if self.default_timeout and "timeout" not in self.config:
            self.timeout = self.default_timeout
        base = str(self.config.get("base_url", self.base_url)).rstrip("/")
        self.base_url = base
        chat_path = str(self.config.get("chat_path", self.chat_path))
        models_path = str(self.config.get("models_path", self.models_path))
        self.chat_url = str(self.config.get("chat_url", f"{base}{chat_path}"))
        self.models_url = str(self.config.get("models_url", f"{base}{models_path}"))
        # ``CloudflareBase.post`` targets ``endpoint``; for a relay that is the chat URL.
        self.endpoint = self.chat_url
        self.catalog = dict(self.config.get("catalog") or self.catalog)
        self.default_model = str(self.config.get("model", self.default_model))
        self.api_key = str(self.config.get("api_key", "")) or self.anonymous_key
        self.include_usage = bool(self.config.get("include_usage", False))
        if self.api_key:
            self.headers["authorization"] = f"Bearer {self.api_key}"
        self._post_json = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self.post_json_once)

    # -- provider API --------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Complete ``messages`` and return one response."""
        payload = self.build_payload(model, messages, kwargs)
        body = await self._post_json(payload)
        return self.parse_completion(body, requested=payload["model"])

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield OpenAI-style delta chunks over SSE."""
        payload = self.build_payload(model, messages, kwargs)
        payload["stream"] = True
        if self.include_usage:
            payload.setdefault("stream_options", {"include_usage": True})
        owns_client, client = self._client()
        model_name = str(payload["model"])
        try:
            request = client.build_request("POST", self.chat_url, headers=self.headers, json=payload)
            response = await client.send(request, stream=True)
            try:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error_for(response, model_name)
                async for chunk in self.iter_sse(response, model_name):
                    yield chunk
            finally:
                await response.aclose()
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"{self.name} stream failed: {exc}", provider=self.name, model=model_name
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def list_models(self) -> list[str]:
        """Return the ``<name>/<alias>`` strings this adapter knows offline."""
        prefix = f"{self.name}/"
        return sorted(prefix + alias for alias in self.catalog)

    async def fetch_models(self) -> list[str]:
        """Read the live catalog and return the upstream ids that can chat."""
        response = await self.get(self.models_url)
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.name} catalog is not JSON: {response.text[:200]!r}",
                provider=self.name,
                model=None,
            ) from exc
        ids: list[str] = []
        for item in catalog_items(body):
            if not isinstance(item, (str, dict)):
                continue
            ident = item if isinstance(item, str) else str(item.get("id") or item.get("name") or "")
            if ident and self.accept(item):
                ids.append(ident)
        return ids

    def accept(self, item: Any) -> bool:
        """Decide whether a catalog row is worth exposing.

        Everything passes by default. Subclasses override it when an id alone
        is not enough: free.ai lists a paid marketplace next to the models it
        will actually serve anonymously, LLM7 mixes keyless and keyed tiers.
        """
        return True

    # -- payload / response helpers ------------------------------------------

    def resolve_model(self, model: str | None) -> str:
        """Strip the provider prefix and map a short alias to an upstream id."""
        name = (model or self.default_model).strip()
        if not name:
            raise ProviderError(f"{self.name} needs a model name", provider=self.name, model=model)
        prefix = f"{self.name}/"
        if name.startswith(prefix):
            name = name[len(prefix) :]
        return self.catalog.get(name, name)

    def build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the chat body: resolved model, messages, forwarded options."""
        if not messages:
            raise ProviderError(
                f"{self.name} requires at least one message", provider=self.name, model=model
            )
        body: dict[str, Any] = dict(kwargs.pop("payload", None) or {})
        body.update(
            {key: value for key, value in kwargs.items() if key in PASSTHROUGH_PARAMS and value is not None}
        )
        body["model"] = self.resolve_model(model)
        body["messages"] = [message.to_dict() for message in messages]
        return body

    async def post_json_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One retryable POST, returning the decoded body with errors mapped."""
        response = await self.post(payload, str(payload.get("model")))
        try:
            body: Any = response.json()
        except ValueError as exc:
            # Relays behind a proxy answer 200 with HTML more often than you would hope.
            raise ProviderError(
                f"{self.name} returned a non-JSON body: {response.text[:200]!r}",
                provider=self.name,
                model=str(payload.get("model")),
                status_code=response.status_code,
            ) from exc
        if not isinstance(body, dict):
            raise ProviderError(
                f"{self.name} returned {type(body).__name__}, expected an object",
                provider=self.name,
                model=str(payload.get("model")),
                status_code=response.status_code,
            )
        if body.get("error") and "choices" not in body:
            message, code = describe_error(body["error"])
            raise ProviderError(
                f"{message} [code={code}]",
                provider=self.name,
                model=str(payload.get("model")),
                status_code=response.status_code,
            )
        return body

    def parse_completion(self, body: dict[str, Any], *, requested: str) -> ChatResponse:
        """Convert a ``chat.completion`` payload into a :class:`ChatResponse`."""
        choices = as_list(body.get("choices"))
        first = as_dict(choices[0]) if choices else {}
        message = as_dict(first.get("message"))
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        usage = Usage.from_dict(body.get("usage"))
        if not usage.total_tokens:
            usage = Usage(completion_tokens=estimate_tokens(content))
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return ChatResponse(
            id=str(body.get("id", "")),
            model=str(body.get("model", requested)),
            content=content,
            finish_reason=first.get("finish_reason"),
            usage=usage,
            raw={
                "requested_model": requested,
                "created": body.get("created"),
                "system_fingerprint": body.get("system_fingerprint"),
                "reasoning": reasoning if isinstance(reasoning, str) else "",
            },
        )

    async def iter_sse(self, response: httpx.Response, model: str) -> AsyncIterator[StreamChunk]:
        """Turn ``data:`` lines into chunks; tolerate relays that finish without one."""
        finished = False
        async for raw_line in response.aiter_lines():
            data = sse_data(raw_line)
            if data is None:
                continue  # comments, blank keep-alives, ``event:``/``id:`` fields
            if data == "[DONE]":
                finished = True
                continue
            try:
                event: Any = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                message, code = describe_error(event["error"])
                raise ProviderError(
                    f"{message} [code={code}]", provider=self.name, model=model
                )
            chunk = self.parse_delta(event, model)
            if chunk is not None:
                if chunk.finish_reason is not None:
                    finished = True
                yield chunk
        if not finished:
            raise StreamInterruptedError(
                f"{self.name} stream ended before a finish reason", provider=self.name, model=model
            )

    def parse_delta(self, event: dict[str, Any], model: str) -> StreamChunk | None:
        """Map one SSE JSON event onto a chunk, or ``None`` when it carries nothing.

        OVH's free tier emits hundreds of ``{"role": "assistant", "content":
        null}`` frames before any text; forwarding those would produce a stream
        of empty chunks, so bookkeeping-only frames are dropped while anything
        carrying text, reasoning, usage or a finish reason survives.
        """
        choices = as_list(event.get("choices"))
        first = as_dict(choices[0]) if choices else {}
        delta = as_dict(first.get("delta"))
        text = delta.get("content")
        text = text if isinstance(text, str) else ""
        finish_reason = first.get("finish_reason")
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        usage = Usage.from_dict(event.get("usage"))
        if not text and finish_reason is None and not reasoning and not usage.total_tokens:
            return None
        raw: dict[str, Any] = {"model": str(event.get("model", model))}
        if isinstance(reasoning, str) and reasoning:
            raw["reasoning"] = reasoning
        if usage.total_tokens:
            raw["usage"] = usage.to_dict()
        return StreamChunk(delta=text, finish_reason=finish_reason, raw=raw)

    # -- error mapping -------------------------------------------------------

    def _error_for(self, response: httpx.Response, model: str | None) -> ProviderError:
        """Status -> exception, with 402 treated as quota and retried.

        Keyless relays run out of per-IP budget in whatever way suits them, so
        an explicit "limit" phrase on a 4xx/5xx becomes a :class:`RateLimitError`
        rather than a hard ``ProviderError``.
        """
        status = response.status_code
        payload = error_payload(response)
        message, code = describe_error(payload)
        detail = f"{message} [code={code}]"
        quota = status in (402, 429) or any(hint in message.lower() for hint in _QUOTA_HINTS)
        if quota:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                detail += f" (retry after {retry_after}s)"
            return RateLimitError(detail, provider=self.name, model=model, status_code=status)
        if status in (401, 403):
            return AuthenticationError(detail, provider=self.name, model=model, status_code=status)
        if status >= 500:
            return UpstreamError(detail, provider=self.name, model=model, status_code=status)
        return ProviderError(detail, provider=self.name, model=model, status_code=status)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} base_url={self.base_url!r}>"


async def verify_models(
    provider: OpenAiRelay,
    models: dict[str, str] | None = None,
    question: str = "reply with exactly: ok",
) -> dict[str, str]:
    """Send one tiny prompt per alias and report what the relay says.

    Failures are captured as strings rather than raised, so a dead model never
    aborts the sweep -- that is the whole point of the walk-the-list pattern.
    """
    table = models if models is not None else provider.catalog
    results: dict[str, str] = {}
    for alias, upstream in table.items():
        try:
            response = await provider.chat(alias, [ChatMessage("user", question)], max_tokens=64)
        except OinyxError as exc:
            results[alias] = f"FAIL {upstream} -> {exc.message[:80]}"
        except Exception as exc:  # noqa: BLE001 - keep sweeping on anything weird
            results[alias] = f"FAIL {upstream} -> {type(exc).__name__}: {str(exc)[:80]}"
        else:
            results[alias] = f"OK   {upstream} -> {response.content.strip()[:30]!r}"
    return results


def relay_main(provider: OpenAiRelay, models: dict[str, str] | None = None) -> int:
    """Shared ``python -m providers.<relay>`` body: live catalog, then alias check.

    ``models`` overrides the alias table for the check -- pass a one-entry dict
    when several aliases are the same upstream model and one call is enough.
    """
    print(f"{provider.name}: {provider.chat_url}")
    try:
        live = asyncio.run(provider.fetch_models())
    except (OinyxError, httpx.HTTPError) as exc:
        live = []
        print(f"  catalog lookup failed: {exc}")
    if live:
        print(f"  upstream ids today ({len(live)}):")
        for ident in live:
            print(f"    {ident}")
    print("  live alias check:")
    results = asyncio.run(verify_models(provider, models))
    for line in results.values():
        print(f"    {line}")
    return 0 if all(line.startswith("OK") for line in results.values()) else 1
