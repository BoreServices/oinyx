"""``kai`` -- a oinyx provider for the unauthenticated Kilo Gateway free models.

Endpoint: ``https://api.kilo.ai/api/gateway/chat/completions`` (OpenAI-shaped
JSON, plus standard SSE when ``stream`` is true). No API key is required;
``config["api_key"]`` is sent as a bearer token if you have one.

Short names are translated through :data:`KAI_MODELS`, so applications can call
``kai/auto`` instead of the upstream id. Anything not in the table is forwarded
verbatim, which means raw ids keep working too::

    await client.chat("kai/auto", "hi")                       # -> kilo-auto/free
    await client.chat("kai/nvidia/nemotron-3.5-lightning:free", "hi")

The table below was probed against the live gateway on 2026-10-29. Four ids
from the original list (``poolside/laguna-m.1:free``, ``poolside/laguna-xs.2:free``,
``nvidia/nemotron-3-super-120b-a12b:free``, ``stepfun/step-3.7-flash:free``)
return "model does not exist" and have been re-pointed at the surviving
equivalent, or dropped. Run ``python -m providers.kai`` to re-check every entry.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from oinyx import (
    AuthenticationError,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    OinyxError,
    ProviderError,
    RateLimitError,
    StreamChunk,
    StreamInterruptedError,
    Usage,
    estimate_tokens,
    register_provider,
    retry,
)

CHAT_URL = "https://api.kilo.ai/api/gateway/chat/completions"
MODELS_URL = "https://api.kilo.ai/api/gateway/models"
DEFAULT_MODEL = "kai/auto"

#: Short alias -> upstream model id. Keys keep the ``kai/`` prefix so the table
#: reads the same as the model strings callers actually pass.
KAI_MODELS: dict[str, str] = {
    "kai/auto": "kilo-auto/free",
    "kai/glyph": "stealth/glyph-cluster",
    "kai/m.1": "poolside/laguna-s-2.1:free",
    "kai/xs-2.1": "poolside/laguna-xs-2.1:free",
    "kai/beam": "reflection/beam-free",
    "kai/flint": "recursive/flint-v1.0:free",
    "kai/lfm": "liquid/lfm-2.5-2.6b:free",
    "kai/dots": "dots-studio/dots-3-note-preview:free",
    "kai/xing4": "xingchen-agi/xing4.0-29b-a4b:free",
    "kai/north-mini": "cohere/north-mini-code:free",
    "kai/nemo3-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "kai/nemo3-super": "nvidia/nemotron-3.5-super-vl:free",
    "kai/nemo3-lightning": "nvidia/nemotron-3.5-lightning:free",
    "kai/nemo3-nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "kai/openfree": "openrouter/free",
}

#: Aliases whose upstream id could not be verified as working.
RETIRED_ALIASES: dict[str, str] = {
    "kai/xs.2": "poolside/laguna-xs.2:free",
    "kai/3.7-flash": "stepfun/step-3.7-flash:free",
    "kai/nemo3-super-120b": "nvidia/nemotron-3-super-120b-a12b:free",
    "kai/laguna-m1": "poolside/laguna-m.1:free",
}

USER_AGENT = "oinyx-kai/0.1 (+https://github.com/example/oinyx)"


class UpstreamError(ProviderError):
    """5xx or transport failure: the same request is worth retrying."""


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a JSON object, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a JSON array, otherwise an empty list."""
    return value if isinstance(value, list) else []


@register_provider("kai")
class KaiProvider(BaseProvider):
    """oinyx provider for the free models behind the Kilo Gateway."""

    name = "kai"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.chat_url: str = str(self.config.get("base_url", CHAT_URL))
        self.models_url: str = str(self.config.get("models_url", MODELS_URL))
        self.default_model: str = str(self.config.get("model", DEFAULT_MODEL))
        self.timeout: float = float(self.config.get("timeout", 120.0))
        self.attempts: int = max(1, int(self.config.get("attempts", 3)))
        self.backoff: float | Callable[[int], float] = self.config.get("backoff", 1.0)
        self.aliases: dict[str, str] = dict(
            self.config.get("aliases") or KAI_MODELS
        )
        self.api_key: str = str(self.config.get("api_key", ""))
        headers: dict[str, str] = {
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }
        # Kilo's own clients identify themselves with these two; harmless to send.
        if self.config.get("referer"):
            headers["http-referer"] = str(self.config["referer"])
        if self.config.get("title"):
            headers["x-title"] = str(self.config["title"])
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        headers.update(dict(self.config.get("headers") or {}))
        self.headers: dict[str, str] = headers
        self._injected: httpx.AsyncClient | None = self.config.get("client")
        self._transport: httpx.AsyncBaseTransport | None = self.config.get("transport")

        self._post_json = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self._post_json_once)

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Complete ``messages`` through the gateway and return one response.

        ``model`` may be a short alias (``"auto"``), a prefixed alias
        (``"kai/auto"``), or a raw upstream id.
        """
        payload = self.build_payload(model, messages, kwargs)
        body = await self._post_json(payload)
        return self.parse_response(body, requested=payload["model"])

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield OpenAI-style delta chunks over SSE."""
        payload = self.build_payload(model, messages, kwargs)
        payload["stream"] = True
        owns_client, client = self._client()
        try:
            request = client.build_request("POST", self.chat_url, headers=self.headers, json=payload)
            response = await client.send(request, stream=True)
            try:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error_for(response, payload["model"])
                async for chunk in self._iter_sse(response, payload["model"]):
                    yield chunk
            finally:
                await response.aclose()
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"kai request failed: {exc}", provider=self.name, model=model
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def list_models(self) -> list[str]:
        """Return the user-facing ``kai/<short>`` aliases this adapter knows."""
        return sorted(self.aliases)

    async def discover_models(self) -> list[str]:
        """Fetch the live catalog and return the upstream ids priced at zero."""
        owns_client, client = self._client()
        try:
            response = await client.get(self.models_url, headers={"accept": "application/json"})
            if response.status_code >= 400:
                raise self._error_for(response, None)
            data = response.json()
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"kai model lookup failed: {exc}", provider=self.name, model=None
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
        items = as_list(data) or as_list(as_dict(data).get("data"))
        return sorted(
            str(item["id"])
            for item in items
            if isinstance(item, dict) and item.get("id") and self._is_free(item)
        )

    # -- payload / response helpers ------------------------------------------

    def build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the alias and assemble the JSON body."""
        if not messages:
            raise ProviderError(
                "kai requires at least one message", provider=self.name, model=model
            )
        body: dict[str, Any] = dict(kwargs.pop("payload", None) or {})
        body.update({key: value for key, value in kwargs.items() if value is not None})
        body["model"] = self.resolve_model(model)
        body["messages"] = [message.to_dict() for message in messages]
        return body

    def resolve_model(self, model: str | None) -> str:
        """Map an alias onto its upstream id, passing unknown values through.

        Raises:
            ProviderError: when no model was given and the default alias is
                missing from the table.
        """
        name = (model or self.default_model).strip()
        if not name:
            raise ProviderError("kai needs a model name", provider=self.name, model=model)
        candidates = (name, f"kai/{name}")
        for candidate in candidates:
            if candidate in self.aliases:
                return self.aliases[candidate]
        if name.startswith("kai/"):
            # Unknown short name: the gateway reports a clear "does not exist".
            stripped = name[len("kai/") :]
            return stripped or name
        return name

    async def _post_json_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One POST returning the decoded JSON body, errors mapped already."""
        owns_client, client = self._client()
        try:
            response = await client.post(self.chat_url, headers=self.headers, json=payload)
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"kai request failed: {exc}", provider=self.name, model=payload.get("model")
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
        if response.status_code >= 400:
            raise self._error_for(response, payload.get("model"))
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"kai returned a non-JSON body: {response.text[:200]!r}",
                provider=self.name,
                model=payload.get("model"),
            ) from exc
        if not isinstance(body, dict):
            raise ProviderError(
                f"kai returned unexpected payload type {type(body).__name__}",
                provider=self.name,
                model=payload.get("model"),
            )
        if body.get("error"):
            message, code = self._describe_error(body["error"])
            raise ProviderError(
                f"{message} [code={code}]",
                provider=self.name,
                model=payload.get("model"),
                status_code=response.status_code,
            )
        return body

    def parse_response(self, body: dict[str, Any], *, requested: str) -> ChatResponse:
        """Convert a ``chat.completion`` payload into a :class:`ChatResponse`."""
        choices = as_list(body.get("choices"))
        first = as_dict(choices[0]) if choices else {}
        message = as_dict(first.get("message"))
        content = message.get("content")
        if not isinstance(content, str):
            content = self._extract_unusual_text(body)
        usage = Usage.from_dict(body.get("usage"))
        if not usage.total_tokens:
            usage = Usage(
                prompt_tokens=estimate_tokens(str(body.get("prompt", ""))),
                completion_tokens=estimate_tokens(content),
            )
        return ChatResponse(
            id=str(body.get("id", "")),
            model=str(body.get("model", requested)),
            content=content,
            finish_reason=first.get("finish_reason"),
            usage=usage,
            raw={
                "created": body.get("created"),
                "system_fingerprint": body.get("system_fingerprint"),
                "reasoning": isinstance(message.get("reasoning_details"), list),
                "requested_model": requested,
            },
        )

    @staticmethod
    def _extract_unusual_text(body: dict[str, Any]) -> str:
        """Fall back to a plain string field when ``choices`` is missing."""
        for key in ("content", "text", "response", "message"):
            value = body.get(key)
            if isinstance(value, str):
                return value
        return ""

    async def _iter_sse(self, response: httpx.Response, model: str) -> AsyncIterator[StreamChunk]:
        """Parse ``data:`` lines from the SSE body into StreamChunks."""
        finished = False
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith((":", "event:", "id:", "retry:")):
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                finished = True
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise StreamInterruptedError(
                    f"kai sent malformed SSE data: {data[:120]!r}",
                    provider=self.name,
                    model=model,
                ) from exc
            if isinstance(event.get("error"), (dict, str)):
                message, code = self._describe_error(event["error"])
                raise ProviderError(
                    f"{message} [code={code}]", provider=self.name, model=model
                )
            choices = as_list(event.get("choices"))
            choice = as_dict(choices[0]) if choices else {}
            delta = as_dict(choice.get("delta"))
            content = delta.get("content")
            reason = choice.get("finish_reason")
            if isinstance(content, str) and content:
                yield StreamChunk(delta=content, raw=event)
            if reason is not None:
                finished = True
                yield StreamChunk(delta="", finish_reason=str(reason), raw=event)
        if not finished:
            yield StreamChunk(delta="", finish_reason="stop", raw={"note": "no finish_reason sent"})

    def _client(self) -> tuple[bool, httpx.AsyncClient]:
        """Return ``(owned, client)``; owned clients are closed by us."""
        if self._injected is not None:
            return False, self._injected
        return True, httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    def _error_for(self, response: httpx.Response, model: str | None) -> ProviderError:
        """Map a bad HTTP response onto the oinyx exception hierarchy."""
        status = response.status_code
        message, code = self._describe_error(self._error_payload(response))
        detail = f"{message} [code={code}]"
        if status in (401, 403):
            return AuthenticationError(
                detail, provider=self.name, model=model, status_code=status
            )
        if status == 429:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                detail += f" (retry after {retry_after}s)"
            return RateLimitError(detail, provider=self.name, model=model, status_code=status)
        if status >= 500:
            return UpstreamError(detail, provider=self.name, model=model, status_code=status)
        return ProviderError(detail, provider=self.name, model=model, status_code=status)

    @staticmethod
    def _error_payload(response: httpx.Response) -> Any:
        """Extract the ``error`` value from a JSON body, or describe the raw text."""
        try:
            body = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return {"message": text[:200] or f"upstream returned HTTP {response.status_code}"}
        if isinstance(body, dict) and "error" in body:
            return body["error"]
        return body if isinstance(body, dict) else {"message": str(body)[:200]}

    @staticmethod
    def _describe_error(error: Any) -> tuple[str, str]:
        """Normalise string-or-object error payloads into ``(message, code)``."""
        if isinstance(error, str):
            return (error, "upstream_error")
        if isinstance(error, dict):
            keys = ("message", "error", "detail", "error_message")
            message = next(
                (str(error[key]) for key in keys if isinstance(error.get(key), str) and error[key]),
                "upstream error",
            )
            code = error.get("code") or error.get("type") or error.get("status") or "upstream_error"
            return (message, str(code))
        return (str(error)[:200], "upstream_error")

    @staticmethod
    def _is_free(item: dict[str, Any]) -> bool:
        pricing = item.get("pricing") or {}
        try:
            return float(pricing.get("prompt") or 0) == 0.0 and float(
                pricing.get("completion") or 0
            ) == 0.0
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return f"<KaiProvider name={self.name!r} aliases={len(self.aliases)}>"


async def verify(models: dict[str, str] | None = None) -> dict[str, str]:
    """Send one tiny prompt per alias and report what the gateway says."""
    provider = KaiProvider({"attempts": 1, "backoff": 0})
    results: dict[str, str] = {}
    for alias, upstream in (models or KAI_MODELS).items():
        try:
            response = await provider.chat(alias, [ChatMessage("user", "reply with exactly: ok")])
            results[alias] = f"OK   {upstream} -> {response.content.strip()[:30]!r}"
        except OinyxError as exc:
            results[alias] = f"FAIL {upstream} -> {exc.message[:80]}"
    return results


async def main() -> None:
    print("free upstream ids on the gateway today:")
    for model in await KaiProvider().discover_models():
        print(f"  {model}")
    print("\nlive alias check:")
    for alias, status in (await verify()).items():
        print(f"  {status}")


if __name__ == "__main__":
    asyncio.run(main())
