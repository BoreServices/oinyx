"""``fable5`` -- a oinyx provider for the public ``https://fable5.io/api/chat`` endpoint.

This is a reverse-engineered adapter for the free chat proxy that backs the
fable5.io web UI. It requires no API key and serves ``openai/gpt-4o``.

The wire format, as observed:

* ``POST`` with ``{"messages": [...], "model": "openai/gpt-4o"}``.
* Success is **``text/plain``** -- the body *is* the completion, no JSON envelope.
* Failures are JSON: ``{"data": null, "error": {"message": ..., "code": ...}}``.
* The served model is echoed in the ``x-chat-model`` response header.
* Limits: 1-32 messages, each at most 8000 characters. No usage numbers.

Usage in an application::

    import providers.fable5io          # registers the "fable5" provider
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("fable5/openai/gpt-4o", "hello")

Calls are retried on rate limits and 5xx/transport failures; 4xx responses are
reported immediately because the payload will not change on retry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from oinyx import (
    AuthenticationError,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ProviderError,
    RateLimitError,
    StreamChunk,
    StreamInterruptedError,
    Usage,
    estimate_tokens,
    register_provider,
    retry,
)

ENDPOINT = "https://fable5.io/api/chat"
DEFAULT_MODEL = "openai/gpt-4o"
#: The endpoint does not expose a model listing; override with ``config["models"]``.
KNOWN_MODELS = ["openai/gpt-4o"]

MAX_MESSAGES = 32
MAX_CONTENT_CHARS = 8000
#: Header the endpoint uses to report which model actually served the request.
MODEL_HEADER = "x-chat-model"

USER_AGENT = "oinyx-fable5/0.1 (+https://github.com/example/oinyx)"


class UpstreamError(ProviderError):
    """5xx or transport failure: the same request is worth retrying."""


@register_provider("fable5")
class Fable5Provider(BaseProvider):
    """oinyx provider backed by the free fable5.io chat endpoint."""

    name = "fable5"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.endpoint: str = str(self.config.get("base_url", ENDPOINT))
        self.default_model: str = str(self.config.get("model", DEFAULT_MODEL))
        self.timeout: float = float(self.config.get("timeout", 60.0))
        self.attempts: int = max(1, int(self.config.get("attempts", 3)))
        self.backoff: float | Callable[[int], float] = self.config.get("backoff", 1.0)
        self.max_messages: int = int(self.config.get("max_messages", MAX_MESSAGES))
        self.max_content_chars: int = int(self.config.get("max_content_chars", MAX_CONTENT_CHARS))
        self.headers: dict[str, str] = {
            "content-type": "application/json",
            "accept": "text/plain, application/json;q=0.9, */*;q=0.8",
            "user-agent": USER_AGENT,
            **dict(self.config.get("headers") or {}),
        }
        self._models: list[str] = list(self.config.get("models") or KNOWN_MODELS)
        #: Supply an ``httpx.AsyncClient`` (or ``transport``) to share pooling or drive tests.
        self._injected: httpx.AsyncClient | None = self.config.get("client")
        self._transport: httpx.AsyncBaseTransport | None = self.config.get("transport")

        self._send = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self._send_once)

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Complete ``messages`` and return the whole reply.

        Extra keyword arguments are merged into the JSON body, so
        ``temperature=`` style options are forwarded even though the upstream
        proxy currently ignores most of them.
        """
        payload = self.build_payload(model, messages, kwargs)
        response = await self._send(payload)
        text = self.extract_text(response)
        return self.build_response(model, messages, text, response)

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield the reply as it arrives.

        The upstream endpoint returns a single ``text/plain`` body rather than
        SSE, so chunk boundaries depend on how the response is transmitted.
        A final empty chunk carrying ``finish_reason="stop"`` is always emitted
        to match the streaming contract.
        """
        payload = self.build_payload(model, messages, kwargs)
        owns_client, client = self._client()
        try:
            request = client.stream("POST", self.endpoint, headers=self.headers, json=payload)
            async with request as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error_for(response, model)
                try:
                    async for piece in response.aiter_text():
                        if piece:
                            yield StreamChunk(delta=piece, raw={"status_code": response.status_code})
                except httpx.TransportError as exc:
                    raise StreamInterruptedError(
                        f"fable5 stream ended early: {exc}",
                        provider=self.name,
                        model=self._served_model(response, model),
                    ) from exc
                yield StreamChunk(
                    delta="",
                    finish_reason="stop",
                    raw={
                        "status_code": response.status_code,
                        MODEL_HEADER: response.headers.get(MODEL_HEADER),
                    },
                )
        finally:
            if owns_client:
                await client.aclose()

    async def list_models(self) -> list[str]:
        """Return the models this adapter knows about.

        The endpoint has no catalog route, so this is the static
        ``KNOWN_MODELS`` list unless overridden through ``config["models"]``.
        """
        return list(self._models)

    # -- request/response plumbing -------------------------------------------

    def build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate the conversation and assemble the JSON body.

        Raises:
            ProviderError: when the conversation breaks the endpoint's limits
                (1-32 turns, 8000 characters each), which saves a round-trip.
        """
        if not messages:
            raise ProviderError(
                f"fable5 requires between 1 and {self.max_messages} messages, got 0",
                provider=self.name,
                model=model,
            )
        if len(messages) > self.max_messages:
            raise ProviderError(
                f"fable5 accepts at most {self.max_messages} messages, got {len(messages)}",
                provider=self.name,
                model=model,
            )
        serialized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            content = message.content or ""
            if len(content) > self.max_content_chars:
                raise ProviderError(
                    f"fable5 message {index} exceeds {self.max_content_chars} characters "
                    f"(got {len(content)})",
                    provider=self.name,
                    model=model,
                )
            turn: dict[str, Any] = {"role": message.role, "content": content}
            if message.name is not None:
                turn["name"] = message.name
            serialized.append(turn)

        body: dict[str, Any] = dict(kwargs.pop("payload", None) or {})
        body.update({k: v for k, v in kwargs.items() if v is not None})
        body["messages"] = serialized
        body["model"] = model or self.default_model
        return body

    async def _send_once(self, payload: dict[str, Any]) -> httpx.Response:
        """POST the payload once, translating failures into oinyx errors."""
        owns_client, client = self._client()
        try:
            response = await client.post(self.endpoint, headers=self.headers, json=payload)
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"fable5 request failed: {exc}",
                provider=self.name,
                model=payload.get("model"),
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise self._error_for(response, str(payload.get("model") or self.default_model))
        return response

    def _client(self) -> tuple[bool, httpx.AsyncClient]:
        """Return ``(owned, client)``; owned clients must be closed by us."""
        if self._injected is not None:
            return False, self._injected
        return True, httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    def _error_for(self, response: httpx.Response, model: str) -> ProviderError:
        """Map a bad response onto the oinyx exception hierarchy."""
        message, code = self._describe_error(response)
        status = response.status_code
        if status in (401, 403):
            return AuthenticationError(message, provider=self.name, model=model, status_code=status)
        detail = f"{message} [code={code}]"
        if status == 429:
            retry_after = response.headers.get("retry-after")
            suffix = f" (retry after {retry_after}s)" if retry_after else ""
            return RateLimitError(
                f"{message}{suffix} [code={code}]",
                provider=self.name,
                model=model,
                status_code=status,
            )
        if status >= 500:
            return UpstreamError(detail, provider=self.name, model=model, status_code=status)
        return ProviderError(detail, provider=self.name, model=model, status_code=status)

    @staticmethod
    def _describe_error(response: httpx.Response) -> tuple[str, str]:
        """Pull ``(message, code)`` out of the JSON error envelope."""
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return (text[:200] or f"upstream returned HTTP {status}", f"http_{status}")
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return (
                    str(error.get("message") or f"upstream returned HTTP {status}"),
                    str(error.get("code") or f"http_{status}"),
                )
            if isinstance(error, str) and error:
                return (error, f"http_{status}")
            for key in ("message", "detail", "error_message"):
                value = body.get(key)
                if isinstance(value, str) and value:
                    return (value, f"http_{status}")
        return (f"upstream returned HTTP {status}", f"http_{status}")

    @staticmethod
    def extract_text(response: httpx.Response) -> str:
        """Return the completion from either a ``text/plain`` or JSON body.

        Today the endpoint replies with plain text; the JSON branches keep the
        adapter working if that changes to an OpenAI-style envelope.
        """
        content_type = response.headers.get("content-type", "")
        body = response.text or ""
        if "json" not in content_type and not body.lstrip().startswith(("{", "[")):
            return body
        try:
            data = response.json()
        except ValueError:
            return body
        if not isinstance(data, dict):
            return body
        for key in ("content", "text", "response", "message", "data"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for holder in ("message", "delta"):
                    part = first.get(holder)
                    if isinstance(part, dict) and isinstance(part.get("content"), str):
                        return str(part["content"])
                if isinstance(first.get("text"), str):
                    return str(first["text"])
        return body

    def build_response(
        self,
        model: str,
        messages: list[ChatMessage],
        text: str,
        response: httpx.Response,
    ) -> ChatResponse:
        """Assemble a :class:`ChatResponse`, estimating usage locally.

        The endpoint reports no token counts, so ``Usage`` is filled from the
        ``chars/4`` heuristic and ``raw`` keeps the HTTP details for debugging.
        """
        served = response.headers.get(MODEL_HEADER) or model or self.default_model
        prompt = sum(estimate_tokens(message.content) for message in messages)
        completion = estimate_tokens(text)
        return ChatResponse(
            id=f"fable5-{response.status_code}",
            model=served,
            content=text,
            finish_reason="stop",
            usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
            raw={
                "status_code": response.status_code,
                MODEL_HEADER: response.headers.get(MODEL_HEADER),
                "content_type": response.headers.get("content-type"),
            },
        )

    @staticmethod
    def _served_model(response: httpx.Response, fallback: str) -> str:
        return response.headers.get(MODEL_HEADER) or fallback

    def __repr__(self) -> str:
        return f"<Fable5Provider name={self.name!r} endpoint={self.endpoint!r}>"
