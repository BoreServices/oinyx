"""``fai`` -- a oinyx provider for ``aifreeforever.com`` (Free Ai Forever).

Two endpoints, both reverse-engineered from the site's own traffic:

* ``GET  https://aifreeforever.com/api/chat-models`` -- the model catalog.
* ``POST https://aifreeforever.com/api/generate-ai-answer-vertex`` -- the chat
  call, which answers ``text/event-stream`` (SSE) even for a plain request.

The request body is not message-shaped. The conversation is split into one
``question`` (the last user turn) plus a ``conversationHistory`` array, and every
call carries an ``interactionProof`` fingerprint the browser derives from how the
user typed::

    {"timeZone": "Asia/Dhaka", "model": "gemini-3-5", "question": "test",
     "tone": "friendly", "format": "paragraph", "file": null,
     "conversationHistory": [],
     "interactionProof": {"nonce": "mvto0lmt.5f2b...3b.d947...f5",
                          "keystrokeCount": 4, "pasteEvents": 0,
                          "totalTypingTime": 2807,
                          "startTime": 1793285739448,
                          "submitTime": 1793285742255},
     "resend": false, "aiRole": "assistant", "aiName": "", "language": "",
     "followUps": true}

Usage::

    import providers.fai
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("fai/gemini-3-5", "hello")
    async for chunk in await OinyxClient().stream("fai/gemini-3-5", "hello"):
        print(chunk.delta, end="")

Two constraints follow from the site being a browser app behind Cloudflare:

* It only answers clients holding a solved ``cf_clearance`` cookie, so
  ``config["cookies"]`` plus the matching ``user_agent`` is required in practice
  -- see :mod:`providers.cloudflare_base`.
* The ``nonce`` is minted client-side as ``<base36 ms>.<32 hex>.<40 hex>``. This
  module synthesises a well-formed one (``keystrokeCount`` mirrors the typed
  characters, exactly as the capture does). If the server checks the nonce
  against a session, capture a real proof from DevTools and pass
  ``config={"proof": {...}}`` or ``interactionProof={...}`` per call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable
from secrets import token_hex
from typing import Any

import httpx

from oinyx import (
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

from .cloudflare_base import CloudflareBase, UpstreamError

ENDPOINT = "https://aifreeforever.com/api/generate-ai-answer-vertex"
MODELS_ENDPOINT = "https://aifreeforever.com/api/chat-models"
ORIGIN = "https://aifreeforever.com"
CHAT_PAGE = f"{ORIGIN}/chat"
DEFAULT_MODEL = "gemini-3-5"

#: The capture that defines this contract was made from Firefox.
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0"

#: Catalog ids known without a live lookup.
KNOWN_MODELS = [DEFAULT_MODEL]

#: Frame keys tried in order for the answer text.
TEXT_FIELDS = ("text", "content", "answer", "message", "output", "response")

#: Catalog keys tried when the payload is a dict rather than a plain list.
LIST_KEYS = ("models", "data", "items", "results")

#: Catalog keys tried for each entry.
NAME_KEYS = ("id", "slug", "value", "model", "name", "title")

#: Conversation options, popped from kwargs so they are never sent verbatim.
OPTION_KEYS = (
    "time_zone",
    "tone",
    "format",
    "file",
    "history",
    "question",
    "resend",
    "follow_ups",
    "proof",
    "max_tokens",
)


def base36(value: int) -> str:
    """Encode ``value`` in lowercase base36, the scheme the site's nonce uses."""
    if value <= 0:
        return "0"
    digits = ""
    while value:
        value, remainder = divmod(value, 36)
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"[remainder] + digits
    return digits


@register_provider("fai")
class FaiProvider(CloudflareBase):
    """oinyx provider for the aifreeforever.com chat endpoint."""

    name = "fai"
    endpoint = ENDPOINT
    origin = ORIGIN
    user_agent = DEFAULT_USER_AGENT

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.default_model: str = str(self.config.get("model", DEFAULT_MODEL))
        self.time_zone: str = str(self.config.get("time_zone", "Asia/Dhaka"))
        self.tone: str = str(self.config.get("tone", "friendly"))
        self.output_format: str = str(self.config.get("format", "paragraph"))
        self.ai_role: str = str(self.config.get("ai_role", "assistant"))
        self.ai_name: str = str(self.config.get("ai_name", ""))
        self.language: str = str(self.config.get("language", ""))
        self.follow_ups: bool = bool(self.config.get("follow_ups", True))
        self.resend: bool = bool(self.config.get("resend", False))
        self.file: Any = self.config.get("file")
        self.max_history: int = int(self.config.get("max_history", 50))
        self.typing_ms_per_char: int = int(self.config.get("typing_ms_per_char", 120))
        self.proof: dict[str, Any] = dict(self.config.get("proof") or {})
        self.nonce: str | None = str(self.proof.get("nonce")) if self.proof.get("nonce") else None
        self._models: list[str] = list(self.config.get("models") or KNOWN_MODELS)
        self._post = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self.post)

    # -- public API -----------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Complete ``messages`` and return the whole answer.

        The endpoint streams, so a buffered ``text/event-stream`` body is parsed
        frame by frame; anything else is read as JSON or plain text.
        """
        payload = self.build_payload(model, messages, kwargs)
        target = str(payload["model"])
        response = await self._post(payload, target, self.request_headers(target))
        chunks = self.parse_body(response, target)
        content = "".join(chunk.delta for chunk in chunks)
        prompt_tokens = sum(estimate_tokens(str(turn.content)) for turn in messages)
        completion_tokens = estimate_tokens(content)
        return ChatResponse(
            id=f"fai-{response.status_code}",
            model=target,
            content=content,
            finish_reason=self._finish_reason(chunks),
            usage=Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            raw={
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "request_id": response.headers.get("x-zeabur-request-id"),
                "frames": len(chunks),
            },
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield answer deltas from the SSE body."""
        payload = self.build_payload(model, messages, kwargs)
        target = str(payload["model"])
        owns_client, client = self._client()
        try:
            request = client.build_request(
                "POST", self.endpoint, headers=self.request_headers(target), json=payload
            )
            response = await client.send(request, stream=True)
            finished = False
            try:
                if response.status_code >= 400:
                    await response.aread()
                    self.raise_for_status(response, target)
                if self._is_event_stream(response):
                    async for line in response.aiter_lines():
                        for chunk in self.parse_line(line, target):
                            finished = finished or bool(chunk.finish_reason)
                            yield chunk
                else:
                    await response.aread()
                    for chunk in self.parse_body(response, target):
                        finished = True
                        yield chunk
            finally:
                await response.aclose()
            if not finished:
                yield StreamChunk(delta="", finish_reason="stop", raw={"synthesized": True})
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"fai request failed: {exc}", provider=self.name, model=target
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def list_models(self) -> list[str]:
        """Return catalog ids without touching the network."""
        return list(self._models)

    async def fetch_models(self) -> list[str]:
        """Fetch ``/api/chat-models`` and return every id it advertises."""
        response = await self.get(MODELS_ENDPOINT, self.default_model)
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise ProviderError(
                "chat-models did not return JSON", provider=self.name, model=None
            ) from exc
        return parse_model_list(data)

    # -- request building -----------------------------------------------------

    def request_headers(self, model: str) -> dict[str, str]:
        """Per-call headers: the chat page for ``model`` must be the referer."""
        headers = dict(self.headers)
        headers["referer"] = f"{CHAT_PAGE}/{model}"
        headers["accept"] = "text/event-stream, application/json, text/plain, */*"
        return headers

    def build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the JSON body the site posts."""
        options = {key: kwargs.pop(key) for key in OPTION_KEYS if key in kwargs}
        target = str(kwargs.pop("model", None) or model or self.default_model)
        question, history = split_conversation(messages)
        question = str(options.get("question") or question)
        override = options.get("proof")
        proof: dict[str, Any] = dict(override) if isinstance(override, dict) else self.proof
        return {
            "timeZone": str(options.get("time_zone", self.time_zone)),
            "model": target.removeprefix(f"{self.name}/"),
            "question": question,
            "tone": str(options.get("tone", self.tone)),
            "format": str(options.get("format", self.output_format)),
            "file": options.get("file", self.file),
            "conversationHistory": self.trim_history(history, options.get("history")),
            "interactionProof": self.interaction_proof(question, proof),
            "resend": bool(options.get("resend", self.resend)),
            "aiRole": self.ai_role,
            "aiName": self.ai_name,
            "language": self.language,
            "followUps": bool(options.get("follow_ups", self.follow_ups)),
            **{key: value for key, value in kwargs.items() if value is not None},
        }

    def trim_history(
        self, history: list[dict[str, str]], override: Any
    ) -> list[dict[str, str]]:
        """Use an explicit history when given, else the last ``max_history`` turns."""
        if isinstance(override, list):
            history = [
                {"role": str(turn["role"]), "content": str(turn["content"])}
                for turn in override
                if isinstance(turn, dict) and "role" in turn and "content" in turn
            ]
        if self.max_history <= 0:
            return history
        return history[-self.max_history :]

    def interaction_proof(self, question: str, override: dict[str, Any]) -> dict[str, Any]:
        """Build the anti-bot fingerprint, honouring a captured proof."""
        submit_time = int(time.time() * 1000)
        typing_time = int(
            self.config.get("total_typing_time", 600 + self.typing_ms_per_char * len(question))
        )
        start_time = submit_time - typing_time
        proof: dict[str, Any] = {
            "nonce": self.nonce or mint_nonce(start_time),
            "keystrokeCount": len(question),
            "pasteEvents": 0,
            "totalTypingTime": typing_time,
            "startTime": start_time,
            "submitTime": submit_time,
        }
        proof.update(override)
        return proof

    # -- response parsing -----------------------------------------------------

    @staticmethod
    def _is_event_stream(response: httpx.Response) -> bool:
        return "event-stream" in (response.headers.get("content-type") or "")

    @staticmethod
    def _finish_reason(chunks: list[StreamChunk]) -> str:
        for chunk in reversed(chunks):
            if chunk.finish_reason:
                return chunk.finish_reason
        return "stop"

    def parse_body(self, response: httpx.Response, model: str) -> list[StreamChunk]:
        """Parse a fully buffered body as SSE frames, JSON, or raw text."""
        lines = (response.text or "").splitlines()
        framed = any(line.strip().startswith("data:") for line in lines)
        if framed or self._is_event_stream(response):
            chunks = [chunk for line in lines for chunk in self.parse_line(line, model)]
            if not any(chunk.finish_reason for chunk in chunks):
                chunks.append(
                    StreamChunk(delta="", finish_reason="stop", raw={"synthesized": True})
                )
            return chunks
        content = self.extract_text(response)
        if not content:
            return [StreamChunk(delta="", finish_reason="stop", raw={"empty": True})]
        return [StreamChunk(delta=content, finish_reason="stop", raw={"body": True})]

    def parse_line(self, line: str, model: str) -> list[StreamChunk]:
        """Turn one SSE line into zero or more chunks.

        Frames are JSON when the site behaves, but a plain-text ``data:`` line is
        treated as a delta rather than rejected -- the frame schema is not
        published.
        """
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return []
        data = stripped[len("data:") :].strip()
        if not data:
            return []
        if data == "[DONE]":
            return [StreamChunk(delta="", finish_reason="stop", raw={"done": True})]
        try:
            event = json.loads(data)
        except ValueError:
            if data[:1] in ("{", "["):
                raise StreamInterruptedError(
                    f"fai sent malformed SSE data: {data[:120]!r}",
                    provider=self.name,
                    model=model,
                ) from None
            return [StreamChunk(delta=data, raw={"text": data})]
        return self.parse_event(event, model)

    def parse_event(self, event: Any, model: str) -> list[StreamChunk]:
        """Extract the delta and finish reason from one decoded frame."""
        if not isinstance(event, dict):
            return [StreamChunk(delta=str(event), raw={"data": event})]
        if event.get("error"):
            message, code = self._describe_error(event["error"])
            raise ProviderError(f"{message} [code={code}]", provider=self.name, model=model)
        delta = event_text(event)
        finish_reason = event.get("finishReason") or event.get("finish_reason")
        chunks: list[StreamChunk] = []
        if delta:
            chunks.append(StreamChunk(delta=delta, raw=event))
        if finish_reason:
            chunks.append(StreamChunk(delta="", finish_reason=str(finish_reason), raw=event))
        return chunks

    @staticmethod
    def extract_text(response: httpx.Response) -> str:
        """Pull an answer out of a JSON or plain-text (non-SSE) body."""
        body = response.text or ""
        try:
            data = response.json()
        except ValueError:
            return body
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            text = event_text(data)
            if text:
                return text
        return body


def mint_nonce(start_time: int) -> str:
    """A nonce in the site's shape: ``<base36 ms>.<32 hex>.<40 hex>``.

    The timestamp sits a second or more before ``startTime``, mirroring a page
    that minted it on load rather than at submit.
    """
    return f"{base36(start_time - 1000 - (start_time % 58_000))}.{token_hex(16)}.{token_hex(20)}"


def split_conversation(messages: list[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    """Split turns into ``(question, history)`` the way the site does.

    The last user turn becomes ``question``; every other turn is replayed as
    ``conversationHistory``, including a trailing assistant turn.
    """
    question = ""
    index = -1
    for position, message in enumerate(messages):
        if message.role == "user":
            question, index = message.content, position
    history = [
        {"role": message.role, "content": message.content}
        for position, message in enumerate(messages)
        if position != index
    ]
    return question, history


def event_text(event: dict[str, Any]) -> str:
    """Return the first text-bearing field of a decoded frame."""
    for field in TEXT_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for key in ("delta", "message"):
                part = choice.get(key)
                if isinstance(part, dict) and isinstance(part.get("content"), str):
                    return str(part["content"])
            if isinstance(choice.get("text"), str):
                return str(choice["text"])
    nested = event.get("data")
    if isinstance(nested, dict):
        return event_text(nested)
    return ""


def parse_model_list(data: Any) -> list[str]:
    """Normalise the catalog payload into a sorted list of model ids."""
    items: Any = data
    if isinstance(data, dict):
        for key in LIST_KEYS:
            if isinstance(data.get(key), (list, dict)):
                items = data[key]
                break
    entries: Iterable[Any] = items if isinstance(items, list) else ()
    found: list[str] = []
    for item in entries:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, dict):
            for key in NAME_KEYS:
                value = item.get(key)
                if isinstance(value, str) and value:
                    found.append(value)
                    break
    if isinstance(items, dict):
        found.extend(str(key) for key in items)
    return sorted(set(found))


async def verify(model: str = DEFAULT_MODEL, question: str = "ping") -> dict[str, Any]:
    """Report the live catalog and one round trip, for manual re-checking."""
    provider = FaiProvider()
    report: dict[str, Any] = {"models": None, "answer": None, "error": None}
    try:
        report["models"] = await provider.fetch_models()
        response = await provider.chat(model, [ChatMessage("user", question)])
        report["answer"] = response.content[:200]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    """``python -m providers.fai`` -- print the catalog and one probe answer."""
    report = asyncio.run(verify())
    print(json.dumps(report, indent=2))
    return 0 if report["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
