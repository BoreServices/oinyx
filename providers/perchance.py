"""``perchance`` -- text generation through ``text-generation.perchance.org``.

The upstream API is *instruction-shaped*, not message-shaped: it takes one free
``instruction`` string plus generation hints and returns the completion::

    {"instruction": "test ", "startWith": "", "stopSequences": [],
     "generatorName": "ai-code-generator",
     "startWithTokenCount": 0, "instructionTokenCount": 2}

:class:`PerchanceProvider` flattens a oinyx conversation into that instruction,
so normal chat calls work::

    import providers.perchance
    reply = await OinyxClient().chat("perchance/ai-chat", "write a haiku about rain")

Note that Perchance's endpoint is behind a Cloudflare managed challenge: see
:mod:`providers.cloudflare_base` for how to supply a browser session through
``config["cookies"]``. Streaming is emulated locally because the endpoint
returns the whole completion in one body.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from oinyx import (
    ChatMessage,
    ChatResponse,
    RateLimitError,
    StreamChunk,
    Usage,
    estimate_tokens,
    register_provider,
    retry,
)

from .cloudflare_base import CloudflareBase, UpstreamError

ENDPOINT = "https://text-generation.perchance.org/api/generate"
ORIGIN = "https://perchance.org"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
DEFAULT_MODEL = "ai-chat"

#: Models are Perchance *generator names*; override with ``config["models"]``.
KNOWN_MODELS = ["ai-chat", "ai-code-generator", "ai-story-writer", "ai-image-describer"]

#: Response keys tried in order when the payload is JSON.
TEXT_FIELDS = ("generatedText", "text", "content", "completion", "result", "output", "message")

_WORD_SPLIT = re.compile(r"(\S+\s*)")


@register_provider("perchance")
class PerchanceProvider(CloudflareBase):
    """oinyx provider for Perchance's free text-generation endpoint."""

    name = "perchance"
    endpoint = ENDPOINT
    origin = ORIGIN
    user_agent = DEFAULT_USER_AGENT

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.default_generator: str = str(self.config.get("generator_name", DEFAULT_MODEL))
        self.label_roles: bool = bool(self.config.get("label_roles", True))
        self.chunk_words: int = max(1, int(self.config.get("stream_chunk_words", 4)))
        self._models: list[str] = list(self.config.get("models") or KNOWN_MODELS)
        self._post = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self.post)

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Send the conversation as one instruction and return the completion."""
        payload = self.build_payload(model, messages, kwargs)
        response = await self._post(payload, payload["generatorName"])
        text = self.extract_text(response)
        prompt_tokens = estimate_tokens(payload["instruction"])
        completion_tokens = estimate_tokens(text)
        return ChatResponse(
            id=f"perchance-{response.status_code}",
            model=payload["generatorName"],
            content=text,
            finish_reason="stop",
            usage=Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            raw={
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "instruction_tokens": payload["instructionTokenCount"],
            },
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Re-emit the completion in word groups.

        Perchance returns the entire text in one body, so these chunks are
        local framings of an already-finished response rather than a true
        server stream.
        """
        response = await self.chat(model, messages, **kwargs)
        words = [piece for piece in _WORD_SPLIT.findall(response.content) if piece]
        for start in range(0, len(words), self.chunk_words):
            yield StreamChunk(delta="".join(words[start : start + self.chunk_words]).rstrip())
        yield StreamChunk(delta="", finish_reason="stop", raw={"emulated": True})

    async def list_models(self) -> list[str]:
        """Return the configured Perchance generator names."""
        return list(self._models)

    def build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the exact JSON shape the endpoint documents."""
        instruction = self.render_instruction(messages)
        generator = str(kwargs.pop("generatorName", None) or model or self.default_generator)
        start_with = str(kwargs.pop("startWith", self.config.get("start_with", "")))
        stop_sequences = list(
            kwargs.pop("stopSequences", self.config.get("stop_sequences") or [])
        )
        payload: dict[str, Any] = {
            "instruction": instruction,
            "startWith": start_with,
            "stopSequences": stop_sequences,
            "generatorName": generator,
            "startWithTokenCount": int(kwargs.pop("startWithTokenCount", 0) or 0),
            "instructionTokenCount": int(
                kwargs.pop("instructionTokenCount", None) or estimate_tokens(instruction)
            ),
        }
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return payload

    def render_instruction(self, messages: list[ChatMessage]) -> str:
        """Flatten turns into one instruction string.

        With ``label_roles`` on, each turn is prefixed with its role so the
        model can tell a system prompt from a user request; otherwise turns are
        joined verbatim.
        """
        if not messages:
            return ""
        if not self.label_roles:
            return "\n".join(message.content for message in messages)
        lines = [f"{message.role.capitalize()}: {message.content}" for message in messages]
        if messages[-1].role != "assistant":
            lines.append("Assistant:")
        return "\n".join(lines) + "\n"

    @staticmethod
    def extract_text(response: httpx.Response) -> str:
        """Pull the completion out of a JSON or plain-text body."""
        body = response.text or ""
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type and not body.lstrip().startswith(("{", "[")):
            return body
        try:
            data = response.json()
        except ValueError:
            return body
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for field in TEXT_FIELDS:
                value = data.get(field)
                if isinstance(value, str) and value:
                    return value
        return body
