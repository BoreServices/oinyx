"""OpenAI-shaped dataclasses shared by every provider plugin.

Every model here round-trips through :meth:`to_dict` / :meth:`from_dict` so
providers can serialize payloads without depending on external libraries.
Field names intentionally match the OpenAI chat-completions schema for
drop-in compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Role = str  # "system" | "user" | "assistant" | "tool" | ...


@dataclass
class ChatMessage:
    """A single turn in a conversation."""

    role: Role
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "") or ""),
            name=data.get("name"),
        )

    @classmethod
    def coerce(cls, value: ChatMessage | dict[str, Any] | str) -> ChatMessage:
        """Accept a ChatMessage, an OpenAI-shaped dict, or a bare string.

        A bare string becomes a ``user`` turn.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(role="user", content=value)
        if isinstance(value, dict):
            return cls.from_dict(value)
        raise TypeError(f"cannot coerce {type(value).__name__} into ChatMessage")


@dataclass
class Usage:
    """Token accounting for a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
        )


@dataclass
class ChatResponse:
    """A complete (non-streamed) chat result."""

    id: str = ""
    model: str = ""
    content: str = ""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        return cls(
            id=str(data.get("id", "")),
            model=str(data.get("model", "")),
            content=str(data.get("content", "")),
            finish_reason=data.get("finish_reason"),
            usage=Usage.from_dict(data.get("usage")),
            raw=dict(data.get("raw") or {}),
        )


@dataclass
class ImageResponse:
    """A generated image, either as a URL or inline base64.

    Exactly one of ``url`` / ``b64`` is usually populated; providers that hand
    back raw bytes should base64-encode them into ``b64`` with a ``mime_type``.
    """

    id: str = ""
    model: str = ""
    url: str | None = None
    b64: str | None = None
    mime_type: str | None = None
    revised_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return bool(self.url or self.b64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageResponse:
        return cls(
            id=str(data.get("id", "")),
            model=str(data.get("model", "")),
            url=data.get("url"),
            b64=data.get("b64"),
            mime_type=data.get("mime_type"),
            revised_prompt=data.get("revised_prompt"),
            width=data.get("width"),
            height=data.get("height"),
            raw=dict(data.get("raw") or {}),
        )


@dataclass
class StreamChunk:
    """One incremental piece of a streamed response.

    ``delta`` carries the newly generated text for this chunk; the final
    chunk typically has an empty ``delta`` and a non-None ``finish_reason``.
    """

    delta: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"delta": self.delta, "finish_reason": self.finish_reason, "raw": self.raw}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamChunk:
        return cls(
            delta=str(data.get("delta", "")),
            finish_reason=data.get("finish_reason"),
            raw=dict(data.get("raw") or {}),
        )
