"""Turning oinyx results into Discord messages: chunking, meta lines, attachments."""

from __future__ import annotations

import base64
import io
import re
from typing import Any

import discord

from oinyx import ChatResponse, ImageResponse, OinyxError

#: Discord's message limit is 2000; leave room for the meta line.
CHUNK_LIMIT = 1900
#: Longest single word we will try to keep unbroken when splitting.
WORD_GUARD = 400

_MIME_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_FENCE = re.compile(r"```")


def chunks(text: str, size: int = CHUNK_LIMIT) -> list[str]:
    """Split ``text`` into message-sized pieces, preferring line boundaries."""
    if not text:
        return []
    pieces: list[str] = []
    rest = text
    while len(rest) > size:
        cut = rest.rfind("\n", 0, size)
        if cut < size // 2:
            cut = rest.rfind(" ", 0, size)
        if cut < size // 2:
            cut = size
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        pieces.append(rest)
    # A hard cut can split a code fence; close and reopen so Discord stays sane.
    fixed: list[str] = []
    for piece in pieces:
        if len(_FENCE.findall(piece)) % 2:
            piece = piece + "\n```"
            if not piece.endswith("```"):
                piece += "\n"
        fixed.append(piece[:size])
    return fixed


def fence(text: str, language: str = "") -> str:
    """Wrap ``text`` in a code fence that survives backticks inside the body."""
    marks = "````" if _FENCE.search(text) else "```"
    return f"{marks}{language}\n{text}\n{marks}"


def meta(
    provider: str,
    model: str,
    *,
    served: str | None = None,
    elapsed_ms: int | None = None,
    usage: tuple[int, int] | None = None,
    finish: str | None = None,
    extra: str = "",
) -> str:
    """Build the one-line provenance stamp above every answer."""
    parts = [f"`{provider}/{model}`"]
    if served and served != model:
        parts.append(f"served `{served}`")
    if elapsed_ms is not None:
        parts.append(f"{elapsed_ms}ms")
    if usage:
        parts.append(f"{usage[0]}→{usage[1]} tok")
    if finish and finish != "stop":
        parts.append(f"_{finish}_")
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def describe_response(response: ChatResponse, provider: str, elapsed_ms: int | None = None) -> str:
    """Return the meta line for a completion, including whatever it hid."""
    usage = response.usage
    requested = str(response.raw.get("requested_model") or response.model or "")
    served = str(response.raw.get("model") or requested)
    reasoning = str(response.raw.get("reasoning") or "")
    extra_bits: list[str] = []
    if reasoning.strip():
        extra_bits.append(f"{len(reasoning)} chars of reasoning")
    if not response.content.strip():
        extra_bits.append("empty answer")
    return meta(
        provider,
        requested or response.model,
        served=served,
        elapsed_ms=elapsed_ms,
        usage=(usage.prompt_tokens, usage.completion_tokens),
        finish=response.finish_reason,
        extra=", ".join(extra_bits),
    )


def error_text(exc: Exception) -> str:
    """Render any exception as one Discord-safe line of blame."""
    if isinstance(exc, OinyxError):
        provider = getattr(exc, "provider", None) or "?"
        status = getattr(exc, "status_code", None)
        retry_after = getattr(exc, "retry_after", None)
        bits = [f"**{type(exc).__name__}** `{provider}`"]
        if status:
            bits.append(f"HTTP {status}")
        if retry_after:
            bits.append(f"retry in {int(retry_after)}s")
        message = str(exc.message).replace("\n", " ")
        return f"{' · '.join(bits)}: {message[:900]}"
    return f"**{type(exc).__name__}**: {str(exc).replace(chr(10), ' ')[:900]}"


def image_attachment(image: ImageResponse, label: str) -> discord.File | None:
    """Return an attachment for inline image bytes, or ``None`` for a hosted url."""
    if not image.b64:
        return None
    extension = _MIME_EXT.get(image.mime_type or "", "png")
    try:
        payload = base64.b64decode(image.b64)
    except Exception:  # noqa: BLE001 - a malformed data URI is the provider's problem
        return None
    safe = re.sub(r"[^a-z0-9_-]+", "-", label.lower())[:40] or "image"
    return discord.File(io.BytesIO(payload), filename=f"{safe}.{extension}")


def image_reference(image: ImageResponse) -> str:
    """Describe where the pixels came from."""
    if image.url:
        return image.url
    return f"inline {image.mime_type} · {len(image.b64 or '')} chars base64"


def size_of(image: ImageResponse) -> str:
    """``WxH`` when the generator reported dimensions, else a readable placeholder."""
    if image.width and image.height:
        return f"{image.width}x{image.height}"
    return "size not reported"


def image_meta(image: ImageResponse, elapsed_ms: int | None) -> str:
    """Build the caption line for a rendered image."""
    bits = [f"`{image.model}`"]
    if image.width and image.height:
        bits.append(f"{image.width}x{image.height}")
    if elapsed_ms is not None:
        bits.append(f"{elapsed_ms}ms")
    if image.revised_prompt:
        bits.append("revised prompt")
    return " · ".join(bits)


def table(rows: list[tuple[str, str]], width: int = 40) -> str:
    """Render aligned ``name -> detail`` rows in a code block."""
    if not rows:
        return "```\n(nothing to show)```"
    names, details = zip(*rows)
    padding = max(len(name) for name in names)
    lines = [f"{name.ljust(padding)}  {truncate(detail, width)}" for name, detail in zip(names, details)]
    return "```text\n" + "\n".join(lines) + "\n```"


def truncate(value: Any, limit: int = 120) -> str:
    """Collapse whitespace and cut ``value`` to a single readable line."""
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def bullet_list(items: list[str], limit: int = 25) -> str:
    """Format up to ``limit`` bullets, noting how many were dropped."""
    shown = items[:limit]
    body = "\n".join(f"- {item}" for item in shown) or "- nothing"
    hidden = len(items) - len(shown)
    if hidden > 0:
        body += f"\n- …{hidden} more"
    return body
