"""``freeai`` -- the keyless self-hosted tier of ``api.free.ai``.

OpenAI-shaped and genuinely unauthenticated, but the catalog is misleading:
``/v1/models`` returns ~470 rows spanning chat, code, vision, image, video,
TTS, OCR and translation, and the vast majority of them are a *marketplace* of
third-party paid models that this endpoint will not serve for free. The rows
marked ``"self_hosted": true`` are the ones that answer, and they are all thin
aliases over a single deployed model -- every reply below came back with
``"model": "Qwen/Qwen3-30B-A3B-Instruct"``, whatever id was sent.

    >>> await client.chat("freeai/coder", "refactor this function")

A rejected id produces a deliberately helpful 400::

    {"error": "'moondream2' is not a model this endpoint serves. Use qwen7b
              (Qwen 3.6 35B, also handles vision), or deepseek-..."}

-- note the bare string under ``error`` rather than an OpenAI object, which the
relay base normalises. ``chat_types`` keeps :meth:`fetch_models` to the text
rows so the marketplace noise does not leak into listings.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://api.free.ai/v1"
DEFAULT_MODEL = "qwen7b"

#: Short alias -> the id sent upstream. All of them resolve to one Qwen deploy.
FREEAI_MODELS: dict[str, str] = {
    "chat": "qwen7b",
    "general": "mistral",
    "small": "qwen3-8b",
    "thinking": "deepseek-r1",
    "coder": "qwen-coder",
    "coder-large": "qwen3-coder",
    "vl": "qwen-vl",
    "vl25": "qwen25-vl",
}

#: Row types the chat endpoint serves, out of the ~470 the catalog lists.
CHAT_TYPES = ("chat", "code", "vision")


@register_provider("freeai")
class FreeAiProvider(OpenAiRelay):
    """oinyx provider for free.ai's self-hosted keyless models."""

    name = "freeai"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    catalog = FREEAI_MODELS

    def accept(self, item: Any) -> bool:
        """Keep self-hosted text rows; the rest of the catalog is paid or not chat.

        ``self_hosted`` is what separates the free tier from the marketplace
        listings, and ``type`` drops the image/video/TTS rows that share it.
        """
        if not isinstance(item, dict):
            return True
        return bool(item.get("self_hosted")) and str(item.get("type")) in CHAT_TYPES


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live catalog and every alias: ``python -m providers.freeai``."""
    provider = FreeAiProvider({"attempts": 1, "backoff": 0, **(config or {})})
    return relay_main(provider)


if __name__ == "__main__":
    raise SystemExit(main())
