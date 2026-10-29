"""``pollinations`` -- the anonymous tier of ``text.pollinations.ai``.

OpenAI-shaped, no credential, one caveat: the anonymous tier is a single model.
``/models`` (which returns a bare JSON array, not the OpenAI envelope) lists
``openai-fast`` with ``"tier": "anonymous"`` and the aliases ``openai``,
``gpt-oss``, ``gpt-oss-20b``, ``ovh-reasoning`` -- the same GPT-OSS 20B behind
all four names. Everything else needs a *pollen key* and is not reachable here.

    >>> await client.chat("pollinations/fast", "hello")

When the shared anonymous budget is spent the endpoint replies **503 with a
JSON body**, not a plain error page::

    {"status": 503, "success": false,
     "error": {"code": "SERVICE_UNAVAILABLE",
               "message": "The Classic API is at its limit right now. Retry after 10s, ..."}}

which this adapter maps to :class:`~oinyx.RateLimitError` -- retryable, but the
honest answer is to move on to another relay. ``flux``/``turbo``/``gptimage``
are image models on a different host (``image.pollinations.ai``) and are not
chat-capable.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://text.pollinations.ai"
DEFAULT_MODEL = "openai-fast"

#: Every anonymous alias maps to the one model the keyless tier serves.
POLLINATIONS_MODELS: dict[str, str] = {
    "fast": "openai-fast",
    "openai": "openai-fast",
    "gpt-oss": "openai-fast",
    "gpt-oss-20b": "openai-fast",
    "ovh-reasoning": "openai-fast",
}


@register_provider("pollinations")
class PollinationsProvider(OpenAiRelay):
    """oinyx provider for Pollinations' keyless text endpoint."""

    name = "pollinations"
    base_url = BASE_URL
    chat_path = "/openai"
    models_path = "/models"
    default_model = DEFAULT_MODEL
    catalog = POLLINATIONS_MODELS


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live model list and the anonymous alias: ``python -m providers.pollinations``."""
    provider = PollinationsProvider({"attempts": 1, "backoff": 0, **(config or {})})
    # One call is enough: every alias in the table is the same upstream model.
    return relay_main(provider, {"fast": "openai-fast"})


if __name__ == "__main__":
    raise SystemExit(main())
