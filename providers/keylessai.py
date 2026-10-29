"""``keylessai`` -- the alias-only Cloudflare Worker relay.

``https://keylessai.thryx.workers.dev/v1`` is a thin OpenAI-shaped shim that
exposes a fixed set of *pseudonymous* aliases -- ``gpt-4o``, ``gpt-4o-mini``,
``claude-3-5-sonnet-latest``, ``openai-fast`` -- rather than a real catalog.
The upstream behind each alias is whoever is currently serving it, so the names
are labels, not guarantees, and ``/v1/models`` returns the alias list itself.

Honesty note: this host did **not** resolve from the machine these adapters
were written on (``curl: (6) Could not resolve host``), so unlike the other
relays in this package the table below is transcribed from the service's own
documentation rather than verified against a live reply. Treat it as the most
likely to be dead of the set, and use :func:`verify` before relying on it.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://keylessai.thryx.workers.dev/v1"
DEFAULT_MODEL = "openai-fast"

#: Alias -> upstream label. Both sides are the same string: nothing to translate.
KEYLESSAI_MODELS: dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-latest",
    "openai-fast": "openai-fast",
}


@register_provider("keylessai")
class KeylessAiProvider(OpenAiRelay):
    """oinyx provider for the keylessai worker relay."""

    name = "keylessai"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    catalog = KEYLESSAI_MODELS


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live alias list: ``python -m providers.keylessai``."""
    provider = KeylessAiProvider({"attempts": 1, "backoff": 0, **(config or {})})
    return relay_main(provider)


if __name__ == "__main__":
    raise SystemExit(main())
