"""``llm7`` -- the keyless tier of ``api.llm7.io``.

A plain OpenAI surface (``/v1/chat/completions``, real SSE, ``usage`` in the
reply) that gates by *tier* rather than by path: catalog rows carry
``"tier": "turbo"`` or ``"tier": "pro"``, and everything in ``pro`` answers

    401 {"error":{"message":"Missing API key.","code":"missing_api_key"}}

so :data:`LLM7_MODELS` lists the turbo ids that were verified working without a
key on 2026-10-29. Pro ids still pass through verbatim -- set
``config={"api_key": ...}`` and they work like any other OpenAI endpoint.

    >>> await client.chat("llm7/codestral", "write a bubble sort")

Three catalog oddities worth knowing: ``minimax-m2.7`` reports
``"stream": false`` and shares a hot pool (it answers ``429 Rate limit
exceeded. Retry after 3183 seconds.`` more often than the others), and
``"reasoning_content"`` on some models means a small ``max_tokens`` can buy you
an empty ``content`` with a finished reasoning block instead.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://api.llm7.io/v1"
DEFAULT_MODEL = "codestral-latest"
KEYLESS_TIER = "turbo"

#: Keyless (``tier: turbo``) chat ids, under short aliases.
LLM7_MODELS: dict[str, str] = {
    "codestral": "codestral-latest",
    "nemo": "mistral-Nemo-Instruct-2407",
    "nemotron": "nemotron-3-nano:30b",
    "gpt-oss": "gpt-oss:20b",
    "minimax": "minimax-m2.7",
}

#: Ids the catalog lists behind an API key; kept for documentation, not routed.
PAID_MODELS = (
    "glm-5.3",
    "glm-5.2",
    "deepseek-v4-pro",
    "deepseek-v4-flash:0731",
    "minimax-m3",
    "gemma4:31b",
    "kimi-k3",
    "mistral-large-3:675b",
)


@register_provider("llm7")
class Llm7Provider(OpenAiRelay):
    """oinyx provider for the free turbo tier of LLM7."""

    name = "llm7"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    catalog = LLM7_MODELS

    def accept(self, item: Any) -> bool:
        """Keep keyless chat rows: ``tier == "turbo"`` and ``model_type == "chat"``."""
        if not isinstance(item, dict):
            return True
        return str(item.get("tier")) == KEYLESS_TIER and str(item.get("model_type")) == "chat"


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live catalog and every turbo alias: ``python -m providers.llm7``."""
    return relay_main(Llm7Provider({"attempts": 1, "backoff": 0, **(config or {})}))


if __name__ == "__main__":
    raise SystemExit(main())
