"""``ovh`` -- OVHcloud AI Endpoints, the keyless ``/v1`` surface.

Plain OpenAI: ``POST https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions``,
standard SSE, no ``Authorization`` header. Two things about the free tier are
easy to get wrong:

* **Model ids are Title-Case.** ``qwen3.5-9b`` 400s; ``Qwen3.5-9B`` works. The
  table below holds the exact ids read from ``/v1/models`` on 2026-10-29, and
  lowercase short aliases that map onto them.
* **The limit is 2 requests/minute *per model*, not per account.** Cycling
  models therefore raises throughput roughly proportionally, which is why this
  adapter exposes a dozen of them rather than one default.

Three further quirks, all verified against the live endpoint:

* reasoning models emit hundreds of ``{"role": "assistant", "content": null}``
  SSE frames before text; the base drops them.
* a small ``max_tokens`` can be consumed entirely by the reasoning block, so
  ``content`` comes back ``null`` with ``finish_reason: "length"`` -- the reply
  is in ``raw["reasoning"]``.
* ``Qwen3Guard-Gen-*`` is a moderation model (it classifies rather than chats)
  and none of the Qwen3Guard rows support ``tools`` or ``response_format``.

Vision ids (``Qwen2.5-VL-72B-Instruct``) are listed for discovery, but
: class:`oinyx.ChatMessage` carries a ``str`` body, so image content parts are
not reachable through the core yet.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
DEFAULT_MODEL = "Meta-Llama-3_3-70B-Instruct"

#: Short alias -> the exact upstream id (case matters).
OVH_MODELS: dict[str, str] = {
    "llama-70b": "Meta-Llama-3_3-70B-Instruct",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.6-27b": "Qwen3.6-27B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "qwen3.5-397b": "Qwen3.5-397B-A17B",
    "mistral-small": "Mistral-Small-3.2-24B-Instruct-2506",
    "mistral-nemo": "Mistral-Nemo-Instruct-2407",
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "qwen2.5-vl-72b": "Qwen2.5-VL-72B-Instruct",
    "guard-8b": "Qwen3Guard-Gen-8B",
    "guard-0.6b": "Qwen3Guard-Gen-0.6B",
}

#: The catalog also lists embeddings, STT, TTS, image and a reranker, none of
#: which answer on ``/chat/completions``: bge-m3, bge-multilingual-gemma2,
#: Qwen3-Embedding-8B, whisper-large-v3{,-turbo}, nvr-tts-*,
#: stable-diffusion-xl-base-v10.


@register_provider("ovh")
class OvhProvider(OpenAiRelay):
    """oinyx provider for OVHcloud's anonymous AI Endpoints tier."""

    name = "ovh"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    catalog = OVH_MODELS


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live catalog and every alias: ``python -m providers.ovh``."""
    provider = OvhProvider({"attempts": 1, "backoff": 0, **(config or {})})
    return relay_main(provider)


if __name__ == "__main__":
    raise SystemExit(main())
