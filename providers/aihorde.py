"""``aihorde`` -- crowd-sourced text workers behind ``oai.aihorde.net``.

The OpenAI-shaped front for the AI Horde: models are whatever someone's GPU is
currently offering, so the catalog is volatile and every id looks like a
runtime prefix (``koboldcpp/``, ``aphrodite/``, ``Text-generation-webui/``).

Two differences from the other relays matter:

* **It insists on an ``Authorization`` header even for anonymous use.** Without
  one: ``401 {"detail":"Authorization header missing"}``. Horde's convention is
  the literal anonymous key ``0000000000``, which :data:`ANONYMOUS_KEY` supplies
  by default -- pass ``config={"api_key": "<your horde key>"}`` to jump the
  queue, since priority follows the operator's kudos.
* **Replies are queued, not served.** The anonymous key sits at the back, so a
  completion can take a couple of minutes. The default timeout here is 300s for
  that reason, and a slow call is normal rather than an error.

There is no stable default model, so :data:`AIHORDE_MODELS` pins a few ids that
were live on 2026-10-29 as a starting point and :meth:`list_models` prefers the
live catalog.
"""

from __future__ import annotations

from typing import Any

from oinyx import register_provider

from .openai_relay import OpenAiRelay, relay_main

BASE_URL = "https://oai.aihorde.net/v1"
ANONYMOUS_KEY = "0000000000"
DEFAULT_MODEL = "koboldcpp/L3-8B-Stheno-v3.2"

#: Ids observed in the live catalog on 2026-10-29. Anything currently being
#: offered works too -- pass the raw id.
AIHORDE_MODELS: dict[str, str] = {
    "stheno": "koboldcpp/L3-8B-Stheno-v3.2",
    "magnum": "koboldcpp/mini-magnum-12b-v1.1",
    "qwen-coder": "koboldcpp/Qwen2.5-Coder-3B-Instruct",
    "cydonia": "koboldcpp/Cydonia-24B-v4.3-heretic-Q4_K_M",
    "behemoth": "aphrodite/TheDrummer/Behemoth-128B-v3-4.25bpw-h6-exl3",
}


@register_provider("aihorde")
class AIHordeProvider(OpenAiRelay):
    """oinyx provider for the AI Horde's OpenAI-compatible front."""

    name = "aihorde"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    catalog = AIHORDE_MODELS
    anonymous_key = ANONYMOUS_KEY
    #: One retry only: a queued request that timed out is still queued.
    default_attempts = 1
    default_timeout = 300.0

    async def list_models(self) -> list[str]:
        """Prefer the live worker catalog; fall back to the pinned aliases."""
        try:
            return sorted(await self.fetch_models())
        except Exception:  # noqa: BLE001 - horde workers come and go; never raise here
            return await super().list_models()


def main(config: dict[str, Any] | None = None) -> int:
    """Probe the live worker catalog and the pinned ids: ``python -m providers.aihorde``."""
    provider = AIHordeProvider({"attempts": 1, "backoff": 0, **(config or {})})
    return relay_main(provider, {"stheno": AIHORDE_MODELS["stheno"]})


if __name__ == "__main__":
    raise SystemExit(main())
