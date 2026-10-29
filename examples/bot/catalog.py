"""What each provider is, which models it will take, and how to autocomplete them."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from discord import app_commands

from oinyx import OinyxClient, ProviderRegistry
from providers.aihorde import AIHORDE_MODELS
from providers.alyx import KNOWN_MODELS as ALYX_MODELS
from providers.fable5io import KNOWN_MODELS as FABLE5_MODELS
from providers.fai import FaiProvider, KNOWN_MODELS as FAI_MODELS
from providers.freeai import FREEAI_MODELS
from providers.kai import KaiProvider
from providers.keylessai import KEYLESSAI_MODELS
from providers.llm7 import LLM7_MODELS
from providers.openai_relay import OpenAiRelay
from providers.ovh import OVH_MODELS
from providers.perchance import KNOWN_MODELS as PERCHANCE_MODELS
from providers.pollinations import POLLINATIONS_MODELS

#: Discord returns at most this many autocomplete choices.
MAX_CHOICES = 25
#: Choice labels are capped by Discord; provider names keep them readable.
MAX_LABEL = 100

#: Alias tables for the relays. ``kai`` reads its own, which config can override.
CATALOGS: dict[str, dict[str, str]] = {
    "llm7": LLM7_MODELS,
    "pollinations": POLLINATIONS_MODELS,
    "ovh": OVH_MODELS,
    "aihorde": AIHORDE_MODELS,
    "freeai": FREEAI_MODELS,
    "keylessai": KEYLESSAI_MODELS,
}

#: Adapters with no catalog route at all: the table below is the whole truth.
STATIC_MODELS: dict[str, tuple[str, ...]] = {
    "fable5": tuple(FABLE5_MODELS),
    "perchance": tuple(PERCHANCE_MODELS),
    "alyx": tuple(ALYX_MODELS),
    "fai": tuple(FAI_MODELS),
}


@dataclass(frozen=True)
class ProviderSpec:
    """Display metadata for one adapter, used by ``/<provider>_info``."""

    name: str
    kind: str
    summary: str
    traps: tuple[str, ...] = ()
    streaming: bool = True
    live_catalog: bool = True
    json_mode: bool = False
    history: str = "keeps per-channel turns"

    @property
    def is_image(self) -> bool:
        """Whether this adapter renders images instead of answering prompts."""
        return self.kind == "image"


SPECS: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec(
            name="fable5",
            kind="chat",
            summary="Free openai/gpt-4o behind fable5.io's site endpoint.",
            traps=(
                "replies are plain text, not OpenAI JSON",
                "the endpoint accepts at most 32 messages and 8000 characters each",
                "no catalog route: the model list is whatever the adapter pins",
            ),
            live_catalog=False,
        ),
        ProviderSpec(
            name="kai",
            kind="chat",
            summary="Kilo Code's free gateway models, unauthenticated.",
            traps=(
                "the free tier churns; most of the pinned aliases are already dead",
                "reasoning models spend max_tokens on reasoning_content first",
                "use /kai_models live to read the current zero-priced ids",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="perchance",
            kind="chat",
            summary="Perchance text generators, instruction-shaped.",
            traps=(
                "Cloudflare-gated: needs a browser cf_clearance cookie and matching user_agent",
                "the whole conversation is flattened into one instruction string",
                "the model name is a Perchance generator, not a model id",
                "streaming is emulated locally; one body arrives from upstream",
            ),
            live_catalog=False,
        ),
        ProviderSpec(
            name="fai",
            kind="chat",
            summary="aifreeforever.com's site API (Gemini and Claude proxies).",
            traps=(
                "Cloudflare-gated and needs a captured interactionProof plus cf_clearance",
                "the Referer header must match the model slug of each call",
                "answers arrive as text/event-stream frames behind a synthetic typing delay",
            ),
            json_mode=False,
        ),
        ProviderSpec(
            name="llm7",
            kind="chat",
            summary="OpenAI-compatible relay; keyless access is tier-gated.",
            traps=(
                "only catalog rows with tier=turbo answer without a key; pro ids return 401",
                "quota is per IP and shows up as 429 with a retry-after in prose",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="pollinations",
            kind="chat",
            summary="text.pollinations.ai's OpenAI route.",
            traps=(
                "the anonymous tier is exactly one chat model behind five aliases",
                "over quota it answers 503 with a JSON body instead of a stream",
                "flux, turbo, and gptimage are image models and will not chat",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="ovh",
            kind="chat",
            summary="OVHcloud's free OpenAI-compatible endpoints.",
            traps=(
                "the free limit is 2 requests/minute per model, so cycling models is the only way up",
                "ids are Title-Case (Qwen3.5-9B); the lowercase alias table maps them",
                "reasoning models can reply with content=null when the budget goes to reasoning",
                "Qwen3Guard-* are classifiers, not chat models",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="aihorde",
            kind="chat",
            summary="AI Horde's OpenAI-compatible front over volunteer workers.",
            traps=(
                "requests carry Authorization: Bearer 0000000000, which the adapter adds",
                "answers queue behind other users; nine to thirty-eight seconds was normal while probing",
                "worker deployments come and go, so the catalog is volatile",
                "the adapter uses one attempt and a 300s timeout because a timed-out request stays queued",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="freeai",
            kind="chat",
            summary="api.free.ai's OpenAI-compatible route.",
            traps=(
                "its ~469-row catalog is mostly a paid marketplace; only self_hosted rows answer",
                "the self-hosted ids all resolve to the same Qwen deployment",
                "moondream2 returns 400 despite being listed",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="keylessai",
            kind="chat",
            summary="keylessai.thryx.workers.dev, an alias-only relay.",
            traps=(
                "the host did not resolve from the machine this library was written on",
                "the alias table is therefore unverified, not confirmed dead",
            ),
            json_mode=True,
        ),
        ProviderSpec(
            name="alyx",
            kind="image",
            summary="Perchance image generators, behind the same Cloudflare session.",
            traps=(
                "Cloudflare-gated: same cf_clearance and user_agent requirements",
                "the model name is a generator; the result field differs per generator",
                "image responses carry a url or inline base64, never text",
            ),
            streaming=False,
            live_catalog=False,
            history="stores the last prompt per channel for /alyx_retry",
        ),
    )
}


def spec_for(name: str) -> ProviderSpec:
    """Return the display metadata for ``name``."""
    return SPECS[name]


def provider_instance(name: str, config: dict[str, Any] | None = None) -> Any:
    """Fetch the registered instance for ``name``, applying ``config`` once."""
    if config:
        ProviderRegistry.configure(name, config)
    return ProviderRegistry.get(name)


def default_model(name: str) -> str:
    """The model this adapter serves when a command does not name one.

    Handlers build ``<provider>/<model>``, and an empty model segment is a routing
    error, so every command that omits ``model`` needs this. Kilo spells its
    default with the provider prefix, which the router would double up -- strip it.
    """
    value = str(getattr(provider_instance(name), "default_model", "") or "")
    prefix = f"{name}/"
    return value[len(prefix):] if value.startswith(prefix) else value


def endpoint_of(instance: Any) -> str:
    """Best-effort upstream URL for an adapter instance."""
    for attribute in ("chat_url", "endpoint", "base_url"):
        value = getattr(instance, attribute, "")
        if isinstance(value, str) and value:
            return value
    return "unknown endpoint"


def offline_models(name: str, instance: Any) -> list[tuple[str, str]]:
    """Return ``(value, label)`` pairs the adapter accepts without a network call."""
    if name == "kai":
        aliases = sorted(getattr(instance, "aliases", {}))
        return [(alias.split("/", 1)[-1], alias) for alias in aliases]
    if name in CATALOGS:
        catalog = dict(getattr(instance, "catalog", {})) or CATALOGS[name]
        return [(alias, f"{alias} -> {upstream}") for alias, upstream in catalog.items()]
    if name in STATIC_MODELS:
        models = list(getattr(instance, "_models", ())) or list(STATIC_MODELS[name])
        label = "generator" if spec_for(name).is_image else "known id"
        return [(model, label) for model in models]
    return []


class CatalogCache:
    """TTL cache over the alias table and the live catalog of each adapter.

    Autocomplete fires on every keystroke, so nothing here may raise or block:
    a failed live read simply leaves the pinned aliases in place.
    """

    def __init__(self, client: OinyxClient, configs: dict[str, dict[str, Any]], ttl: float = 300.0) -> None:
        self.client = client
        self.configs = configs
        self.ttl = ttl
        self._entries: dict[tuple[str, bool], tuple[float, list[tuple[str, str]]]] = {}

    async def models(self, name: str, live: bool = False) -> list[tuple[str, str]]:
        """Return cached ``(value, label)`` pairs for ``name``.

        ``live`` also merges the upstream catalog when the adapter publishes one;
        the alias table is always present so a dead cache never hides a working
        shortcut.
        """
        key = (name, live)
        cached = self._entries.get(key)
        if cached and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        instance = provider_instance(name, self.configs.get(name))
        pairs = offline_models(name, instance)
        if live:
            pairs = pairs + await self._fetch_live(name, instance)
        merged: dict[str, str] = {}
        for value, label in pairs:
            merged.setdefault(value, label)
        result = sorted(merged.items())
        self._entries[key] = (time.monotonic(), result)
        return result

    async def _fetch_live(self, name: str, instance: Any) -> list[tuple[str, str]]:
        """Await the adapter's catalog route, swallowing every failure."""
        if not spec_for(name).live_catalog:
            return []
        try:
            if isinstance(instance, KaiProvider):
                ids = await instance.discover_models()
            elif isinstance(instance, (OpenAiRelay, FaiProvider)):
                ids = await instance.fetch_models()
            else:
                ids = await instance.list_models()
        except Exception:  # noqa: BLE001 - autocomplete must not surface upstream errors
            return []
        return [(str(ident), f"upstream: {ident}") for ident in ids]

    def invalidate(self, name: str | None = None) -> None:
        """Drop cached entries for one provider, or all of them."""
        for key in list(self._entries):
            if name is None or key[0] == name:
                del self._entries[key]

    async def values(self, name: str, live: bool = False) -> list[str]:
        """Return just the model values for ``name``."""
        return [value for value, _ in await self.models(name, live)]


def choices(models: list[tuple[str, str]], current: str) -> list[app_commands.Choice[str]]:
    """Filter and cap ``models`` into Discord autocomplete choices."""
    needle = (current or "").lower()
    picked = [
        app_commands.Choice(name=f"{value} ({label})"[:MAX_LABEL], value=value)
        for value, label in models
        if not needle or needle in value.lower() or needle in label.lower()
    ]
    return picked[:MAX_CHOICES]
