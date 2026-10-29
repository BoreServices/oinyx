"""Offline tests for the six thin relay adapters.

The transport behaviour lives in :mod:`providers.openai_relay` and is covered
there; what matters here is that each subclass points at the right host, ships
the catalog we verified against the live service, and completes a round trip
with the payload shape that service actually accepts.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from oinyx import ChatMessage, OinyxClient, ProviderRegistry
from providers.aihorde import AIHORDE_MODELS, ANONYMOUS_KEY, AIHordeProvider
from providers.freeai import CHAT_TYPES, FREEAI_MODELS, FreeAiProvider
from providers.keylessai import KEYLESSAI_MODELS, KeylessAiProvider
from providers.llm7 import KEYLESS_TIER, LLM7_MODELS, PAID_MODELS, Llm7Provider
from providers.openai_relay import OpenAiRelay
from providers.ovh import OVH_MODELS, OvhProvider
from providers import stillworks
from providers.pollinations import POLLINATIONS_MODELS, PollinationsProvider

PROVIDERS: dict[str, type[OpenAiRelay]] = {
    "llm7": Llm7Provider,
    "ovh": OvhProvider,
    "pollinations": PollinationsProvider,
    "freeai": FreeAiProvider,
    "aihorde": AIHordeProvider,
    "keylessai": KeylessAiProvider,
}

CATALOGS: dict[str, dict[str, str]] = {
    "llm7": LLM7_MODELS,
    "ovh": OVH_MODELS,
    "pollinations": POLLINATIONS_MODELS,
    "freeai": FREEAI_MODELS,
    "aihorde": AIHORDE_MODELS,
    "keylessai": KEYLESSAI_MODELS,
}

#: Expected host and chat path, exactly as probed on 2026-10-29.
HOSTS: dict[str, tuple[str, str]] = {
    "llm7": ("https://api.llm7.io/v1", "/chat/completions"),
    "ovh": ("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "/chat/completions"),
    "pollinations": ("https://text.pollinations.ai", "/openai"),
    "freeai": ("https://api.free.ai/v1", "/chat/completions"),
    "aihorde": ("https://oai.aihorde.net/v1", "/chat/completions"),
    "keylessai": ("https://keylessai.thryx.workers.dev/v1", "/chat/completions"),
}


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://relay.test/"),
    )


def completion(content: str = "pong", model: str = "upstream-id") -> dict[str, Any]:
    return {
        "id": "chatcmpl-9",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def make(cls: type[OpenAiRelay], handler: Callable[[httpx.Request], httpx.Response]) -> OpenAiRelay:
    return cls({"transport": httpx.MockTransport(handler), "backoff": 0, "attempts": 1})


@pytest.fixture(autouse=True)
def registered() -> None:
    for cls in PROVIDERS.values():
        ProviderRegistry.register(cls)


class TestRegistration:
    def test_every_relay_is_registered_on_import(self) -> None:
        listing = ProviderRegistry.list_providers()
        for name in PROVIDERS:
            assert name in listing

    def test_adapters_do_not_collide_with_the_reverse_engineered_ones(self) -> None:
        assert set(PROVIDERS).isdisjoint({"kai", "fai", "fable5", "perchance", "alyx"})


@pytest.mark.parametrize("name", sorted(PROVIDERS))
class TestAdapterShape:
    def test_hosts_match_the_probed_endpoints(self, name: str) -> None:
        provider = PROVIDERS[name]()
        base, chat_path = HOSTS[name]
        assert provider.base_url == base
        assert provider.chat_url == base + chat_path
        assert provider.endpoint == provider.chat_url

    def test_default_model_is_known(self, name: str) -> None:
        provider = PROVIDERS[name]()
        assert provider.default_model
        assert provider.resolve_model(provider.default_model)

    async def test_list_models_prefers_the_catalog_when_it_is_volatile(self, name: str) -> None:
        cls = PROVIDERS[name]
        provider = cls(
            {
                "transport": httpx.MockTransport(
                    lambda request: json_response({"data": [{"id": "live-id"}]})
                )
            }
        )
        listing = await provider.list_models()
        if name == "aihorde":
            # Workers come and go, so this adapter always asks the live catalog.
            assert listing == ["live-id"]
        else:
            assert listing == sorted(f"{name}/{alias}" for alias in cls.catalog)
            assert "live-id" not in listing

    def test_catalog_aliases_resolve_to_live_ids(self, name: str) -> None:
        provider = PROVIDERS[name]()
        assert provider.catalog == CATALOGS[name]
        for alias, upstream in CATALOGS[name].items():
            assert provider.resolve_model(alias) == upstream
            assert provider.resolve_model(f"{name}/{alias}") == upstream

    async def test_a_round_trip_reaches_the_expected_url(self, name: str) -> None:
        cls = PROVIDERS[name]
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response(completion())

        alias = next(iter(cls.catalog))
        response = await make(cls, handler).chat(alias, [ChatMessage("user", "ping")])
        assert response.content == "pong"
        assert response.usage.total_tokens == 7
        assert len(seen) == 1
        base, chat_path = HOSTS[name]
        assert str(seen[0].url) == base + chat_path
        body = json.loads(seen[0].content)
        assert body["messages"] == [{"role": "user", "content": "ping"}]
        assert body["model"] == cls.catalog[alias]


class TestAdapterQuirks:
    def test_aihorde_sends_the_anonymous_bearer_token(self) -> None:
        provider = AIHordeProvider()
        assert provider.headers["authorization"] == f"Bearer {ANONYMOUS_KEY}"
        assert provider.api_key == ANONYMOUS_KEY

    def test_aihorde_allows_a_real_key_to_jump_the_queue(self) -> None:
        assert AIHordeProvider({"api_key": "my-key"}).headers["authorization"] == "Bearer my-key"

    def test_aihorde_waits_long_enough_for_the_queue(self) -> None:
        provider = AIHordeProvider()
        assert provider.timeout == pytest.approx(300.0)
        assert provider.attempts == 1

    async def test_aihorde_lists_live_workers(self) -> None:
        provider = make(
            AIHordeProvider,
            lambda request: json_response({"data": [{"id": "koboldcpp/New-Model-7B"}]}),
        )
        assert await provider.list_models() == ["koboldcpp/New-Model-7B"]

    async def test_aihorde_falls_back_to_pinned_ids_when_the_catalog_is_down(self) -> None:
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        offline = make(AIHordeProvider, broken)
        assert all(model.startswith("aihorde/") for model in await offline.list_models())

    def test_pollinations_chat_path_is_not_the_openai_default(self) -> None:
        assert PollinationsProvider().chat_url == "https://text.pollinations.ai/openai"

    def test_pollinations_aliases_all_collapse_to_one_model(self) -> None:
        provider = PollinationsProvider()
        assert {provider.resolve_model(alias) for alias in provider.catalog} == {"openai-fast"}

    def test_ovh_catalog_ids_keep_their_case(self) -> None:
        provider = OvhProvider()
        assert provider.resolve_model("llama-70b") == "Meta-Llama-3_3-70B-Instruct"
        assert provider.resolve_model("Meta-Llama-3_3-70B-Instruct") == "Meta-Llama-3_3-70B-Instruct"
        for upstream in provider.catalog.values():
            assert upstream == upstream.strip()

    def test_ovh_can_be_told_to_cycle_models(self) -> None:
        provider = OvhProvider({"model": "Qwen3.5-9B"})
        assert provider.resolve_model("") == "Qwen3.5-9B"

    def test_llm7_accepts_only_the_keyless_tier(self) -> None:
        provider = Llm7Provider()
        assert provider.accept({"id": "a", "tier": KEYLESS_TIER, "model_type": "chat"})
        assert not provider.accept({"id": "b", "tier": "pro", "model_type": "chat"})
        assert not provider.accept({"id": "c", "tier": KEYLESS_TIER, "model_type": "image"})
        assert not provider.accept({"id": "d"})

    def test_freeai_accepts_only_self_hosted_text_rows(self) -> None:
        provider = FreeAiProvider()
        assert provider.accept({"id": "qwen7b", "type": "chat", "self_hosted": True})
        assert not provider.accept({"id": "x", "type": "chat", "self_hosted": False})
        assert not provider.accept({"id": "sdxl", "type": "image", "self_hosted": True})
        assert set(CHAT_TYPES) == {"chat", "code", "vision"}

    def test_paid_llm7_ids_are_documented_but_not_routed(self) -> None:
        assert "glm-5.3" in PAID_MODELS
        assert "glm-5.3" not in LLM7_MODELS.values()

    def test_keylessai_aliases_are_identity_mappings(self) -> None:
        provider = KeylessAiProvider()
        assert all(alias == upstream for alias, upstream in KEYLESSAI_MODELS.items())
        assert provider.resolve_model("gpt-4o") == "gpt-4o"


class TestCatalogParsing:
    async def test_llm7_filters_the_live_envelope(self) -> None:
        provider = make(
            Llm7Provider,
            lambda request: json_response(
                {
                    "object": "list",
                    "data": [
                        {"id": "codestral-latest", "tier": "turbo", "model_type": "chat"},
                        {"id": "glm-5.3", "tier": "pro", "model_type": "chat"},
                        {"id": "gpt-image-2", "tier": "pro", "model_type": "image"},
                    ],
                }
            ),
        )
        assert await provider.fetch_models() == ["codestral-latest"]

    async def test_pollinations_reads_a_bare_array(self) -> None:
        provider = make(
            PollinationsProvider,
            lambda request: json_response(
                [
                    {"name": "openai-fast", "tier": "anonymous", "aliases": ["openai"]},
                    {"name": "flux", "tier": "anonymous"},
                ]
            ),
        )
        assert await provider.fetch_models() == ["openai-fast", "flux"]

    async def test_freeai_filters_the_marketplace(self) -> None:
        provider = make(
            FreeAiProvider,
            lambda request: json_response(
                {
                    "models": [
                        {"id": "qwen7b", "type": "chat", "self_hosted": True},
                        {"id": "sdxl", "type": "image", "self_hosted": True},
                        {"id": "openai/gpt-6.1-terra", "type": "chat", "self_hosted": False},
                    ]
                }
            ),
        )
        assert await provider.fetch_models() == ["qwen7b"]

    async def test_ovh_keeps_every_id_it_lists(self) -> None:
        provider = make(
            OvhProvider,
            lambda request: json_response({"data": [{"id": "Qwen3.5-9B"}, {"id": "bge-m3"}]}),
        )
        assert await provider.fetch_models() == ["Qwen3.5-9B", "bge-m3"]


class TestClientRouting:
    async def test_router_reaches_a_relay(self) -> None:
        client = OinyxClient(
            config={"ovh": {"transport": httpx.MockTransport(lambda request: json_response(completion("routed")))}}
        )
        response = await client.chat("ovh/llama-70b", "hi")
        assert response.content == "routed"
        assert response.model == "upstream-id"

    async def test_raw_upstream_ids_route_too(self) -> None:
        client = OinyxClient(
            config={
                "llm7": {
                    "transport": httpx.MockTransport(lambda request: json_response(completion("x")))
                }
            }
        )
        response = await client.chat("llm7/mistral-Nemo-Instruct-2407", "hi")
        assert response.content == "x"


class TestAdapterMains:
    """Each adapter's ``python -m`` entry point must build its own provider."""

    @pytest.mark.parametrize("name", sorted(PROVIDERS))
    def test_main_probes_its_own_adapter(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[OpenAiRelay] = []

        def fake_relay_main(provider: OpenAiRelay, models: dict[str, str] | None = None) -> int:
            captured.append(provider)
            return 0

        module = importlib.import_module(f"providers.{name}")
        monkeypatch.setattr(module, "relay_main", fake_relay_main)
        assert module.main() == 0
        assert type(captured[0]) is PROVIDERS[name]
        # The sweeps must not sit in a retry loop waiting on a dead relay.
        assert captured[0].attempts == 1


class TestCatalogTables:
    """Pin the ids verified against the live services on 2026-10-29."""

    def test_llm7_turbo_set(self) -> None:
        assert set(LLM7_MODELS.values()) == {
            "codestral-latest",
            "mistral-Nemo-Instruct-2407",
            "nemotron-3-nano:30b",
            "gpt-oss:20b",
            "minimax-m2.7",
        }

    def test_ovh_chat_models(self) -> None:
        assert set(OVH_MODELS.values()) == {
            "Meta-Llama-3_3-70B-Instruct",
            "Qwen3.5-9B",
            "Qwen3.6-27B",
            "Qwen3.8-27B",
            "Qwen3.5-397B-A17B",
            "Mistral-Small-3.2-24B-Instruct-2506",
            "Mistral-Nemo-Instruct-2407",
            "gpt-oss-20b",
            "gpt-oss-120b",
            "Qwen2.5-VL-72B-Instruct",
            "Qwen3Guard-Gen-8B",
            "Qwen3Guard-Gen-0.6B",
        }

    def test_freeai_self_hosted_subset(self) -> None:
        assert set(FREEAI_MODELS.values()) == {
            "qwen7b",
            "mistral",
            "qwen3-8b",
            "deepseek-r1",
            "qwen-coder",
            "qwen3-coder",
            "qwen-vl",
            "qwen25-vl",
        }

    def test_pollinations_anonymous_tier(self) -> None:
        assert set(POLLINATIONS_MODELS) == {"fast", "openai", "gpt-oss", "gpt-oss-20b", "ovh-reasoning"}


#: The rest of this module is offline; these deliberately hit the network.
#: Run with `pytest -m live`, and expect per-IP quota failures -- that is what
#: the stillworks walk below is for.
LIVE_RELAYS = ("llm7", "ovh", "pollinations", "freeai", "aihorde")


@pytest.mark.live
@pytest.mark.parametrize("name", LIVE_RELAYS)
async def test_live_catalog_still_lists_the_default(name: str) -> None:
    """Catalog drift is the failure mode these tables age into."""
    provider = PROVIDERS[name]()
    ids = await provider.fetch_models()
    assert ids, "upstream catalog came back empty"
    assert provider.default_model in ids, f"{provider.default_model} gone from {name}"


@pytest.mark.live
async def test_live_relay_round_trip() -> None:
    provider = PollinationsProvider({"attempts": 1})
    response = await provider.chat("fast", [ChatMessage("user", "reply with exactly: ok")])
    assert response.content.strip()


@pytest.mark.live
async def test_live_stillworks_walk_answers_within_the_limit() -> None:
    registry = await stillworks.fetch()
    route, reply = await stillworks.walk(
        "reply with exactly: ok", routes=registry, limit=6, max_tokens=16
    )
    assert reply.content.strip()
    assert route.provider in ProviderRegistry.list_providers()
    assert "stillworks" in reply.raw
