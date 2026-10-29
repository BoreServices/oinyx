"""Offline tests for the ``stillworks`` discovery registry.

The registry payload below is a trimmed copy of what
``GET /api/llm/up`` actually returned on 2026-10-29, including its own
disclaimer wording, so the parsers are tested against the real shape.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from oinyx import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ProviderError,
    ProviderRegistry,
    RateLimitError,
    StreamChunk,
)
from providers import stillworks
from providers.pollinations import PollinationsProvider
from providers.stillworks import (
    CAPABILITIES,
    ENDPOINT_NAMES,
    REGISTRY_URL,
    Route,
    fetch,
    main,
    model_strings,
    rank,
    walk,
)

REGISTRY_BODY: dict[str, Any] = {
    "count": 4,
    "disclaimer": (
        "Every entry was verified by a real call carrying no Authorization "
        "header, at the time shown, from this service's own IP address. Keyless "
        "quotas are commonly per-IP: an endpoint verified here can still answer "
        "402 or 429 from yours."
    ),
    "generated_at": "2026-10-29T14:27:40.769387Z",
    "ranked_by": "recent success rate, then latency",
    "models": [
        {
            "endpoint": "pollinations",
            "base_url": "https://text.pollinations.ai",
            "chat_path": "/openai",
            "auth_mode": "none",
            "model": "openai-fast",
            "tier": "anonymous",
            "chat_capable": True,
            "openai_compatible": True,
            "last_ok": "2026-10-29T14:27:40.769387Z",
            "latency_ms": 441,
            "capabilities": {
                "json_object": {"claimed": None, "supported": True},
                "tools": {"claimed": True, "supported": False, "last_error": "answered without calling the tool"},
                "vision": {"claimed": False, "supported": None},
            },
            "recent_attempts": 168,
            "recent_successes": 168,
            "openai_base_url": "https://text.pollinations.ai/openai",
            "auth": "none",
            "answered": "168/168",
            "proved": ["json_schema", "json_object"],
            "claimed_unproved": [],
            "failed": ["tools"],
        },
        {
            "endpoint": "free-ai",
            "base_url": "https://api.free.ai",
            "chat_path": "/v1/chat/completions",
            "auth_mode": "none",
            "model": "qwen3-coder",
            "tier": "",
            "chat_capable": True,
            "openai_compatible": True,
            "latency_ms": 470,
            "openai_base_url": "https://api.free.ai/v1",
            "auth": "none",
            "answered": "9/9",
            "proved": ["tools", "json_object", "vision"],
            "failed": [],
            "capabilities": {"vision": {"supported": True}},
        },
        {
            "endpoint": "ovh-anonymous",
            "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net",
            "chat_path": "/v1/chat/completions",
            "model": "gpt-oss-120b",
            "latency_ms": 771,
            "openai_base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            "auth": "none",
            "answered": "10/10",
            "proved": ["tools"],
            "failed": ["json_object"],
        },
        {
            "endpoint": "some-relay-we-do-not-ship",
            "base_url": "https://unknown.test",
            "model": "mystery-1",
            "answered": "1/3",
            "latency_ms": 50,
            "proved": [],
        },
    ],
}


class GoodProvider(BaseProvider):
    """Relay stand-in that always answers, used to drive :func:`walk`."""

    name = "goodrelay"

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(id="g1", model=model, content="answered", finish_reason="stop")

    def stream(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise AssertionError("walk never streams")


class BadProvider(BaseProvider):
    """Relay stand-in whose IP has already used today's quota."""

    name = "badrelay"

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise RateLimitError("per-IP budget spent", provider=self.name, model=model, status_code=429)

    def stream(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise AssertionError("walk never streams")


class QuietProvider(BadProvider):
    """Answers with an empty completion, which is not an answer."""

    name = "quietrelay"

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(id="q1", model=model, content="   \n", finish_reason="stop")


@pytest.fixture(autouse=True)
def registered_providers() -> None:
    for cls in (GoodProvider, BadProvider, QuietProvider, PollinationsProvider):
        ProviderRegistry.register(cls)


def registry_transport(body: Any = REGISTRY_BODY, status: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            status, json=body, headers={"content-type": "application/json"}, request=request
        )
    )


def route(provider: str, model: str, *, answered: str = "10/10", latency: int = 100) -> Route:
    return Route.from_row(
        {
            "endpoint": provider,
            "model": model,
            "openai_base_url": "https://x.test/v1",
            "chat_path": "/chat/completions",
            "answered": answered,
            "latency_ms": latency,
            "proved": ["tools"],
        }
    )


class TestRouteParsing:
    def test_from_row_maps_registry_names_onto_our_providers(self) -> None:
        row = REGISTRY_BODY["models"][0]
        parsed = Route.from_row(row)
        assert parsed.provider == "pollinations"
        assert parsed.model == "openai-fast"
        assert parsed.model_string == "pollinations/openai-fast"
        assert parsed.base_url == "https://text.pollinations.ai/openai"
        assert parsed.successes == 168 and parsed.attempts == 168
        assert parsed.success_rate == pytest.approx(1.0)

    def test_unknown_endpoints_keep_their_own_name(self) -> None:
        parsed = Route.from_row(REGISTRY_BODY["models"][3])
        assert parsed.provider == "some-relay-we-do-not-ship"
        assert parsed.model_string == "some-relay-we-do-not-ship/mystery-1"
        assert parsed.success_rate == pytest.approx(1 / 3)

    def test_missing_and_malformed_fields_default_cleanly(self) -> None:
        parsed = Route.from_row({"endpoint": "llm7", "model": "m", "answered": "n/a", "latency_ms": None})
        assert parsed.successes == 0 and parsed.attempts == 0
        assert parsed.success_rate == 0.0
        assert parsed.latency_ms == 0
        assert parsed.base_url == ""

    def test_capabilities_are_flattened_to_supported_flags(self) -> None:
        parsed = Route.from_row(REGISTRY_BODY["models"][0])
        assert parsed.capabilities == {"json_object": True, "tools": False, "vision": False}
        assert parsed.supports("json_object")
        assert not parsed.supports("tools")
        assert set(parsed.failed) == {"tools"}

    def test_to_dict_is_json_safe(self) -> None:
        parsed = Route.from_row(REGISTRY_BODY["models"][1])
        data = parsed.to_dict()
        assert json.loads(json.dumps(data))["model_string"] == "freeai/qwen3-coder"
        assert data["proved"] == ["tools", "json_object", "vision"]

    def test_endpoint_table_covers_the_documented_registry(self) -> None:
        assert set(ENDPOINT_NAMES) == {
            "kilo-gateway",
            "llm7",
            "ovh-anonymous",
            "free-ai",
            "pollinations",
            "aihorde",
            "keylessai",
        }
        assert set(CAPABILITIES) == {"tools", "json_schema", "json_object", "vision"}


class TestFetch:
    async def test_parses_the_documented_envelope(self) -> None:
        registry = await fetch(transport=registry_transport())
        assert len(registry.routes) == 4
        assert registry.generated_at == "2026-10-29T14:27:40.769387Z"
        assert "per-IP" in registry.disclaimer
        assert registry.ranked_by == "recent success rate, then latency"

    async def test_sends_a_browser_user_agent(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=REGISTRY_BODY, request=request)

        await fetch(transport=httpx.MockTransport(handler))
        assert "Firefox" in seen[0].headers["user-agent"]
        assert str(seen[0].url) == REGISTRY_URL

    async def test_headers_can_be_overridden(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=REGISTRY_BODY, request=request)

        await fetch(transport=httpx.MockTransport(handler), headers={"user-agent": "custom/1"})
        assert seen[0].headers["user-agent"] == "custom/1"

    async def test_rows_that_are_not_objects_are_dropped(self) -> None:
        body = dict(REGISTRY_BODY, models=[{"endpoint": "llm7", "model": "a"}, "junk", 7, None])
        registry = await fetch(transport=registry_transport(body))
        assert [route.model for route in registry.routes] == ["a"]

    async def test_403_is_reported_as_a_provider_error(self) -> None:
        with pytest.raises(ProviderError, match="HTTP 403"):
            await fetch(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(403, text="Forbidden", request=request)
                ),
                attempts=1,
            )

    async def test_non_object_body_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="expected an object"):
            await fetch(transport=registry_transport([1, 2, 3]), attempts=1)

    async def test_transport_failures_are_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("flap", request=request)
            return httpx.Response(200, json=REGISTRY_BODY, request=request)

        registry = await fetch(
            transport=httpx.MockTransport(handler), attempts=2, backoff=0, sleep=None
        )
        assert calls["n"] == 2
        assert len(registry.routes) == 4

    async def test_gives_up_after_the_attempts_are_spent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        with pytest.raises(ProviderError):
            await fetch(transport=httpx.MockTransport(handler), attempts=2, backoff=0)


class TestRank:
    def test_orders_by_success_rate_then_latency(self) -> None:
        routes = [
            route("ovh", "slow", answered="10/10", latency=900),
            route("llm7", "flaky", answered="5/10", latency=10),
            route("freeai", "fast", answered="9/9", latency=100),
        ]
        assert [item.model for item in rank(routes)] == ["fast", "slow", "flaky"]

    def test_unroutable_providers_are_dropped_by_default(self) -> None:
        routes = [route("pollinations", "here"), route("unknown-thing", "there")]
        assert [item.model for item in rank(routes)] == ["here"]
        assert [item.model for item in rank(routes, routable_only=False)] == ["here", "there"]

    def test_provider_filter(self) -> None:
        routes = [route("ovh", "a"), route("freeai", "b")]
        assert [item.model for item in rank(routes, provider="ovh")] == ["a"]

    def test_capability_filter_uses_proved_not_claimed(self) -> None:
        routes = [route("ovh", "a"), route("freeai", "b")]
        assert [item.model for item in rank(routes, capability="tools")] == ["a", "b"]
        assert rank(routes, capability="vision") == []

    def test_rank_is_stable_for_equal_scores(self) -> None:
        routes = [route("ovh", "first", latency=5), route("freeai", "second", latency=5)]
        assert [item.model for item in rank(routes)] == ["first", "second"]


class TestRegistryGrouping:
    async def test_grouped_by_provider(self) -> None:
        registry = await fetch(transport=registry_transport())
        grouped = registry.providers()
        assert sorted(grouped) == ["freeai", "ovh", "pollinations", "some-relay-we-do-not-ship"]
        assert grouped["ovh"][0].model == "gpt-oss-120b"


class TestWalk:
    async def test_returns_the_first_candidate_that_answers(self) -> None:
        candidates = [route("badrelay", "m1"), route("goodrelay", "m2")]
        picked, response = await walk("hello", candidates=candidates)
        assert picked.model_string == "goodrelay/m2"
        assert response.content == "answered"

    async def test_records_why_the_winner_was_picked(self) -> None:
        _picked, response = await walk("hello", candidates=[route("goodrelay", "m1")])
        provenance = response.raw["stillworks"]
        assert provenance["registry_latency_ms"] == 100
        assert provenance["answered"] == "10/10"
        assert provenance["tried_ms"] >= 0

    async def test_blank_completions_count_as_no_answer(self) -> None:
        picked, _response = await walk("hello", candidates=[route("quietrelay", "m1"), route("goodrelay", "m2")])
        assert picked.provider == "goodrelay"

    async def test_gives_up_after_the_limit(self) -> None:
        candidates = [route("badrelay", f"m{index}") for index in range(4)]
        with pytest.raises(ProviderError, match="no stillworks candidate answered"):
            await walk("hello", candidates=candidates, limit=2)

    async def test_failure_reasons_are_reported(self) -> None:
        candidates = [route("badrelay", "m1"), route("quietrelay", "m2")]
        with pytest.raises(ProviderError, match="empty completion") as captured:
            await walk("hello", candidates=candidates)
        assert "per-IP budget spent" in str(captured.value)

    async def test_nothing_to_try(self) -> None:
        with pytest.raises(ProviderError, match="nothing to try"):
            await walk("hello", candidates=[])

    async def test_extra_kwargs_reach_the_provider(self) -> None:
        seen: list[dict[str, Any]] = []

        class Recorder(GoodProvider):
            name = "recordrelay"

            async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
                seen.append(dict(kwargs))
                return await super().chat(model, messages, **kwargs)

        ProviderRegistry.register(Recorder)
        await walk("hello", candidates=[route("recordrelay", "m1")], temperature=0.1)
        assert seen[0]["max_tokens"] == 64
        assert seen[0]["temperature"] == 0.1

    async def test_a_live_registry_route_is_called_through_the_client(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
                request=request,
            )

        ProviderRegistry.configure("pollinations", {"transport": httpx.MockTransport(handler)})
        registry = await fetch(transport=registry_transport())
        picked, response = await walk("hello", candidates=rank(registry.routes)[:1], limit=1)
        assert picked.model_string == "pollinations/openai-fast"
        assert response.content == "hi"
        assert str(calls[0].url) == "https://text.pollinations.ai/openai"


class TestConvenienceAndCli:
    async def test_model_strings_are_the_routable_candidates(self) -> None:
        listing = await model_strings(transport=registry_transport())
        assert listing == ["pollinations/openai-fast", "freeai/qwen3-coder", "ovh/gpt-oss-120b"]

    @pytest.fixture
    def offline_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve the canned payload from ``stillworks.fetch`` without touching the network."""

        async def fake_fetch(*args: Any, **kwargs: Any) -> Any:
            return await fetch(transport=registry_transport(), attempts=1)

        monkeypatch.setattr(stillworks, "fetch", fake_fetch)

    def test_main_prints_the_table(
        self, offline_registry: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 0
        out = capsys.readouterr().out
        assert "pollinations/openai-fast" in out
        assert "per-IP" in out
        assert "168/168" in out

    def test_main_reports_a_winner(
        self,
        offline_registry: None,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def winner(*args: Any, **kwargs: Any) -> Any:
            return route("ovh", "gpt-oss-120b"), ChatResponse(content="  done  ")

        monkeypatch.setattr(stillworks, "walk", winner)
        assert main(["hello"]) == 0
        assert "ovh/gpt-oss-120b -> 'done'" in capsys.readouterr().out

    def test_main_exits_non_zero_when_nothing_answers(
        self,
        offline_registry: None,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def no_one(*args: Any, **kwargs: Any) -> Any:
            raise ProviderError("all dead", provider="stillworks", model=None)

        monkeypatch.setattr(stillworks, "walk", no_one)
        assert main(["hello"]) == 1
        assert "all dead" in capsys.readouterr().out
