"""Offline tests for the shared OpenAI-relay base and its helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from oinyx import (
    AuthenticationError,
    ChatMessage,
    OinyxClient,
    ProviderError,
    ProviderRegistry,
    RateLimitError,
    StreamInterruptedError,
)
from providers.http_utils import describe_error, error_payload, sse_data
from providers.openai_relay import OpenAiRelay, catalog_items, relay_main, verify_models

CHAT_URL = "https://relay.test/v1/chat/completions"


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class RelayStub(OpenAiRelay):
    """Concrete subclass standing in for the six thin adapters."""

    name = "testrelay"
    base_url = "https://relay.test/v1"
    default_model = "small"
    catalog = {"small": "tiny-model", "big": "huge-model"}


@pytest.fixture(autouse=True)
def registered() -> None:
    ProviderRegistry.register(RelayStub)


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> RelayStub:
    return RelayStub({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


def json_response(payload: Any, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json", **headers},
        request=httpx.Request("POST", CHAT_URL),
    )


def completion(content: str = "pong", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1793283577,
        "model": "tiny-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }
    payload.update(overrides)
    return payload


def sse(events: list[Any]) -> list[bytes]:
    chunks: list[bytes] = []
    for event in events:
        text = event if isinstance(event, str) else json.dumps(event)
        chunks.append(f"data: {text}\n\n".encode())
    return chunks


class TestConstruction:
    def test_urls_are_assembled_from_the_base(self) -> None:
        provider = RelayStub()
        assert provider.chat_url == "https://relay.test/v1/chat/completions"
        assert provider.models_url == "https://relay.test/v1/models"
        assert provider.endpoint == provider.chat_url

    def test_custom_paths_and_overrides(self) -> None:
        provider = RelayStub(
            {"base_url": "https://other.test/api/", "chat_path": "/go", "models_path": "/ids"}
        )
        assert provider.chat_url == "https://other.test/api/go"
        assert provider.models_url == "https://other.test/api/ids"

    def test_catalog_can_be_replaced_per_instance(self) -> None:
        provider = RelayStub({"catalog": {"only": "single-model"}})
        assert provider.resolve_model("only") == "single-model"
        # The built-in table is gone, so a former alias is now a raw id.
        assert provider.resolve_model("small") == "small"

    def test_default_attempts_are_lower_than_the_cloudflare_base(self) -> None:
        assert RelayStub().attempts == 2
        assert RelayStub({"attempts": 5}).attempts == 5

    def test_repr(self) -> None:
        assert "testrelay" in repr(RelayStub())


class TestHeaders:
    def test_no_authorization_without_a_key(self) -> None:
        assert "authorization" not in RelayStub().headers

    def test_api_key_becomes_a_bearer_token(self) -> None:
        assert RelayStub({"api_key": "abc"}).headers["authorization"] == "Bearer abc"

    def test_anonymous_key_is_used_when_nothing_is_configured(self) -> None:
        class Queued(RelayStub):
            name = "queuedrelay"
            anonymous_key = "0000000000"

        assert Queued().headers["authorization"] == "Bearer 0000000000"
        assert Queued({"api_key": "real"}).headers["authorization"] == "Bearer real"


class TestResolveModel:
    def test_provider_prefix_is_stripped_before_lookup(self) -> None:
        assert RelayStub().resolve_model("testrelay/big") == "huge-model"

    def test_unknown_ids_pass_through(self) -> None:
        assert RelayStub().resolve_model("raw-upstream-id") == "raw-upstream-id"

    def test_empty_model_falls_back_to_the_default(self) -> None:
        assert RelayStub().resolve_model("") == "tiny-model"

    def test_no_default_raises(self) -> None:
        class Bare(RelayStub):
            name = "barerelay"
            default_model = ""
            catalog = {}

        with pytest.raises(ProviderError, match="needs a model"):
            Bare().resolve_model("")


class TestBuildPayload:
    def test_only_whitelisted_options_are_forwarded(self) -> None:
        provider = RelayStub()
        payload = provider.build_payload(
            "small",
            [ChatMessage("user", "hi")],
            {"temperature": 0.4, "max_tokens": 10, "internal_debug": True, "seed": None},
        )
        assert payload["temperature"] == 0.4
        assert payload["max_tokens"] == 10
        assert "internal_debug" not in payload
        assert "seed" not in payload

    def test_explicit_payload_is_merged(self) -> None:
        provider = RelayStub()
        payload = provider.build_payload(
            "small", [ChatMessage("user", "hi")], {"payload": {"custom": 1}}
        )
        assert payload["custom"] == 1
        assert payload["model"] == "tiny-model"

    def test_messages_are_serialised(self) -> None:
        provider = RelayStub()
        payload = provider.build_payload(
            "small", [ChatMessage("system", "be terse"), ChatMessage("user", "hi")], {}
        )
        assert payload["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="at least one message"):
            RelayStub().build_payload("small", [], {})


class TestChat:
    async def test_round_trip(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return json_response(completion())

        provider = make_provider(handler)
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.content == "pong"
        assert response.usage.total_tokens == 12
        assert seen[0] == {"model": "tiny-model", "messages": [{"role": "user", "content": "hello"}]}

    async def test_upstream_model_is_reported_alongside_the_requested_one(self) -> None:
        provider = make_provider(lambda request: json_response(completion()))
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.model == "tiny-model"
        assert response.raw["requested_model"] == "tiny-model"

    async def test_missing_usage_is_estimated(self) -> None:
        provider = make_provider(lambda request: json_response(completion("x" * 400, usage=None)))
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.usage.completion_tokens > 0

    async def test_null_content_becomes_empty_text(self) -> None:
        body = completion()
        body["choices"][0]["message"]["content"] = None
        provider = make_provider(lambda request: json_response(body))
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.content == ""

    async def test_reasoning_is_surfaced_in_raw(self) -> None:
        body = completion()
        body["choices"][0]["message"]["reasoning"] = "thoughts"
        provider = make_provider(lambda request: json_response(body))
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.raw["reasoning"] == "thoughts"

    async def test_no_choices_at_all(self) -> None:
        provider = make_provider(lambda request: json_response({"id": "x"}))
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.content == ""
        assert response.finish_reason is None

    async def test_json_body_carrying_an_error_object(self) -> None:
        provider = make_provider(
            lambda request: json_response({"error": {"message": "nope", "code": "bad"}})
        )
        with pytest.raises(ProviderError, match=r"nope \[code=bad\]"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_non_json_body(self) -> None:
        provider = make_provider(
            lambda request: httpx.Response(200, text="not json", request=request)
        )
        with pytest.raises(ProviderError, match="non-JSON"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_array_body_is_rejected(self) -> None:
        provider = make_provider(lambda request: json_response([1, 2]))
        with pytest.raises(ProviderError, match="expected an object"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_transport_error_is_retried_then_raised(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("boom", request=request)

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="request failed"):
            await provider.chat("small", [ChatMessage("user", "hello")])
        assert calls["n"] == 2

    async def test_quota_429_is_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return json_response({"error": "slow down"}, status=429)
            return json_response(completion())

        provider = make_provider(handler)
        response = await provider.chat("small", [ChatMessage("user", "hello")])
        assert response.content == "pong"
        assert calls["n"] == 2


class TestErrorMapping:
    async def test_401_is_authentication(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "denied"}, status=401))
        with pytest.raises(AuthenticationError):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_403_is_authentication(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "blocked"}, status=403))
        with pytest.raises(AuthenticationError):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_402_counts_as_quota(self) -> None:
        provider = make_provider(
            lambda request: json_response({"error": {"message": "budget exhausted"}}, status=402)
        )
        with pytest.raises(RateLimitError):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_429_reports_retry_after(self) -> None:
        provider = make_provider(
            lambda request: httpx.Response(
                429,
                json={"error": {"message": "calm down"}},
                headers={"retry-after": "30", "content-type": "application/json"},
                request=request,
            )
        )
        with pytest.raises(RateLimitError, match="retry after 30s"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_prose_that_admits_a_limit_becomes_rate_limit(self) -> None:
        provider = make_provider(
            lambda request: json_response(
                {"error": {"message": "The Classic API is at its limit right now"}}, status=503
            )
        )
        with pytest.raises(RateLimitError, match="at its limit"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_plain_500_is_upstream(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "gone"}, status=500))
        with pytest.raises(ProviderError, match="gone"):
            await provider.chat("small", [ChatMessage("user", "hello")])

    async def test_400_is_not_retryable(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return json_response({"error": {"message": "bad model", "code": 400}}, status=400)

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match=r"bad model \[code=400\]"):
            await provider.chat("small", [ChatMessage("user", "hello")])
        assert calls["n"] == 1


class TestStream:
    async def test_deltas_and_done_marker(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(
                    sse(
                        [
                            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                            {"choices": [{"index": 0, "delta": {"content": "pin"}}]},
                            {"choices": [{"index": 0, "delta": {"content": "g"}}]},
                            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                            "[DONE]",
                        ]
                    )
                ),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        chunks = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]
        assert "".join(chunk.delta for chunk in chunks) == "ping"
        assert chunks[-1].finish_reason == "stop"

    async def test_role_only_frames_are_dropped(self) -> None:
        frames = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]}] * 5
        frames.append({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
        frames.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse(frames)),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        chunks = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]
        assert len(chunks) == 2
        assert chunks[0].delta == "hi"

    async def test_stream_payload_sets_stream_true(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=ChunkStream(sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"])),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        provider = make_provider(handler, include_usage=True)
        _ = [chunk async for chunk in provider.stream("small", [ChatMessage("user", "hi")])]
        assert seen[0]["stream"] is True
        assert seen[0]["stream_options"] == {"include_usage": True}

    async def test_usage_frame_is_kept_and_labelled(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse(events)),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        chunks = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]
        assert chunks[1].raw["usage"]["total_tokens"] == 7

    async def test_reasoning_delta_lands_in_raw(self) -> None:
        events = [
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse(events)),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        chunks = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]
        assert chunks[0].raw["reasoning"] == "thinking"
        assert chunks[0].delta == ""

    async def test_truncated_stream_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse([{"choices": [{"delta": {"content": "half"}}]}])),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        with pytest.raises(StreamInterruptedError):
            _ = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]

    async def test_error_frame_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse([{"error": {"message": "mid-stream fail"}}])),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        with pytest.raises(ProviderError, match="mid-stream fail"):
            _ = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]

    async def test_error_status_on_stream(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "nope"}, status=429))
        with pytest.raises(RateLimitError):
            _ = [chunk async for chunk in provider.stream("small", [ChatMessage("user", "hi")])]

    async def test_transport_error_on_stream(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("drop", request=request)

        with pytest.raises(ProviderError, match="stream failed"):
            _ = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]

    async def test_garbage_and_comment_lines_are_ignored(self) -> None:
        chunks = [
            b": keep-alive\n\n",
            b"event: message\n\n",
            b"data: not-json\n\n",
            b"data: 7\n\n",
            b"data: \n\n",
        ] + sse([{"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]}])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(chunks),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        received = [chunk async for chunk in make_provider(handler).stream("small", [ChatMessage("user", "hi")])]
        assert [chunk.delta for chunk in received] == ["ok", ""]
        assert received[-1].finish_reason == "stop"


class TestModels:
    async def test_list_models_is_prefixed(self) -> None:
        assert await RelayStub().list_models() == ["testrelay/big", "testrelay/small"]

    async def test_fetch_models_accepts_the_data_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/models"
            return json_response({"object": "list", "data": [{"id": "a"}, {"id": "b"}]})

        assert await make_provider(handler).fetch_models() == ["a", "b"]

    async def test_fetch_models_accepts_the_models_key(self) -> None:
        provider = make_provider(
            lambda request: json_response({"models": [{"id": "x", "type": "chat"}]})
        )
        assert await provider.fetch_models() == ["x"]

    async def test_fetch_models_accepts_a_bare_list_of_strings(self) -> None:
        provider = make_provider(lambda request: json_response(["one", "two"]))
        assert await provider.fetch_models() == ["one", "two"]

    async def test_rows_without_ids_are_skipped(self) -> None:
        provider = make_provider(
            lambda request: json_response({"data": [{"name": "n"}, {"id": ""}, 5, None, {"id": "y"}]})
        )
        assert await provider.fetch_models() == ["n", "y"]

    async def test_accept_hook_filters_rows(self) -> None:
        class Filtered(RelayStub):
            name = "filteredrelay"

            def accept(self, item: Any) -> bool:
                return isinstance(item, dict) and item.get("tier") == "turbo"

        handler = lambda request: json_response(
            {"data": [{"id": "free", "tier": "turbo"}, {"id": "paid", "tier": "pro"}]}
        )
        assert await make_provider(handler).fetch_models() == ["free", "paid"]
        filtered = Filtered({"transport": httpx.MockTransport(handler), "backoff": 0})
        assert await filtered.fetch_models() == ["free"]

    async def test_non_json_catalog(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200, text="<html>", request=request))
        with pytest.raises(ProviderError, match="not JSON"):
            await provider.fetch_models()


class TestHttpUtils:
    def test_describe_error_shapes(self) -> None:
        assert describe_error("boom") == ("boom", "upstream_error")
        assert describe_error({"detail": "d"}) == ("d", "upstream_error")
        assert describe_error({"code": 429}) == ("upstream error", "429")
        assert describe_error(["raw"]) == ("['raw']", "upstream_error")

    def test_error_payload_unwraps_the_error_key(self) -> None:
        response = httpx.Response(400, json={"error": {"message": "m"}}, request=httpx.Request("GET", CHAT_URL))
        assert error_payload(response) == {"message": "m"}

    def test_error_payload_stringifies_a_bare_error(self) -> None:
        response = httpx.Response(400, json={"error": "plain"}, request=httpx.Request("GET", CHAT_URL))
        assert error_payload(response) == {"message": "plain"}

    def test_error_payload_falls_back_to_text(self) -> None:
        response = httpx.Response(500, text="kaboom", request=httpx.Request("GET", CHAT_URL))
        assert error_payload(response) == {"message": "kaboom"}

    def test_error_payload_on_an_empty_body(self) -> None:
        response = httpx.Response(502, text="", request=httpx.Request("GET", CHAT_URL))
        assert "HTTP 502" in error_payload(response)["message"]

    def test_error_payload_stringifies_an_array_body(self) -> None:
        response = httpx.Response(400, json=[1, 2], request=httpx.Request("GET", CHAT_URL))
        assert error_payload(response) == {"message": "[1, 2]"}

    def test_catalog_items_shapes(self) -> None:
        assert catalog_items({"data": [1]}) == [1]
        assert catalog_items({"models": [2]}) == [2]
        assert catalog_items([3]) == [3]
        assert catalog_items("nope") == []

    def test_sse_data(self) -> None:
        assert sse_data("data: a") == "a"
        assert sse_data("data:b") == "b"
        assert sse_data("event: x") is None
        assert sse_data(": comment") is None
        assert sse_data("data:") is None


class TestClientIntegration:
    async def test_routing_through_the_client(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response(completion("routed"))

        client = OinyxClient(
            provider="testrelay", config={"transport": httpx.MockTransport(handler)}
        )
        response = await client.chat("small", [ChatMessage("user", "hi")])
        assert response.content == "routed"
        assert response.model == "tiny-model"

    async def test_collect_via_client_stream(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ChunkStream(sse([
                    {"choices": [{"delta": {"content": "ab"}}]},
                    {"choices": [{"delta": {"content": "cd"}, "finish_reason": "stop"}]},
                    "[DONE]",
                ])),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        client = OinyxClient(config={"transport": httpx.MockTransport(handler)})
        collected = await client.collect("testrelay/small", "hi")
        assert collected.content == "abcd"


class TestVerifyHelpers:
    async def test_verify_models_reports_each_alias(self) -> None:
        provider = make_provider(lambda request: json_response(completion("ok")))
        results = await verify_models(provider, {"small": "tiny-model"})
        assert results["small"].startswith("OK")

    async def test_verify_models_captures_failures(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "dead"}, status=400))
        results = await verify_models(provider, {"small": "tiny-model"})
        assert results["small"].startswith("FAIL")

    async def test_verify_models_captures_non_oinyx_failures(self) -> None:
        class Exploding(RelayStub):
            name = "explodingrelay"

            async def chat(self, model: str, messages: Any, **kwargs: Any) -> Any:
                raise RuntimeError("socket hung up mid-parse")

        results = await verify_models(Exploding(), {"small": "tiny-model"})
        assert results["small"].startswith("FAIL tiny-model -> RuntimeError")

    def test_relay_main_prints_and_returns_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return json_response({"data": [{"id": "tiny-model"}]})
            return json_response(completion("ok"))

        provider = RelayStub({"transport": httpx.MockTransport(handler), "backoff": 0, "attempts": 1})
        assert relay_main(provider, {"small": "tiny-model"}) == 0
        out = capsys.readouterr().out
        assert "tiny-model" in out and "OK" in out

    def test_relay_main_fails_loudly_on_a_dead_alias(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        provider = RelayStub(
            {
                "transport": httpx.MockTransport(
                    lambda request: json_response({"error": "nope"}, status=400)
                ),
                "backoff": 0,
                "attempts": 1,
            }
        )
        assert relay_main(provider, {"small": "tiny-model"}) == 1
        assert "FAIL" in capsys.readouterr().out
