"""Offline tests for the bundled ``kai`` provider (mocked transport only).

Live gateway checks are marked ``live`` and excluded by default; run them with
``pytest -m live``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from oinyx import OinyxClient, ProviderRegistry, RateLimitError
from oinyx.exceptions import AuthenticationError, ProviderError, StreamInterruptedError
from oinyx.models import ChatMessage, ChatResponse, StreamChunk, Usage
from providers.kai import (
    CHAT_URL,
    DEFAULT_MODEL,
    KAI_MODELS,
    MODELS_URL,
    RETIRED_ALIASES,
    KaiProvider,
    UpstreamError,
)


class ChunkStream(httpx.AsyncByteStream):
    """ replays fixed byte chunks so SSE behaviour is deterministic """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> KaiProvider:
    return KaiProvider({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", CHAT_URL),
    )


def completion(content: str = "ok", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "gen_1",
        "object": "chat.completion",
        "created": 1793283577,
        "model": "stealth/glyph-cluster",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 127, "completion_tokens": 153, "total_tokens": 280},
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def registered() -> None:
    ProviderRegistry.register(KaiProvider)


def sse(events: list[Any]) -> list[bytes]:
    chunks: list[bytes] = []
    for event in events:
        text = event if isinstance(event, str) else json.dumps(event)
        chunks.append(f"data: {text}\n\n".encode())
    return chunks


class TestTable:
    def test_alias_keys_are_prefixed_and_unique(self) -> None:
        assert all(key.startswith("kai/") for key in KAI_MODELS)
        assert len(set(KAI_MODELS.values())) == len(KAI_MODELS), "two aliases share one upstream id"
        assert KAI_MODELS["kai/auto"] == "kilo-auto/free"

    def test_retired_ids_are_not_in_the_live_table(self) -> None:
        live_targets = set(KAI_MODELS.values())
        assert not live_targets & set(RETIRED_ALIASES.values())

    def test_default_model_exists_in_the_table(self) -> None:
        assert DEFAULT_MODEL in KAI_MODELS


class TestRegistration:
    def test_provider_is_registered(self) -> None:
        assert ProviderRegistry.list_providers() == ["kai"]
        assert isinstance(ProviderRegistry.get("kai"), KaiProvider)

    def test_defaults(self) -> None:
        provider = KaiProvider()
        assert provider.chat_url == CHAT_URL
        assert provider.models_url == MODELS_URL
        assert provider.default_model == DEFAULT_MODEL
        assert repr(provider) == f"<KaiProvider name='kai' aliases={len(KAI_MODELS)}>"


class TestModelResolution:
    def test_short_alias_and_prefixed_alias(self) -> None:
        provider = KaiProvider()
        assert provider.resolve_model("auto") == "kilo-auto/free"
        assert provider.resolve_model("kai/auto") == "kilo-auto/free"

    def test_raw_upstream_id_passes_through(self) -> None:
        provider = KaiProvider()
        assert provider.resolve_model("nvidia/nemotron-3.5-lightning:free") == (
            "nvidia/nemotron-3.5-lightning:free"
        )

    def test_unknown_alias_falls_back_to_the_stripped_name(self) -> None:
        provider = KaiProvider()
        assert provider.resolve_model("kai/nope") == "nope"

    def test_missing_model_uses_the_default(self) -> None:
        assert KaiProvider().resolve_model("") == "kilo-auto/free"

    def test_custom_aliases_override_the_table(self) -> None:
        provider = KaiProvider({"aliases": {"kai/tiny": "vendor/tiny:free"}})
        assert provider.resolve_model("tiny") == "vendor/tiny:free"
        assert provider.resolve_model("auto") == "auto"

    def test_model_without_a_value_raises(self) -> None:
        provider = KaiProvider({"model": ""})
        with pytest.raises(ProviderError, match="needs a model name"):
            provider.resolve_model("   ")


class TestPayload:
    async def test_body_shape(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response(completion())

        provider = make_provider(handler)
        await provider.chat(
            "auto",
            [ChatMessage("system", "be terse"), ChatMessage("user", "hi")],
            max_tokens=32,
            dropped=None,
        )
        body = captured[0]
        assert body["model"] == "kilo-auto/free"
        assert body["max_tokens"] == 32
        assert "dropped" not in body
        assert body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    async def test_empty_messages_are_rejected_locally(self) -> None:
        provider = make_provider(lambda request: json_response(completion()))
        with pytest.raises(ProviderError, match="at least one message"):
            await provider.chat("auto", [])

    async def test_headers_are_sent(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response(completion())

        await make_provider(handler, api_key="secret", referer="https://app.test", title="my app").chat(
            "auto", [ChatMessage("user", "hi")]
        )
        assert seen[0].headers["authorization"] == "Bearer secret"
        assert seen[0].headers["http-referer"] == "https://app.test"
        assert seen[0].headers["x-title"] == "my app"

    async def test_extra_headers_merge(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response(completion())

        await make_provider(handler, headers={"x-trace": "abc"}).chat(
            "auto", [ChatMessage("user", "hi")]
        )
        assert seen[0].headers["x-trace"] == "abc"


class TestChatParsing:
    async def test_full_completion(self) -> None:
        provider = make_provider(lambda request: json_response(completion("all good")))
        response = await provider.chat("auto", [ChatMessage("user", "hi")])
        assert isinstance(response, ChatResponse)
        assert response.id == "gen_1"
        assert response.model == "stealth/glyph-cluster"
        assert response.content == "all good"
        assert response.finish_reason == "stop"
        assert response.usage == Usage(127, 153, 280)
        assert response.raw["requested_model"] == "kilo-auto/free"
        assert response.raw["system_fingerprint"] is None

    async def test_reasoning_flag_when_encrypted_details_present(self) -> None:
        body = completion("ok")
        body["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted"}]
        provider = make_provider(lambda request: json_response(body))
        response = await provider.chat("auto", [ChatMessage("user", "hi")])
        assert response.raw["reasoning"] is True

    async def test_usage_is_estimated_when_absent(self) -> None:
        body = completion("aaaa")
        body["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        provider = make_provider(lambda request: json_response(body))
        response = await provider.chat("auto", [ChatMessage("user", "bbbb bbbb")])
        assert response.usage.completion_tokens == 1
        assert response.usage.total_tokens > 0

    async def test_flat_content_fallback(self) -> None:
        provider = make_provider(lambda request: json_response({"content": "flat"}))
        response = await provider.chat("auto", [ChatMessage("user", "hi")])
        assert response.content == "flat"

    async def test_missing_choices_yields_empty_content(self) -> None:
        provider = make_provider(lambda request: json_response({"id": "x", "choices": []}))
        response = await provider.chat("auto", [ChatMessage("user", "hi")])
        assert response.content == ""
        assert response.finish_reason is None

    async def test_non_json_success_body_is_reported(self) -> None:
        provider = make_provider(
            lambda request: httpx.Response(
                200, text="not json", headers={"content-type": "text/plain"},
                request=httpx.Request("POST", CHAT_URL),
            )
        )
        with pytest.raises(ProviderError, match="non-JSON body"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_json_array_body_is_reported(self) -> None:
        provider = make_provider(lambda request: json_response([1, 2, 3]))
        with pytest.raises(ProviderError, match="unexpected payload type"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_error_inside_a_200_body_is_raised(self) -> None:
        provider = make_provider(lambda request: json_response({"error": {"message": "no quota"}}))
        with pytest.raises(ProviderError, match="no quota"):
            await provider.chat("auto", [ChatMessage("user", "hi")])


class TestErrorMapping:
    async def test_unknown_model_is_a_plain_provider_error(self) -> None:
        payload = {"error": "The requested model 'kai/gone' does not exist"}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json=payload, headers={"content-type": "application/json"},
                request=httpx.Request("POST", CHAT_URL),
            )

        provider = make_provider(handler, attempts=3)
        with pytest.raises(ProviderError) as excinfo:
            await provider.chat("auto", [ChatMessage("user", "hi")])
        assert excinfo.value.status_code == 404
        assert "does not exist" in str(excinfo.value)
        assert not isinstance(excinfo.value, (RateLimitError, AuthenticationError, UpstreamError))

    async def test_auth_errors(self) -> None:
        provider = make_provider(lambda request: json_response({"error": {"message": "denied"}}, status=401))
        with pytest.raises(AuthenticationError):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            response = json_response({"error": {"message": "slow", "code": 429}}, status=429)
            return response

        provider = make_provider(handler, attempts=1)
        with pytest.raises(RateLimitError, match="slow"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_rate_limit_retry_after_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429, json={"error": {"message": "busy"}},
                headers={"content-type": "application/json", "retry-after": "12"},
                request=httpx.Request("POST", CHAT_URL),
            )

        provider = make_provider(handler, attempts=1)
        with pytest.raises(RateLimitError, match="retry after 12s"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_server_errors_are_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return json_response({"error": {"message": "down"}}, status=503)
            return json_response(completion("back"))

        response = await make_provider(handler, attempts=3).chat(
            "auto", [ChatMessage("user", "hi")]
        )
        assert response.content == "back"
        assert len(calls) == 3

    async def test_retries_are_exhausted(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "boom"}, status=500), attempts=2)
        with pytest.raises(UpstreamError) as excinfo:
            await provider.chat("auto", [ChatMessage("user", "hi")])
        assert excinfo.value.status_code == 500

    async def test_transport_error_becomes_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        provider = make_provider(handler, attempts=2)
        with pytest.raises(UpstreamError, match="nope"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_html_error_body_is_summarised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, text="<html>bad request</html>", headers={"content-type": "text/html"},
                request=httpx.Request("POST", CHAT_URL),
            )

        provider = make_provider(handler, attempts=1)
        with pytest.raises(ProviderError, match="bad request"):
            await provider.chat("auto", [ChatMessage("user", "hi")])

    async def test_nested_error_dictionary_shapes(self) -> None:
        assert KaiProvider._describe_error({"message": "m", "type": "t"}) == ("m", "t")
        assert KaiProvider._describe_error({"detail": "d"}) == ("d", "upstream_error")
        assert KaiProvider._describe_error({"code": 429})[1] == "429"
        assert KaiProvider._describe_error({"nothing": 1}) == ("upstream error", "upstream_error")
        assert KaiProvider._describe_error(["raw"]) == ("['raw']", "upstream_error")

    async def test_error_payload_without_error_key(self) -> None:
        provider = make_provider(lambda request: json_response({"detail": "nope"}, status=400), attempts=1)
        with pytest.raises(ProviderError, match="nope"):
            await provider.chat("auto", [ChatMessage("user", "hi")])


class TestStreaming:
    async def test_deltas_and_done_marker(self) -> None:
        events = sse(
            [
                {"id": "g1", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"id": "g1", "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
                {
                    "id": "g1",
                    "choices": [
                        {"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}
                    ],
                },
                "[DONE]",
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content.decode())["stream"] is True
            return httpx.Response(
                200,
                stream=ChunkStream(events),
                headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        chunks = [
            chunk
            async for chunk in make_provider(handler).stream("auto", [ChatMessage("user", "hi")])
        ]
        assert "".join(chunk.delta for chunk in chunks) == "Hello world"
        assert chunks[-1].finish_reason == "stop"

    async def test_finish_reason_is_taken_from_the_gateway(self) -> None:
        events = sse(
            [
                {"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "length"}]},
                "[DONE]",
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=ChunkStream(events), headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        chunks = [
            chunk
            async for chunk in make_provider(handler).stream("auto", [ChatMessage("user", "hi")])
        ]
        assert [chunk.finish_reason for chunk in chunks if chunk.finish_reason] == ["length"]

    async def test_missing_finish_reason_is_synthesised(self) -> None:
        events = sse([{"choices": [{"index": 0, "delta": {"content": "solo"}}]}])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=ChunkStream(events), headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        chunks = [
            chunk
            async for chunk in make_provider(handler).stream("auto", [ChatMessage("user", "hi")])
        ]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].raw["note"]

    async def test_comment_and_event_lines_are_skipped(self) -> None:
        events = [b": keepalive\n\n", b"event: message\n", b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "kept"}}]}
        ).encode() + b"\n\n", b"data: [DONE]\n\n"]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=ChunkStream(events), headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        chunks = [
            chunk
            async for chunk in make_provider(handler).stream("auto", [ChatMessage("user", "hi")])
        ]
        assert "".join(chunk.delta for chunk in chunks) == "kept"

    async def test_malformed_sse_payload_raises(self) -> None:
        events = [b"data: {not json}\n\n"]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=ChunkStream(events), headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        with pytest.raises(StreamInterruptedError, match="malformed SSE"):
            async for _ in make_provider(handler).stream("auto", [ChatMessage("user", "hi")]):
                pass

    async def test_error_event_raises(self) -> None:
        events = sse([{"error": {"message": "upstream died", "code": "e"}}])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=ChunkStream(events), headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )

        with pytest.raises(ProviderError, match="upstream died"):
            async for _ in make_provider(handler).stream("auto", [ChatMessage("user", "hi")]):
                pass

    async def test_http_error_before_the_first_chunk(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response({"error": {"message": "denied"}}, status=403)

        with pytest.raises(AuthenticationError):
            async for _ in make_provider(handler).stream("auto", [ChatMessage("user", "hi")]):
                pass

    async def test_transport_failure_mid_stream(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("reset")

        with pytest.raises(UpstreamError, match="reset"):
            async for _ in make_provider(handler).stream("auto", [ChatMessage("user", "hi")]):
                pass

    async def test_stream_validation_is_local(self) -> None:
        provider = make_provider(lambda request: json_response(completion()))
        with pytest.raises(ProviderError, match="at least one message"):
            async for _ in provider.stream("auto", []):
                pass


class TestModelDiscovery:
    async def test_free_models_are_filtered_and_sorted(self) -> None:
        catalog = {
            "data": [
                {"id": "z/late:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "a/early:free", "pricing": {"prompt": 0, "completion": 0}},
                {"id": "paid/model", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                {"id": "broken", "pricing": {"prompt": "n/a", "completion": "0"}},
                {"no_pricing": True},
                "string-entry",
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert str(request.url) == MODELS_URL
            return json_response(catalog)

        provider = make_provider(handler)
        assert await provider.discover_models() == ["a/early:free", "z/late:free"]

    async def test_list_models_returns_aliases(self) -> None:
        models = await KaiProvider().list_models()
        assert models == sorted(KAI_MODELS)
        assert "kai/auto" in models

    async def test_discovery_errors_are_mapped(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "gone"}, status=500), attempts=1)
        with pytest.raises(UpstreamError):
            await provider.discover_models()

    async def test_discovery_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        with pytest.raises(UpstreamError, match="offline"):
            await make_provider(handler).discover_models()


class TestClientIntegration:
    @staticmethod
    def client(**extra: Any) -> OinyxClient:
        return OinyxClient(
            config={
                "kai": {
                    "transport": httpx.MockTransport(lambda request: json_response(completion("routed"))),
                    "backoff": 0,
                    **extra,
                }
            }
        )

    async def test_chat_through_the_client(self) -> None:
        response = await self.client().chat("kai/auto", "hi")
        assert response.content == "routed"
        assert response.usage.total_tokens == 280

    async def test_alias_table_works_with_a_bare_default_provider(self) -> None:
        client = OinyxClient(
            provider="kai",
            aliases={"fast": "kai/xs-2.1"},
            config={"kai": {"transport": httpx.MockTransport(lambda r: json_response(completion())), "backoff": 0}},
        )
        response = await client.chat("fast", "hi")
        assert response.model == "stealth/glyph-cluster"

    async def test_streaming_through_the_client(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=ChunkStream(sse([{"choices": [{"delta": {"content": "go"}}]}, "[DONE]"])),
                headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )
        )
        client = OinyxClient(config={"kai": {"transport": transport, "backoff": 0}})
        chunks = [chunk async for chunk in client.stream("kai/auto", "hi")]
        assert "".join(chunk.delta for chunk in chunks) == "go"
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)

    async def test_collect_through_the_client(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=ChunkStream(sse([{"choices": [{"delta": {"content": "ab"}, "finish_reason": "stop"}]}])),
                headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            )
        )
        client = OinyxClient(config={"kai": {"transport": transport, "backoff": 0}})
        response = await client.collect("kai/auto", "hi")
        assert response.content == "ab"
        assert response.finish_reason == "stop"


@pytest.mark.live
async def test_live_gateway_chat() -> None:
    """Opt-in smoke test against the real, unauthenticated gateway."""
    response = await KaiProvider({"attempts": 2, "backoff": 0.5}).chat(
        "auto", [ChatMessage("user", "reply with exactly: ok")]
    )
    assert response.content.strip()
    assert response.usage.total_tokens > 0


@pytest.mark.live
async def test_live_gateway_stream() -> None:
    provider = KaiProvider({"attempts": 1})
    text = ""
    async for chunk in provider.stream("auto", [ChatMessage("user", "count 1 to 5")]):
        text += chunk.delta
    assert text.strip()


@pytest.mark.live
async def test_live_catalog_contains_the_default_alias() -> None:
    models = await KaiProvider().discover_models()
    assert KAI_MODELS["kai/auto"] in models
