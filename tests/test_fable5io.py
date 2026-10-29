"""Offline tests for the bundled ``fable5`` provider.

All traffic is served by :class:`httpx.MockTransport`, so the suite never
reaches the network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from oinyx import OinyxClient, ProviderRegistry, RateLimitError
from oinyx.exceptions import AuthenticationError, ProviderError, StreamInterruptedError
from oinyx.models import ChatMessage, ChatResponse, StreamChunk
from providers.fable5io import DEFAULT_MODEL, ENDPOINT, Fable5Provider, UpstreamError


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> Fable5Provider:
    """Provider wired to a mock transport, with retries that do not sleep."""
    return Fable5Provider({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


def json_response(payload: Any, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json", **headers},
        request=httpx.Request("POST", ENDPOINT),
    )


def text_response(body: str, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        text=body,
        headers={"content-type": "text/plain; charset=utf-8", **headers},
        request=httpx.Request("POST", ENDPOINT),
    )


class ExplodingStream(httpx.AsyncByteStream):
    """Streams one chunk, then dies mid-response."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"partial "
        raise httpx.ReadError("connection reset")


@pytest.fixture(autouse=True)
def registered() -> None:
    """Re-register the provider after the global registry was cleared."""
    ProviderRegistry.register(Fable5Provider)


class TestRegistration:
    def test_it_is_registered_under_fable5(self) -> None:
        assert ProviderRegistry.list_providers() == ["fable5"]
        assert isinstance(ProviderRegistry.get("fable5"), Fable5Provider)

    def test_defaults(self) -> None:
        provider = Fable5Provider()
        assert provider.endpoint == ENDPOINT
        assert provider.default_model == DEFAULT_MODEL
        assert provider.name == "fable5"
        assert repr(provider).startswith("<Fable5Provider")


class TestChat:
    async def test_plain_text_reply_becomes_a_response(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return text_response("Hello there!", **{"x-chat-model": "openai/gpt-4o"})

        provider = make_provider(handler)
        response = await provider.chat("openai/gpt-4o", [ChatMessage("user", "hi")])

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello there!"
        assert response.model == "openai/gpt-4o"
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens > 0
        assert response.raw["status_code"] == 200
        assert response.raw["content_type"].startswith("text/plain")

        assert seen[0].method == "POST"
        assert seen[0].url == ENDPOINT
        assert seen[0].headers["content-type"] == "application/json"

    async def test_payload_matches_the_documented_shape(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update({"body": request.content.decode(), "url": str(request.url)})
            return text_response("ok")

        await make_provider(handler).chat(
            "openai/gpt-4o",
            [ChatMessage("system", "be brief"), ChatMessage("user", "test")],
        )
        assert captured["url"] == ENDPOINT
        assert json_loads(captured["body"]) == {
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "test"},
            ],
            "model": "openai/gpt-4o",
        }

    async def test_json_envelopes_are_also_understood(self) -> None:
        cases = [
            ({"choices": [{"message": {"content": "from choices"}}]}, "from choices"),
            ({"choices": [{"delta": {"content": "from delta"}}]}, "from delta"),
            ({"choices": [{"text": "from text"}]}, "from text"),
            ({"content": "from content"}, "from content"),
            ({"response": "from response"}, "from response"),
            ({"data": "from data"}, "from data"),
            ([1, 2], None),
            ({"unexpected": 1}, None),
        ]
        for payload, expected in cases:
            response = json_response(payload)
            extracted = Fable5Provider.extract_text(response)
            assert extracted == (expected if expected is not None else response.text)

    async def test_kwargs_are_merged_into_the_body(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json_loads(request.content.decode()))
            return text_response("ok")

        await make_provider(handler).chat(
            "m", [ChatMessage("user", "hi")], temperature=0.5, ignored=None
        )
        assert captured[0]["temperature"] == 0.5
        assert "ignored" not in captured[0]

    async def test_extra_headers_are_sent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-test"] == "1"
            return text_response("ok")

        await make_provider(handler, headers={"x-test": "1"}).chat("m", [ChatMessage("user", "hi")])


class TestValidation:
    async def test_empty_conversation_is_rejected_before_the_request(self) -> None:
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return text_response("ok")

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="between 1 and 32 messages"):
            await provider.chat("m", [])
        assert not called

    async def test_too_many_messages(self) -> None:
        provider = make_provider(lambda request: text_response("ok"), max_messages=3)
        turns = [ChatMessage("user", "x") for _ in range(4)]
        with pytest.raises(ProviderError, match="at most 3 messages"):
            await provider.chat("m", turns)

    async def test_message_too_long(self) -> None:
        provider = make_provider(lambda request: text_response("ok"), max_content_chars=10)
        with pytest.raises(ProviderError, match="exceeds 10 characters"):
            await provider.chat("m", [ChatMessage("user", "y" * 11)])

    async def test_message_names_are_preserved(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json_loads(request.content.decode()))
            return text_response("ok")

        await make_provider(handler).chat("m", [ChatMessage("tool", "result", name="calc")])
        assert captured[0]["messages"][0] == {"role": "tool", "content": "result", "name": "calc"}


class TestErrorMapping:
    async def test_bad_request_maps_to_provider_error_with_code(self) -> None:
        payload = {
            "data": None,
            "error": {"message": "messages must be 1-32 each", "code": "INVALID_MESSAGES"},
        }
        provider = make_provider(lambda request: json_response(payload, status=400))
        with pytest.raises(ProviderError) as excinfo:
            await provider.chat("m", [ChatMessage("user", "hi")])
        error = excinfo.value
        assert not isinstance(error, (RateLimitError, AuthenticationError, UpstreamError))
        assert error.status_code == 400
        assert error.provider == "fable5"
        assert "INVALID_MESSAGES" in str(error)

    async def test_unauthorised_maps_to_authentication_error(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "denied"}, status=401))
        with pytest.raises(AuthenticationError):
            await provider.chat("m", [ChatMessage("user", "hi")])

    async def test_rate_limit_maps_with_retry_after(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response({"error": {"message": "slow down"}}, status=429, **{"retry-after": "7"})

        provider = make_provider(handler, attempts=1)
        with pytest.raises(RateLimitError, match="retry after 7s"):
            await provider.chat("m", [ChatMessage("user", "hi")])

    async def test_server_error_is_upstream_and_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return text_response("gateway trouble", status=503)
            return text_response("recovered")

        provider = make_provider(handler, attempts=3)
        response = await provider.chat("m", [ChatMessage("user", "hi")])
        assert response.content == "recovered"
        assert len(calls) == 3

    async def test_exhausted_retries_raise_upstream_error(self) -> None:
        provider = make_provider(lambda request: text_response("boom", status=500), attempts=2)
        with pytest.raises(UpstreamError) as excinfo:
            await provider.chat("m", [ChatMessage("user", "hi")])
        assert excinfo.value.status_code == 500

    async def test_4xx_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return text_response("nope", status=400)

        provider = make_provider(handler, attempts=4)
        with pytest.raises(ProviderError):
            await provider.chat("m", [ChatMessage("user", "hi")])
        assert len(calls) == 1

    async def test_non_json_error_body_is_truncated_into_the_message(self) -> None:
        provider = make_provider(lambda request: text_response("<html>oops</html>", status=400))
        with pytest.raises(ProviderError, match="oops"):
            await provider.chat("m", [ChatMessage("user", "hi")])

    async def test_transport_failure_becomes_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        provider = make_provider(handler, attempts=2)
        with pytest.raises(UpstreamError, match="no route to host"):
            await provider.chat("m", [ChatMessage("user", "hi")])


class TestStreaming:
    async def test_chunks_are_forwarded_and_terminated(self) -> None:
        provider = make_provider(lambda request: text_response("hello stream"))
        chunks = [chunk async for chunk in provider.stream("m", [ChatMessage("user", "hi")])]
        assert "".join(chunk.delta for chunk in chunks) == "hello stream"
        assert chunks[-1].finish_reason == "stop"
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)

    async def test_validation_happens_before_the_request(self) -> None:
        provider = make_provider(lambda request: text_response("ok"), max_messages=2)
        with pytest.raises(ProviderError):
            async for _ in provider.stream("m", [ChatMessage("user", "a")] * 3):
                pass

    async def test_stream_errors_are_mapped(self) -> None:
        provider = make_provider(lambda request: json_response({"error": {"message": "no"}}, status=403))
        with pytest.raises(AuthenticationError):
            async for _ in provider.stream("m", [ChatMessage("user", "hi")]):
                pass

    async def test_interruption_raises_stream_interrupted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                stream=ExplodingStream(),
                headers={"content-type": "text/plain"},
                request=httpx.Request("POST", ENDPOINT),
            )

        provider = make_provider(handler)
        parts: list[str] = []
        with pytest.raises(StreamInterruptedError, match="ended early"):
            async for chunk in provider.stream("openai/gpt-4o", [ChatMessage("user", "hi")]):
                parts.append(chunk.delta)
        assert parts == ["partial "]


class TestModelListing:
    async def test_static_catalog_is_returned(self) -> None:
        assert await Fable5Provider().list_models() == [DEFAULT_MODEL]

    async def test_catalog_is_overridable(self) -> None:
        provider = Fable5Provider({"models": ["openai/gpt-4o", "openai/gpt-4.1"]})
        assert await provider.list_models() == ["openai/gpt-4o", "openai/gpt-4.1"]


class TestClientIntegration:
    @staticmethod
    def client_for(handler: Callable[[httpx.Request], httpx.Response], **extra: Any) -> OinyxClient:
        return OinyxClient(
            config={"fable5": {"transport": httpx.MockTransport(handler), "backoff": 0, **extra}}
        )

    async def test_routing_through_oinyx_client(self) -> None:
        client = self.client_for(lambda request: text_response("via client"))
        response = await client.chat("fable5/openai/gpt-4o", "hi")
        assert response.content == "via client"
        assert response.model == DEFAULT_MODEL

    async def test_bare_model_name_uses_an_alias(self) -> None:
        client = self.client_for(lambda request: text_response("x"), )
        client.router.add_alias("chat", "fable5/openai/gpt-4o")
        assert (await client.chat("chat", "hi")).content == "x"

    async def test_streaming_via_client(self) -> None:
        client = self.client_for(lambda request: text_response("abc"))
        chunks = [chunk async for chunk in client.stream("fable5/m", "hi")]
        assert "".join(chunk.delta for chunk in chunks) == "abc"

    async def test_collect_via_client(self) -> None:
        client = self.client_for(lambda request: text_response("whole reply"))
        response = await client.collect("fable5/openai/gpt-4o", "hi")
        assert response.content == "whole reply"
        assert response.finish_reason == "stop"


def json_loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)
