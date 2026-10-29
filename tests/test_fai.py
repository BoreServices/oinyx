"""Offline tests for the bundled ``fai`` provider (aifreeforever.com).

The request body and the ``text/event-stream`` reply are pinned to the captured
browser traffic; every frame shape the parser accepts is exercised here with an
:class:`httpx.MockTransport`, so the suite never reaches the network.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from oinyx import (
    ChatMessage,
    OinyxClient,
    ProviderError,
    ProviderRegistry,
    RateLimitError,
    StreamInterruptedError,
)
from oinyx.exceptions import AuthenticationError
from oinyx import ChatResponse
from providers.cloudflare_base import CloudflareChallengeError, UpstreamError
from providers.fai import (
    CHAT_PAGE,
    DEFAULT_MODEL,
    DEFAULT_USER_AGENT,
    ENDPOINT,
    KNOWN_MODELS,
    MODELS_ENDPOINT,
    ORIGIN,
    FaiProvider,
    base36,
    event_text,
    mint_nonce,
    parse_model_list,
    split_conversation,
    verify,
)

NONCE = re.compile(r"^[0-9a-z]{8}\.[0-9a-f]{32}\.[0-9a-f]{40}$")

BODY_KEYS = {
    "timeZone",
    "model",
    "question",
    "tone",
    "format",
    "file",
    "conversationHistory",
    "interactionProof",
    "resend",
    "aiRole",
    "aiName",
    "language",
    "followUps",
}

PROOF_KEYS = {"nonce", "keystrokeCount", "pasteEvents", "totalTypingTime", "startTime", "submitTime"}


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> FaiProvider:
    """Provider wired to a mock transport, with retries that do not sleep."""
    return FaiProvider({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


def sse_response(frames: list[str], status: int = 200, **headers: str) -> httpx.Response:
    body = "".join(f"data: {frame}\n\n" for frame in frames)
    return httpx.Response(
        status,
        text=body,
        headers={"content-type": "text/event-stream", **headers},
        request=httpx.Request("POST", ENDPOINT),
    )


def json_response(payload: Any, status: int = 200, url: str = ENDPOINT) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", url),
    )


def text_response(body: str, status: int = 200, content_type: str = "text/plain") -> httpx.Response:
    return httpx.Response(
        status,
        text=body,
        headers={"content-type": content_type},
        request=httpx.Request("POST", ENDPOINT),
    )


class ChunkStream(httpx.AsyncByteStream):
    """Replays fixed byte chunks so SSE behaviour is deterministic."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def stream_response(chunks: list[bytes], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        stream=ChunkStream(chunks),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", ENDPOINT),
    )


@pytest.fixture(autouse=True)
def registered() -> None:
    """Re-register the provider after the global registry was cleared."""
    ProviderRegistry.register(FaiProvider)


class TestRegistration:
    def test_it_is_registered_under_fai(self) -> None:
        assert ProviderRegistry.list_providers() == ["fai"]
        assert isinstance(ProviderRegistry.get("fai"), FaiProvider)

    async def test_defaults(self) -> None:
        provider = FaiProvider()
        assert provider.name == "fai"
        assert provider.endpoint == ENDPOINT
        assert provider.origin == ORIGIN
        assert provider.user_agent == DEFAULT_USER_AGENT
        assert provider.default_model == DEFAULT_MODEL
        assert provider.time_zone == "Asia/Dhaka"
        assert provider.tone == "friendly"
        assert provider.output_format == "paragraph"
        assert provider.follow_ups is True
        assert provider.resend is False
        assert await provider.list_models() == KNOWN_MODELS
        assert repr(provider).startswith("<FaiProvider")

    async def test_catalog_is_overridable(self) -> None:
        provider = FaiProvider({"models": ["gemini-3-5", "claude-4"]})
        assert await provider.list_models() == ["gemini-3-5", "claude-4"]


class TestPayload:
    def test_key_set_matches_the_capture(self) -> None:
        provider = FaiProvider()
        payload = provider.build_payload(DEFAULT_MODEL, [ChatMessage("user", "test")], {})
        assert set(payload) == BODY_KEYS
        assert set(payload["interactionProof"]) == PROOF_KEYS

    def test_values_match_the_capture(self) -> None:
        provider = FaiProvider()
        payload = provider.build_payload("gemini-3-5", [ChatMessage("user", "test")], {})
        assert payload["timeZone"] == "Asia/Dhaka"
        assert payload["model"] == "gemini-3-5"
        assert payload["question"] == "test"
        assert payload["tone"] == "friendly"
        assert payload["format"] == "paragraph"
        assert payload["file"] is None
        assert payload["conversationHistory"] == []
        assert payload["resend"] is False
        assert payload["aiRole"] == "assistant"
        assert payload["aiName"] == ""
        assert payload["language"] == ""
        assert payload["followUps"] is True

    def test_the_prefix_is_stripped_from_the_model(self) -> None:
        provider = FaiProvider()
        payload = provider.build_payload("fai/gemini-3-5", [ChatMessage("user", "hi")], {})
        assert payload["model"] == "gemini-3-5"

    def test_a_missing_model_falls_back_to_the_default(self) -> None:
        assert FaiProvider().build_payload("", [ChatMessage("user", "hi")], {})["model"] == DEFAULT_MODEL

    def test_proof_mirrors_the_typed_question(self) -> None:
        provider = FaiProvider()
        proof = provider.build_payload(DEFAULT_MODEL, [ChatMessage("user", "test")], {})[
            "interactionProof"
        ]
        assert proof["keystrokeCount"] == 4
        assert proof["pasteEvents"] == 0
        assert NONCE.match(proof["nonce"])
        assert proof["submitTime"] - proof["startTime"] == proof["totalTypingTime"]

    def test_configured_options_reach_the_body(self) -> None:
        provider = FaiProvider(
            {
                "time_zone": "UTC",
                "tone": "concise",
                "format": "bullets",
                "ai_role": "helper",
                "ai_name": "Fai",
                "language": "bn",
                "follow_ups": False,
                "resend": True,
                "file": "photo.png",
                "total_typing_time": 2807,
            }
        )
        payload = provider.build_payload(DEFAULT_MODEL, [ChatMessage("user", "hi")], {})
        assert payload["timeZone"] == "UTC"
        assert payload["tone"] == "concise"
        assert payload["format"] == "bullets"
        assert payload["aiRole"] == "helper"
        assert payload["aiName"] == "Fai"
        assert payload["language"] == "bn"
        assert payload["followUps"] is False
        assert payload["resend"] is True
        assert payload["file"] == "photo.png"
        assert payload["interactionProof"]["totalTypingTime"] == 2807

    def test_per_call_options_win_over_config(self) -> None:
        provider = FaiProvider({"tone": "friendly"})
        payload = provider.build_payload(
            DEFAULT_MODEL,
            [ChatMessage("user", "hi")],
            {"tone": "formal", "format": "table", "question": "override", "follow_ups": False},
        )
        assert payload["tone"] == "formal"
        assert payload["format"] == "table"
        assert payload["question"] == "override"
        assert payload["followUps"] is False

    def test_unknown_kwargs_are_forwarded_and_none_is_dropped(self) -> None:
        provider = FaiProvider()
        payload = provider.build_payload(
            DEFAULT_MODEL, [ChatMessage("user", "hi")], {"temperature": 0.4, "skip": None}
        )
        assert payload["temperature"] == 0.4
        assert "skip" not in payload

    def test_a_captured_proof_is_used_verbatim(self) -> None:
        captured = {
            "nonce": "mvto0lmt.5f2b01730dd28c48ddbc529e2aed103b."
            "d9474916a504027e59e1bba87c058a3c5da6654887488f21d9ddd16171c46bf5",
            "keystrokeCount": 4,
            "pasteEvents": 0,
            "totalTypingTime": 2807,
            "startTime": 1793285739448,
            "submitTime": 1793285742255,
        }
        provider = FaiProvider({"proof": captured})
        payload = provider.build_payload(DEFAULT_MODEL, [ChatMessage("user", "hello")], {})
        assert payload["interactionProof"] == captured

        per_call = provider.build_payload(
            DEFAULT_MODEL, [ChatMessage("user", "hello")], {"proof": {**captured, "nonce": "x"}}
        )
        assert per_call["interactionProof"]["nonce"] == "x"


class TestConversation:
    def test_the_last_user_turn_is_the_question(self) -> None:
        turns = [
            ChatMessage("system", "be brief"),
            ChatMessage("user", "first"),
            ChatMessage("assistant", "answer"),
            ChatMessage("user", "second"),
        ]
        question, history = split_conversation(turns)
        assert question == "second"
        assert history == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_an_assistant_only_thread_has_no_question(self) -> None:
        question, history = split_conversation([ChatMessage("assistant", "hi")])
        assert question == ""
        assert history == [{"role": "assistant", "content": "hi"}]

    def test_history_is_trimmed_to_the_newest_turns(self) -> None:
        provider = FaiProvider({"max_history": 1})
        turns = [ChatMessage("user", "a"), ChatMessage("assistant", "b"), ChatMessage("user", "c")]
        payload = provider.build_payload(DEFAULT_MODEL, turns, {})
        assert payload["conversationHistory"] == [{"role": "assistant", "content": "b"}]

    def test_history_can_be_supplied_per_call(self) -> None:
        provider = FaiProvider()
        payload = provider.build_payload(
            DEFAULT_MODEL,
            [ChatMessage("user", "hi")],
            {"history": [{"role": "user", "content": "kept"}, {"bogus": True}]},
        )
        assert payload["conversationHistory"] == [{"role": "user", "content": "kept"}]

    def test_trimming_can_be_disabled(self) -> None:
        provider = FaiProvider({"max_history": 0})
        long = [{"role": "user", "content": str(index)} for index in range(60)]
        assert len(provider.trim_history(long, None)) == 60


class TestHeaders:
    async def test_referer_tracks_the_chat_page(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return sse_response(['{"text": "ok"}'])

        await make_provider(handler).chat("claude-4", [ChatMessage("user", "hi")])
        assert seen[0].url == ENDPOINT
        assert seen[0].headers["referer"] == f"{CHAT_PAGE}/claude-4"
        assert "text/event-stream" in seen[0].headers["accept"]
        assert seen[0].headers["origin"] == ORIGIN
        assert seen[0].headers["user-agent"] == DEFAULT_USER_AGENT

    async def test_browser_cookies_are_replayed(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return sse_response(['{"text": "ok"}'])

        clearance = "CthEYuEg_utYuZChdXEjmNMF9jx7NtiY"
        await make_provider(handler, cookies={"cf_clearance": clearance, "aiff_dsid": "e2767bd9"}).chat(
            DEFAULT_MODEL, [ChatMessage("user", "hi")]
        )
        assert seen[0].headers["cookie"] == f"cf_clearance={clearance}; aiff_dsid=e2767bd9"


class TestChat:
    async def test_sse_frames_are_joined_into_one_answer(self) -> None:
        provider = make_provider(
            lambda request: sse_response(
                ['{"text": "Hel"}', '{"text": "lo"}', "[DONE]"],
                **{"x-zeabur-request-id": "a32aac85"},
            )
        )
        response = await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.content == "Hello"
        assert response.model == DEFAULT_MODEL
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens > 0
        assert response.raw["request_id"] == "a32aac85"
        assert response.raw["content_type"] == "text/event-stream"

    @pytest.mark.parametrize(
        "frame",
        [
            '{"text": "plain field"}',
            '{"content": "content field"}',
            '{"answer": "answer field"}',
            '{"data": {"message": "nested field"}}',
            '{"choices": [{"delta": {"content": "openai delta"}}]}',
            '{"choices": [{"message": {"content": "openai message"}}]}',
            '{"choices": [{"text": "openai text"}]}',
            "bare text",
        ],
    )
    async def test_every_frame_shape_yields_text(self, frame: str) -> None:
        provider = make_provider(lambda request: sse_response([frame]))
        response = await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.content

    async def test_a_finish_reason_in_a_frame_is_honoured(self) -> None:
        provider = make_provider(
            lambda request: sse_response(['{"text": "stop", "finishReason": "length"}'])
        )
        response = await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.finish_reason == "length"

    async def test_a_json_body_is_understood_too(self) -> None:
        provider = make_provider(lambda request: json_response({"answer": "from json"}))
        response = await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.content == "from json"

    async def test_a_plain_body_is_returned_verbatim(self) -> None:
        provider = make_provider(lambda request: text_response("raw words"))
        assert (await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])).content == "raw words"

    async def test_an_empty_body_yields_an_empty_answer(self) -> None:
        provider = make_provider(lambda request: text_response(""))
        response = await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.content == ""
        assert response.finish_reason == "stop"

    async def test_an_error_frame_becomes_a_provider_error(self) -> None:
        provider = make_provider(lambda request: sse_response(['{"error": {"message": "quota exhausted"}}']))
        with pytest.raises(ProviderError, match="quota exhausted"):
            await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])


class TestStreaming:
    async def test_deltas_arrive_in_order_and_terminate(self) -> None:
        frames = [
            b'data: {"text": "one "}\n\n',
            b'event: ping\n\n',
            b'data: {"text": "two"}\n\n',
            b"data: [DONE]\n\n",
        ]
        provider = make_provider(lambda request: stream_response(frames))
        chunks = [chunk async for chunk in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")])]
        assert "".join(chunk.delta for chunk in chunks) == "one two"
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].raw == {"done": True}

    async def test_a_stream_without_a_termination_gets_one(self) -> None:
        provider = make_provider(lambda request: stream_response([b'data: {"text": "x"}\n\n']))
        chunks = [chunk async for chunk in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")])]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].raw == {"synthesized": True}

    async def test_a_non_streaming_reply_is_still_delivered(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"answer": "buffered"},
                headers={"content-type": "application/json"},
                request=httpx.Request("POST", ENDPOINT),
            )

        provider = make_provider(handler)
        chunks = [chunk async for chunk in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")])]
        assert [chunk.delta for chunk in chunks] == ["buffered"]
        assert chunks[-1].finish_reason == "stop"

    async def test_malformed_json_frame_raises_stream_interrupted(self) -> None:
        provider = make_provider(lambda request: stream_response([b"data: {\"text\": \n\n"]))
        with pytest.raises(StreamInterruptedError, match="malformed SSE"):
            async for _ in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")]):
                pass

    async def test_errors_are_mapped_before_the_first_chunk(self) -> None:
        provider = make_provider(lambda request: text_response("denied", status=403))
        with pytest.raises(AuthenticationError):
            async for _ in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")]):
                pass

    async def test_transport_failure_becomes_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("reset by peer")

        provider = make_provider(handler)
        with pytest.raises(UpstreamError, match="reset by peer"):
            async for _ in provider.stream(DEFAULT_MODEL, [ChatMessage("user", "hi")]):
                pass


class TestErrorMapping:
    async def test_challenge_is_reported_actionably(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return text_response("<html>Just a moment...</html>", status=403, content_type="text/html")

        provider = make_provider(handler, attempts=1)
        with pytest.raises(CloudflareChallengeError) as excinfo:
            await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert "cf_clearance" in str(excinfo.value)

    async def test_rate_limit_and_server_errors(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "slow"}, status=429), attempts=1)
        with pytest.raises(RateLimitError):
            await provider.chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])

        calls: list[int] = []

        def flaky(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return text_response("busy", status=503)
            return sse_response(['{"text": "late"}'])

        response = await make_provider(flaky, attempts=3).chat(DEFAULT_MODEL, [ChatMessage("user", "hi")])
        assert response.content == "late"
        assert len(calls) == 2


class TestModelCatalog:
    async def test_the_catalog_is_parsed(self) -> None:
        payload = {"models": [{"id": "gemini-3-5", "name": "Gemini"}, {"slug": "claude-4"}, "plain"]}
        provider = make_provider(lambda request: json_response(payload, url=MODELS_ENDPOINT))
        assert await provider.fetch_models() == ["claude-4", "gemini-3-5", "plain"]

    async def test_a_non_json_catalog_is_an_error(self) -> None:
        provider = make_provider(lambda request: text_response("<html>models</html>"))
        with pytest.raises(ProviderError, match="did not return JSON"):
            await provider.fetch_models()

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (["b", "a", "a"], ["a", "b"]),
            ({"data": [{"name": "n"}]}, ["n"]),
            ({"keyed": {"x": 1}}, ["keyed"]),
            ({"items": {"m1": {}, "m2": {}}}, ["m1", "m2"]),
            ([{"untitled": True}], []),
            ("not a list", []),
        ],
    )
    def test_parse_model_list_shapes(self, payload: Any, expected: list[str]) -> None:
        assert parse_model_list(payload) == expected

    def test_event_text_prefers_known_fields(self) -> None:
        assert event_text({"text": "t", "content": "c"}) == "t"
        assert event_text({"choices": ["not a dict"]}) == ""
        assert event_text({"other": 1}) == ""


class TestNonce:
    def test_base36_round_trip(self) -> None:
        assert base36(int("mvto0lmt", 36)) == "mvto0lmt"
        assert base36(0) == "0"

    def test_minted_nonce_is_earlier_than_the_submit(self) -> None:
        start = 1793285739448
        nonce = mint_nonce(start)
        assert NONCE.match(nonce)
        assert int(nonce.split(".", 1)[0], 36) < start


class TestProbe:
    async def test_verify_reports_models_and_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fetch(self: FaiProvider) -> list[str]:
            return ["gemini-3-5"]

        async def fake_chat(self: FaiProvider, model: str, messages: Any, **kwargs: Any) -> ChatResponse:
            return ChatResponse(model=model, content="pong")

        monkeypatch.setattr(FaiProvider, "fetch_models", fake_fetch)
        monkeypatch.setattr(FaiProvider, "chat", fake_chat)
        assert await verify() == {"models": ["gemini-3-5"], "answer": "pong", "error": None}

    async def test_verify_reports_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(self: FaiProvider) -> list[str]:
            raise ProviderError("blocked", provider="fai")

        monkeypatch.setattr(FaiProvider, "fetch_models", boom)
        report = await verify()
        assert report["models"] is None
        assert "blocked" in (report["error"] or "")

    def test_main_reports_a_blocked_probe(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from providers.fai import main

        async def fake_verify(model: str = DEFAULT_MODEL, question: str = "ping") -> dict[str, Any]:
            return {"models": [], "answer": None, "error": "CloudflareChallengeError: blocked"}

        monkeypatch.setattr("providers.fai.verify", fake_verify)
        assert main() == 1
        assert "blocked" in capsys.readouterr().out


class TestClientIntegration:
    @staticmethod
    def client_for(handler: Callable[[httpx.Request], httpx.Response], **extra: Any) -> OinyxClient:
        return OinyxClient(
            config={"fai": {"transport": httpx.MockTransport(handler), "backoff": 0, **extra}}
        )

    async def test_routing_through_oinyx_client(self) -> None:
        client = self.client_for(lambda request: sse_response(['{"text": "via client"}', "[DONE]"]))
        response = await client.chat("fai/gemini-3-5", "hi")
        assert response.content == "via client"
        assert response.model == DEFAULT_MODEL

    async def test_streaming_via_client(self) -> None:
        client = self.client_for(lambda request: stream_response([b'data: {"text": "abc"}\n\n']))
        chunks = [chunk async for chunk in client.stream("fai/gemini-3-5", "hi")]
        assert "".join(chunk.delta for chunk in chunks) == "abc"

    def test_chat_sync_round_trip(self) -> None:
        client = self.client_for(lambda request: sse_response(['{"text": "sync"}', "[DONE]"]))
        assert client.chat_sync("fai/gemini-3-5", "hi").content == "sync"
