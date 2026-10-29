"""Offline tests for the bundled ``perchance`` text provider.

Traffic is served by :class:`httpx.MockTransport`; the Cloudflare paths are
simulated with the same headers the real endpoint returns.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
    estimate_tokens,
)
from providers.perchance import DEFAULT_MODEL, ENDPOINT, KNOWN_MODELS, PerchanceProvider
from providers.cloudflare_base import CloudflareChallengeError, UpstreamError

CHALLENGE_BODY = "<html><head>Just a moment...</head><body>checking your browser</body></html>"


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> PerchanceProvider:
    """Provider wired to a mock transport, with retries that do not sleep."""
    return PerchanceProvider({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


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


def challenge_response() -> httpx.Response:
    return text_response(CHALLENGE_BODY, status=403, **{"cf-mitigated": "challenge"})


@pytest.fixture(autouse=True)
def registered() -> None:
    """Re-register the provider after the global registry was cleared."""
    ProviderRegistry.register(PerchanceProvider)


class TestRegistration:
    def test_it_is_registered_under_perchance(self) -> None:
        assert ProviderRegistry.list_providers() == ["perchance"]
        assert isinstance(ProviderRegistry.get("perchance"), PerchanceProvider)

    def test_defaults(self) -> None:
        provider = PerchanceProvider()
        assert provider.name == "perchance"
        assert provider.endpoint == ENDPOINT
        assert provider.default_generator == DEFAULT_MODEL
        assert provider.label_roles is True
        assert repr(provider).startswith("<PerchanceProvider")


class TestPayload:
    async def test_body_matches_the_documented_shape(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"generatedText": "ok"})

        instruction = "User: test\nAssistant:\n"
        await make_provider(handler).chat(
            "ai-code-generator", [ChatMessage("user", "test")]
        )
        assert captured[0] == {
            "instruction": instruction,
            "startWith": "",
            "stopSequences": [],
            "generatorName": "ai-code-generator",
            "startWithTokenCount": 0,
            "instructionTokenCount": estimate_tokens(instruction),
        }

    async def test_model_is_used_as_the_generator_name(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"generatedText": "x"})

        await make_provider(handler).chat("ai-story-writer", [ChatMessage("user", "once")])
        assert captured[0]["generatorName"] == "ai-story-writer"

    async def test_explicit_generator_name_wins(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"generatedText": "x"})

        await make_provider(handler).chat(
            "ai-chat", [ChatMessage("user", "hi")], generatorName="ai-code-generator", temperature=1
        )
        assert captured[0]["generatorName"] == "ai-code-generator"
        assert captured[0]["temperature"] == 1

    async def test_config_can_seed_the_generators_fields(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"generatedText": "x"})

        provider = make_provider(
            handler,
            generator_name="ai-code-generator",
            start_with="def ",
            stop_sequences=["\n\n"],
        )
        await provider.chat("", [ChatMessage("user", "hi")])
        assert captured[0]["generatorName"] == "ai-code-generator"
        assert captured[0]["startWith"] == "def "
        assert captured[0]["stopSequences"] == ["\n\n"]

    async def test_request_carries_browser_headers_and_cookies(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response({"generatedText": "x"})

        provider = make_provider(
            handler,
            cookies={"cf_clearance": "abc", "cf_bm": "def"},
            user_agent="Browser/1.0",
            headers={"x-test": "1"},
        )
        await provider.chat("ai-chat", [ChatMessage("user", "hi")])
        assert seen[0].headers["cookie"] == "cf_clearance=abc; cf_bm=def"
        assert seen[0].headers["user-agent"] == "Browser/1.0"
        assert seen[0].headers["origin"] == "https://perchance.org"
        assert seen[0].headers["x-test"] == "1"

    async def test_cookie_string_is_passed_through(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response({"generatedText": "x"})

        await make_provider(handler, cookies=" cf_bm=xyz; cf_clearance=q ").chat(
            "ai-chat", [ChatMessage("user", "hi")]
        )
        assert seen[0].headers["cookie"] == "cf_bm=xyz; cf_clearance=q"


class TestInstruction:
    def test_roles_are_labelled_and_the_reply_is_seeded(self) -> None:
        provider = PerchanceProvider()
        turns = [ChatMessage("system", "be brief"), ChatMessage("user", "hi")]
        assert provider.render_instruction(turns) == "System: be brief\nUser: hi\nAssistant:\n"

    def test_an_assistant_turn_is_not_followed_by_a_seed(self) -> None:
        provider = PerchanceProvider()
        turns = [ChatMessage("user", "hi"), ChatMessage("assistant", "yo")]
        assert provider.render_instruction(turns).endswith("Assistant: yo\n")

    def test_labels_can_be_disabled(self) -> None:
        provider = PerchanceProvider({"label_roles": False})
        turns = [ChatMessage("user", "hi"), ChatMessage("assistant", "yo")]
        assert provider.render_instruction(turns) == "hi\nyo"

    def test_no_messages_no_instruction(self) -> None:
        assert PerchanceProvider().render_instruction([]) == ""


class TestResponseParsing:
    async def test_json_completion_becomes_a_chat_response(self) -> None:
        provider = make_provider(lambda request: json_response({"generatedText": "hello world"}))
        response = await provider.chat("ai-chat", [ChatMessage("user", "hi")])
        assert response.content == "hello world"
        assert response.model == "ai-chat"
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens > 0
        assert response.raw["status_code"] == 200
        assert response.raw["instruction_tokens"] == estimate_tokens("User: hi\nAssistant:\n")

    async def test_plain_text_body_is_returned_verbatim(self) -> None:
        provider = make_provider(lambda request: text_response("raw words"))
        response = await provider.chat("ai-chat", [ChatMessage("user", "hi")])
        assert response.content == "raw words"

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"text": "from text"}, "from text"),
            ({"content": "from content"}, "from content"),
            ({"completion": "from completion"}, "from completion"),
            (["not", "a", "dict"], None),
            ({"unexpected": 1}, None),
        ],
    )
    async def test_field_fallbacks(self, payload: Any, expected: str | None) -> None:
        response = json_response(payload)
        extracted = PerchanceProvider.extract_text(response)
        assert extracted == (expected if expected is not None else response.text)

    async def test_a_json_string_body_is_the_completion(self) -> None:
        assert PerchanceProvider.extract_text(json_response("quoted")) == "quoted"

    async def test_unparseable_json_falls_back_to_the_body(self) -> None:
        response = text_response("{not json", **{"content-type": "application/json"})
        assert PerchanceProvider.extract_text(response) == "{not json"


class TestStreaming:
    async def test_the_completion_is_re_emitted_in_word_groups(self) -> None:
        provider = make_provider(
            lambda request: json_response({"generatedText": "one two three four five"})
        )
        chunks = [chunk async for chunk in provider.stream("ai-chat", [ChatMessage("user", "hi")])]
        assert "".join(f"{chunk.delta} " for chunk in chunks).split()[:5] == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]
        assert chunks[-1].delta == ""
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].raw == {"emulated": True}

    async def test_chunk_size_is_configurable(self) -> None:
        provider = make_provider(
            lambda request: json_response({"generatedText": "a b c d"}), stream_chunk_words=2
        )
        deltas = [
            chunk.delta async for chunk in provider.stream("ai-chat", [ChatMessage("user", "hi")])
        ]
        assert deltas == ["a b", "c d", ""]


class TestErrorMapping:
    async def test_challenge_is_reported_actionably(self) -> None:
        provider = make_provider(lambda request: challenge_response(), attempts=1)
        with pytest.raises(CloudflareChallengeError) as excinfo:
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])
        assert "cf_clearance" in str(excinfo.value)
        assert excinfo.value.status_code == 403

    async def test_challenge_detected_from_the_body_alone(self) -> None:
        provider = make_provider(lambda request: text_response(CHALLENGE_BODY, status=503), attempts=1)
        with pytest.raises(CloudflareChallengeError):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_forbidden_without_challenge_markers_is_authentication(self) -> None:
        provider = make_provider(lambda request: json_response({"error": "denied"}, status=403), attempts=1)
        with pytest.raises(AuthenticationError):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_rate_limit_reports_retry_after(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response({"error": {"message": "slow"}}, status=429, **{"retry-after": "5"})

        with pytest.raises(RateLimitError, match="retry after 5s"):
            await make_provider(handler, attempts=1).chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_bad_request_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return text_response("bad generator", status=400)

        with pytest.raises(ProviderError, match="bad generator"):
            await make_provider(handler, attempts=4).chat("ai-chat", [ChatMessage("user", "hi")])
        assert len(calls) == 1

    async def test_server_errors_are_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return text_response("flaky", status=500)
            return json_response({"generatedText": "late reply"})

        response = await make_provider(handler, attempts=3).chat("ai-chat", [ChatMessage("user", "hi")])
        assert response.content == "late reply"
        assert len(calls) == 2

    async def test_transport_failure_becomes_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure")

        with pytest.raises(UpstreamError, match="dns failure"):
            await make_provider(handler, attempts=1).chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_non_json_error_detail_is_truncated(self) -> None:
        provider = make_provider(lambda request: text_response("<b>nope</b>", status=400), attempts=1)
        with pytest.raises(ProviderError, match="nope"):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_a_json_list_error_body_is_stringified(self) -> None:
        provider = make_provider(lambda request: json_response([1, 2], status=400), attempts=1)
        with pytest.raises(ProviderError, match=r"\[1, 2\]"):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_a_non_string_error_object_is_stringified(self) -> None:
        provider = make_provider(lambda request: json_response({"error": 5}, status=400), attempts=1)
        with pytest.raises(ProviderError, match="upstream_error"):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_rate_limit_without_retry_after(self) -> None:
        provider = make_provider(lambda request: json_response({"message": "slow"}, status=429), attempts=1)
        with pytest.raises(RateLimitError):
            await provider.chat("ai-chat", [ChatMessage("user", "hi")])

    async def test_an_injected_client_is_reused_and_not_closed(self) -> None:
        handler = httpx.MockTransport(lambda request: json_response({"generatedText": "shared"}))
        async with httpx.AsyncClient(transport=handler) as shared:
            provider = PerchanceProvider({"client": shared, "backoff": 0})
            assert (await provider.chat("ai-chat", [ChatMessage("user", "hi")])).content == "shared"
            assert (await provider.chat("ai-chat", [ChatMessage("user", "hi")])).content == "shared"


class TestModelListing:
    async def test_static_catalog_is_returned(self) -> None:
        assert await PerchanceProvider().list_models() == KNOWN_MODELS

    async def test_catalog_is_overridable(self) -> None:
        provider = PerchanceProvider({"models": ["ai-chat"]})
        assert await provider.list_models() == ["ai-chat"]


class TestClientIntegration:
    @staticmethod
    def client_for(handler: Callable[[httpx.Request], httpx.Response], **extra: Any) -> OinyxClient:
        return OinyxClient(
            config={"perchance": {"transport": httpx.MockTransport(handler), "backoff": 0, **extra}}
        )

    async def test_routing_through_oinyx_client(self) -> None:
        client = self.client_for(lambda request: json_response({"generatedText": "via client"}))
        response = await client.chat("perchance/ai-chat", "hi")
        assert response.content == "via client"
        assert response.model == "ai-chat"

    async def test_streaming_via_client(self) -> None:
        client = self.client_for(lambda request: json_response({"generatedText": "abc def"}))
        chunks = [chunk async for chunk in client.stream("perchance/ai-chat", "hi")]
        assert "".join(chunk.delta for chunk in chunks) == "abc def"

    async def test_collect_via_client(self) -> None:
        client = self.client_for(lambda request: json_response({"generatedText": "whole thing"}))
        response = await client.collect("perchance/ai-chat", "hi")
        assert response.content == "whole thing"
        assert response.finish_reason == "stop"
