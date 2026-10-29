"""Tests for the OpenAI-shaped dataclasses in :mod:`oinyx.models`."""

from __future__ import annotations

import pytest

from oinyx.models import ChatMessage, ChatResponse, StreamChunk, Usage


class TestChatMessage:
    def test_defaults(self) -> None:
        message = ChatMessage(role="user", content="hi")
        assert message.name is None

    def test_to_dict_omits_absent_name(self) -> None:
        assert ChatMessage("user", "hi").to_dict() == {"role": "user", "content": "hi"}

    def test_to_dict_keeps_name(self) -> None:
        assert ChatMessage("tool", "ok", name="calc").to_dict() == {
            "role": "tool",
            "content": "ok",
            "name": "calc",
        }

    def test_round_trip(self) -> None:
        message = ChatMessage("assistant", "done", name="bot")
        assert ChatMessage.from_dict(message.to_dict()) == message

    def test_from_dict_tolerates_missing_fields(self) -> None:
        message = ChatMessage.from_dict({})
        assert (message.role, message.content) == ("user", "")

    def test_from_dict_handles_null_content(self) -> None:
        assert ChatMessage.from_dict({"role": "assistant", "content": None}).content == ""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hello", ChatMessage("user", "hello")),
            ({"role": "system", "content": "be brief"}, ChatMessage("system", "be brief")),
        ],
    )
    def test_coerce(self, value: object, expected: ChatMessage) -> None:
        assert ChatMessage.coerce(value) == expected  # type: ignore[arg-type]

    def test_coerce_passes_instances_through(self) -> None:
        message = ChatMessage("user", "x")
        assert ChatMessage.coerce(message) is message

    def test_coerce_rejects_other_types(self) -> None:
        with pytest.raises(TypeError, match="cannot coerce int"):
            ChatMessage.coerce(7)  # type: ignore[arg-type]


class TestUsage:
    def test_total_is_derived_when_missing(self) -> None:
        assert Usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_explicit_total_is_kept(self) -> None:
        assert Usage(1, 2, 99).total_tokens == 99

    def test_from_dict_partial(self) -> None:
        usage = Usage.from_dict({"prompt_tokens": 4})
        assert usage == Usage(4, 0, 4)

    def test_from_dict_none(self) -> None:
        assert Usage.from_dict(None) == Usage(0, 0, 0)

    def test_round_trip(self) -> None:
        usage = Usage(3, 4, 7)
        assert Usage.from_dict(usage.to_dict()) == usage


class TestChatResponse:
    def test_round_trip(self) -> None:
        response = ChatResponse(
            id="abc",
            model="m",
            content="hello",
            finish_reason="stop",
            usage=Usage(1, 2, 3),
            raw={"k": "v"},
        )
        assert ChatResponse.from_dict(response.to_dict()) == response

    def test_defaults(self) -> None:
        response = ChatResponse()
        assert response.content == "" and response.finish_reason is None
        assert response.usage == Usage(0, 0, 0) and response.raw == {}

    def test_from_dict_missing_usage(self) -> None:
        assert ChatResponse.from_dict({"id": "x"}).usage == Usage(0, 0, 0)


class TestStreamChunk:
    def test_round_trip(self) -> None:
        chunk = StreamChunk(delta="tok", finish_reason=None, raw={"a": 1})
        assert StreamChunk.from_dict(chunk.to_dict()) == chunk

    def test_defaults(self) -> None:
        assert StreamChunk() == StreamChunk(delta="", finish_reason=None, raw={})
