"""Tests for the :class:`oinyx.BaseProvider` contract."""

from __future__ import annotations

from typing import Any

import pytest

from oinyx.base import BaseProvider
from oinyx.models import ChatMessage, ChatResponse


class Partial(BaseProvider):
    """Implements only the required members, leaving the optional default."""

    name = "partial"

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(model=model)

    async def stream(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        if False:
            yield None


def test_base_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]


def test_missing_stream_is_rejected() -> None:
    class NoStream(BaseProvider):
        name = "nostream"

        async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> None: ...

    with pytest.raises(TypeError, match="abstract"):
        NoStream()  # type: ignore[abstract]


async def test_default_list_models_is_empty() -> None:
    assert await Partial().list_models() == []


def test_config_is_copied_not_aliased() -> None:
    source = {"api_key": "secret"}
    provider = Partial(source)
    source["api_key"] = "changed"
    assert provider.config["api_key"] == "secret"

    empty = Partial()
    assert empty.config == {}
    empty.config["x"] = 1
    assert Partial().config == {}


def test_repr_names_the_provider() -> None:
    assert repr(Partial()) == "<Partial name='partial'>"
