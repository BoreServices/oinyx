"""Tests for the :class:`oinyx.BaseProvider` contract."""

from __future__ import annotations

from typing import Any

import pytest

from oinyx.base import BaseImageProvider, BaseProvider
from oinyx.exceptions import ProviderError
from oinyx.models import ChatMessage, ChatResponse, ImageResponse


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


class Pixel(BaseImageProvider):
    """Minimal image provider: only :meth:`generate_image` is implemented."""

    name = "pixel"

    async def generate_image(self, model: str, prompt: str, **kwargs: Any) -> ImageResponse:
        return ImageResponse(model=model, url=f"https://pixel/{model}.png", raw=dict(kwargs))


class TestImageProvider:
    def test_it_can_be_instantiated_without_chat(self) -> None:
        assert Pixel().supports_images is True
        assert isinstance(Pixel(), BaseProvider)

    def test_generate_image_is_still_abstract(self) -> None:
        class Incomplete(BaseImageProvider):
            name = "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    async def test_generate_image_returns_the_response(self) -> None:
        image = await Pixel().generate_image("small", "a fox", width=8)
        assert image.url == "https://pixel/small.png"
        assert image.raw == {"width": 8}

    async def test_chat_is_replaced_with_a_clear_error(self) -> None:
        with pytest.raises(ProviderError, match="pixel generates images"):
            await Pixel().chat("m", [])

    async def test_stream_is_replaced_with_a_clear_error(self) -> None:
        with pytest.raises(ProviderError, match="use generate_image"):
            async for _ in Pixel().stream("m", []):
                pass

    async def test_text_providers_do_not_advertise_images(self) -> None:
        assert Partial().supports_images is False
