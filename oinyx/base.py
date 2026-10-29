"""The abstract provider interface every plugin implements.

A provider is a lightweight adapter between oinyx's unified call shape and
one backend's wire protocol. Providers are constructed with a plain config
dict so the registry can lazily instantiate them at call time::

    class MyProvider(BaseProvider):
        name = "myprovider"

        def __init__(self, config: dict | None = None) -> None:
            self.api_key = (config or {}).get("api_key", "")

        async def chat(self, model, messages, **kwargs):
            ...

        async def stream(self, model, messages, **kwargs):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from .exceptions import ProviderError
from .models import ChatMessage, ChatResponse, ImageResponse, StreamChunk


class BaseProvider(ABC):
    """Abstract base class for provider plugins."""

    #: Unique registry key, e.g. ``"myprovider"``. Must be set by subclasses.
    name: str = ""

    #: True for providers that can render images (see :class:`BaseImageProvider`).
    supports_images: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Return a complete response for ``messages``.

        Args:
            model: Bare model name, already stripped of its ``provider/`` prefix.
            messages: Normalized conversation turns.
            **kwargs: Provider-specific options (temperature, tools, ...).
        """

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Return an async iterator of incremental :class:`StreamChunk` items.

        Implement this as an async generator, which satisfies the contract::

            async def stream(self, model, messages, **kwargs):
                yield StreamChunk(delta="hello")
        """

    async def list_models(self) -> list[str]:
        """Return model identifiers this provider can serve.

        Optional; providers that cannot enumerate models may leave the
        default, which reports an empty list.
        """
        return []

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class BaseImageProvider(BaseProvider):
    """Base class for providers whose job is image generation.

    ``chat`` and ``stream`` are satisfied with clear failures so an image-only
    plugin can be registered through the same :class:`ProviderRegistry`; only
    :meth:`generate_image` must be implemented::

        @register_provider("myimagegen")
        class MyImageProvider(BaseImageProvider):
            async def generate_image(self, model, prompt, **kwargs):
                return ImageResponse(url="https://...")
    """

    supports_images = True

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise ProviderError(
            f"{self.name or type(self).__name__} generates images; use generate_image()",
            provider=self.name,
            model=model,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        raise ProviderError(
            f"{self.name or type(self).__name__} generates images; use generate_image()",
            provider=self.name,
            model=model,
        )

    @abstractmethod
    async def generate_image(self, model: str, prompt: str, **kwargs: Any) -> ImageResponse:
        """Render ``prompt`` into an image.

        Args:
            model: Bare model name, already stripped of its ``provider/`` prefix.
            prompt: The text description of the image to generate.
            **kwargs: Provider options such as ``width``, ``height``, ``seed``.
        """
