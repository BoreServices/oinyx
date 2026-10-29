"""The user-facing entry point: :class:`OinyxClient`.

The client owns no provider logic. It parses the model string, looks the
provider up in the registry, normalizes messages, and forwards the call::

    from oinyx import OinyxClient

    client = OinyxClient()
    reply = await client.chat("myprovider/gpt-mini", "hello")
    print(reply.content)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, overload

from .base import BaseImageProvider, BaseProvider
from .exceptions import OinyxError
from .models import ChatMessage, ChatResponse, ImageResponse, StreamChunk
from .registry import ProviderRegistry
from .router import ModelRouter, RoutedModel
from .utils import collect_stream

MessageLike = ChatMessage | dict[str, Any] | str
MessagesLike = MessageLike | Sequence[MessageLike]


class OinyxClient:
    """Provider-agnostic chat client.

    Args:
        provider: Default provider name, used when a model string has no
            ``provider/`` prefix.
        config: Provider configuration. Either nested
            ``{"myprovider": {"api_key": "..."}}`` or a flat dict, which is
            handed to the resolved provider as-is.
        aliases: Bare-name -> ``provider/model`` map for the built-in router.
        router: Custom :class:`~oinyx.router.ModelRouter`; overrides ``aliases``.
        registry: Registry to resolve providers from, for clients that want a
            private registry instead of the global one.

    All keyword arguments after ``model`` and ``messages`` are forwarded to the
    provider untouched (``temperature``, ``max_tokens``, ...).
    """

    def __init__(
        self,
        provider: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        aliases: dict[str, str] | None = None,
        router: ModelRouter | None = None,
        registry: type[ProviderRegistry] | None = None,
    ) -> None:
        self.default_provider = provider
        self.config: dict[str, Any] = dict(config or {})
        self.router = router or ModelRouter(aliases)
        self._registry: type[ProviderRegistry] = registry or ProviderRegistry

    def route(self, model: str) -> RoutedModel:
        """Resolve ``model`` to provider + bare model name.

        Raises:
            RoutingError: if no provider can be determined.
        """
        return self.router.resolve(model, self.default_provider)

    def provider_for(self, model: str) -> BaseProvider:
        """Return the lazily instantiated provider backing ``model``."""
        routed = self.route(model)
        return self._registry.get(routed.provider, self._config_for(routed.provider))

    def _config_for(self, provider_name: str) -> dict[str, Any] | None:
        nested = self.config.get(provider_name)
        if isinstance(nested, dict):
            return dict(nested)
        if not self.config:
            return None
        return dict(self.config)

    @staticmethod
    def normalize_messages(messages: MessagesLike | None) -> list[ChatMessage]:
        """Coerce dicts, strings, and :class:`ChatMessage` objects into a list.

        A single message may be passed without wrapping it in a list, and
        ``None`` means an empty conversation.

        Raises:
            OinyxError: if an element cannot be coerced.
        """
        if messages is None:
            return []
        if isinstance(messages, (str, dict, ChatMessage)):
            items: Sequence[Any] = [messages]
        else:
            items = list(messages)
        try:
            return [ChatMessage.coerce(item) for item in items]
        except TypeError as exc:
            raise OinyxError(f"invalid messages payload: {exc}") from exc

    @overload
    async def chat(
        self,
        model: str,
        messages: MessagesLike | None = None,
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def chat(
        self,
        model: str,
        messages: MessagesLike | None = None,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...

    async def chat(
        self,
        model: str,
        messages: MessagesLike | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        """Call a model and return a response or a stream.

        Args:
            model: ``"provider/model"`` string, a bare name resolvable through
                an alias or the client's default provider.
            messages: Conversation turns as ``ChatMessage`` objects, dicts,
                strings, or a mix.
            stream: When ``True`` returns an async iterator of
                :class:`~oinyx.models.StreamChunk` instead of a full response.
            **kwargs: Forwarded to the provider.

        Raises:
            RoutingError: if the model string cannot be resolved.
            ProviderNotFoundError: if the provider is not registered.
            ProviderError: for backend failures.
        """
        routed = self.route(model)
        provider = self._registry.get(routed.provider, self._config_for(routed.provider))
        turns = self.normalize_messages(messages)

        if stream:
            return provider.stream(routed.model, turns, **kwargs)
        return await provider.chat(routed.model, turns, **kwargs)

    def stream(
        self,
        model: str,
        messages: MessagesLike | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Return the chunk iterator for ``model`` without the ``stream=True`` dance.

        ``async for chunk in client.stream("provider/model", "hi")`` is the
        equivalent of ``await client.chat(..., stream=True)``.
        """
        routed = self.route(model)
        provider = self._registry.get(routed.provider, self._config_for(routed.provider))
        return provider.stream(routed.model, self.normalize_messages(messages), **kwargs)

    async def image(self, model: str, prompt: str, **kwargs: Any) -> ImageResponse:
        """Generate an image with an image-capable provider.

        Args:
            model: ``"provider/model"`` string, routed exactly like :meth:`chat`.
            prompt: Text description of the image.
            **kwargs: Forwarded to the provider (``width``, ``height``, ...).

        Raises:
            ProviderError: if the resolved provider cannot generate images.
        """
        routed = self.route(model)
        provider = self._registry.get(routed.provider, self._config_for(routed.provider))
        if not isinstance(provider, BaseImageProvider):
            raise OinyxError(
                f"provider {routed.provider!r} does not support image generation",
                provider=routed.provider,
                model=routed.model,
            )
        return await provider.generate_image(routed.model, prompt, **kwargs)

    def image_sync(self, model: str, prompt: str, **kwargs: Any) -> ImageResponse:
        """Synchronous wrapper around :meth:`image`."""
        if self._running_loop() is not None:
            raise OinyxError(
                "image_sync() cannot be used inside a running event loop; await image() instead"
            )
        return asyncio.run(self.image(model, prompt, **kwargs))

    def chat_sync(
        self,
        model: str,
        messages: MessagesLike | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | list[StreamChunk]:
        """Synchronous wrapper around :meth:`chat`.

        Non-streaming calls return a :class:`~oinyx.models.ChatResponse`.
        Streaming calls are fully collected into a list of chunks, since an
        async iterator is not usable without a loop.

        Raises:
            OinyxError: if called from inside a running event loop.
        """
        if self._running_loop() is not None:
            raise OinyxError(
                "chat_sync() cannot be used inside a running event loop; await chat() instead"
            )
        if not stream:
            return asyncio.run(self.chat(model, messages, stream=False, **kwargs))

        iterator = self.stream(model, messages, **kwargs)

        async def drain() -> list[StreamChunk]:
            return [chunk async for chunk in iterator]

        return asyncio.run(drain())

    async def collect(
        self,
        model: str,
        messages: MessagesLike | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Stream a response and collapse it into one :class:`ChatResponse`."""
        turns = self.normalize_messages(messages)
        return await collect_stream(
            self.stream(model, turns, **kwargs), model=model, messages=turns
        )

    async def list_models(self, provider: str | None = None) -> list[str]:
        """List models, for ``provider`` or for every registered provider.

        Names are returned in ``provider/model`` form when listing all.
        """
        names = [provider] if provider else self._registry.list_providers()
        models: list[str] = []
        for name in names:
            instance = self._registry.get(name, self._config_for(name))
            for model in await instance.list_models():
                models.append(model if provider else f"{name}/{model}")
        return models

    @staticmethod
    def _running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None
