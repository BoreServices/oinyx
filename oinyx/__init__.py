"""oinyx -- a provider-agnostic client library for chat-completion models.

The package ships an interface and a registry only; providers are contributed
by third-party packages::

    from oinyx import BaseProvider, OinyxClient, register_provider

    @register_provider("myprovider")
    class MyProvider(BaseProvider):
        async def chat(self, model, messages, **kwargs): ...
        async def stream(self, model, messages, **kwargs): ...

    client = OinyxClient()
    response = await client.chat("myprovider/my-model", "hello")
"""

from __future__ import annotations

from .base import BaseImageProvider, BaseProvider
from .client import MessageLike, MessagesLike, OinyxClient
from .exceptions import (
    AuthenticationError,
    OinyxError,
    ProviderError,
    ProviderNotFoundError,
    RateLimitError,
    RoutingError,
    StreamInterruptedError,
)
from .models import ChatMessage, ChatResponse, ImageResponse, StreamChunk, Usage
from .registry import ProviderRegistry, register_provider
from .router import ModelRouter, RoutedModel
from .utils import collect_stream, estimate_tokens, fill_usage, retry

__version__ = "0.1.0"

__all__ = [
    "AuthenticationError",
    "BaseImageProvider",
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "ImageResponse",
    "MessageLike",
    "MessagesLike",
    "ModelRouter",
    "OinyxClient",
    "OinyxError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "RateLimitError",
    "RoutedModel",
    "RoutingError",
    "StreamChunk",
    "StreamInterruptedError",
    "Usage",
    "__version__",
    "collect_stream",
    "estimate_tokens",
    "fill_usage",
    "register_provider",
    "retry",
]
