"""Minimal third-party provider plugin: the shape a real package would ship.

Run it directly to see registration, routing, streaming, and error handling
with no network access::

    python examples/myprovider_plugin.py

A published plugin would live in its own distribution, e.g.
``oinyx_myprovider`` with the single line ``from . import provider`` in its
``__init__.py`` so that ``import oinyx_myprovider`` registers the provider.
"""

from __future__ import annotations

import asyncio
import collections.abc
import random
import uuid
from typing import Any

from oinyx import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    OinyxClient,
    ProviderError,
    ProviderRegistry,
    StreamChunk,
    Usage,
    register_provider,
)

MODEL_CATALOG = {"echo": "echo-1", "chatter": "chatter-2"}


@register_provider("myprovider")
class MyProvider(BaseProvider):
    """Fake provider that echoes the last user turn back to you."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key: str = str(self.config.get("api_key", ""))
        self.latency: float = float(self.config.get("latency", 0.0))
        self.fail_rate: float = float(self.config.get("fail_rate", 0.0))

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        await self._check_auth()
        await self._settle()
        content = self._compose(messages, model)
        return ChatResponse(
            id=f"my-{uuid.uuid4().hex[:12]}",
            model=model,
            content=content,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=sum(len(m.content) // 4 + 1 for m in messages),
                completion_tokens=len(content) // 4 + 1,
                total_tokens=0,
            ),
            raw={"echoed": len(messages), "kwargs": kwargs},
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> collections.abc.AsyncIterator[StreamChunk]:
        await self._check_auth()
        for word in self._compose(messages, model).split(" "):
            if self.latency:
                await asyncio.sleep(self.latency)
            yield StreamChunk(delta=word + " ", raw={"word": word})
        yield StreamChunk(delta="", finish_reason="stop")

    async def list_models(self) -> list[str]:
        return sorted(MODEL_CATALOG)

    async def _check_auth(self) -> None:
        if self.config.get("require_key") and not self.api_key:
            raise ProviderError("missing api_key", provider=self.name, model=None, status_code=401)

    async def _settle(self) -> None:
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.fail_rate and random.random() < self.fail_rate:
            raise ProviderError("backend hiccup", provider=self.name, model=None)

    @staticmethod
    def _compose(messages: list[ChatMessage], model: str) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"[{model}] you said: {last_user}"


async def main() -> None:
    client = OinyxClient(aliases={"echo": "myprovider/echo-1"})

    print("registered:", ProviderRegistry.list_providers())
    print("models:    ", await client.list_models("myprovider"))

    response = await client.chat("echo", "hello oinyx")
    print("chat:      ", response.content, response.usage.to_dict())

    streamed = client.stream("myprovider/echo-1", "streaming works")
    async for chunk in streamed:
        print(chunk.delta, end="")
    print()


def sync_demo() -> None:
    """Blocking callers use chat_sync, but never from inside a running loop."""
    client = OinyxClient()
    print("sync:      ", client.chat_sync("myprovider/echo-1", "blocking call").content)
    chunks = client.chat_sync("myprovider/echo-1", "blocking stream", stream=True)
    print("sync chunks:", [chunk.delta for chunk in chunks])


if __name__ == "__main__":
    asyncio.run(main())
    sync_demo()
