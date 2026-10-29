"""End-to-end tests for :class:`oinyx.OinyxClient` against the MockProvider."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oinyx.client import OinyxClient
from oinyx.exceptions import (
    OinyxError,
    ProviderNotFoundError,
    RateLimitError,
    RoutingError,
)
from oinyx.models import ChatMessage, ChatResponse, StreamChunk
from oinyx.registry import ProviderRegistry
from oinyx.router import ModelRouter
from oinyx.utils import retry


@pytest.fixture
def client(mock_provider: type[Any]) -> OinyxClient:
    """Client with the ``mock`` provider registered and no config."""
    return OinyxClient()


class TestRouting:
    def test_route_splits_the_prefix(self, client: OinyxClient) -> None:
        routed = client.route("mock/gpt")
        assert (routed.provider, routed.model) == ("mock", "gpt")

    def test_default_provider_fills_a_bare_name(self) -> None:
        assert OinyxClient(provider="mock").route("gpt").provider == "mock"

    def test_aliases_are_honoured(self, client: OinyxClient) -> None:
        client.router.add_alias("fast", "mock/turbo")
        assert client.route("fast") == client.route("mock/turbo")

    def test_custom_router_is_used(self) -> None:
        client = OinyxClient(router=ModelRouter({"x": "mock/y"}))
        assert client.route("x").model == "y"

    async def test_missing_prefix_without_default_raises(self, client: OinyxClient) -> None:
        with pytest.raises(RoutingError):
            await client.chat("gpt", "hi")

    async def test_unknown_provider_raises_with_context(self, client: OinyxClient) -> None:
        with pytest.raises(ProviderNotFoundError) as excinfo:
            await client.chat("ghost/model", "hi")
        assert excinfo.value.provider == "ghost"

    def test_provider_for_returns_the_instance(self, client: OinyxClient, mock_provider: Any) -> None:
        assert isinstance(client.provider_for("mock/m"), mock_provider)


class TestChat:
    async def test_returns_the_provider_response(self, client: OinyxClient, mock_provider: Any) -> None:
        response = await client.chat("mock/model-a", "hello", temperature=0.2)
        assert isinstance(response, ChatResponse)
        assert response.content == "mock-reply"
        assert response.model == "model-a"

        provider = ProviderRegistry.get("mock")
        assert provider.calls[0]["model"] == "model-a"
        assert provider.calls[0]["kwargs"] == {"temperature": 0.2}

    async def test_messages_are_normalized(self, client: OinyxClient) -> None:
        payload: list[Any] = [
            {"role": "system", "content": "be nice"},
            ChatMessage("user", "hi"),
            "raw string",
        ]
        await client.chat("mock/m", payload)
        turns = ProviderRegistry.get("mock").calls[0]["messages"]
        assert [turn.role for turn in turns] == ["system", "user", "user"]
        assert all(isinstance(turn, ChatMessage) for turn in turns)

    async def test_single_dict_message_is_wrapped(self, client: OinyxClient) -> None:
        await client.chat("mock/m", {"role": "user", "content": "one"})
        assert ProviderRegistry.get("mock").calls[0]["messages"] == [ChatMessage("user", "one")]

    async def test_missing_messages_means_an_empty_conversation(self, client: OinyxClient) -> None:
        await client.chat("mock/m")
        assert ProviderRegistry.get("mock").calls[0]["messages"] == []

    async def test_invalid_messages_raise_oinyx_error(self, client: OinyxClient) -> None:
        with pytest.raises(OinyxError, match="invalid messages payload"):
            await client.chat("mock/m", [123])  # type: ignore[list-item]

    async def test_bare_name_uses_default_provider(self, mock_provider: Any) -> None:
        client = OinyxClient(provider="mock")
        response = await client.chat("model-b", "hi")
        assert response.model == "model-b"


class TestStreaming:
    async def test_stream_flag_returns_an_async_iterator(self, client: OinyxClient) -> None:
        stream = await client.chat("mock/m", "hi", stream=True)
        chunks = [chunk async for chunk in stream]
        assert [chunk.delta for chunk in chunks] == ["mock", "-", "reply", ""]
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)
        assert chunks[-1].finish_reason == "stop"

    async def test_collect_assembles_a_response(self, client: OinyxClient) -> None:
        response = await client.collect("mock/m", "abcd abcd")
        assert response.content == "mock-reply"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens > 0

    async def test_custom_chunks_are_used(self, mock_provider: Any) -> None:
        ProviderRegistry.configure("mock", {"chunks": ["a", "b", "c"]})
        response = await OinyxClient().collect("mock/m", "hi")
        assert response.content == "abc"


class TestConfig:
    async def test_nested_config_reaches_the_provider(self, mock_provider: Any) -> None:
        client = OinyxClient(config={"mock": {"response": "from nested"}})
        assert (await client.chat("mock/m", "hi")).content == "from nested"

    async def test_flat_config_reaches_the_provider(self, mock_provider: Any) -> None:
        client = OinyxClient(config={"response": "from flat"})
        assert (await client.chat("mock/m", "hi")).content == "from flat"

    async def test_client_config_does_not_mutate_the_registry(self, mock_provider: Any) -> None:
        client = OinyxClient(config={"mock": {"response": "stored"}})
        assert (await client.chat("mock/m", "hi")).content == "stored"
        assert ProviderRegistry.get("mock").config == {}

    async def test_stored_registry_config_is_the_baseline(self, mock_provider: Any) -> None:
        ProviderRegistry.configure("mock", {"response": "registry level"})
        assert (await OinyxClient().chat("mock/m", "hi")).content == "registry level"


class TestSyncWrapper:
    def test_chat_sync_returns_a_response(self, client: OinyxClient) -> None:
        assert client.chat_sync("mock/m", "hi").content == "mock-reply"

    def test_chat_sync_streaming_returns_chunks(self, client: OinyxClient) -> None:
        chunks = client.chat_sync("mock/m", "hi", stream=True)
        assert isinstance(chunks, list)
        assert "".join(chunk.delta for chunk in chunks) == "mock-reply"

    async def test_chat_sync_inside_a_loop_is_rejected(self, client: OinyxClient) -> None:
        with pytest.raises(OinyxError, match="running event loop"):
            client.chat_sync("mock/m", "hi")

    def test_sync_errors_propagate(self, client: OinyxClient, mock_provider: Any) -> None:
        from oinyx.exceptions import ProviderError

        ProviderRegistry.configure("mock", {"fail_times": 10, "fail_with": "generic"})
        with pytest.raises(ProviderError):
            client.chat_sync("mock/m", "hi")


class TestRetryIntegration:
    async def test_transient_rate_limits_are_retried(self, mock_provider: Any) -> None:
        ProviderRegistry.configure("mock", {"fail_times": 2})
        client = OinyxClient()

        @retry(attempts=5, backoff=0, on=(RateLimitError,))
        async def call() -> ChatResponse:
            return await client.chat("mock/m", "hi")

        response = await call()
        assert response.content == "mock-reply"
        assert ProviderRegistry.get("mock").attempts == 3

    async def test_retries_give_up_and_surface_the_error(self, mock_provider: Any) -> None:
        ProviderRegistry.configure("mock", {"fail_times": 99})
        client = OinyxClient()

        @retry(attempts=2, backoff=0)
        async def call() -> ChatResponse:
            return await client.chat("mock/m", "hi")

        with pytest.raises(RateLimitError):
            await call()
        assert ProviderRegistry.get("mock").attempts == 2


class TestListModels:
    async def test_single_provider_is_unprefixed(self, client: OinyxClient, mock_provider: Any) -> None:
        assert await client.list_models("mock") == ["mock-model"]

    async def test_all_providers_are_prefixed(self, client: OinyxClient, mock_provider: Any) -> None:
        ProviderRegistry.register(type("Second", (mock_provider,), {"name": "second"}))
        assert await client.list_models() == ["mock/mock-model", "second/mock-model"]

    async def test_provider_may_skip_list_models(self, client: OinyxClient) -> None:
        from oinyx.base import BaseProvider

        class NoListing(BaseProvider):
            name = "nolisting"

            async def chat(self, model: str, messages: Any, **kwargs: Any) -> ChatResponse:
                return ChatResponse(model=model)

            async def stream(self, model: str, messages: Any, **kwargs: Any) -> Any:
                if False:
                    yield StreamChunk()

        ProviderRegistry.register(NoListing)
        assert await client.list_models("nolisting") == []


class TestLifecycle:
    async def test_close_is_not_required_but_loop_reuse_is_safe(self, client: OinyxClient) -> None:
        for _ in range(3):
            assert (await client.chat("mock/m", "hi")).content == "mock-reply"

    async def test_concurrent_calls_are_isolated(self, client: OinyxClient, mock_provider: Any) -> None:
        ProviderRegistry.configure("mock", {"chunks": ["x"]})
        responses = await asyncio.gather(*(client.chat("mock/m", "hi") for _ in range(5)))
        assert all(response.content == "mock-reply" for response in responses)
