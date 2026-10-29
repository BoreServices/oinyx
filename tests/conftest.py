"""Shared fixtures and a MockProvider used across the test-suite."""

from __future__ import annotations

import collections.abc
from typing import Any

import pytest

from oinyx.base import BaseProvider
from oinyx.exceptions import ProviderError, RateLimitError
from oinyx.models import ChatMessage, ChatResponse, StreamChunk, Usage
from oinyx.registry import ProviderRegistry


class MockProvider(BaseProvider):
    """Deterministic in-memory provider used to exercise the core.

    Behaviour is driven entirely by the config dict so tests never need to
    monkeypatch methods:

    ``response``: text returned by :meth:`chat` (default ``"mock-reply"``).
    ``chunks``: deltas yielded by :meth:`stream`.
    ``usage``: usage payload, omitted from the response when ``None``.
    ``fail_times`` / ``fail_with``: raise before succeeding.
    ``models``: value returned by :meth:`list_models`.
    """

    name = "mock"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.calls: list[dict[str, Any]] = []
        self.attempts = 0

    @property
    def _response_text(self) -> str:
        return str(self.config.get("response", "mock-reply"))

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        self.attempts += 1
        if self.attempts <= int(self.config.get("fail_times", 0)):
            raise self._make_error()
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        usage = self.config.get("usage")
        return ChatResponse(
            id=self.config.get("id", "mock-id"),
            model=model,
            content=self._response_text,
            finish_reason=self.config.get("finish_reason", "stop"),
            usage=Usage.from_dict(usage) if usage else Usage(),
            raw=self.config.get("raw", {}),
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> collections.abc.AsyncIterator[StreamChunk]:
        self.attempts += 1
        if self.attempts <= int(self.config.get("fail_times", 0)):
            raise self._make_error()
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs, "stream": True})
        deltas: list[str] = list(self.config.get("chunks", ["mock", "-", "reply"]))
        for delta in deltas:
            yield StreamChunk(delta=delta, raw={"delta": delta})
        yield StreamChunk(
            delta="",
            finish_reason=self.config.get("finish_reason", "stop"),
            raw={"usage": self.config["usage"]} if self.config.get("usage") else {},
        )

    def _make_error(self) -> ProviderError:
        error = self.config.get("fail_with", "rate_limit")
        message = "mock provider failure"
        if error == "generic":
            return ProviderError(message, provider=self.name, model="mock-model")
        return RateLimitError(message, provider=self.name, model="mock-model", status_code=429)

    async def list_models(self) -> list[str]:
        return list(self.config.get("models", ["mock-model"]))


@pytest.fixture(autouse=True)
def isolated_registry() -> collections.abc.Iterator[None]:
    """Snapshot and restore the global registry around every test."""
    classes = dict(ProviderRegistry._classes)
    configs = dict(ProviderRegistry._configs)
    instances = dict(ProviderRegistry._instances)
    yield
    ProviderRegistry._classes.clear()
    ProviderRegistry._instances.clear()
    ProviderRegistry._configs.clear()
    ProviderRegistry._classes.update(classes)
    ProviderRegistry._configs.update(configs)
    ProviderRegistry._instances.update(instances)


@pytest.fixture
def mock_provider() -> type[MockProvider]:
    """Register :class:`MockProvider` under the name ``mock``."""
    ProviderRegistry.register(MockProvider)
    return MockProvider
