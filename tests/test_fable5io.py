"""Offline tests for the bundled ``fable5`` provider.

Every request is served by :class:`httpx.MockTransport`, so this module never
touches the network.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import pytest

from oinyx import OinyxClient, ProviderRegistry, RateLimitError
from oinyx.exceptions import AuthenticationError, ProviderError, StreamInterruptedError
from oinyx.models import ChatMessage, ChatResponse, StreamChunk
from providers.fable5io import (
    DEFAULT_MODEL,
    ENDPOINT,
    Fable5Provider,
    UpstreamError,
)


def make_provider(handler: Any, **config: Any) -> Fable5Provider:
    transport = httpx.MockTransport(handler)
    return Fable5Provider({"client": httpx.AsyncClient(transport=transport), **config})


@pytest.fixture
def provider_client() -> AsyncIterator[httpx.AsyncClient]:
    """Shared client closed at the end of each test."""
    holders: list[httpx.AsyncClient] = []

    class Recorder(dict):
        def record(self, client: httpx.AsyncClient) -> None:
            holders.append(client)

    yield from ()  # placeholder replaced below
