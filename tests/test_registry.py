"""Tests for :class:`oinyx.ProviderRegistry` and the registration decorator."""

from __future__ import annotations

from typing import Any

import pytest

from oinyx.base import BaseProvider
from oinyx.exceptions import ProviderNotFoundError
from oinyx.models import ChatMessage
from oinyx.registry import ProviderRegistry, register_provider


class DummyProvider(BaseProvider):
    name = "dummy"

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        return None

    async def stream(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        return None


def test_register_returns_the_class() -> None:
    assert ProviderRegistry.register(DummyProvider) is DummyProvider


def test_get_lazy_instantiates_and_caches() -> None:
    ProviderRegistry.register(DummyProvider)
    first = ProviderRegistry.get("dummy")
    assert isinstance(first, DummyProvider)
    assert ProviderRegistry.get("dummy") is first


def test_get_with_inline_config_builds_a_fresh_instance() -> None:
    ProviderRegistry.register(DummyProvider)
    cached = ProviderRegistry.get("dummy")
    scoped = ProviderRegistry.get("dummy", {"api_key": "secret"})
    assert scoped is not cached
    assert scoped.config == {"api_key": "secret"}


def test_unknown_provider_raises_with_context() -> None:
    with pytest.raises(ProviderNotFoundError) as excinfo:
        ProviderRegistry.get("nope")
    assert excinfo.value.provider == "nope"
    assert "no provider registered" in str(excinfo.value)


def test_class_without_name_is_rejected() -> None:
    class Nameless(BaseProvider):
        name = ""

        async def chat(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

        async def stream(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

    with pytest.raises(ValueError, match="non-empty 'name'"):
        ProviderRegistry.register(Nameless)


def test_conflicting_registration_is_rejected() -> None:
    ProviderRegistry.register(DummyProvider)

    class Other(DummyProvider):
        pass

    with pytest.raises(ValueError, match="already registered"):
        ProviderRegistry.register(Other)


def test_re_registering_same_class_is_allowed() -> None:
    ProviderRegistry.register(DummyProvider)
    ProviderRegistry.register(DummyProvider)
    assert ProviderRegistry.list_providers() == ["dummy"]


def test_decorator_with_explicit_name() -> None:
    @register_provider("fancy")
    class Fancy(BaseProvider):
        async def chat(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

        async def stream(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

    assert Fancy.name == "fancy"
    assert isinstance(ProviderRegistry.get("fancy"), Fancy)


def test_decorator_without_arguments_uses_class_name() -> None:
    @register_provider()
    class Plain(BaseProvider):
        name = "plain"

        async def chat(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

        async def stream(self, model: str, messages: list[ChatMessage], **k: Any) -> None: ...

    assert isinstance(ProviderRegistry.get("plain"), Plain)


def test_configure_replaces_config_and_instance() -> None:
    ProviderRegistry.register(DummyProvider)
    ProviderRegistry.configure("dummy", {"a": 1})
    assert ProviderRegistry.get("dummy").config == {"a": 1}
    ProviderRegistry.configure("dummy", {"a": 2})
    assert ProviderRegistry.get("dummy").config == {"a": 2}


def test_configure_merges_under_inline_config() -> None:
    ProviderRegistry.register(DummyProvider)
    ProviderRegistry.configure("dummy", {"api_key": "stored", "region": "eu"})
    provider = ProviderRegistry.get("dummy", {"api_key": "inline"})
    assert provider.config == {"api_key": "inline", "region": "eu"}


def test_list_providers_is_sorted() -> None:
    ProviderRegistry.register(DummyProvider)

    @register_provider("alpha")
    class Alpha(DummyProvider):
        pass

    assert ProviderRegistry.list_providers() == ["alpha", "dummy"]


def test_unregister_removes_everything() -> None:
    ProviderRegistry.register(DummyProvider)
    ProviderRegistry.get("dummy")
    ProviderRegistry.unregister("dummy")
    assert ProviderRegistry.list_providers() == []
    with pytest.raises(ProviderNotFoundError):
        ProviderRegistry.get("dummy")


def test_unregister_unknown_is_a_no_op() -> None:
    ProviderRegistry.unregister("ghost")


def test_clear_empties_the_registry() -> None:
    ProviderRegistry.register(DummyProvider)
    ProviderRegistry.clear()
    assert ProviderRegistry.list_providers() == []


def test_registration_is_thread_safe() -> None:
    """Concurrent registrations of distinct providers must all survive."""
    import threading

    ProviderRegistry.register(DummyProvider)
    classes = []
    for index in range(16):
        classes.append(type(f"P{index}", (DummyProvider,), {"name": f"p{index}"}))

    threads = [threading.Thread(target=ProviderRegistry.register, args=(cls,)) for cls in classes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ProviderRegistry.list_providers()) == 17  # 16 new + "dummy"
