"""Thread-safe provider registry with lazy instantiation.

Third-party packages register their provider class at import time::

    from oinyx import BaseProvider, register_provider

    @register_provider("myprovider")
    class MyProvider(BaseProvider):
        ...

The registry stores *classes*, not instances. A provider is constructed the
first time it is requested, using the config dict supplied via
:meth:`ProviderRegistry.configure` or :class:`~oinyx.client.OinyxClient`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from .base import BaseProvider
from .exceptions import ProviderNotFoundError

T = TypeVar("T", bound=type[BaseProvider])


class ProviderRegistry:
    """Global mapping of provider names to provider classes.

    All mutating and reading operations are guarded by a lock, so the
    registry is safe to use from multiple threads and event loops.
    """

    _lock = threading.RLock()
    _classes: dict[str, type[BaseProvider]] = {}
    _instances: dict[str, BaseProvider] = {}
    _configs: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, provider_cls: T) -> T:
        """Register ``provider_cls`` under its ``name`` attribute.

        Raises:
            ValueError: if the class has no ``name`` or a different class is
                already registered under that name.
        """
        name = getattr(provider_cls, "name", "")
        if not name:
            raise ValueError(f"{provider_cls.__name__} must define a non-empty 'name'")
        with cls._lock:
            existing = cls._classes.get(name)
            if existing is not None and existing is not provider_cls:
                raise ValueError(
                    f"provider {name!r} is already registered to {existing.__qualname__}"
                )
            cls._classes[name] = provider_cls
            cls._instances.pop(name, None)
        return provider_cls

    @classmethod
    def configure(cls, name: str, config: dict[str, Any] | None = None) -> None:
        """Attach (or replace) the config dict for provider ``name``.

        Replacing the config discards any cached instance so the next call
        rebuilds the provider with the new settings.
        """
        with cls._lock:
            cls._configs[name] = dict(config or {})
            cls._instances.pop(name, None)

    @classmethod
    def get(cls, name: str, config: dict[str, Any] | None = None) -> BaseProvider:
        """Return the provider instance for ``name``, building it on demand.

        Args:
            name: Registry key of the provider.
            config: Per-call config merged over the registered config. When
                supplied it does *not* permanently replace the stored config.

        Raises:
            ProviderNotFoundError: if nothing is registered under ``name``.
        """
        with cls._lock:
            provider_cls = cls._classes.get(name)
            if provider_cls is None:
                raise ProviderNotFoundError(
                    f"no provider registered as {name!r}; "
                    f"known providers: {sorted(cls._classes) or 'none'}",
                    provider=name,
                )
            if config is not None:
                merged = {**cls._configs.get(name, {}), **config}
                return provider_cls(merged)
            instance = cls._instances.get(name)
            if instance is None:
                instance = provider_cls(cls._configs.get(name, {}))
                cls._instances[name] = instance
            return instance

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return the sorted names of every registered provider."""
        with cls._lock:
            return sorted(cls._classes)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a provider, its cached instance, and its config.

        Intended for tests and plugin teardown; missing names are ignored.
        """
        with cls._lock:
            cls._classes.pop(name, None)
            cls._instances.pop(name, None)
            cls._configs.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Drop all registrations. Intended for tests."""
        with cls._lock:
            cls._classes.clear()
            cls._instances.clear()
            cls._configs.clear()


def register_provider(name: str | None = None) -> Callable[[T], T]:
    """Class decorator that registers a provider.

    Usable with or without parentheses::

        @register_provider("myprovider")
        class MyProvider(BaseProvider): ...

        @register_provider()
        class OtherProvider(BaseProvider):
            name = "other"
    """

    def decorator(provider_cls: T) -> T:
        if name is not None:
            provider_cls.name = name
        return ProviderRegistry.register(provider_cls)

    return decorator
