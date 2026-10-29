"""Model-string routing: ``"provider/model-name"`` -> (provider, model).

Bare names such as ``"gpt-4o"`` carry no provider information, so callers can
install an alias map that expands them, e.g. ``{"gpt-4o": "openai/gpt-4o"}``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import RoutingError

SEPARATOR = "/"


@dataclass(frozen=True)
class RoutedModel:
    """The outcome of parsing a model string."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}{SEPARATOR}{self.model}"


class ModelRouter:
    """Parses model strings and resolves aliases.

    Args:
        aliases: Optional mapping of bare model names to fully qualified
            ``provider/model`` strings. Entries may themselves point at other
            aliases; resolution follows at most ``max_hops`` links and rejects
            cycles.
    """

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases: dict[str, str] = dict(aliases or {})

    @property
    def aliases(self) -> dict[str, str]:
        """A copy of the active alias map."""
        return dict(self._aliases)

    def add_alias(self, name: str, target: str) -> None:
        """Map bare name ``name`` onto a ``provider/model`` string."""
        self._aliases[name] = target

    def resolve(self, model: str, default_provider: str | None = None) -> RoutedModel:
        """Turn a model string into a :class:`RoutedModel`.

        Resolution order:

        1. Expand ``model`` through the alias map while it maps to a bare name.
        2. If the result contains ``provider/model``, split on the first ``/``.
        3. Otherwise use ``default_provider`` when one was given.

        Raises:
            RoutingError: if the string is empty, has an empty provider or
                model segment, forms an alias cycle, or has no provider and no
                default was supplied.
        """
        if not model or not model.strip():
            raise RoutingError("model string is empty")
        name = model.strip()
        name = self._expand(name)

        if SEPARATOR in name:
            provider, _, rest = name.partition(SEPARATOR)
            provider, rest = provider.strip(), rest.strip()
            if not provider:
                raise RoutingError(f"model string {model!r} has an empty provider segment")
            if not rest:
                raise RoutingError(f"model string {model!r} has an empty model segment")
            return RoutedModel(provider=provider, model=rest)

        if default_provider:
            return RoutedModel(provider=default_provider, model=name)

        raise RoutingError(
            f"model {model!r} has no 'provider/' prefix and no default provider is set; "
            f"pass 'provider/model' or register an alias"
        )

    def _expand(self, name: str) -> str:
        seen: set[str] = set()
        while name in self._aliases:
            if name in seen:
                raise RoutingError(f"alias cycle detected at {name!r}")
            seen.add(name)
            name = self._aliases[name]
        return name
