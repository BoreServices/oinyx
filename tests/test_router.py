"""Tests for :class:`oinyx.ModelRouter`."""

from __future__ import annotations

import pytest

from oinyx.exceptions import RoutingError
from oinyx.router import ModelRouter, RoutedModel


def test_parses_provider_and_model() -> None:
    assert ModelRouter().resolve("mock/gpt-mini") == RoutedModel(provider="mock", model="gpt-mini")


def test_splits_on_the_first_separator_only() -> None:
    routed = ModelRouter().resolve("mock/nested/model-name")
    assert (routed.provider, routed.model) == ("mock", "nested/model-name")


def test_str_renders_the_canonical_form() -> None:
    assert str(RoutedModel("mock", "m")) == "mock/m"


def test_whitespace_is_stripped() -> None:
    assert ModelRouter().resolve("  mock / m  ") == RoutedModel("mock", "m")


@pytest.mark.parametrize("model", ["", "   ", "\t"])
def test_empty_model_string_raises(model: str) -> None:
    with pytest.raises(RoutingError, match="empty"):
        ModelRouter().resolve(model)


@pytest.mark.parametrize("model", ["/only-model", " /m", "mock/", "mock/   "])
def test_missing_segments_raise(model: str) -> None:
    with pytest.raises(RoutingError):
        ModelRouter().resolve(model)


def test_bare_name_without_default_raises_and_explains() -> None:
    with pytest.raises(RoutingError, match="no 'provider/' prefix") as excinfo:
        ModelRouter().resolve("gpt-4o")
    assert excinfo.value.message.startswith("model 'gpt-4o'")


def test_default_provider_is_applied_to_bare_names() -> None:
    routed = ModelRouter().resolve("gpt-4o", default_provider="mock")
    assert routed == RoutedModel("mock", "gpt-4o")


def test_alias_expands_a_bare_name() -> None:
    router = ModelRouter({"gpt-4o": "mock/gpt-4o-2024"})
    assert router.resolve("gpt-4o") == RoutedModel("mock", "gpt-4o-2024")


def test_add_alias_and_aliases_copy() -> None:
    router = ModelRouter({"a": "mock/a"})
    router.add_alias("b", "mock/b")
    assert router.aliases == {"a": "mock/a", "b": "mock/b"}
    router.aliases.clear()
    assert router.resolve("b") == RoutedModel("mock", "b")


def test_alias_chain_is_followed() -> None:
    router = ModelRouter({"short": "mid", "mid": "mock/real", "mid2": "mock/x"})
    assert router.resolve("short") == RoutedModel("mock", "real")


def test_alias_to_bare_name_falls_back_to_default_provider() -> None:
    router = ModelRouter({"best": "mock/gpt"})
    assert router.resolve("best") == RoutedModel("mock", "gpt")
    assert router.resolve("unknown", default_provider="mock") == RoutedModel("mock", "unknown")


def test_alias_cycle_is_detected() -> None:
    router = ModelRouter({"a": "b", "b": "a"})
    with pytest.raises(RoutingError, match="alias cycle"):
        router.resolve("a")


def test_self_referential_alias_is_detected() -> None:
    with pytest.raises(RoutingError, match="alias cycle"):
        ModelRouter({"loop": "loop"}).resolve("loop")


def test_alias_wins_over_default_provider() -> None:
    router = ModelRouter({"gpt": "mock/gpt-alias"})
    assert router.resolve("gpt", default_provider="other") == RoutedModel("mock", "gpt-alias")
