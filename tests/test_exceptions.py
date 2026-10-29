"""Tests for the exception hierarchy and its provider/model context."""

from __future__ import annotations

import pytest

from oinyx.exceptions import (
    AuthenticationError,
    OinyxError,
    ProviderError,
    ProviderNotFoundError,
    RateLimitError,
    RoutingError,
    StreamInterruptedError,
)

ALL_ERRORS = [
    ProviderNotFoundError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    StreamInterruptedError,
    RoutingError,
]


@pytest.mark.parametrize("error", ALL_ERRORS)
def test_every_error_inherits_from_oinyx_error(error: type[Exception]) -> None:
    assert issubclass(error, OinyxError)


def test_context_is_attached() -> None:
    error = RateLimitError("slow down", provider="mock", model="gpt-4o", status_code=429)
    assert (error.provider, error.model, error.status_code) == ("mock", "gpt-4o", 429)
    assert error.message == "slow down"


def test_context_is_rendered_in_the_message() -> None:
    error = ProviderError("boom", provider="mock", model="m")
    assert str(error) == "boom (provider='mock', model='m')"


def test_bare_error_renders_plainly() -> None:
    assert str(OinyxError("plain")) == "plain"


def test_partial_context_only_shows_what_is_set() -> None:
    assert str(RoutingError("bad model string")) == "bad model string"
    assert str(StreamInterruptedError("cut off", provider="mock")) == "cut off (provider='mock')"


def test_rate_limit_and_auth_are_provider_errors() -> None:
    for error in (RateLimitError("x"), AuthenticationError("x")):
        assert isinstance(error, ProviderError)
        assert error.status_code is None


def test_errors_are_catchable_as_base_class() -> None:
    with pytest.raises(OinyxError):
        raise AuthenticationError("missing api key", provider="mock")
