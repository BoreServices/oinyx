"""Exception hierarchy for :mod:`oinyx`.

All errors raised by the library inherit from :class:`OinyxError` and carry
``provider`` and ``model`` context so callers can log or branch on origin.
"""

from __future__ import annotations


class OinyxError(Exception):
    """Base class for every error raised by oinyx."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.model:
            parts.append(f"model={self.model!r}")
        return " [".join(parts[:1]) + (" (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else "")


class ProviderNotFoundError(OinyxError):
    """No provider is registered under the requested name."""


class ProviderError(OinyxError):
    """A provider backend returned an error (HTTP failure, bad response, ...)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.status_code = status_code


class RateLimitError(ProviderError):
    """The provider rate-limited the request (HTTP 429 and friends)."""


class AuthenticationError(ProviderError):
    """Credentials were missing, invalid, or rejected."""


class StreamInterruptedError(OinyxError):
    """A streamed response ended before the provider signalled completion."""


class RoutingError(OinyxError):
    """A model string could not be parsed or resolved to a provider."""
