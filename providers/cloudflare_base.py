"""Shared plumbing for browser-backed, Cloudflare-guarded JSON endpoints.

Several free AI sites front their API with a Cloudflare *managed challenge*.
A plain HTTP client is rejected with ``403`` + ``cf-mitigated: challenge`` and
an HTML "Just a moment..." body, which no amount of header spoofing fixes --
the challenge needs a real browser to execute its script and set the cookies.

Two things follow from that, and both are handled here:

* :class:`CloudflareChallengeError` is raised with an actionable message
  instead of a confusing HTML parse failure.
* A browser session can be replayed through ``config["cookies"]`` -- pass the
  ``cf_clearance`` and friends copied out of a logged-in browser (or a full
  ``Cookie`` header string) together with the *same* ``user-agent`` that solved
  the challenge, since the clearance cookie is bound to it.

:class:`CloudflareBase` also owns the boring HTTP mechanics: an injectable
``httpx`` client or transport, retryable ``post()``, and status-to-exception
mapping. Subclasses set ``endpoint`` and ``origin``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from oinyx import (
    AuthenticationError,
    BaseProvider,
    ProviderError,
    RateLimitError,
)

from .http_utils import describe_error, error_payload

CHALLENGE_MARKER = "just a moment"


class CloudflareChallengeError(ProviderError):
    """Cloudflare blocked the request with a browser challenge."""


class UpstreamError(ProviderError):
    """5xx or transport failure: the same request is worth retrying."""


class CloudflareBase(BaseProvider):
    """HTTP mechanics shared by browser-backed providers."""

    #: Subclasses point these at their own hosted site.
    endpoint: str = ""
    origin: str = ""
    referer: str = ""
    #: The user-agent that solved the challenge; clearance cookies are bound to it.
    user_agent: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.endpoint = str(self.config.get("base_url", self.endpoint))
        self.origin = str(self.config.get("origin", self.origin))
        self.referer = str(self.config.get("referer", self.referer or f"{self.origin}/"))
        self.timeout: float = float(self.config.get("timeout", 120.0))
        self.attempts: int = max(1, int(self.config.get("attempts", 3)))
        self.backoff: float | Callable[[int], float] = self.config.get("backoff", 1.0)
        self._injected: httpx.AsyncClient | None = self.config.get("client")
        self._transport: httpx.AsyncBaseTransport | None = self.config.get("transport")
        self.headers: dict[str, str] = self._build_headers()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/plain, */*",
            "origin": self.origin,
            "referer": self.referer,
        }
        user_agent = str(self.config.get("user_agent", self.user_agent))
        if user_agent:
            headers["user-agent"] = user_agent
        cookies = self.config.get("cookies")
        if isinstance(cookies, str) and cookies.strip():
            headers["cookie"] = cookies.strip()
        elif isinstance(cookies, dict) and cookies:
            headers["cookie"] = "; ".join(f"{key}={value}" for key, value in cookies.items())
        extra = self.config.get("headers")
        if isinstance(extra, dict):
            headers.update({str(key): str(value) for key, value in extra.items()})
        return headers

    # -- request helpers ------------------------------------------------------

    def _client(self) -> tuple[bool, httpx.AsyncClient]:
        """Return ``(owned, client)``; owned clients are closed by us."""
        if self._injected is not None:
            return False, self._injected
        return True, httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    async def post(
        self,
        payload: dict[str, Any],
        model: str | None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST ``payload`` once, mapping every failure mode to an oinyx error."""
        owns_client, client = self._client()
        try:
            response = await client.post(
                self.endpoint, headers=headers or self.headers, json=payload
            )
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"{self.name} request failed: {exc}", provider=self.name, model=model
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
        self.raise_for_status(response, model)
        return response

    async def get(self, url: str, model: str | None = None) -> httpx.Response:
        """GET ``url`` with the same headers and error mapping as :meth:`post`."""
        owns_client, client = self._client()
        try:
            response = await client.get(url, headers=self.headers)
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"{self.name} request failed: {exc}", provider=self.name, model=model
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
        self.raise_for_status(response, model)
        return response

    def raise_for_status(self, response: httpx.Response, model: str | None) -> None:
        """Turn a bad (or challenged) response into the right oinyx exception."""
        if self._is_challenge(response):
            raise CloudflareChallengeError(
                "Cloudflare blocked this request with a browser challenge. This "
                "endpoint only answers clients that have solved it, so pass a real "
                "browser session: config={'cookies': {'cf_clearance': '...', "
                "'cf_bm': '...'}, 'user_agent': '<the same UA that solved the challenge>'}.",
                provider=self.name,
                model=model,
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise self._error_for(response, model)

    @staticmethod
    def _is_challenge(response: httpx.Response) -> bool:
        if response.headers.get("cf-mitigated", "").lower() == "challenge":
            return True
        if response.status_code in (403, 429, 503):
            body = (response.text or "")[:2000].lower()
            return CHALLENGE_MARKER in body or "challenge-platform" in body
        return False

    def _error_for(self, response: httpx.Response, model: str | None) -> ProviderError:
        status = response.status_code
        message, code = describe_error(error_payload(response))
        detail = f"{message} [code={code}]"
        if status in (401, 403):
            return AuthenticationError(
                detail, provider=self.name, model=model, status_code=status
            )
        if status == 429:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                detail += f" (retry after {retry_after}s)"
            return RateLimitError(detail, provider=self.name, model=model, status_code=status)
        if status >= 500:
            return UpstreamError(detail, provider=self.name, model=model, status_code=status)
        return ProviderError(detail, provider=self.name, model=model, status_code=status)

    @staticmethod
    def _describe_error(error: Any) -> tuple[str, str]:
        """Normalise string-or-object error payloads into ``(message, code)``."""
        return describe_error(error)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} endpoint={self.endpoint!r}>"
