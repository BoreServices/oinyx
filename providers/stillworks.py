"""``stillworks`` -- a discovery registry, not a chat provider.

``GET https://stillworks.supercapybara.com/api/llm/up`` returns every keyless
relay the service knows about, *proved by an actual keyless call* and ranked by
recent success rate then latency. It is not an inference endpoint: it answers
"which model, on which base URL, is replying right now", which is exactly the
question a per-IP free tier makes hard -- an entry that answered from their
datacenter can still hand your address ``402`` or ``429``.

    >>> from providers import stillworks
    >>> routes = await stillworks.routes()
    >>> routes[0]
    Route(provider='pollinations', model='openai-fast', answered='168/168', ...)

:func:`walk` turns that ranking into the intended pattern -- try candidates in
order until one actually answers, and report which one did:

    >>> route, response = await stillworks.walk("summarise this in one line")
    >>> route.model_string
    'ovh/gpt-oss-120b'

The registry labels endpoints with its own names (``ovh-anonymous``,
``free-ai``, ``kilo-gateway``); :data:`ENDPOINT_NAMES` maps them onto the
provider names registered in this package, so a route is directly usable as an
``OinyxClient`` model string. Anything unmapped keeps its upstream name and
simply will not route.

Note the registry itself is behind Cloudflare's *bot* protection rather than
the managed challenge: a request with no ``User-Agent`` gets a bare ``403``,
so the browser agent below is sent by default.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from oinyx import ChatResponse, OinyxClient, OinyxError, ProviderError, retry

from .http_utils import as_dict, as_list

REGISTRY_URL = "https://stillworks.supercapybara.com/api/llm/up"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0"

#: Registry endpoint label -> provider name registered in this package.
ENDPOINT_NAMES: dict[str, str] = {
    "kilo-gateway": "kai",
    "llm7": "llm7",
    "ovh-anonymous": "ovh",
    "free-ai": "freeai",
    "pollinations": "pollinations",
    "aihorde": "aihorde",
    "keylessai": "keylessai",
}

#: Capability keys the registry checks, in the order it reports them.
CAPABILITIES = ("tools", "json_schema", "json_object", "vision")


@dataclass(frozen=True)
class Route:
    """One proved (endpoint, model) pair, ready to be tried."""

    provider: str
    model: str
    base_url: str
    chat_path: str
    auth: str = "none"
    tier: str = ""
    answered: str = ""
    successes: int = 0
    attempts: int = 0
    latency_ms: int = 0
    last_ok: str = ""
    proved: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    capabilities: dict[str, bool] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def model_string(self) -> str:
        """The ``provider/model`` value to hand to :class:`OinyxClient`."""
        return f"{self.provider}/{self.model}"

    @property
    def success_rate(self) -> float:
        """Recent successes over recent attempts, ``0.0`` when nothing was tried."""
        return self.successes / self.attempts if self.attempts else 0.0

    def supports(self, capability: str) -> bool:
        """Whether the registry *observed* ``capability`` working, not whether it was claimed."""
        return capability in self.proved

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy for logging or caching."""
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "chat_path": self.chat_path,
            "auth": self.auth,
            "tier": self.tier,
            "answered": self.answered,
            "successes": self.successes,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "last_ok": self.last_ok,
            "proved": list(self.proved),
            "failed": list(self.failed),
            "model_string": self.model_string,
        }
        return data

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Route:
        """Build a route from one registry entry.

        Unknown endpoint labels keep their own name, so ``model_string`` stays
        honest instead of silently pointing at a provider that does not exist.
        """
        endpoint = str(row.get("endpoint", ""))
        answered = str(row.get("answered", ""))
        successes, _, attempts = answered.partition("/")
        caps = as_dict(row.get("capabilities"))
        return cls(
            provider=ENDPOINT_NAMES.get(endpoint, endpoint),
            model=str(row.get("model", "")),
            base_url=str(row.get("openai_base_url") or row.get("base_url") or ""),
            chat_path=str(row.get("chat_path", "")),
            auth=str(row.get("auth") or row.get("auth_mode") or "none"),
            tier=str(row.get("tier", "")),
            answered=answered,
            successes=int(successes) if successes.isdigit() else 0,
            attempts=int(attempts) if attempts.isdigit() else 0,
            latency_ms=int(row.get("latency_ms") or 0),
            last_ok=str(row.get("last_ok", "")),
            proved=tuple(str(value) for value in as_list(row.get("proved"))),
            failed=tuple(str(value) for value in as_list(row.get("failed"))),
            capabilities={
                name: bool(as_dict(caps.get(name)).get("supported"))
                for name in caps
            },
            raw=row,
        )


@dataclass(frozen=True)
class Registry:
    """The whole payload: ranked routes plus the service's own disclaimer."""

    routes: tuple[Route, ...]
    generated_at: str = ""
    disclaimer: str = ""
    ranked_by: str = ""

    def providers(self) -> dict[str, list[Route]]:
        """Group routes by provider name, preserving the registry's ranking."""
        grouped: dict[str, list[Route]] = {}
        for route in self.routes:
            grouped.setdefault(route.provider, []).append(route)
        return grouped


async def fetch(
    url: str = REGISTRY_URL,
    *,
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
    headers: dict[str, str] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    attempts: int = 2,
    backoff: float = 1.0,
) -> Registry:
    """Download and parse the registry, retrying transient failures.

    Raises:
        ProviderError: when the registry is unreachable, refuses, or returns a
            body that is not the documented object.
    """
    request_headers = {"accept": "application/json", "user-agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    async def once() -> Registry:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.get(url, headers=request_headers)
        if response.status_code >= 400:
            raise ProviderError(
                f"stillworks returned HTTP {response.status_code}: {response.text[:200]}",
                provider="stillworks",
                model=None,
                status_code=response.status_code,
            )
        body = response.json()
        if not isinstance(body, dict):
            raise ProviderError(
                f"stillworks returned {type(body).__name__}, expected an object",
                provider="stillworks",
                model=None,
            )
        rows = [row for row in as_list(body.get("models")) if isinstance(row, dict)]
        return Registry(
            routes=tuple(Route.from_row(row) for row in rows),
            generated_at=str(body.get("generated_at", "")),
            disclaimer=str(body.get("disclaimer", "")),
            ranked_by=str(body.get("ranked_by", "")),
        )

    retried = retry(
        attempts=max(1, attempts),
        backoff=backoff,
        on=(httpx.HTTPError, ValueError),
        sleep=sleep,
    )(once)
    try:
        return await retried()
    except (httpx.HTTPError, ValueError) as exc:
        # Callers should only ever have to handle oinyx errors, even on a flap.
        raise ProviderError(
            f"stillworks lookup failed: {exc}", provider="stillworks", model=None
        ) from exc


def rank(
    routes: Sequence[Route],
    *,
    provider: str | None = None,
    capability: str | None = None,
    routable_only: bool = True,
) -> list[Route]:
    """Filter and sort candidates the way the registry itself ranks them.

    Args:
        routes: Candidates, usually straight from :class:`Registry`.
        provider: Keep only this provider name (``"ovh"``, ...).
        capability: Keep only routes the registry *proved* (``"tools"``).
        routable_only: Drop entries whose endpoint has no adapter in this
            package, since their ``model_string`` cannot be called.
    """
    kept = list(routes)
    if routable_only:
        known = set(ENDPOINT_NAMES.values())
        kept = [route for route in kept if route.provider in known]
    if provider:
        kept = [route for route in kept if route.provider == provider]
    if capability:
        kept = [route for route in kept if route.supports(capability)]
    return sorted(kept, key=lambda route: (-route.success_rate, route.latency_ms))


async def walk(
    question: str,
    *,
    candidates: Sequence[Route] | None = None,
    client: OinyxClient | None = None,
    limit: int = 5,
    max_tokens: int = 64,
    routes: Registry | None = None,
    **kwargs: Any,
) -> tuple[Route, ChatResponse]:
    """Try candidates in rank order and return the first one that answers.

    Every keyless relay here throttles per IP, so the point is not to pick the
    best model but the one currently willing to reply. Failures are collected
    rather than raised; the last error is re-raised only when nothing answered.

    Args:
        question: Prompt sent to each candidate.
        candidates: Routes to try; defaults to the live registry, ranked.
        client: Client to route through; a fresh :class:`OinyxClient` by default.
        limit: How many candidates to try before giving up.
        max_tokens: Small by design -- this is a liveness check, not a answer.
        routes: Pre-fetched registry to read candidates from, to avoid a second
            lookup when the caller already has one.
        **kwargs: Forwarded to ``client.chat`` verbatim (``temperature``, ...).

    Returns:
        The route that answered, and its response.

    Raises:
        ProviderError: when no candidate produced a completion. The message
            lists what each one said.
    """
    if candidates is None:
        routes = routes or await fetch()
        candidates = rank(routes.routes)
    active = client or OinyxClient()
    failures: list[str] = []
    for route in list(candidates)[: max(1, limit)]:
        started = time.perf_counter()
        try:
            response = await active.chat(
                route.model_string, question, max_tokens=max_tokens, **kwargs
            )
        except OinyxError as exc:
            failures.append(f"{route.model_string}: {type(exc).__name__} {exc.message[:80]}")
            continue
        if not response.content.strip():
            failures.append(f"{route.model_string}: empty completion")
            continue
        response.raw["stillworks"] = {
            "answered": route.answered,
            "registry_latency_ms": route.latency_ms,
            "tried_ms": int((time.perf_counter() - started) * 1000),
        }
        return route, response
    raise ProviderError(
        "no stillworks candidate answered: " + " | ".join(failures or ["nothing to try"]),
        provider="stillworks",
        model=None,
    )


async def model_strings(url: str = REGISTRY_URL, **kwargs: Any) -> list[str]:
    """Convenience: the ranked, routable candidates as ``provider/model`` strings."""
    registry = await fetch(url, **kwargs)
    return [route.model_string for route in rank(registry.routes)]


def main(argv: list[str] | None = None) -> int:
    """``python -m providers.stillworks`` -- print the ranked table."""
    argv = list(argv or [])
    question = argv[0] if argv else ""
    registry = asyncio.run(fetch())
    print(f"generated_at={registry.generated_at} ranked_by={registry.ranked_by!r}")
    print(registry.disclaimer)
    print()
    for route in rank(registry.routes):
        proved = ",".join(route.proved) or "-"
        print(
            f"  {route.model_string:52} {route.answered:>9} "
            f"{route.latency_ms:>6}ms proved={proved}"
        )
    if not question:
        return 0
    print(f"\nwalking candidates with {question!r} ...")
    try:
        route, response = asyncio.run(walk(question, routes=registry, limit=3))
    except OinyxError as exc:
        print(f"  none answered: {exc.message}")
        return 1
    print(f"  {route.model_string} -> {response.content.strip()[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
