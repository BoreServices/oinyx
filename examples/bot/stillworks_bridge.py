"""Bridge to :mod:`providers.stillworks`, whose registry says what is answering."""

from __future__ import annotations

from typing import Any

from oinyx import OinyxError, OinyxClient

from providers import stillworks


def _failures(message: str) -> list[str]:
    """Split stillworks' joined failure list back into bullets."""
    return [part.strip() for part in message.split("|") if part.strip()]


async def fetch_routes(limit: int = 12) -> list[Any]:
    """Return the ranked, routable routes, or an empty list if the registry is down."""
    try:
        registry = await stillworks.fetch()
    except Exception:  # noqa: BLE001 - the caller reports the outage
        return []
    return list(stillworks.rank(registry.routes))[: max(1, limit)]


async def walk(
    client: OinyxClient,
    prompt: str,
    *,
    limit: int = 5,
    max_tokens: int = 512,
) -> tuple[str, str, list[str]]:
    """Ask the registry until something answers.

    Returns ``(model_string, answer, failures)``; ``answer`` is empty when every
    candidate refused, in which case ``failures`` says why.
    """
    candidates = await fetch_routes(limit=limit)
    if not candidates:
        return ("", "", ["stillworks gave nothing routable to try (registry unreachable?)"])
    try:
        route, response = await stillworks.walk(
            prompt, candidates=candidates, client=client, limit=limit, max_tokens=max_tokens
        )
    except Exception as exc:  # noqa: BLE001 - walk reports refusals as errors
        message = getattr(exc, "message", None) or str(exc)
        return ("", "", _failures(str(message)) or [str(message)[:200]])
    return (route.model_string, response.content, [])


async def registry_table(limit: int = 12) -> list[tuple[str, str, str]]:
    """Return ``(model_string, success rate, latency)`` rows for reporting."""
    rows: list[tuple[str, str, str]] = []
    for route in await fetch_routes(limit=limit):
        rows.append(
            (
                route.model_string,
                f"{route.success_rate:.2f} ({route.answered})",
                f"{route.latency_ms}ms",
            )
        )
    return rows
