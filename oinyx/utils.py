"""Retry/backoff decorators and streaming assembly helpers.

Nothing in this module performs network I/O; :func:`retry` simply waits via
:func:`asyncio.sleep` (or an injected sleeper) between attempts.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from typing import Any, ParamSpec, TypeVar, cast

from .exceptions import ProviderError, RateLimitError
from .models import ChatResponse, StreamChunk, Usage

P = ParamSpec("P")
R = TypeVar("R")

#: Exceptions retried by default: transient provider failures and rate limits.
DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (RateLimitError, ProviderError)

BackoffLike = float | Callable[[int], float]


def compute_delay(backoff: BackoffLike, attempt: int) -> float:
    """Delay before retry ``attempt`` (1-based).

    A number grows exponentially -- ``backoff * 2 ** (attempt - 1)`` -- while a
    callable receives the attempt number and returns the delay in seconds.
    Negative results are clamped to ``0.0``.
    """
    if isinstance(backoff, (int, float)):
        return max(0.0, float(backoff) * 2.0 ** (attempt - 1))
    return max(0.0, float(backoff(attempt)))


def retry(
    attempts: int = 3,
    backoff: BackoffLike = 1.0,
    on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator for async (and sync) callables.

    Args:
        attempts: Total tries, including the first. Must be >= 1.
        backoff: Base delay in seconds (exponential) or a callable mapping the
            1-based attempt number to a delay.
        on: Exception types that trigger a retry. Anything else propagates
            immediately.
        sleep: Async sleep override, injectable so tests avoid real waiting.
        on_retry: Optional sync hook called as ``on_retry(attempt, exc, delay)``.

    Raises:
        ValueError: if ``attempts < 1``.
        TypeError: if ``on`` is not a tuple of exception types.

    The final exception is re-raised once all attempts are exhausted::

        @retry(attempts=4, backoff=0.5, on=(RateLimitError,))
        async def chat(...): ...
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    if not isinstance(on, tuple) or not all(
        isinstance(exc, type) and issubclass(exc, BaseException) for exc in on
    ):
        raise TypeError("on must be a tuple of exception types")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_async_gen = inspect.isasyncgenfunction(func)
        is_sync_gen = inspect.isgeneratorfunction(func)
        coroutine_func = cast("Callable[P, Awaitable[Any]]", func)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            for attempt in range(1, attempts + 1):
                try:
                    return await coroutine_func(*args, **kwargs)
                except on as exc:
                    if attempt == attempts:
                        raise
                    delay = compute_delay(backoff, attempt)
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    if delay:
                        await (sleep or asyncio.sleep)(delay)
            raise AssertionError("retry loop exited unexpectedly")  # pragma: no cover

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except on as exc:
                    if attempt == attempts:
                        raise
                    delay = compute_delay(backoff, attempt)
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    if delay:
                        time.sleep(delay)
            raise AssertionError("retry loop exited unexpectedly")  # pragma: no cover

        if is_async_gen or is_sync_gen:
            raise TypeError(
                f"retry cannot wrap generator function {func.__qualname__}; "
                "wrap the coroutine it awaits instead"
            )
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator


def estimate_tokens(text: str) -> int:
    """Rough token count using the common ``len(text) / 4`` heuristic.

    Used as a fallback when a provider omits usage information. Whitespace and
    punctuation are counted like words, which is what makes the divide-by-four
    rule a reasonable approximation for English prose.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def fill_usage(
    usage: Usage | None,
    messages: Iterable[Any],
    completion: str,
) -> Usage:
    """Return ``usage``, or an estimate when it is empty.

    Args:
        usage: Provider-reported usage, possibly ``None`` or all zeros.
        messages: Conversation turns used to estimate prompt tokens. Any of
            :class:`~oinyx.models.ChatMessage`, dict, or str is accepted.
        completion: Generated text used to estimate completion tokens.
    """
    if usage is not None and (usage.prompt_tokens or usage.completion_tokens):
        return usage
    prompt = sum(estimate_tokens(_text_of(item)) for item in messages)
    done = estimate_tokens(completion)
    return Usage(prompt_tokens=prompt, completion_tokens=done)


def _text_of(item: Any) -> str:
    content = getattr(item, "content", None)
    if content is None and isinstance(item, dict):
        content = item.get("content", "")
    if content is None:
        content = item
    return content if isinstance(content, str) else str(content)


async def collect_stream(
    stream: AsyncIterable[StreamChunk] | AsyncIterable[str],
    *,
    model: str = "",
    id: str | None = None,
    provider: str | None = None,
    messages: Iterable[Any] | None = None,
) -> ChatResponse:
    """Concatenate a stream into a single :class:`~oinyx.models.ChatResponse`.

    Accepts chunks or bare strings. ``finish_reason`` is taken from the last
    chunk that carries one, usage is estimated from ``messages`` when the
    stream does not report it, and raw dicts are merged into ``raw`` keyed by
    chunk position. Errors raised by the underlying stream propagate as-is.
    """
    parts: list[str] = []
    raws: dict[str, Any] = {}
    finish_reason: str | None = None
    usage: Usage | None = None
    index = 0

    async for chunk in stream:
        index += 1
        if isinstance(chunk, str):
            parts.append(chunk)
            continue
        if chunk.delta:
            parts.append(chunk.delta)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.raw:
            raws[f"chunk_{index}"] = chunk.raw
        candidate = chunk.raw.get("usage") if isinstance(chunk.raw, dict) else None
        if candidate:
            usage = Usage.from_dict(dict(candidate))

    content = "".join(parts)
    final_usage = fill_usage(usage, messages or [], content)
    return ChatResponse(
        id=id or f"stream-{uuid.uuid4().hex[:12]}",
        model=model or (provider or ""),
        content=content,
        finish_reason=finish_reason,
        usage=final_usage,
        raw=raws,
    )
