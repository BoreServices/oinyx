"""Tests for :mod:`oinyx.utils` -- retry, backoff, stream assembly, estimates."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from oinyx.exceptions import OinyxError, ProviderError, RateLimitError
from oinyx.models import ChatMessage, StreamChunk, Usage
from oinyx.utils import collect_stream, compute_delay, estimate_tokens, fill_usage, retry


@pytest.fixture
def clock() -> list[float]:
    """Records requested delays instead of sleeping."""
    return []


@pytest.fixture
def fake_sleep(clock: list[float]):
    async def sleep(delay: float) -> None:
        clock.append(delay)

    return sleep


class TestComputeDelay:
    def test_exponential_growth_from_a_number(self) -> None:
        assert [compute_delay(0.5, attempt) for attempt in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 4.0]

    def test_callable_backoff_receives_the_attempt(self) -> None:
        assert compute_delay(lambda attempt: attempt * 3, 2) == 6.0

    def test_negative_delays_are_clamped(self) -> None:
        assert compute_delay(-5, 1) == 0.0
        assert compute_delay(lambda attempt: -1, 1) == 0.0


class TestRetryAsync:
    async def test_returns_first_success_without_sleeping(self, fake_sleep: Any, clock: list[float]) -> None:
        calls = 0

        @retry(attempts=3, backoff=1, sleep=fake_sleep)
        async def work() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert await work() == "ok"
        assert calls == 1 and clock == []

    async def test_retries_listed_exceptions_then_succeeds(
        self, fake_sleep: Any, clock: list[float]
    ) -> None:
        attempts: list[int] = []

        @retry(attempts=4, backoff=1, on=(RateLimitError,), sleep=fake_sleep)
        async def work(value: int) -> int:
            attempts.append(value)
            if len(attempts) < 3:
                raise RateLimitError("slow down")
            return value

        assert await work(7) == 7
        assert len(attempts) == 3
        assert clock == [1.0, 2.0]

    async def test_reraises_the_final_exception(self, fake_sleep: Any) -> None:
        @retry(attempts=2, backoff=0, sleep=fake_sleep)
        async def work() -> None:
            raise RateLimitError("nope", provider="mock", model="m")

        with pytest.raises(RateLimitError) as excinfo:
            await work()
        assert excinfo.value.provider == "mock"

    async def test_unlisted_exceptions_propagate_immediately(self, fake_sleep: Any, clock: list[float]) -> None:
        calls = 0

        @retry(attempts=5, backoff=1, on=(ProviderError,), sleep=fake_sleep)
        async def work() -> None:
            nonlocal calls
            calls += 1
            raise OinyxError("unretryable")

        with pytest.raises(OinyxError):
            await work()
        assert calls == 1 and clock == []

    async def test_forwards_arguments_and_preserves_metadata(self, fake_sleep: Any) -> None:
        @retry(attempts=2, sleep=fake_sleep)
        async def work(a: int, *, b: int = 0) -> int:
            return a + b

        assert await work(1, b=2) == 3
        assert work.__name__ == "work"

    async def test_on_retry_hook_is_invoked(self, fake_sleep: Any) -> None:
        seen: list[tuple[int, float]] = []

        @retry(
            attempts=3,
            backoff=0.25,
            sleep=fake_sleep,
            on_retry=lambda attempt, exc, delay: seen.append((attempt, delay)),
        )
        async def work() -> str:
            if len(seen) < 2:
                raise RateLimitError("retry me")
            return "done"

        assert await work() == "done"
        assert seen == [(1, 0.25), (2, 0.5)]

    async def test_zero_delay_skips_sleep(self, fake_sleep: Any, clock: list[float]) -> None:
        calls = 0

        @retry(attempts=2, backoff=0, sleep=fake_sleep)
        async def work() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("x")

        await work()
        assert clock == []


class TestRetrySync:
    def test_sync_function_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import oinyx.utils as utils

        slept: list[float] = []
        monkeypatch.setattr(utils.time, "sleep", lambda delay: slept.append(delay))
        calls = 0

        @retry(attempts=3, backoff=1)
        def work() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RateLimitError("busy")
            return "sync-ok"

        assert work() == "sync-ok"
        assert slept == [1.0, 2.0]

    def test_sync_reraises_after_the_last_attempt(self) -> None:
        @retry(attempts=2, backoff=0)
        def work() -> None:
            raise ProviderError("dead")

        with pytest.raises(ProviderError, match="dead"):
            work()

    def test_generator_functions_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="cannot wrap generator"):

            @retry(attempts=2)
            async def streamer() -> Any:
                yield 1

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(ValueError, match="attempts must be >= 1"):
            retry(attempts=0)
        with pytest.raises(TypeError, match="on must be a tuple"):
            retry(attempts=2, on=RateLimitError)  # type: ignore[arg-type]


class TestTokenEstimate:
    @pytest.mark.parametrize(
        "text,expected",
        [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2), ("hello world", 3)],
    )
    def test_char_over_four(self, text: str, expected: int) -> None:
        assert estimate_tokens(text) == expected

    def test_fill_usage_prefers_provider_numbers(self) -> None:
        given = Usage(100, 200, 300)
        assert fill_usage(given, ["prompt"], "completion") is given

    def test_fill_usage_estimates_when_empty(self) -> None:
        usage = fill_usage(Usage(), [ChatMessage("user", "abcd")], "abcdefghij")
        assert usage == Usage(1, 3, 4)

    def test_fill_usage_accepts_dicts_and_none(self) -> None:
        usage = fill_usage(None, [{"role": "user", "content": "abcd"}, "abcd"], "")
        assert usage.prompt_tokens == 2
        assert usage.completion_tokens == 0


class TestCollectStream:
    async def test_joins_deltas_and_keeps_finish_reason(self) -> None:
        async def source() -> Any:
            for delta in ("he", "llo"):
                yield StreamChunk(delta=delta, raw={"delta": delta})
            yield StreamChunk(delta="", finish_reason="stop")

        response = await collect_stream(source(), model="mock/m", id="id-1")
        assert response.content == "hello"
        assert (response.id, response.model, response.finish_reason) == ("id-1", "mock/m", "stop")
        assert response.raw["chunk_1"] == {"delta": "he"}

    async def test_accepts_bare_strings(self) -> None:
        async def source() -> Any:
            yield "a"
            yield "b"

        assert (await collect_stream(source())).content == "ab"

    async def test_uses_usage_from_the_final_chunk(self) -> None:
        async def source() -> Any:
            yield StreamChunk(delta="hi", raw={"usage": {"prompt_tokens": 5, "completion_tokens": 2}})

        response = await collect_stream(source(), messages=["hello there"])
        assert response.usage == Usage(5, 2, 7)

    async def test_estimates_usage_when_absent(self) -> None:
        async def source() -> Any:
            yield StreamChunk(delta="abcd")

        response = await collect_stream(source(), messages=[ChatMessage("user", "abcd abcd")])
        assert response.usage == Usage(3, 1, 4)

    async def test_generated_id_when_missing(self) -> None:
        async def source() -> Any:
            yield StreamChunk(delta="x")

        response = await collect_stream(source())
        assert response.id.startswith("stream-")
        assert response.model == ""

    async def test_empty_stream_yields_empty_response(self) -> None:
        async def source() -> Any:
            return
            yield StreamChunk()

        response = await collect_stream(source())
        assert response.content == "" and response.finish_reason is None

    async def test_stream_errors_propagate(self) -> None:
        async def source() -> Any:
            yield StreamChunk(delta="partial")
            raise RateLimitError("boom")

        with pytest.raises(RateLimitError):
            await collect_stream(source())

    async def test_respects_external_cancellation(self) -> None:
        async def source() -> Any:
            yield StreamChunk(delta="one")
            await asyncio.sleep(10)

        task = asyncio.create_task(collect_stream(source()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
