"""Retry helper: exponential backoff, jitter, retryable filter."""

from __future__ import annotations

import pytest

from athenai.resilience import retry as retry_mod
from athenai.resilience.retry import RetryPolicy, with_retry


class _Flaky:
    def __init__(self, failures: int, exc: type[Exception] = ValueError) -> None:
        self.calls = 0
        self._failures = failures
        self._exc = exc

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._exc("boom")
        return "ok"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(d: float) -> None:
        recorded.append(d)

    monkeypatch.setattr(retry_mod.asyncio, "sleep", fake_sleep)
    return recorded


async def test_succeeds_after_transient_failures(sleeps: list[float]) -> None:
    fn = _Flaky(2)
    assert await with_retry(fn, RetryPolicy(max_attempts=3, jitter=False)) == "ok"
    assert fn.calls == 3


async def test_backoff_is_exponential_without_jitter(sleeps: list[float]) -> None:
    fn = _Flaky(2)
    await with_retry(fn, RetryPolicy(max_attempts=3, base_delay_s=0.5, jitter=False))
    assert sleeps == [0.5, 1.0]


async def test_delay_capped_at_max(sleeps: list[float]) -> None:
    fn = _Flaky(4)
    policy = RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=2.0, jitter=False)
    await with_retry(fn, policy)
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


async def test_jitter_stays_within_exponential_bound(sleeps: list[float]) -> None:
    fn = _Flaky(3)
    await with_retry(fn, RetryPolicy(max_attempts=4, base_delay_s=1.0, jitter=True))
    assert len(sleeps) == 3
    for attempt, delay in enumerate(sleeps):
        assert 0 <= delay <= 1.0 * 2**attempt


async def test_exhausted_reraises_last_exception(sleeps: list[float]) -> None:
    fn = _Flaky(10)
    with pytest.raises(ValueError, match="boom"):
        await with_retry(fn, RetryPolicy(max_attempts=3, jitter=False))
    assert fn.calls == 3


async def test_non_retryable_propagates_immediately(sleeps: list[float]) -> None:
    fn = _Flaky(1, exc=KeyError)
    with pytest.raises(KeyError):
        await with_retry(fn, RetryPolicy(max_attempts=3), retryable=ValueError)
    assert fn.calls == 1
    assert sleeps == []
