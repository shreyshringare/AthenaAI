"""P2 gate: token bucket rate limiter."""

from __future__ import annotations

import asyncio

import pytest

from athenai.core.exceptions import RateLimitError
from athenai.resilience.rate_limiter import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_capacity_5_fifth_succeeds() -> None:
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate=100.0)
    for _ in range(5):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_sixth_acquire_raises_rate_limit() -> None:
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate=0.0)
    for _ in range(5):
        await limiter.acquire()
    with pytest.raises(RateLimitError):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_refill_after_sleep() -> None:
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1000.0)
    for _ in range(5):
        await limiter.acquire()
    await asyncio.sleep(0.01)  # enough for 10 tokens at 1000/s
    for _ in range(5):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_concurrent_exactly_3_succeed() -> None:
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate=0.0)
    successes = 0
    failures = 0

    async def try_acquire() -> None:
        nonlocal successes, failures
        try:
            await limiter.acquire()
            successes += 1
        except RateLimitError:
            failures += 1

    await asyncio.gather(*[try_acquire() for _ in range(10)])
    assert successes == 3
    assert failures == 7
