"""
WHY FULL JITTER:
Thundering herd — without jitter, N failing clients all retry at the same
instant after backoff, causing correlated load spikes. Full jitter spreads
retries randomly across [0, delay] so aggregate load stays flat.

WHY NOT FIXED BACKOFF:
Fixed intervals give predictable retry storms. Exponential backoff with jitter
is the AWS/Google SRE recommendation for rate-limited API calls.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: bool = True


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    retryable: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> T:
    last_exc: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except retryable as exc:
            last_exc = exc
            if attempt == policy.max_attempts - 1:
                break

            delay = min(policy.base_delay_s * (2**attempt), policy.max_delay_s)
            if policy.jitter:
                delay = random.uniform(0, delay)

            await asyncio.sleep(delay)

    raise last_exc or RuntimeError("with_retry exhausted without exception")
