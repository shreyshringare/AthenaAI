"""
Retry helper — re-runs an async call with exponential backoff and jitter.

HOW IT FITS IN THE SYSTEM:
Wraps a single async operation (typically an LLM provider call or tool
invocation) that may fail transiently. Composes with the other resilience
primitives rather than replacing them: a caller might wrap a
circuit_breaker.CircuitBreaker.call() in with_retry(), or catch
RateLimitError/ToolTimeoutError as part of the `retryable` exception set so
transient throttling/timeouts get retried while permanent failures surface
immediately.

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
    """Retry configuration. Delay per attempt is base_delay_s * 2**attempt,
    capped at max_delay_s, then randomised to [0, delay] if jitter is True.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: bool = True


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    retryable: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> T:
    """Call fn(), retrying on `retryable` exceptions up to policy.max_attempts.

    Exceptions not in `retryable` propagate immediately without retrying.
    After the final attempt fails, the last exception is re-raised (not
    wrapped) so callers see the original error type/traceback.
    """
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
