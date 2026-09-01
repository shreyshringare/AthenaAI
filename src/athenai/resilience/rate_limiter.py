"""
Token bucket rate limiter — caps the throughput of calls to a flaky/quota-
limited external dependency (e.g. per-provider LLM API rate limits).

HOW IT FITS IN THE SYSTEM:
Intended to guard outbound calls the same way circuit_breaker.CircuitBreaker
guards against failures: a caller awaits limiter.acquire() immediately before
making the protected call, and handles RateLimitError the way it would any
other backpressure signal (e.g. surface to the caller, or feed into a retry
policy from athenai.resilience.retry). Unlike a typical rate limiter,
acquire() never sleeps/waits for tokens — see WHY RAISE, NOT BLOCK below.

WHY RAISE, NOT BLOCK:
Blocking until tokens are available would let async callers pile up waiting
on a shared lock with no bound, and hides backpressure from the caller who
may want to fail fast, shed load, or retry with backoff instead of queueing.
Raising immediately keeps the decision of what to do next with the caller.

WHY ONE LOCK COVERS BOTH FIELDS:
token_count and last_refill must be updated atomically. If two coroutines
interleave between the refill check and the token decrement, both could see
a full bucket and both decrement past the capacity limit. One asyncio.Lock
covering both fields prevents this — no TOCTOU between refill and consume.

WHY TOKEN BUCKET (not leaky bucket):
Token bucket allows bursts up to capacity while enforcing a long-run rate.
Leaky bucket smooths bursts, which is undesirable for batch operations that
legitimately need short high-throughput windows.
"""

from __future__ import annotations

import asyncio
import time

from athenai.core.exceptions import RateLimitError


class TokenBucketRateLimiter:
    """Lazily-refilling token bucket; refill happens on acquire(), not on a timer.

    There is no background task ticking the bucket — elapsed time since the
    last acquire() is computed on demand, so an idle limiter costs nothing
    and naturally catches up to full capacity once enough time has passed.
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        """
        Args:
            capacity: Maximum token count (burst size).
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Consume `tokens` from the bucket, refilling first based on elapsed time.

        Raises RateLimitError immediately if insufficient tokens are
        available — does not wait/retry (see module WHY RAISE, NOT BLOCK).
        On failure, no tokens are deducted.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + refill)
            self._last_refill = now

            if self._tokens < tokens:
                raise RateLimitError(
                    f"Rate limit exceeded: {self._tokens:.1f} tokens available, {tokens} requested"
                )

            self._tokens -= tokens
