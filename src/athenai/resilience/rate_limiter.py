"""
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
