"""Per-client inbound rate limiting for the gateway's business endpoints.

HOW IT FITS IN THE SYSTEM:
`rate_limit` is a FastAPI dependency attached next to `require_api_key` on
every /v1/* route (never /health, /ready, /metrics). It keeps one
TokenBucketRateLimiter per client — keyed by API key when one is sent,
else by client IP — and turns the limiter's RateLimitError into HTTP 429
with a Retry-After hint.

`ClientRateLimiter` lives on `app.state.rate_limiter`, built once in
app.py's lifespan from AthenaConfig. If it's absent (tests that build
app.state by hand) or capacity is 0, the dependency is a no-op.

WHY BOUNDED KEY STORE:
Keys come from the network, so an attacker rotating IPs/keys could grow the
dict without bound. Least-recently-used buckets are evicted past `max_clients`;
an evicted client just starts again with a full bucket.
"""

from __future__ import annotations

import math
from collections import OrderedDict

from fastapi import HTTPException, Request

from athenai.core.exceptions import RateLimitError
from athenai.observability.metrics import rate_limited_total
from athenai.resilience.rate_limiter import TokenBucketRateLimiter


class ClientRateLimiter:
    def __init__(self, capacity: int, refill_per_s: float, max_clients: int = 10_000) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self._max_clients = max_clients
        self._buckets: OrderedDict[str, TokenBucketRateLimiter] = OrderedDict()

    def _bucket(self, key: str) -> TokenBucketRateLimiter:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucketRateLimiter(self.capacity, self.refill_per_s)
            self._buckets[key] = bucket
            if len(self._buckets) > self._max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return bucket

    async def acquire(self, key: str) -> None:
        await self._bucket(key).acquire()

    def retry_after_s(self) -> int:
        if self.refill_per_s <= 0:
            return 1
        return max(1, math.ceil(1.0 / self.refill_per_s))


async def rate_limit(request: Request) -> None:
    limiter: ClientRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None or limiter.capacity <= 0:
        return
    client = request.headers.get("x-api-key") or (request.client.host if request.client else "?")
    try:
        await limiter.acquire(client)
    except RateLimitError as exc:
        rate_limited_total.inc()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(limiter.retry_after_s())},
        ) from exc
