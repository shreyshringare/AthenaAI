"""InMemoryCache — process-local CacheBackend implementation, no external
dependency.

HOW IT FITS IN THE SYSTEM:
Used by gateway/app.py when ATHENA_REDIS_URL is unset or unreachable, so
embedding caching still works (within a single process) without forcing
Redis on local/demo deployments — same degrade-to-permissive pattern as
every other optional infra dependency in this gateway.

WHY A DICT, NOT functools.lru_cache:
lru_cache has no TTL support and is sync-only; CacheBackend is async by
contract (so RedisCache and InMemoryCache are interchangeable behind the
same interface) and embedding cache entries should expire eventually
(the underlying provider's embedding model could change).
"""

from __future__ import annotations

import time
from typing import Any


class InMemoryCache:
    """Dict-backed cache with optional per-entry TTL. Not shared across
    processes — see RedisCache for that."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and time.monotonic() >= expires_at

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if self._is_expired(expires_at):
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None
