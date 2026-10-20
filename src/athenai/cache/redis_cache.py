"""RedisCache — Redis-backed CacheBackend implementation.

HOW IT FITS IN THE SYSTEM:
Used by gateway/app.py when ATHENA_REDIS_URL (AthenaConfig.redis_url) is
reachable at startup, shared across process restarts and multiple API
instances (unlike InMemoryCache, which is process-local). Values are
JSON-serialised — embedding vectors (list[float]) round-trip through JSON
without precision loss relevant to cosine similarity.

WHY redis.asyncio OVER a sync client:
Every I/O path in AthenaAI is async (same reasoning as CloudModel using
httpx.AsyncClient over requests) — a sync Redis client would block the
event loop on every cache lookup.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis


class RedisCache:
    """Wraps a redis.asyncio.Redis client behind the CacheBackend protocol."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        await self._client.set(key, json.dumps(value), ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))
