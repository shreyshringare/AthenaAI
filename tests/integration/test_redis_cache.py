"""Integration tests for athenai.cache.redis_cache.RedisCache against a real
Redis (requires the `redis` docker-compose service / ATHENA_TEST_REDIS_URL).
"""

from __future__ import annotations

import asyncio

import pytest

from athenai.cache.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_set_then_get_roundtrips_through_real_redis(redis_client) -> None:
    cache = RedisCache(redis_client)
    await cache.set("embedding:test:k1", [0.1, 0.2, 0.3])
    assert await cache.get("embedding:test:k1") == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_get_missing_key_returns_none(redis_client) -> None:
    cache = RedisCache(redis_client)
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_exists_reflects_presence(redis_client) -> None:
    cache = RedisCache(redis_client)
    assert await cache.exists("k1") is False
    await cache.set("k1", "v1")
    assert await cache.exists("k1") is True


@pytest.mark.asyncio
async def test_delete_removes_key(redis_client) -> None:
    cache = RedisCache(redis_client)
    await cache.set("k1", "v1")
    await cache.delete("k1")
    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_ttl_expiry_via_real_redis(redis_client) -> None:
    """Uses Redis's own TTL/expiry mechanism (EX), not an in-process timer --
    proves the ttl_seconds param actually reaches Redis's EXPIRE, not just
    this process's clock."""
    cache = RedisCache(redis_client)
    await cache.set("k1", "v1", ttl_seconds=1)
    assert await cache.get("k1") == "v1"
    await asyncio.sleep(1.5)
    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_complex_value_roundtrips(redis_client) -> None:
    """Embedding vectors are lists of floats -- confirm JSON round-trip
    preserves type and precision well enough for cosine similarity use."""
    cache = RedisCache(redis_client)
    vector = [0.123456789, -0.987654321, 0.0]
    await cache.set("embedding:test:vec", vector)
    result = await cache.get("embedding:test:vec")
    assert result == vector
