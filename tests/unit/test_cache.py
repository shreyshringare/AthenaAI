"""Unit tests for athenai.cache.in_memory.InMemoryCache (no Redis needed --
see tests/integration/test_redis_cache.py for RedisCache against real Redis)
and athenai.rag.embedder.CachingEmbedder.
"""

from __future__ import annotations

import asyncio

import pytest

from athenai.cache.in_memory import InMemoryCache


@pytest.mark.asyncio
async def test_set_then_get_returns_value() -> None:
    cache = InMemoryCache()
    await cache.set("k1", [1.0, 2.0, 3.0])
    assert await cache.get("k1") == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_get_missing_key_returns_none() -> None:
    cache = InMemoryCache()
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_exists_reflects_presence() -> None:
    cache = InMemoryCache()
    assert await cache.exists("k1") is False
    await cache.set("k1", "v1")
    assert await cache.exists("k1") is True


@pytest.mark.asyncio
async def test_delete_removes_key() -> None:
    cache = InMemoryCache()
    await cache.set("k1", "v1")
    await cache.delete("k1")
    assert await cache.get("k1") is None
    assert await cache.exists("k1") is False


@pytest.mark.asyncio
async def test_delete_missing_key_is_noop() -> None:
    cache = InMemoryCache()
    await cache.delete("nonexistent")  # must not raise


@pytest.mark.asyncio
async def test_ttl_expiry() -> None:
    cache = InMemoryCache()
    await cache.set("k1", "v1", ttl_seconds=0.05)
    assert await cache.get("k1") == "v1"
    await asyncio.sleep(0.1)
    assert await cache.get("k1") is None
    assert await cache.exists("k1") is False


@pytest.mark.asyncio
async def test_no_ttl_never_expires() -> None:
    cache = InMemoryCache()
    await cache.set("k1", "v1")
    await asyncio.sleep(0.05)
    assert await cache.get("k1") == "v1"


# ---------------------------------------------------------------------------
# CachingEmbedder
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    """Deterministic embedder that counts how many times embed() actually ran,
    so tests can assert the cache prevented a redundant call."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(len(t))] for t in texts]


@pytest.mark.asyncio
async def test_caching_embedder_hits_cache_on_repeat_text() -> None:
    from athenai.rag.embedder import CachingEmbedder

    inner = _CountingEmbedder()
    cache = InMemoryCache()
    embedder = CachingEmbedder(inner, cache, model_name="test-model")

    first = await embedder.embed(["hello world"])
    second = await embedder.embed(["hello world"])

    assert first == second == [[11.0]]
    assert inner.call_count == 1  # second call was a pure cache hit


@pytest.mark.asyncio
async def test_caching_embedder_only_embeds_uncached_texts() -> None:
    from athenai.rag.embedder import CachingEmbedder

    inner = _CountingEmbedder()
    cache = InMemoryCache()
    embedder = CachingEmbedder(inner, cache, model_name="test-model")

    await embedder.embed(["a", "b"])
    result = await embedder.embed(["a", "b", "c"])

    # Two embed() calls total: first ingests "a","b"; second only needs "c".
    assert inner.call_count == 2
    assert result == [[1.0], [1.0], [1.0]]


@pytest.mark.asyncio
async def test_caching_embedder_different_models_do_not_collide() -> None:
    from athenai.rag.embedder import CachingEmbedder

    cache = InMemoryCache()
    inner_a = _CountingEmbedder()
    inner_b = _CountingEmbedder()
    embedder_a = CachingEmbedder(inner_a, cache, model_name="model-a")
    embedder_b = CachingEmbedder(inner_b, cache, model_name="model-b")

    await embedder_a.embed(["same text"])
    await embedder_b.embed(["same text"])

    assert inner_a.call_count == 1
    assert inner_b.call_count == 1  # not served from model-a's cache entry
