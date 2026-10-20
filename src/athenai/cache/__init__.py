"""
Cache implementations for athenai.core.protocols.CacheBackend.

HOW IT FITS IN THE SYSTEM:
CacheBackend (core/protocols.py) was defined but had zero implementations
until now. Two are provided:
  - in_memory.py — InMemoryCache: process-local dict + per-key expiry,
                   used when ATHENA_REDIS_URL is unset/unreachable.
  - redis_cache.py — RedisCache: wraps redis.asyncio, used when
                      ATHENA_REDIS_URL is reachable (see gateway/app.py).

Both are consumed by athenai.rag.embedder.CachingEmbedder, the only current
caller — embeddings are a deterministic function of (text, model), so
caching them is always safe. Caching full chat/agent responses was
deliberately NOT built here: ModelRequest.temperature defaults to 0.7
(real sampling), so a cached chat response can be a different answer than
the model would actually give on a repeat call — caching that is a
correctness risk, not just a performance tradeoff.
"""
