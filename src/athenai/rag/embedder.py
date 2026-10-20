"""Text embedders — CloudEmbedder (OpenAI-compatible API), MockEmbedder, and
CachingEmbedder (wraps either behind a CacheBackend).

HOW IT FITS IN THE SYSTEM:
Third stage of the ingest pipeline (see athenai/rag/__init__.py). Called by
DocumentLoader.ingest() to turn Chunk text into vectors before
PgVectorRetriever.store_chunks() persists them. The same embedder type must
also be used to embed queries at search time, since PgVectorRetriever.search()
does cosine similarity against whatever embedding space was used at ingest.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import httpx

from athenai.core.exceptions import EmbeddingError
from athenai.core.protocols import CacheBackend, EmbedderProtocol


class CloudEmbedder:
    """Batch-aware embedder calling any OpenAI-compatible /v1/embeddings endpoint.

    WHY BATCH-AWARE:
    Embedding APIs charge per token and have per-request overhead. Batching
    reduces round-trips by up to batch_size x. With 96 texts per batch and
    ~200ms per call, throughput improves from ~5 to ~480 embeddings/second.

    WHY OPENAI-COMPATIBLE (NOT ANTHROPIC-NATIVE):
    Anthropic embeddings are served via Voyage AI with an OpenAI-compatible
    interface. A single client covers both, and future providers (Cohere,
    Mistral) use the same schema.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com",
        batch_size: int = 96,
        dimensions: int = 1536,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                result = await self._embed_batch(client, batch)
                all_embeddings.extend(result)
        return all_embeddings

    async def _embed_batch(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> list[list[float]]:
        """POST one batch and return embeddings in input order.

        Sorts the response by its `index` field before extracting embeddings —
        the API contract doesn't guarantee `data` is returned in request order,
        so relying on response order here would silently mis-pair embeddings
        with the wrong chunk text.
        """
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        response = await client.post(
            f"{self._base_url}/v1/embeddings",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code != 200:
            raise EmbeddingError(
                f"embedding API returned {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        items: list[dict[str, Any]] = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


class MockEmbedder:
    """Deterministic embedder for tests — no network, reproducible vectors."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._make_embedding(text) for text in texts]

    def _make_embedding(self, text: str) -> list[float]:
        seed = float(sum(ord(c) for c in text))
        base = [math.sin(seed + i * 0.01) for i in range(self._dimensions)]
        magnitude = math.sqrt(sum(x * x for x in base))
        return [x / magnitude for x in base]


class CachingEmbedder:
    """Wraps any EmbedderProtocol-conforming embedder behind a CacheBackend.

    WHY THIS IS SAFE TO CACHE (UNLIKE CHAT RESPONSES):
    embed() is a deterministic function of (text, model) — no sampling, no
    temperature. The same text always produces the same vector for a given
    model. Chat completions are NOT cached anywhere in this codebase for
    the opposite reason: ModelRequest.temperature defaults to 0.7, so a
    cached chat response can be a different answer than the model would
    actually give on a repeat call.

    WHY model_name IS PART OF THE CACHE KEY:
    Different models produce different vector spaces for the same text —
    without the model name in the key, switching embedder/model would read
    stale vectors from a different space as if they were still valid.

    WHY sha256(text) NOT raw text:
    Redis/cache keys should be bounded-length and free of characters that
    could misbehave in a key (newlines, unicode edge cases, arbitrary
    length for long documents) — hashing sidesteps all of that.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        cache: CacheBackend,
        model_name: str,
        ttl_seconds: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._model_name = model_name
        self._ttl_seconds = ttl_seconds

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{self._model_name}:{digest}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns embeddings in input order, fetching only cache misses
        from the wrapped embedder (in a single batched call) and writing
        them back to the cache before returning."""
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = await self._cache.get(self._key(text))
            results.append(cached)
            if cached is None:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh = await self._embedder.embed(miss_texts)
            for i, text, vector in zip(miss_indices, miss_texts, fresh, strict=True):
                results[i] = vector
                await self._cache.set(self._key(text), vector, self._ttl_seconds)

        return results  # type: ignore[return-value]  # every slot filled by now
