"""Text embedders — CloudEmbedder (OpenAI-compatible API) and MockEmbedder."""

from __future__ import annotations

import math
from typing import Any

import httpx

from athenai.core.exceptions import EmbeddingError


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
