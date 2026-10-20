"""Unit tests for DocumentLoader.search()'s RelevanceRanker wiring.

Uses a stub retriever (no DB needed) that deliberately returns candidates
in an order that does NOT match true cosine similarity to the query, so
these tests prove RelevanceRanker's independent Python-side recomputation
actually determines final order/score -- not just a pass-through of
whatever order/score the stub (standing in for SQL) handed it.
"""

from __future__ import annotations

import math

import pytest

from athenai.rag.loader import DocumentLoader
from athenai.rag.retriever import RetrievedChunk


def _unit_embedding(seed: float, dim: int = 8) -> list[float]:
    base = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in base))
    return [x / norm for x in base]


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def search(self, query_embedding, k=5, metadata_filter=None):
        return self._chunks[:k]


class _StubEmbedder:
    """Returns a fixed, caller-supplied embedding regardless of query text,
    so the test controls exactly what "the query" looks like in vector
    space without depending on MockEmbedder's text-hash-based scheme."""

    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embedding for _ in texts]


@pytest.mark.asyncio
async def test_search_reorders_by_true_cosine_not_stub_order() -> None:
    query_embedding = _unit_embedding(0.0)
    # chunk_a is deliberately placed first by the stub (simulating a wrong
    # SQL/ANN ordering) but is actually the least similar to the query;
    # chunk_b is last but actually most similar.
    chunk_a = RetrievedChunk(
        chunk_id="a", document_id="d1", content="least relevant",
        chunk_index=0, score=0.99, embedding=_unit_embedding(50.0),
    )
    chunk_b = RetrievedChunk(
        chunk_id="b", document_id="d1", content="most relevant",
        chunk_index=1, score=0.01, embedding=_unit_embedding(0.01),
    )
    retriever = _StubRetriever([chunk_a, chunk_b])
    loader = DocumentLoader(
        parser=None, chunker=None, embedder=_StubEmbedder(query_embedding), retriever=retriever
    )

    results = await loader.search("irrelevant query text", top_k=2)

    assert results[0].chunk_id == "b"  # RelevanceRanker's own cosine math wins
    assert results[0].content == "most relevant"


@pytest.mark.asyncio
async def test_search_truncates_to_top_k_via_relevance_ranker() -> None:
    query_embedding = _unit_embedding(0.0)
    chunks = [
        RetrievedChunk(
            chunk_id=str(i), document_id="d1", content=f"chunk {i}",
            chunk_index=i, score=0.5, embedding=_unit_embedding(float(i)),
        )
        for i in range(5)
    ]
    retriever = _StubRetriever(chunks)
    loader = DocumentLoader(
        parser=None, chunker=None, embedder=_StubEmbedder(query_embedding), retriever=retriever
    )

    results = await loader.search("query", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_handles_missing_embeddings_gracefully() -> None:
    """A candidate with no embedding (e.g. older schema row) must not
    crash the ranking step -- RelevanceRanker scores it 0.0 and it simply
    ranks last, per its own documented behavior."""
    query_embedding = _unit_embedding(0.0)
    chunk_no_embedding = RetrievedChunk(
        chunk_id="x", document_id="d1", content="no embedding", chunk_index=0, score=0.9
    )
    chunk_with_embedding = RetrievedChunk(
        chunk_id="y", document_id="d1", content="has embedding",
        chunk_index=1, score=0.1, embedding=_unit_embedding(0.0),
    )
    retriever = _StubRetriever([chunk_no_embedding, chunk_with_embedding])
    loader = DocumentLoader(
        parser=None, chunker=None, embedder=_StubEmbedder(query_embedding), retriever=retriever
    )

    results = await loader.search("query", top_k=2)
    assert results[0].chunk_id == "y"
