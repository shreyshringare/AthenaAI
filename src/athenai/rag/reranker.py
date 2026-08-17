"""Reranker — cosine similarity re-ordering of retrieved chunks."""

from __future__ import annotations

from athenai.rag.retriever import RetrievedChunk


class CosineReranker:
    """Re-ranks retrieved chunks by score descending.

    WHY RERANK AFTER RETRIEVAL:
    pgvector ANN (approximate nearest neighbour) trades exactness for speed.
    ANN may return slightly suboptimal ordering. A cheap O(k) sort over the
    small candidate set corrects ordering without a second DB round-trip.
    """

    def rerank(
        self,
        query_embedding: list[float],
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        if top_k is not None:
            return sorted_chunks[:top_k]
        return sorted_chunks
