"""Reranker — cosine similarity re-ordering of retrieved chunks.

HOW IT FITS IN THE SYSTEM:
Last step of the query-side RAG path, downstream of
PgVectorRetriever.search() (see athenai/rag/__init__.py). Called by
athenai.rag.loader.DocumentLoader.search(), which the /v1/documents/search
gateway route uses — DocumentLoader constructs a default instance if none
is injected.
"""

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
        """Sort `chunks` by their existing `.score` descending and optionally
        truncate to `top_k`.

        Note `query_embedding` is accepted but unused: this reranker only
        re-sorts the similarity scores retriever.search() already computed,
        it doesn't recompute similarity itself. The parameter exists to keep
        the interface stable for a future reranker that re-scores candidates
        (e.g. a cross-encoder) against the query.
        """
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        if top_k is not None:
            return sorted_chunks[:top_k]
        return sorted_chunks
