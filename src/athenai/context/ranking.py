"""
Relevance ranking via cosine similarity (numpy).

HOW IT FITS IN THE SYSTEM:
This is the ranking utility for the context package: given a query embedding
and a set of candidate chunks (each optionally carrying its own embedding),
it scores and truncates to the top_k most relevant before content is handed
to ContextEngine/ContextPacker for token-budgeted assembly. It operates
purely in-memory on already-fetched candidates — it does not perform vector
search itself (that's PgVectorRetriever in athenai.rag.retriever) and is
independent of athenai.rag.reranker.CosineReranker, which re-sorts chunks
that already carry a similarity score from a DB-side ANN query.

WHY SCORE 0.0 INSTEAD OF RAISING ON MISSING/ZERO EMBEDDINGS:
A chunk with no embedding or a zero-norm vector has no defined cosine
similarity. Raising would let one malformed chunk fail an entire ranking
pass; scoring it 0.0 (lowest possible relevance) lets ranking degrade
gracefully and keeps the chunk visible for debugging instead of silently
disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RankedChunk:
    """A chunk after scoring; `metadata` excludes `content`/`embedding` (already surfaced separately)."""

    content: str
    score: float
    metadata: dict[str, object]


class RelevanceRanker:
    """Ranks retrieved chunks by cosine similarity to the query embedding."""

    def rank(
        self,
        query_embedding: list[float],
        chunks: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Scores and sorts `chunks` by cosine similarity, returning the top_k.

        If `query_embedding` is a zero vector (no defined direction), cosine
        similarity can't be computed against anything — this returns the
        first `top_k` chunks unscored (score=0.0) rather than raising, so
        callers still get a bounded result set.
        """
        if not chunks:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [
                RankedChunk(
                    content=str(c.get("content", "")),
                    score=0.0,
                    metadata={k: v for k, v in c.items() if k not in ("content", "embedding")},
                )
                for c in chunks[:top_k]
            ]

        q_unit = q / q_norm
        scored: list[RankedChunk] = []

        for chunk in chunks:
            embedding = chunk.get("embedding")
            content = str(chunk.get("content", ""))
            meta = {k: v for k, v in chunk.items() if k not in ("content", "embedding")}

            if embedding is None:
                scored.append(RankedChunk(content=content, score=0.0, metadata=meta))
                continue

            v = np.array(embedding, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q_unit, v / v_norm))

            scored.append(RankedChunk(content=content, score=score, metadata=meta))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
