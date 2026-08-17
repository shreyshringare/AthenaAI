"""Relevance ranking via cosine similarity (numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RankedChunk:
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
