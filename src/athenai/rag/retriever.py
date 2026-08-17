"""PgVectorRetriever — pgvector-backed chunk store and similarity search."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from athenai.rag.chunker import Chunk

_CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    content      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    embedding    vector(1536),
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PgVectorRetriever:
    """Stores chunks with embeddings and retrieves by cosine similarity.

    WHY COSINE (NOT L2):
    Consistent with SemanticMemory. Cosine normalises for embedding magnitude,
    giving better semantic ordering across variable-length chunks.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> PgVectorRetriever:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        instance = cls(pool)
        await instance._ensure_schema()
        return instance

    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)

    async def store_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        now = time.time()
        async with self._pool.acquire() as conn:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
                await conn.execute(
                    """
                    INSERT INTO chunks
                        (id, document_id, content, chunk_index, embedding, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, $7)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    emb_str,
                    json.dumps(chunk.metadata),
                    now,
                )

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, document_id, content, chunk_index, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                emb_str,
                k,
            )
        results = []
        for row in rows:
            raw_meta = row["metadata"]
            if isinstance(raw_meta, str):
                meta: dict[str, Any] = json.loads(raw_meta) if raw_meta else {}
            elif raw_meta:
                meta = dict(raw_meta)
            else:
                meta = {}
            if metadata_filter and not all(
                meta.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    score=float(row["score"]),
                    metadata=meta,
                )
            )
        return results

    async def count_by_document(self, document_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
                document_id,
            )
        return int(result or 0)

    async def delete_document(self, document_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1", document_id
            )

    async def close(self) -> None:
        await self._pool.close()
