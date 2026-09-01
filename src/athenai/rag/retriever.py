"""PgVectorRetriever — pgvector-backed chunk store and similarity search.

HOW IT FITS IN THE SYSTEM:
Final stage of the ingest pipeline (see athenai/rag/__init__.py) — receives
Chunk objects and their embeddings from DocumentLoader.ingest() and persists
them. It also owns the query-side entry point (search()), which a future
retrieval endpoint would call with a query embedding; CosineReranker.rerank()
is meant to post-process its results, though no gateway route wires that path
yet.

WHY THE EMBEDDING COLUMN IS A FIXED vector(1536):
pgvector requires a fixed dimensionality per column. 1536 matches
CloudEmbedder's default `dimensions` (text-embedding-3-small). Swapping to an
embedder with a different dimension count requires a matching schema/migration
change here — this class does not adapt the column at runtime.
"""

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
    """Search result row. `score` is cosine similarity in [-1, 1] (1 = identical
    direction), computed as `1 - cosine_distance` from pgvector's `<=>` operator."""

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
        """Build a retriever with its own connection pool and ensure the
        `chunks` table (and the pgvector extension) exist. Prefer this over
        `__init__` unless you're sharing an existing pool, e.g. in tests."""
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
        """Insert chunks paired positionally with `embeddings` (zip, strict=True
        so a length mismatch raises immediately rather than silently truncating).
        Uses `ON CONFLICT (id) DO NOTHING`, so re-inserting a chunk_id that
        already exists is a no-op — this is what makes DocumentLoader.ingest()
        safe to retry."""
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
        """Return up to `k` chunks ordered by cosine distance (ANN, via
        pgvector's `<=>` operator).

        `metadata_filter` is applied in Python *after* the SQL LIMIT k, not
        pushed into the query — so if few of the top-k ANN candidates match
        the filter, this can return fewer than k results even when more
        matching chunks exist elsewhere in the table. Fine for small/coarse
        filters; not a substitute for a real pre-filtered query at scale.
        """
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
