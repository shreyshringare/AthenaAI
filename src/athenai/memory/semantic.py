"""
SemanticMemory: pgvector cosine similarity search over stored facts.

HOW IT FITS IN THE SYSTEM:
Unlike ConversationMemory (raw message log), this layer stores discrete
facts/embeddings a caller has chosen to persist (e.g. extracted user
preferences) and retrieves them by semantic similarity to a query embedding
rather than recency — typically fed into a memory_fn wired into
athenai.context.engine.ContextEngine alongside RAG results.

WHY PGVECTOR:
pgvector integrates directly with PostgreSQL — same connection pool, same
transaction semantics, same backup/restore process as ConversationMemory.
No separate vector DB infrastructure required.

WHY COSINE (NOT L2):
Cosine similarity measures angle between embeddings, not magnitude. Embedding
magnitude varies with text length; cosine normalises this out, giving more
reliable semantic similarity scores for variable-length facts.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from athenai.memory.base import MemoryEntry, MemoryType

_CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS semantic_memories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    created_at  DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sem_session
    ON semantic_memories (session_id);
"""


class SemanticMemory:
    """Stores facts with embeddings and retrieves them by cosine similarity.

    Facts are addressed by caller-supplied or generated id; re-storing under
    the same id overwrites content/embedding (see store()), unlike
    ConversationMemory's append-only log.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> SemanticMemory:
        """Async factory: __init__ can't await pool creation/schema setup,
        so construction goes through this classmethod instead of __init__."""
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        instance = cls(pool)
        await instance._ensure_schema()
        return instance

    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)

    async def store(
        self,
        session_id: str,
        content: str,
        embedding: list[float],
        memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Insert a fact, or overwrite content/embedding of an existing one
        with the same memory_id (ON CONFLICT DO UPDATE) — this is the
        mechanism for correcting a previously stored fact in place rather
        than accumulating stale duplicates."""
        import time
        import uuid

        mem_id = memory_id or str(uuid.uuid4())
        now = time.time()
        meta = metadata or {}
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO semantic_memories
                    (id, session_id, content, embedding, created_at, metadata)
                VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                mem_id, session_id, content, embedding_str, now, json.dumps(meta),
            )

        return MemoryEntry(
            id=mem_id,
            session_id=session_id,
            memory_type=MemoryType.SEMANTIC,
            content=content,
            embedding=embedding,
            metadata=meta,
            created_at=now,
        )

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        session_id: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return up to k facts ordered by similarity (highest first),
        each paired with its score. Score is `1 - cosine_distance` (via
        pgvector's `<=>` operator) so 1.0 is a perfect match and the value
        can be used directly as a relevance/confidence signal — callers
        don't need to know pgvector returns distance, not similarity.
        session_id=None searches across all sessions.
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        async with self._pool.acquire() as conn:
            if session_id:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, content, metadata, created_at,
                           1 - (embedding <=> $1::vector) AS score
                    FROM semantic_memories
                    WHERE session_id = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    embedding_str, session_id, k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, content, metadata, created_at,
                           1 - (embedding <=> $1::vector) AS score
                    FROM semantic_memories
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    embedding_str, k,
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
            entry = MemoryEntry(
                id=row["id"],
                session_id=row["session_id"],
                memory_type=MemoryType.SEMANTIC,
                content=row["content"],
                metadata=meta,
                created_at=row["created_at"],
            )
            results.append((entry, float(row["score"])))

        return results

    async def delete(self, memory_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM semantic_memories WHERE id = $1",
                memory_id,
            )

    async def close(self) -> None:
        await self._pool.close()
