"""
SemanticMemory: pgvector cosine similarity search over stored facts.

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
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> SemanticMemory:
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
