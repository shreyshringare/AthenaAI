"""
ConversationMemory: PostgreSQL-backed recent message store.

WHY POSTGRESQL (NOT REDIS):
Conversation history needs durability across server restarts and deployments.
Redis is ephemeral by default — a restart wipes all sessions. PostgreSQL gives
ACID guarantees and runs alongside pgvector (SemanticMemory), so we need only
one DB dependency rather than two.

WHY NOT IN-PROCESS:
In-process memory is lost on restart. Multi-instance deployments (load-balanced
API) require a shared store — each instance must see the same session history.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from athenai.memory.base import MemoryEntry, MemoryType

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_conv_session_created
    ON conversation_messages (session_id, created_at DESC);
"""


class ConversationMemory:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> ConversationMemory:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        instance = cls(pool)
        await instance._ensure_schema()
        return instance

    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        import time
        import uuid

        msg_id = message_id or str(uuid.uuid4())
        now = time.time()
        meta = metadata or {}

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_messages
                    (id, session_id, role, content, created_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (id) DO NOTHING
                """,
                msg_id, session_id, role, content, now, str(meta).replace("'", '"'),
            )

        return MemoryEntry(
            id=msg_id,
            session_id=session_id,
            memory_type=MemoryType.CONVERSATION,
            content=f"{role}: {content}",
            metadata={**meta, "role": role},
            created_at=now,
        )

    async def get_recent(self, session_id: str, n: int = 10) -> list[MemoryEntry]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, role, content, created_at
                FROM conversation_messages
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id, n,
            )

        # Return in chronological order (oldest first)
        entries = [
            MemoryEntry(
                id=row["id"],
                session_id=row["session_id"],
                memory_type=MemoryType.CONVERSATION,
                content=f"{row['role']}: {row['content']}",
                metadata={"role": row["role"]},
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]
        return entries

    async def clear(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conversation_messages WHERE session_id = $1",
                session_id,
            )

    async def count(self, session_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id = $1",
                session_id,
            )
        return int(result or 0)

    async def close(self) -> None:
        await self._pool.close()
