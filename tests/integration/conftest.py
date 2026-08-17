"""Shared fixtures for integration tests requiring PostgreSQL."""

from __future__ import annotations

import os

import asyncpg
import pytest

TEST_DB_DSN = os.environ.get(
    "ATHENA_TEST_DB_DSN",
    "postgresql://postgres:test@localhost:5433/athenai_test",
)


@pytest.fixture
async def db_pool():
    """Function-scoped asyncpg connection pool — one per test, avoids loop mismatch."""
    pool = await asyncpg.create_pool(TEST_DB_DSN, min_size=1, max_size=5)
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
async def clean_tables(db_pool: asyncpg.Pool):
    """Clean test tables after each test."""
    yield
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT FROM pg_tables WHERE tablename = 'conversation_messages') THEN
                    TRUNCATE TABLE conversation_messages;
                END IF;
                IF EXISTS (SELECT FROM pg_tables WHERE tablename = 'semantic_memories') THEN
                    TRUNCATE TABLE semantic_memories;
                END IF;
            END $$;
            """
        )
