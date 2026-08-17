"""P7 gate (integration): SQLTool + HTTPTool live execution."""

from __future__ import annotations

import asyncpg
import httpx
import pytest
import respx

from athenai.core.exceptions import ToolDeniedError
from athenai.tools.http import HTTPTool
from athenai.tools.sql import SQLTool

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# SQLTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_select_runs(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool)
    result = await tool.execute({"query": "SELECT 1 AS value"})
    assert isinstance(result, list)
    assert result[0]["value"] == 1


@pytest.mark.asyncio
async def test_sql_select_current_timestamp(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool)
    result = await tool.execute({"query": "SELECT current_timestamp AS ts"})
    assert len(result) == 1
    assert "ts" in result[0]


@pytest.mark.asyncio
async def test_sql_rejects_insert(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool)
    with pytest.raises(ToolDeniedError, match="only SELECT"):
        await tool.execute({"query": "INSERT INTO chunks VALUES ('x', 'y', 'z', 0, NULL, '{}', 0)"})


@pytest.mark.asyncio
async def test_sql_rejects_drop(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool)
    with pytest.raises(ToolDeniedError, match="only SELECT"):
        await tool.execute({"query": "DROP TABLE chunks"})


@pytest.mark.asyncio
async def test_sql_rejects_update(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool)
    with pytest.raises(ToolDeniedError, match="only SELECT"):
        await tool.execute({"query": "UPDATE chunks SET content = 'x' WHERE id = '1'"})


@pytest.mark.asyncio
async def test_sql_limit_enforced(db_pool: asyncpg.Pool) -> None:
    tool = SQLTool(db_pool, max_rows=2)
    result = await tool.execute(
        {"query": "SELECT n FROM (VALUES (1),(2),(3),(4),(5)) AS t(n)"}
    )
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# HTTPTool (mocked via respx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_get_success() -> None:
    tool = HTTPTool(allowed_domains=["api.example.com"])
    with respx.mock:
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, text='{"result": "ok"}')
        )
        result = await tool.execute({"url": "https://api.example.com/data"})
    assert result["status"] == 200
    assert "ok" in result["body"]


@pytest.mark.asyncio
async def test_http_get_blocked_domain() -> None:
    tool = HTTPTool(allowed_domains=["api.example.com"])
    with pytest.raises(ToolDeniedError, match="not in the allowed domains list"):
        await tool.execute({"url": "https://attacker.com/steal"})


@pytest.mark.asyncio
async def test_http_get_body_truncated() -> None:
    tool = HTTPTool(allowed_domains=["example.com"])
    big_body = "x" * 20000
    with respx.mock:
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(200, text=big_body)
        )
        result = await tool.execute({"url": "https://example.com/big"})
    assert len(result["body"]) <= 8192
