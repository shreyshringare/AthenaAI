"""SQLTool — read-only SELECT execution via asyncpg.

WHY SELECT-ONLY:
Tools run inside an agent loop that may call them in parallel. Allowing writes
creates race conditions and irreversible side effects the agent cannot undo.
A read-only connection at the DB layer enforces this even if the query check
is bypassed.

WHY READONLY TRANSACTION (NOT JUST CHECK):
Query-string checks can be bypassed with comment injection or semicolon chaining.
asyncpg's readonly=True transaction rolls back immediately on any write attempt,
providing a second enforcement layer independent of the string check.
"""

from __future__ import annotations

from typing import Any, ClassVar

import asyncpg

from athenai.core.exceptions import ToolDeniedError

_WRITE_KEYWORDS = frozenset(
    ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
)


def _reject_writes(query: str) -> None:
    first_token = query.strip().split()[0].upper() if query.strip() else ""
    if first_token != "SELECT":
        raise ToolDeniedError(
            f"only SELECT queries are permitted; got {first_token!r}"
        )


class SQLTool:
    name = "sql_query"
    description = "Execute read-only SQL SELECT queries against the database."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query to execute",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return (default 100)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, pool: asyncpg.Pool, max_rows: int = 100) -> None:
        self._pool = pool
        self._max_rows = max_rows

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = arguments["query"].strip()
        _reject_writes(query)

        limit = min(int(arguments.get("limit", self._max_rows)), self._max_rows)
        bounded_query = f"SELECT * FROM ({query}) _q LIMIT {limit}"

        async with self._pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                rows = await conn.fetch(bounded_query)

        return [dict(row) for row in rows]
