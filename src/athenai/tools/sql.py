"""SQLTool — read-only SELECT execution via asyncpg.

HOW IT FITS IN THE SYSTEM:
Registered into a ToolRegistry (athenai.tools.registry) wherever the host
application wires up a database pool, and invoked by AgentExecutor
(athenai.agents.executor) when the model issues a "sql_query" tool call.
Needs an asyncpg.Pool supplied by the caller — this module owns no
connection lifecycle of its own.

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
    """First line of defense: reject any query not starting with SELECT.

    This is a cheap string check, not a parser — it catches the common case
    but is not the sole safeguard. See module docstring: the readonly=True
    transaction in SQLTool.execute is what actually stops writes that dodge
    this check (e.g. via comment injection or semicolon chaining).
    """
    first_token = query.strip().split()[0].upper() if query.strip() else ""
    if first_token != "SELECT":
        raise ToolDeniedError(
            f"only SELECT queries are permitted; got {first_token!r}"
        )


class SQLTool:
    """Tool protocol implementation for bounded, read-only SQL SELECT queries."""

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
        """Run the caller's query wrapped as a subquery so a row cap always
        applies, even if the caller's own query has no LIMIT (or a larger
        one). The requested limit is clamped to self._max_rows rather than
        trusted outright, so a caller can lower but never raise the ceiling.
        """
        query = arguments["query"].strip()
        _reject_writes(query)

        limit = min(int(arguments.get("limit", self._max_rows)), self._max_rows)
        bounded_query = f"SELECT * FROM ({query}) _q LIMIT {limit}"

        async with self._pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                rows = await conn.fetch(bounded_query)

        return [dict(row) for row in rows]
