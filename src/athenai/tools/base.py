"""
Shared types for the tool system.

WHAT IS A TOOL?
A Tool is any capability an AI agent can call during its reasoning loop —
a calculator, a SQL query runner, an HTTP fetcher. Each tool has:
  - name: unique identifier the model uses to call it
  - description: natural-language explanation so the model knows when to use it
  - input_schema: JSON Schema dict describing required/optional arguments
  - execute(arguments): the actual implementation

ToolResult wraps the output uniformly so the agent executor doesn't need to
special-case errors from each tool — it checks success and routes accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Immutable result returned by any Tool.execute() call.

    success=False + error set means the tool failed but the agent can continue.
    success=False + error="fatal: ..." signals the agent should abort the loop.
    output may be None on failure.
    """

    tool_name: str
    success: bool
    output: Any
    error: str | None = None
