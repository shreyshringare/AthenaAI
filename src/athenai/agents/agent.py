"""
Agent — high-level facade over AgentExecutor.

Constructs the executor with sane defaults and exposes a single run() entry
point. Callers that need fine-grained control (custom registry, iteration
cap) can use AgentExecutor directly.
"""

from __future__ import annotations

from typing import Any

from athenai.agents.executor import AgentExecutor
from athenai.agents.state import AgentResult
from athenai.tools.registry import ToolRegistry


class Agent:
    """Autonomous agent that plans and executes tasks using tools."""

    def __init__(
        self,
        model: Any,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._executor = AgentExecutor(
            model=model,
            tool_registry=tool_registry or ToolRegistry(),
            max_iterations=max_iterations,
        )

    async def run(self, task: str, user_id: str = "default") -> AgentResult:
        """Execute a task. Returns AgentResult with final answer + full step trace."""
        return await self._executor.run(task, user_id=user_id)
