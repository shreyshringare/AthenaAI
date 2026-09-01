"""
Agent — high-level facade over AgentExecutor.

HOW IT FITS IN THE SYSTEM:
The gateway (athenai.gateway.app) constructs one Agent per app with a model
and a ToolRegistry, then calls run(task) per incoming request. Agent itself
holds no loop logic — it just constructs an AgentExecutor with sane defaults
and forwards run() to it. Callers that need fine-grained control (custom
registry, iteration cap) can use AgentExecutor directly instead.
"""

from __future__ import annotations

from typing import Any

from athenai.agents.executor import AgentExecutor
from athenai.agents.state import AgentResult
from athenai.tools.registry import ToolRegistry


class Agent:
    """Thin wrapper that owns an AgentExecutor and exposes only run().

    Stateless across calls other than the executor's fixed config (model,
    registry, max_iterations) — each run() call gets its own fresh message
    history, so one Agent instance is safe to reuse across concurrent
    requests.
    """

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
