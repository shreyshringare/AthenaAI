"""In-flight agent execution state — mutable, in-process only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """
    WHY MUTABLE DATACLASS:
    Working memory tracks in-flight execution state (current step, tool
    results accumulated so far). Unlike value types, this state is
    intentionally mutable — the agent executor writes to it on every
    tool call iteration. Not frozen, not persisted, not shared across requests.
    """

    task: str
    session_id: str
    steps: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    status: str = "CREATED"
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, description: str) -> None:
        self.steps.append(description)

    def add_tool_result(self, tool: str, result: Any, error: str | None = None) -> None:
        self.tool_results.append({
            "tool": tool,
            "result": result,
            "error": error,
            "iteration": self.iteration,
        })

    def snapshot(self) -> dict[str, Any]:
        """Serialisable snapshot for persistence or logging."""
        return {
            "task": self.task,
            "session_id": self.session_id,
            "steps": list(self.steps),
            "tool_results": list(self.tool_results),
            "status": self.status,
            "iteration": self.iteration,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> AgentState:
        state = cls(task=data["task"], session_id=data["session_id"])
        state.steps = data.get("steps", [])
        state.tool_results = data.get("tool_results", [])
        state.status = data.get("status", "CREATED")
        state.iteration = data.get("iteration", 0)
        state.metadata = data.get("metadata", {})
        return state
