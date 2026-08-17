"""
Agent state machine types.

WHY STATE MACHINE:
Agent execution is inherently stateful and asynchronous. Encoding valid
transitions in a lookup table makes illegal state changes detectable at
call time rather than at some later point when behaviour becomes undefined.
The states match the execution lifecycle exactly: created → running →
awaiting_tool → completed/failed. Terminal states (completed, failed) have
no outgoing edges — transition() raises ValueError if you try.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    AWAITING_TOOL = "awaiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"


# Legal state transitions. Terminal states map to empty sets.
_TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.CREATED: frozenset({AgentStatus.RUNNING}),
    AgentStatus.RUNNING: frozenset(
        {AgentStatus.AWAITING_TOOL, AgentStatus.COMPLETED, AgentStatus.FAILED}
    ),
    AgentStatus.AWAITING_TOOL: frozenset({AgentStatus.RUNNING, AgentStatus.FAILED}),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
}


def transition(current: AgentStatus, target: AgentStatus) -> AgentStatus:
    """Validate and return the target status. Raises ValueError on illegal transition."""
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"illegal transition: {current} → {target}")
    return target


@dataclass
class AgentStep:
    """Single iteration of the agent loop — model call + optional tool round-trip."""

    iteration: int
    model_response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Complete result returned after the agent loop terminates."""

    final_answer: str
    steps: list[AgentStep]
    total_iterations: int
    status: AgentStatus
