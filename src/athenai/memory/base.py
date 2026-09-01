"""
Core memory types shared across all memory layers.

HOW IT FITS IN THE SYSTEM:
Every concrete memory implementation (ConversationMemory, SemanticMemory,
SummaryMemory, AgentState) returns or consumes MemoryEntry, tagged with a
MemoryType. This lets callers (e.g. the context engine's memory_fn) treat
results from different backends uniformly instead of branching per source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    """Identifies which memory layer produced a MemoryEntry."""

    WORKING = "working"
    CONVERSATION = "conversation"
    SUMMARY = "summary"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class MemoryEntry:
    """
    WHY FROZEN:
    Memory entries are read by multiple pipeline stages (context engine,
    agent executor). Immutability prevents accidental mutation and makes
    entries safe to cache and pass across async boundaries.
    """

    id: str
    session_id: str
    memory_type: MemoryType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: float = 0.0
