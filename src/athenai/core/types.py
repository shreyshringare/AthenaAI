"""
Core value types for AthenaAI. All types are frozen dataclasses — immutable
by design so they can be safely passed across async boundaries and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class AIRequest:
    """
    WHY FROZEN DATACLASS:
    Requests flow through the entire pipeline — routing, context, model, observability.
    Immutability prevents accidental mutation mid-pipeline and makes the type
    safe to use as a dict key for caching.

    WHY NOT PYDANTIC MODEL:
    Pydantic is used at the API boundary (FastAPI) for parsing and validation.
    Once validated, the internal representation should be a plain frozen dataclass
    to avoid Pydantic overhead on the hot path.
    """

    messages: tuple[Message, ...]
    user_id: str
    session_id: str
    request_id: str = ""
    trace_id: str = ""
    model_role: str = "default"
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("AIRequest requires at least one message")

    def __hash__(self) -> int:
        return hash((self.messages, self.user_id, self.session_id, self.request_id))


@dataclass(frozen=True)
class AIResponse:
    """Single response from the AI runtime."""

    content: str
    model: str
    trace_id: str
    usage: TokenUsage
    request_id: str = ""
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceSpan:
    """
    WHY STRUCTURED TRACING:
    Log tailing for AI debugging is impractical — a single request involves
    routing decisions, parallel retrieval, model selection, and tool dispatch.
    A structured span tree lets you replay the exact execution path for any
    request_id without filtering thousands of log lines.
    """

    name: str
    start_ms: float
    end_ms: float
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingContext:
    """
    WHY SEPARATE ROUTING CONTEXT:
    The router needs a subset of the request plus derived signals (complexity,
    estimated tokens) that are computed once and reused. Carrying these as a
    separate frozen type avoids recomputing them on every routing decision and
    prevents the router from reaching back into the full AIRequest.
    """

    request_id: str
    user_id: str
    estimated_input_tokens: int
    task_description: str
    complexity: str = "MEDIUM"  # LOW | MEDIUM | HIGH
    required_capabilities: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
