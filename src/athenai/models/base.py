"""Model-layer request/response types, separate from core AIRequest/AIResponse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRequest:
    """
    WHY SEPARATE FROM AIRequest:
    AIRequest carries user-facing metadata (session_id, trace_id, user_id, routing hints).
    ModelRequest is the stripped-down payload sent to the LLM adapter — only what the
    model API needs. This separation means adapters stay ignorant of routing and auth.
    """

    messages: list[dict[str, str]]
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    system: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Response returned by any Model adapter."""

    content: str
    model_name: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)
