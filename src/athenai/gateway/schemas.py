"""
Pydantic models for the HTTP API boundary.

HOW IT FITS IN THE SYSTEM:
These are the request/response contracts FastAPI uses to validate incoming
JSON and serialise outgoing JSON for routes.py. They are intentionally kept
separate from the internal dataclasses in athenai.core.types (Message,
AIRequest, AIResponse, ...): the internal types are immutable and shaped
around the pipeline's needs, while these schemas are shaped around a stable
public API and Pydantic's validation/OpenAPI-doc generation. routes.py is the
only place that converts between the two.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One entry in ChatRequest.messages. `role` is a plain str here (not the
    MessageRole enum) so FastAPI can return a normal 422 validation error for
    typos; routes.py converts it to MessageRole and raises there if invalid.
    """

    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str = "default"
    stream: bool = False
    user_id: str = "default"
    session_id: str = Field(default="default")


class UsageStats(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    model: str
    trace_id: str
    usage: UsageStats
    request_id: str


class AgentRunRequest(BaseModel):
    """Request body for POST /v1/agents/run.

    `tools` and `max_iterations` are validated here but currently unused by
    the /v1/agents/run handler — the running agent's tool registry and
    iteration cap are fixed at server startup (see app.py's lifespan), not
    per request. They exist for API forward-compatibility; do not assume
    setting them changes agent behavior today.
    """

    task: str
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1, le=25)
    user_id: str = "default"


class AgentStepSchema(BaseModel):
    """One iteration of the agent loop: the model's raw response plus any
    tool calls it requested and the results returned for them.
    """

    iteration: int
    model_response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)


class AgentRunResponse(BaseModel):
    final_answer: str
    steps: list[AgentStepSchema]
    total_iterations: int
    status: str


class DocumentIngestRequest(BaseModel):
    content: str
    document_id: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    status: str
    model_healthy: bool
