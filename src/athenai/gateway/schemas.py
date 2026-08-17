"""Pydantic models for the HTTP API boundary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
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
    task: str
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1, le=25)
    user_id: str = "default"


class AgentStepSchema(BaseModel):
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
