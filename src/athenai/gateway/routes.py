"""
HTTP route handlers.

Route layout:
  GET  /health              — liveness probe (always 200 if process up)
  GET  /ready               — readiness probe (checks model health)
  GET  /metrics             — Prometheus text format metrics
  POST /v1/chat             — single-turn chat, returns full response
  POST /v1/chat/stream      — single-turn chat, Server-Sent Events stream
  POST /v1/agents/run       — multi-step agent with tool use
  POST /v1/documents/ingest — RAG document ingest
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from athenai.core.types import AIRequest, Message, MessageRole
from athenai.gateway.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStepSchema,
    ChatRequest,
    ChatResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    ReadyResponse,
    UsageStats,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    healthy = await request.app.state.runtime._model.health_check()
    return ReadyResponse(status="ready" if healthy else "not_ready", model_healthy=healthy)


@router.get("/metrics")
async def metrics() -> StreamingResponse:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    data = generate_latest()
    return StreamingResponse(iter([data]), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    from athenai.observability.metrics import requests_total

    req = AIRequest(
        messages=tuple(
            Message(role=MessageRole(m.role), content=m.content) for m in body.messages
        ),
        user_id=body.user_id,
        session_id=body.session_id,
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        model_role=body.model,
    )

    try:
        response = await request.app.state.runtime.execute(req)
        requests_total.labels(endpoint="/v1/chat", status="success").inc()
    except Exception as exc:
        requests_total.labels(endpoint="/v1/chat", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        content=response.content,
        model=response.model,
        trace_id=response.trace_id,
        usage=UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        request_id=response.request_id,
    )


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    from athenai.observability.metrics import requests_total

    req = AIRequest(
        messages=tuple(
            Message(role=MessageRole(m.role), content=m.content) for m in body.messages
        ),
        user_id=body.user_id,
        session_id=body.session_id,
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        model_role=body.model,
        stream=True,
    )

    async def event_stream():
        try:
            async for token in request.app.state.runtime.stream(req):
                yield f"data: {token}\n\n"
            requests_total.labels(endpoint="/v1/chat/stream", status="success").inc()
        except Exception as exc:
            requests_total.labels(endpoint="/v1/chat/stream", status="error").inc()
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/v1/agents/run", response_model=AgentRunResponse)
async def agent_run(body: AgentRunRequest, request: Request) -> AgentRunResponse:
    from athenai.observability.metrics import agent_iterations_total, requests_total

    try:
        result = await request.app.state.agent.run(task=body.task, user_id=body.user_id)
        requests_total.labels(endpoint="/v1/agents/run", status="success").inc()
        agent_iterations_total.inc(result.total_iterations)
    except Exception as exc:
        requests_total.labels(endpoint="/v1/agents/run", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentRunResponse(
        final_answer=result.final_answer,
        steps=[
            AgentStepSchema(
                iteration=s.iteration,
                model_response=s.model_response,
                tool_calls=s.tool_calls,
                tool_results=s.tool_results,
            )
            for s in result.steps
        ],
        total_iterations=result.total_iterations,
        status=result.status.value,
    )


@router.post("/v1/documents/ingest", response_model=DocumentIngestResponse)
async def document_ingest(
    body: DocumentIngestRequest, request: Request
) -> DocumentIngestResponse:
    loader = getattr(request.app.state, "document_loader", None)
    if loader is None:
        raise HTTPException(
            status_code=503,
            detail="Document loader not configured. Set ATHENA_DB_URL and ATHENA_EMBEDDER_URL.",
        )

    chunks = await loader.ingest(
        content=body.content,
        document_id=body.document_id,
        source=body.source,
        metadata=body.metadata,
    )

    return DocumentIngestResponse(document_id=body.document_id, chunks_stored=chunks)
