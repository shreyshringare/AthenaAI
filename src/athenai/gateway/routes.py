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

HOW IT FITS IN THE SYSTEM:
This module defines the `router` that app.py mounts onto the FastAPI app.
Handlers are thin translators: decode the Pydantic request (schemas.py) into
the internal core types (AIRequest, Message), delegate to the long-lived
singletons built once at startup in app.py's lifespan (`request.app.state.runtime`
for /v1/chat*, `request.app.state.agent` for /v1/agents/run), then re-encode
the internal result back into a Pydantic response schema. No business logic
lives here — routing failures into HTTPException and bumping metrics counters
is the extent of it.
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
    """Liveness probe. Deliberately does not touch the model or DB — it only
    proves the process is alive and serving. Use /ready to check dependencies.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    """Readiness probe: pings the model backend's health_check(). Reaches into
    runtime._model directly (not a public AthenaRuntime method) since the
    runtime itself has no notion of "ready" beyond its model being reachable.
    """
    healthy = await request.app.state.runtime._model.health_check()
    return ReadyResponse(status="ready" if healthy else "not_ready", model_healthy=healthy)


@router.get("/metrics")
async def metrics() -> StreamingResponse:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    data = generate_latest()
    return StreamingResponse(iter([data]), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Single-turn, non-streaming chat. Mints a fresh request_id/trace_id per
    call (the client cannot supply its own) so every request is independently
    traceable in logs/metrics even if the client reuses a session_id.
    """
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
    """Streams the chat response as Server-Sent Events.

    Wire format: each model token is sent as `data: {token}\\n\\n`; a failure
    mid-stream is surfaced as `data: [ERROR] {exc}\\n\\n` rather than an HTTP
    error status, since the response headers (200 OK) are already flushed by
    the time an error can occur. The stream always ends with `data: [DONE]\\n\\n`
    (success or failure) via the `finally` block, so clients have one
    consistent signal to stop reading.
    """
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
    """Run the shared `app.state.agent` to completion on `body.task`.

    Note: `body.tools` and `body.max_iterations` are accepted by the schema
    but not forwarded here — the agent's tool registry and iteration cap are
    fixed at process startup (see app.py's lifespan). Per-request overrides
    would require either a fresh AgentExecutor per call or a registry that
    supports request-scoped filtering; neither is wired up yet.
    """
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
    """Chunk + embed + store a document for later retrieval.

    Returns 503 (not 500) when the RAG subsystem was never configured —
    see app.py's _maybe_add_document_loader, which leaves document_loader as
    None instead of raising when ATHENA_DB_URL/ATHENA_EMBEDDER_URL are unset
    or unreachable. 503 tells the caller "try again once configured", which
    is more accurate than a generic server error.
    """
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
