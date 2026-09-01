"""
AthenaRuntime — the top-level orchestration layer between the HTTP gateway
and the model adapters.

HOW IT FITS IN THE SYSTEM:
`athenai.gateway.app.create_app()` builds one `AthenaRuntime` per process
(in the FastAPI `lifespan`) and stores it on `app.state.runtime`. The
`/v1/chat` and `/v1/chat/stream` route handlers in `athenai.gateway.routes`
call `execute()` / `stream()` on it per request. AthenaRuntime itself calls
down into a model adapter (`athenai.models.base.Model`, e.g. MockModel or
CloudModel) — it has no knowledge of tools or the multi-step agent loop
(that's `athenai.agents.agent.Agent`, which wraps a model directly instead
of going through this class).

WHY SEMAPHORE:
Unbounded concurrent model calls exhaust API rate limits and memory.
BoundedExecutor caps in-flight model calls with an asyncio.Semaphore so
back-pressure propagates to the HTTP layer as latency, not crashes. The
semaphore is created once per AthenaRuntime instance and shared across all
requests handled by that instance, so `max_concurrent` is a process-wide cap.

WHY SEPARATE FROM GATEWAY:
The runtime does not know about HTTP — it works with AIRequest/AIResponse.
This lets the same pipeline be called from tests, batch jobs, or a CLI
without pulling in FastAPI.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from athenai.core.types import AIRequest, AIResponse, TokenUsage
from athenai.models.base import ModelRequest


class AthenaRuntime:
    """Orchestrates a single (non-agentic) model call with concurrency control.

    One instance is meant to be created per process and reused across
    requests — the `asyncio.Semaphore` bounding `max_concurrent` lives on
    the instance, so constructing a new AthenaRuntime per request would
    defeat the concurrency cap.
    """

    def __init__(self, model: Any, max_concurrent: int = 20) -> None:
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, request: AIRequest) -> AIResponse:
        """Run a non-streaming chat request. Blocks until the model responds."""
        messages = [{"role": str(m.role), "content": m.content} for m in request.messages]
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
        )

        async with self._semaphore:
            response = await self._model.generate(model_req)

        return AIResponse(
            content=response.content,
            model=response.model_name,
            trace_id=request.trace_id or str(uuid.uuid4()),
            usage=TokenUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.input_tokens + response.output_tokens,
            ),
            request_id=request.request_id or str(uuid.uuid4()),
            finish_reason=response.finish_reason,
        )

    async def stream(self, request: AIRequest) -> AsyncIterator[str]:
        """Run a streaming chat request. Yields tokens as they arrive."""
        messages = [{"role": str(m.role), "content": m.content} for m in request.messages]
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
        )

        async with self._semaphore:
            async for token in self._model.stream(model_req):
                yield token
