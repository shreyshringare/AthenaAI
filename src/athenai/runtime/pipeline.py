"""
AthenaRuntime — the top-level orchestration layer between the HTTP gateway
and the model adapters.

WHY SEMAPHORE:
Unbounded concurrent model calls exhaust API rate limits and memory.
BoundedExecutor caps in-flight model calls with an asyncio.Semaphore so
back-pressure propagates to the HTTP layer as latency, not crashes.

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
    """Orchestrates a single model call with concurrency control."""

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
