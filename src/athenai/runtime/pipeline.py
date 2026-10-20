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

WHY ConversationMemory HERE AND NOT athenai.context.engine.ContextEngine:
ContextEngine (the "designed" integration point for memory, per
athenai.memory's own docstrings) has no caller anywhere in the live request
path — wiring through it would mean also inventing a serialization scheme
between its list[str] content model and the list[dict] message history
used here, plus pulling in token-budget/packing machinery with nothing yet
exercising it. Talking to ConversationMemory directly is smaller and
testable today; revisit once ContextEngine has a second real consumer
(e.g. the RAG query route) that justifies the shared pipeline.

CONTRACT WHEN memory IS SET:
Callers (gateway routes) should send only the *new* turn(s) in
`request.messages`, not replayed history — execute()/stream() prepend the
session's stored history themselves. Sending full history from the client
as well as enabling memory would duplicate turns in the model's context.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from athenai.core.types import AIRequest, AIResponse, TokenUsage
from athenai.models.base import ModelRequest


def _memory_role_and_content(entry: Any) -> tuple[str, str]:
    """Split a ConversationMemory MemoryEntry back into (role, content).

    MemoryEntry.content is stored as "{role}: {content}" (see
    athenai.memory.conversation.ConversationMemory.get_recent) with the role
    repeated in metadata["role"] — reconstructing a clean message dict means
    stripping that prefix back off using the known role, rather than
    re-splitting on ": " (which would mangle content that legitimately
    contains ": ").
    """
    role = entry.metadata.get("role", "user")
    prefix = f"{role}: "
    content = entry.content[len(prefix):] if entry.content.startswith(prefix) else entry.content
    return role, content


class AthenaRuntime:
    """Orchestrates a single (non-agentic) model call with concurrency control.

    One instance is meant to be created per process and reused across
    requests — the `asyncio.Semaphore` bounding `max_concurrent` lives on
    the instance, so constructing a new AthenaRuntime per request would
    defeat the concurrency cap.
    """

    def __init__(self, model: Any, max_concurrent: int = 20, memory: Any = None) -> None:
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._memory = memory

    async def _build_messages(self, request: AIRequest) -> list[dict[str, str]]:
        """New turn(s) from the request, with stored session history (if any
        memory backend is configured) prepended."""
        new_messages = [{"role": str(m.role), "content": m.content} for m in request.messages]
        if self._memory is None:
            return new_messages

        history = await self._memory.get_recent(request.session_id)
        prior_messages = [
            {"role": role, "content": content}
            for entry in history
            for role, content in (_memory_role_and_content(entry),)
        ]
        return prior_messages + new_messages

    async def _persist_turn(self, request: AIRequest, assistant_content: str) -> None:
        """Append this turn's user message(s) and the assistant's reply to
        session memory. No-op if no memory backend is configured."""
        if self._memory is None:
            return
        for m in request.messages:
            await self._memory.add_message(request.session_id, str(m.role), m.content)
        await self._memory.add_message(request.session_id, "assistant", assistant_content)

    async def execute(self, request: AIRequest) -> AIResponse:
        """Run a non-streaming chat request. Blocks until the model responds."""
        messages = await self._build_messages(request)
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
        )

        async with self._semaphore:
            response = await self._model.generate(model_req)

        await self._persist_turn(request, response.content)

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
        """Run a streaming chat request. Yields tokens as they arrive.

        The full assistant reply is only known once streaming finishes, so
        persistence happens after the last token is yielded, not per-token.
        """
        messages = await self._build_messages(request)
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
        )

        chunks: list[str] = []
        async with self._semaphore:
            async for token in self._model.stream(model_req):
                chunks.append(token)
                yield token

        await self._persist_turn(request, "".join(chunks))
