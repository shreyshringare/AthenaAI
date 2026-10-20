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

CONTEXT ENGINE INTEGRATION (optional, via context_engine/document_loader):
When context_engine is set, execute()/stream() route conversation history,
RAG retrieval, and the current turn through athenai.context.engine
.ContextEngine instead of the plain prepend-and-send logic below —
token-budgeted, with RelevanceRanker-adjacent RAG content and the lowest-
priority buckets dropped first under pressure (system > conversation >
memory > rag, per ContextPacker). context_engine=None (the default) keeps
today's behavior unchanged, so this is purely additive.

WHY A CUSTOM \x00 ENCODING, NOT ConversationMemory's OWN "{role}: {content}"
FORMAT:
ContextEngine.build()'s memory_fn/conversation params are list[str] —
plain strings, no side-channel for role. ConversationMemory.get_recent()
already encodes role as a "{role}: " text prefix, which is fine when you
still have the MemoryEntry object (role is also in .metadata, the real
ground truth) but becomes ambiguous once ContextPacker has flattened
everything down to bare ContextItem.content strings: there's no way to
tell "a message that happens to start with the text 'user: '" apart from
"a message whose role is user" using text-prefix matching alone. Encoding
role\x00content ourselves for every string handed to ContextEngine (NUL
can't appear in real text) makes decoding after packing unambiguous,
rather than guessing from a small set of known role names.

WHY MEMORY (PAST TURNS) MUST BE REORDERED BEFORE CONVERSATION (CURRENT
TURN) AT RECONSTRUCTION TIME:
ContextPacker groups items by bucket in PRIORITY_ORDER (system >
conversation > memory > rag > tools) — that order is about what to drop
first under budget pressure, not chronological position. The current
turn (conversation bucket) is higher priority than history (memory
bucket), so packed.items would place it *before* older history if used
as-is — backwards for a chat transcript. _build_messages_with_context_engine
reassembles by bucket into the correct chronological order (memory then
conversation) rather than trusting packed.items' overall sequence.

CONTRACT WHEN memory IS SET (either path):
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
from athenai.observability.logger import get_logger

_logger = get_logger(__name__)
_DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
_DEFAULT_RAG_TOP_K = 3


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


def _encode_role_content(role: str, content: str) -> str:
    """See module docstring's "WHY A CUSTOM \\x00 ENCODING" note."""
    return f"{role}\x00{content}"


def _decode_role_content(encoded: str) -> dict[str, str]:
    role, _, content = encoded.partition("\x00")
    return {"role": role, "content": content}


class AthenaRuntime:
    """Orchestrates a single (non-agentic) model call with concurrency control.

    One instance is meant to be created per process and reused across
    requests — the `asyncio.Semaphore` bounding `max_concurrent` lives on
    the instance, so constructing a new AthenaRuntime per request would
    defeat the concurrency cap.
    """

    def __init__(
        self,
        model: Any,
        max_concurrent: int = 20,
        memory: Any = None,
        context_engine: Any = None,
        document_loader: Any = None,
        rag_top_k: int = _DEFAULT_RAG_TOP_K,
    ) -> None:
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._memory = memory
        self._context_engine = context_engine
        self._document_loader = document_loader
        self._rag_top_k = rag_top_k

    async def _build_messages(self, request: AIRequest) -> list[dict[str, str]]:
        """New turn(s) from the request, with stored session history (if any
        memory backend is configured) prepended. Used when context_engine is
        not set — see _build_messages_with_context_engine for the other path.
        """
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

    async def _build_messages_with_context_engine(
        self, request: AIRequest
    ) -> tuple[list[dict[str, str]], str]:
        """Route history + RAG + the current turn through ContextEngine for
        token-budgeted, priority-ordered assembly. Returns (messages, system)
        rather than just messages, since the packed system+rag content needs
        to go on ModelRequest.system, not into the message list.
        """

        async def memory_fn() -> list[str]:
            if self._memory is None:
                return []
            history = await self._memory.get_recent(request.session_id)
            return [_encode_role_content(*_memory_role_and_content(e)) for e in history]

        async def rag_fn() -> list[str]:
            if self._document_loader is None or not request.messages:
                return []
            query = request.messages[-1].content
            try:
                results = await self._document_loader.search(query, top_k=self._rag_top_k)
            except Exception as exc:
                _logger.warning("runtime.rag_fn_failed", reason=str(exc))
                return []
            return [r.content for r in results]

        conversation = [_encode_role_content(str(m.role), m.content) for m in request.messages]

        built = await self._context_engine.build(
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            conversation=conversation,
            memory_fn=memory_fn,
            rag_fn=rag_fn,
        )

        # Reassemble in chronological order (memory/history, then the
        # current turn) — packed.items is grouped by bucket priority, not
        # presentation order. See module docstring.
        memory_messages = [
            _decode_role_content(item.content)
            for item in built.packed.items
            if item.bucket == "memory"
        ]
        conversation_messages = [
            _decode_role_content(item.content)
            for item in built.packed.items
            if item.bucket == "conversation"
        ]
        messages = memory_messages + conversation_messages

        system_parts = [item.content for item in built.packed.items if item.bucket == "system"]
        rag_parts = [item.content for item in built.packed.items if item.bucket == "rag"]
        if rag_parts:
            system_parts.append("Relevant context:\n" + "\n\n".join(rag_parts))
        system = "\n\n".join(system_parts)

        return messages, system

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
        system = ""
        if self._context_engine is not None:
            messages, system = await self._build_messages_with_context_engine(request)
        else:
            messages = await self._build_messages(request)
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
            system=system,
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
        system = ""
        if self._context_engine is not None:
            messages, system = await self._build_messages_with_context_engine(request)
        else:
            messages = await self._build_messages(request)
        model_req = ModelRequest(
            messages=messages,
            model_name=request.model_role or "default",
            system=system,
        )

        chunks: list[str] = []
        async with self._semaphore:
            async for token in self._model.stream(model_req):
                chunks.append(token)
                yield token

        await self._persist_turn(request, "".join(chunks))
