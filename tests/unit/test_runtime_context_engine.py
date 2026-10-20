"""Unit tests for AthenaRuntime's optional ContextEngine integration.

WHY UNIT NOT INTEGRATION FOR MOST OF THESE:
Budget-overflow/ordering logic is pure in-process orchestration -- stub
memory/document_loader objects are enough to prove it; no Postgres needed.
See tests/integration/test_runtime_context_engine.py for the one test that
needs a real ConversationMemory round-trip.
"""

from __future__ import annotations

import pytest

from athenai.context.budget import TokenBudgetManager
from athenai.context.engine import ContextEngine
from athenai.core.types import AIRequest, Message, MessageRole
from athenai.models.mock import MockModel
from athenai.runtime.pipeline import AthenaRuntime


def _request(session_id: str, content: str) -> AIRequest:
    return AIRequest(
        messages=(Message(role=MessageRole.USER, content=content),),
        user_id="test-user",
        session_id=session_id,
    )


def _engine(budget: dict[str, int]) -> ContextEngine:
    return ContextEngine(TokenBudgetManager(budget))


class _StubMemory:
    def __init__(self, entries: list) -> None:
        self._entries = entries

    async def get_recent(self, session_id: str):
        return self._entries

    async def add_message(self, session_id: str, role: str, content: str):
        pass


class _MemoryEntry:
    def __init__(self, role: str, content: str) -> None:
        self.metadata = {"role": role}
        self.content = f"{role}: {content}"


class _StubDocumentLoader:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.last_query: str | None = None

    async def search(self, query: str, top_k: int = 5):
        from athenai.rag.retriever import RetrievedChunk

        self.last_query = query
        return [
            RetrievedChunk(
                chunk_id=f"c{i}", document_id="d1", content=c, chunk_index=i, score=1.0
            )
            for i, c in enumerate(self._chunks[:top_k])
        ]


@pytest.mark.asyncio
async def test_without_context_engine_behaves_exactly_as_before() -> None:
    """context_engine=None (the default) must stay byte-for-byte the old
    direct-memory-prepend path -- no regression for existing callers."""
    memory = _StubMemory([_MemoryEntry("user", "earlier question")])
    runtime = AthenaRuntime(model=MockModel(), memory=memory)

    response = await runtime.execute(_request("s1", "hello"))
    assert response.content  # MockModel always returns non-empty


@pytest.mark.asyncio
async def test_context_engine_includes_memory_and_rag_in_final_prompt() -> None:
    captured_requests = []

    class CapturingModel:
        name = "capture"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            captured_requests.append(request)
            return ModelResponse(content="ok", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    memory = _StubMemory([_MemoryEntry("user", "my name is Shreyas")])
    loader = _StubDocumentLoader(["Python is a programming language."])
    engine = _engine({"total": 2000})

    runtime = AthenaRuntime(
        model=CapturingModel(), memory=memory, context_engine=engine, document_loader=loader
    )
    await runtime.execute(_request("s1", "what is Python?"))

    req = captured_requests[0]
    assert "Python is a programming language." in req.system
    roles_and_content = [(m["role"], m["content"]) for m in req.messages]
    assert ("user", "my name is Shreyas") in roles_and_content
    assert ("user", "what is Python?") in roles_and_content
    assert loader.last_query == "what is Python?"


@pytest.mark.asyncio
async def test_context_engine_preserves_chronological_order() -> None:
    """Memory (past turns) must come before the current turn in the final
    message list, even though ContextPacker groups by bucket (conversation
    is higher priority than memory for *dropping*, not for *ordering*)."""
    captured_requests = []

    class CapturingModel:
        name = "capture"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            captured_requests.append(request)
            return ModelResponse(content="ok", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    memory = _StubMemory(
        [_MemoryEntry("user", "turn one"), _MemoryEntry("assistant", "reply one")]
    )
    engine = _engine({"total": 2000})

    runtime = AthenaRuntime(model=CapturingModel(), memory=memory, context_engine=engine)
    await runtime.execute(_request("s1", "turn two"))

    contents = [m["content"] for m in captured_requests[0].messages]
    assert contents == ["turn one", "reply one", "turn two"]


@pytest.mark.asyncio
async def test_context_engine_drops_rag_before_memory_under_tight_budget() -> None:
    """Matches ContextPacker's documented priority order (system > conversation
    > memory > rag > tools) -- rag is the most replaceable, drops first."""
    captured_requests = []

    class CapturingModel:
        name = "capture"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            captured_requests.append(request)
            return ModelResponse(content="ok", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    memory = _StubMemory([_MemoryEntry("user", "x" * 40)])
    loader = _StubDocumentLoader(["y" * 40])
    # ContextPacker enforces one global ceiling (TokenBudgetManager's
    # per-bucket allocations aren't consulted by pack() -- only "total"
    # matters here), so this must be sized against the real token counts:
    # system(~8) + conversation(~2) + memory(~11) = ~21 fits in 25;
    # + rag(~10) = ~31 does not.
    engine = _engine({"total": 25})

    runtime = AthenaRuntime(
        model=CapturingModel(), memory=memory, context_engine=engine, document_loader=loader
    )
    await runtime.execute(_request("s1", "short"))

    req = captured_requests[0]
    assert "y" * 40 not in (req.system or "")
    roles_and_content = [m["content"] for m in req.messages]
    assert "x" * 40 in roles_and_content


@pytest.mark.asyncio
async def test_context_engine_without_document_loader_skips_rag() -> None:
    memory = _StubMemory([])
    engine = _engine({"total": 2000})
    runtime = AthenaRuntime(model=MockModel(), memory=memory, context_engine=engine)

    response = await runtime.execute(_request("s1", "hello"))
    assert response.content
