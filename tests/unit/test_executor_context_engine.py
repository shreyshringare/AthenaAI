"""Unit tests for AgentExecutor's optional ContextEngine integration.

Unlike AthenaRuntime, there's no cross-call session memory for agent runs
today (no session_id anywhere in Agent/AgentExecutor/the /v1/agents/run
route) -- memory_fn is always [] here. What ContextEngine buys the agent
loop instead: RAG augmentation of the task, and token-budgeting of the
*in-loop* transcript that grows every iteration a tool gets called.
"""

from __future__ import annotations

import pytest

from athenai.context.budget import TokenBudgetManager
from athenai.context.engine import ContextEngine
from athenai.agents.executor import AgentExecutor
from athenai.models.base import ModelResponse
from athenai.tools.calculator import CalculatorTool
from athenai.tools.registry import ToolRegistry


def _engine(budget: dict[str, int]) -> ContextEngine:
    return ContextEngine(TokenBudgetManager(budget))


class _StubDocumentLoader:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.last_query: str | None = None

    async def search(self, query: str, top_k: int = 5):
        from athenai.rag.retriever import RetrievedChunk

        self.last_query = query
        return [
            RetrievedChunk(chunk_id=f"c{i}", document_id="d1", content=c, chunk_index=i, score=1.0)
            for i, c in enumerate(self._chunks[:top_k])
        ]


@pytest.mark.asyncio
async def test_without_context_engine_behaves_exactly_as_before():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = AgentExecutor(model=_mock_once("hello"), tool_registry=registry)
    result = await executor.run("hi")
    assert result.final_answer


def _mock_once(reply: str):
    class _M:
        name = "m"

        async def generate(self, request):
            return ModelResponse(content=reply, model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    return _M()


@pytest.mark.asyncio
async def test_context_engine_includes_rag_in_system_and_queries_task():
    captured = []

    class CapturingModel:
        name = "capture"
        _calls = 0

        async def generate(self, request):
            captured.append(request)
            self.__class__._calls += 1
            return ModelResponse(content="done", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    loader = _StubDocumentLoader(["Python is great for AI."])
    engine = _engine({"total": 5000})
    registry = ToolRegistry()

    executor = AgentExecutor(
        model=CapturingModel(), tool_registry=registry, context_engine=engine, document_loader=loader
    )
    await executor.run("tell me about Python")

    req = captured[0]
    assert "Python is great for AI." in req.system
    assert loader.last_query == "tell me about Python"


@pytest.mark.asyncio
async def test_context_engine_preserves_tool_name_through_round_trip():
    """Tool-result messages carry a "name" field the model shape relies on
    -- must survive the encode/decode round trip, not just role+content."""
    captured = []

    class ToolCallingModel:
        name = "tc"
        _calls = 0

        async def generate(self, request):
            captured.append(request)
            self.__class__._calls += 1
            if self.__class__._calls == 1:
                return ModelResponse(
                    content='TOOL_CALL: {"name": "calculator", "arguments": {"expression": "2+2"}}',
                    model_name=self.name, input_tokens=1, output_tokens=1,
                )
            return ModelResponse(content="The answer is 4.", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    engine = _engine({"total": 5000})

    executor = AgentExecutor(model=ToolCallingModel(), tool_registry=registry, context_engine=engine)
    result = await executor.run("what is 2+2?")

    assert result.status.value == "completed"
    second_request_messages = captured[1].messages
    tool_messages = [m for m in second_request_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "calculator"
    assert tool_messages[0]["content"] == "4"


@pytest.mark.asyncio
async def test_context_engine_drops_oldest_transcript_first_not_newest():
    """ContextPacker is best-effort greedy, not a strict cutoff -- it skips
    (doesn't stop at) items that don't fit and keeps scanning, so a small
    old item can survive alongside a big new one being dropped. The
    property that actually matters (and that the reversal fix provides):
    processing newest-first means a newest item competes for budget
    *before* the oldest one, so when both can't fit, the newest wins and
    the oldest (here: the original task itself) is what gets dropped --
    not the reverse. Verified by picking a budget where this is only true
    with the fix (hand-computed against the real token counts below)."""
    captured = []

    class ThreeToolCallsModel:
        name = "three-calls"
        _calls = 0
        _expressions = ["1+1", "100+100", "100000+100000"]

        async def generate(self, request):
            captured.append(request)
            call_index = self.__class__._calls
            self.__class__._calls += 1
            if call_index < len(self._expressions):
                expr = self._expressions[call_index]
                return ModelResponse(
                    content=f'TOOL_CALL: {{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}',
                    model_name=self.name, input_tokens=1, output_tokens=1,
                )
            return ModelResponse(content="done", model_name=self.name, input_tokens=1, output_tokens=1)

        async def health_check(self):
            return True

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    # Hand-computed against athenai.context.encoding.encode_role_content's
    # real output for this exact scenario: with newest-first processing,
    # the three tool-result items (~4-5 tokens each) fit but the original
    # task (~6 tokens) does not. With the old oldest-first bug, the task
    # would survive and the newest tool result (200000) would not.
    engine = _engine({"total": 14})

    executor = AgentExecutor(model=ThreeToolCallsModel(), tool_registry=registry, context_engine=engine)
    await executor.run("original task marker")

    fourth_request_messages = captured[3].messages
    contents = [m["content"] for m in fourth_request_messages]
    assert "200000" in contents  # newest tool result -- must survive
    assert "original task marker" not in contents  # oldest -- drops first


@pytest.mark.asyncio
async def test_context_engine_without_document_loader_skips_rag():
    registry = ToolRegistry()
    executor = AgentExecutor(
        model=_mock_once("hi"), tool_registry=registry, context_engine=_engine({"total": 5000})
    )
    result = await executor.run("hello")
    assert result.final_answer
