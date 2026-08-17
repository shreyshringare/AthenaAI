"""P5 gate: memory layer integration tests (requires PostgreSQL + pgvector)."""

from __future__ import annotations

import asyncpg
import pytest

from athenai.memory.base import MemoryType
from athenai.memory.conversation import ConversationMemory
from athenai.memory.semantic import SemanticMemory
from athenai.memory.summary import SummaryMemory
from athenai.memory.working import AgentState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def conv_memory(db_pool: asyncpg.Pool) -> ConversationMemory:
    mem = ConversationMemory(db_pool)
    await mem._ensure_schema()
    return mem


@pytest.fixture
async def sem_memory(db_pool: asyncpg.Pool) -> SemanticMemory:
    mem = SemanticMemory(db_pool)
    await mem._ensure_schema()
    return mem


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_recent_order(conv_memory: ConversationMemory) -> None:
    session_id = "test-session-order"
    for i in range(10):
        await conv_memory.add_message(session_id, "user", f"message {i}")

    recent = await conv_memory.get_recent(session_id, n=5)
    assert len(recent) == 5
    # get_recent returns chronological order — last 5 messages
    contents = [e.content for e in recent]
    assert "user: message 5" in contents
    assert "user: message 9" in contents


@pytest.mark.asyncio
async def test_get_recent_all_messages_in_order(conv_memory: ConversationMemory) -> None:
    session_id = "test-session-all"
    messages = ["hello", "world", "foo", "bar", "baz"]
    for msg in messages:
        await conv_memory.add_message(session_id, "user", msg)

    recent = await conv_memory.get_recent(session_id, n=10)
    assert len(recent) == 5
    # Chronological order
    contents = [e.content.split(": ", 1)[1] for e in recent]
    assert contents == messages


@pytest.mark.asyncio
async def test_clear_session(conv_memory: ConversationMemory) -> None:
    session_id = "test-session-clear"
    await conv_memory.add_message(session_id, "user", "hello")
    await conv_memory.add_message(session_id, "assistant", "hi")

    await conv_memory.clear(session_id)
    recent = await conv_memory.get_recent(session_id)
    assert recent == []


@pytest.mark.asyncio
async def test_memory_entry_type(conv_memory: ConversationMemory) -> None:
    session_id = "test-session-type"
    entry = await conv_memory.add_message(session_id, "user", "test")
    assert entry.memory_type == MemoryType.CONVERSATION
    assert entry.session_id == session_id


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_memory_under_threshold(conv_memory: ConversationMemory) -> None:
    from athenai.models.mock import MockModel
    session_id = "test-summary-under"
    model = MockModel()
    summary_mem = SummaryMemory(conv_memory, model, max_raw_messages=20, keep_recent=5)

    for i in range(5):
        await conv_memory.add_message(session_id, "user", f"msg {i}")

    result = await summary_mem.get_recent(session_id)
    # Under threshold — no summary, just raw messages
    types = [e.memory_type for e in result]
    assert MemoryType.SUMMARY not in types
    assert len(result) == 5


@pytest.mark.asyncio
async def test_summary_memory_over_threshold_generates_summary(
    conv_memory: ConversationMemory,
) -> None:
    from athenai.models.mock import MockModel
    session_id = "test-summary-over"
    model = MockModel()
    summary_mem = SummaryMemory(conv_memory, model, max_raw_messages=10, keep_recent=3)

    for i in range(25):
        await conv_memory.add_message(session_id, "user", f"message number {i}")

    result = await summary_mem.get_recent(session_id)
    types = [e.memory_type for e in result]
    # Should have a SUMMARY entry + recent raw messages
    assert MemoryType.SUMMARY in types
    assert len([e for e in result if e.memory_type == MemoryType.CONVERSATION]) == 3


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


def _make_embedding(seed: float, dim: int = 1536) -> list[float]:
    """Generate deterministic unit embedding for testing."""
    import math
    base = [math.sin(seed + i * 0.01) for i in range(dim)]
    magnitude = math.sqrt(sum(x * x for x in base))
    return [x / magnitude for x in base]


@pytest.mark.asyncio
async def test_semantic_store_and_search(sem_memory: SemanticMemory) -> None:
    session_id = "test-semantic"

    facts = [
        ("Python is a programming language", 1.0),
        ("The sky is blue", 2.0),
        ("Machine learning requires data", 3.0),
        ("Paris is the capital of France", 4.0),
        ("Neural networks learn patterns", 5.0),
    ]

    for content, seed in facts:
        await sem_memory.store(session_id, content, _make_embedding(seed))

    # Search with embedding closest to seed=3.0 (machine learning)
    results = await sem_memory.search(_make_embedding(3.0), k=3)
    assert len(results) == 3

    top_entry, top_score = results[0]
    assert "Machine learning" in top_entry.content
    assert top_score > 0.9  # Should be nearly identical


@pytest.mark.asyncio
async def test_semantic_delete(sem_memory: SemanticMemory) -> None:
    session_id = "test-sem-delete"
    entry = await sem_memory.store(session_id, "fact to delete", _make_embedding(99.0))
    await sem_memory.delete(entry.id)

    results = await sem_memory.search(_make_embedding(99.0), k=5)
    ids = [e.id for e, _ in results]
    assert entry.id not in ids


# ---------------------------------------------------------------------------
# AgentState (working memory — in-process, no DB)
# ---------------------------------------------------------------------------


def test_agent_state_add_steps() -> None:
    state = AgentState(task="compute 2+2", session_id="s1")
    state.add_step("planning")
    state.add_step("tool: calculator")
    assert len(state.steps) == 2
    assert state.steps[0] == "planning"


def test_agent_state_add_tool_results() -> None:
    state = AgentState(task="fetch data", session_id="s1")
    state.add_tool_result("http", {"status": 200, "body": "ok"})
    state.add_tool_result("calculator", 42, error=None)
    assert len(state.tool_results) == 2
    assert state.tool_results[0]["tool"] == "http"


def test_agent_state_snapshot_round_trip() -> None:
    state = AgentState(task="test task", session_id="s2")
    state.add_step("step 1")
    state.add_tool_result("tool_a", "result_a")
    state.status = "EXECUTING_TOOL"
    state.iteration = 3

    snapshot = state.snapshot()
    restored = AgentState.from_snapshot(snapshot)

    assert restored.task == state.task
    assert restored.steps == state.steps
    assert restored.tool_results == state.tool_results
    assert restored.status == "EXECUTING_TOOL"
    assert restored.iteration == 3
