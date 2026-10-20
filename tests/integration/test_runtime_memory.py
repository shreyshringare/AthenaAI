"""Integration tests for AthenaRuntime's ConversationMemory wiring (Option B).

WHY INTEGRATION NOT UNIT:
Exercises a real ConversationMemory backed by PostgreSQL (see conftest.py's
db_pool fixture) rather than a mock, since the behavior under test — history
actually round-tripping through INSERT/SELECT and the stored "{role}: "
prefix being stripped back off correctly — can't be verified against a fake.
"""

from __future__ import annotations

import asyncpg
import pytest

from athenai.core.types import AIRequest, Message, MessageRole
from athenai.memory.conversation import ConversationMemory
from athenai.models.mock import MockModel
from athenai.runtime.pipeline import AthenaRuntime


@pytest.fixture
async def conv_memory(db_pool: asyncpg.Pool) -> ConversationMemory:
    mem = ConversationMemory(db_pool)
    await mem._ensure_schema()
    return mem


def _request(session_id: str, content: str) -> AIRequest:
    return AIRequest(
        messages=(Message(role=MessageRole.USER, content=content),),
        user_id="test-user",
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_execute_persists_user_and_assistant_turns(conv_memory: ConversationMemory) -> None:
    session_id = "runtime-session-persist"
    runtime = AthenaRuntime(model=MockModel(), memory=conv_memory)

    await runtime.execute(_request(session_id, "hello there"))

    stored = await conv_memory.get_recent(session_id)
    assert len(stored) == 2
    assert stored[0].metadata["role"] == "user"
    assert stored[0].content == "user: hello there"
    assert stored[1].metadata["role"] == "assistant"


@pytest.mark.asyncio
async def test_execute_second_turn_sees_first_turn_as_history(
    conv_memory: ConversationMemory,
) -> None:
    session_id = "runtime-session-history"
    runtime = AthenaRuntime(model=MockModel(), memory=conv_memory)

    await runtime.execute(_request(session_id, "my name is Shreyas"))
    await runtime.execute(_request(session_id, "what did I just tell you?"))

    stored = await conv_memory.get_recent(session_id)
    assert len(stored) == 4
    roles = [e.metadata["role"] for e in stored]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "my name is Shreyas" in stored[0].content


@pytest.mark.asyncio
async def test_execute_without_memory_does_not_persist(db_pool: asyncpg.Pool) -> None:
    """memory=None (the default) must stay fully stateless — no table writes
    at all, not even a skipped-but-attempted one."""
    session_id = "runtime-session-no-memory"
    runtime = AthenaRuntime(model=MockModel())  # no memory passed

    response = await runtime.execute(_request(session_id, "hello"))

    assert response.content
    mem = ConversationMemory(db_pool)
    await mem._ensure_schema()
    stored = await mem.get_recent(session_id)
    assert stored == []


@pytest.mark.asyncio
async def test_stream_persists_full_assembled_content(conv_memory: ConversationMemory) -> None:
    session_id = "runtime-session-stream"
    runtime = AthenaRuntime(model=MockModel(), memory=conv_memory)

    chunks = [tok async for tok in runtime.stream(_request(session_id, "stream this"))]
    assert chunks  # MockModel.stream yields at least one token

    stored = await conv_memory.get_recent(session_id)
    assert len(stored) == 2
    assistant_entry = stored[1]
    assert assistant_entry.metadata["role"] == "assistant"
    # The persisted assistant content is the join of every yielded chunk.
    persisted_content = assistant_entry.content[len("assistant: "):]
    assert persisted_content == "".join(chunks)
