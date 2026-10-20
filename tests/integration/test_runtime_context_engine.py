"""Integration test for AthenaRuntime's ContextEngine path against a real
ConversationMemory (Postgres) — proves the \x00-encode/decode round trip
survives actual DB storage, not just in-process stubs.
"""

from __future__ import annotations

import asyncpg
import pytest

from athenai.context.budget import TokenBudgetManager
from athenai.context.engine import ContextEngine
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
async def test_context_engine_round_trips_real_conversation_memory(
    conv_memory: ConversationMemory,
) -> None:
    session_id = "ctx-engine-session"
    engine = ContextEngine(TokenBudgetManager({"total": 5000}))

    captured = []

    class CapturingModel:
        name = "capture"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            captured.append(request)
            return ModelResponse(
                content="nice to meet you, Shreyas", model_name=self.name,
                input_tokens=1, output_tokens=1,
            )

        async def health_check(self):
            return True

    runtime = AthenaRuntime(model=CapturingModel(), memory=conv_memory, context_engine=engine)

    await runtime.execute(_request(session_id, "my name is Shreyas"))
    await runtime.execute(_request(session_id, "what is my name?"))

    second_request = captured[1]
    roles_and_content = [(m["role"], m["content"]) for m in second_request.messages]
    assert ("user", "my name is Shreyas") in roles_and_content
    assert ("assistant", "nice to meet you, Shreyas") in roles_and_content
    assert ("user", "what is my name?") in roles_and_content
    # Chronological: first turn's history comes before the current turn.
    contents = [c for _, c in roles_and_content]
    assert contents.index("my name is Shreyas") < contents.index("what is my name?")
