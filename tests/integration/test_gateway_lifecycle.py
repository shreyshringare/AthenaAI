"""Integration tests for gateway/app.py's real lifespan() — AthenaConfig +
core.lifecycle.lifespan() wiring, and the ConversationMemory it builds.

WHY NOT httpx.ASGITransport (see tests/unit/test_gateway.py's docstring):
ASGITransport never sends the lifespan scope, so FastAPI's lifespan context
manager doesn't run under it — that's why the unit gateway tests set
app.state manually instead. Here we want the *opposite*: to actually
exercise lifespan()'s startup/shutdown wiring against a real database, so
we call athenai.gateway.app.lifespan(app) directly as the async context
manager it is, bypassing HTTP entirely.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from athenai.core.types import AIRequest, Message, MessageRole
from athenai.gateway.app import lifespan
from tests.integration.conftest import TEST_DB_DSN


@pytest.fixture(autouse=True)
def _configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the app's AthenaConfig at the integration test database, and
    make sure no real Anthropic key / RAG config leaks in from the host
    environment — these tests assert the mock-model, memory-enabled shape."""
    monkeypatch.setenv("ATHENA_DB_URL", TEST_DB_DSN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ATHENA_EMBEDDER_URL", raising=False)
    monkeypatch.delenv("ATHENA_ALLOWED_DOMAINS", raising=False)


@pytest.mark.asyncio
async def test_lifespan_wires_conversation_memory_into_runtime() -> None:
    app = FastAPI()

    async with lifespan(app):
        assert app.state.runtime is not None
        assert app.state.runtime._memory is not None

        response = await app.state.runtime.execute(_make_request())
        assert response.content


@pytest.mark.asyncio
async def test_lifespan_closes_memory_pool_on_shutdown() -> None:
    app = FastAPI()
    memory_ref = None

    async with lifespan(app):
        memory_ref = app.state.runtime._memory
        assert memory_ref is not None

    # Pool is closed after the context exits -- a subsequent query must fail.
    with pytest.raises(Exception):  # noqa: B017 - asyncpg raises on a closed pool
        await memory_ref.get_recent("any-session")


def _make_request() -> AIRequest:
    return AIRequest(
        messages=(Message(role=MessageRole.USER, content="hello"),),
        user_id="test-user",
        session_id="lifecycle-test-session",
    )
