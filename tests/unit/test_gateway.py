"""P9 gate tests — FastAPI gateway routes.

WHY MANUAL STATE SETUP:
httpx.ASGITransport sends HTTP scopes to the ASGI app but never sends the
lifespan scope. FastAPI's lifespan context manager therefore does not run
during tests. We initialise app.state manually in the fixture so routes can
access runtime, agent, and document_loader without needing the full startup
sequence or any environment variables.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from athenai.agents.agent import Agent
from athenai.gateway.app import create_app
from athenai.models.mock import MockModel
from athenai.runtime.pipeline import AthenaRuntime
from athenai.tools.registry import ToolRegistry


@pytest.fixture
async def client():
    app = create_app()

    # Manually wire app.state — lifespan does not fire via ASGITransport
    model = MockModel()
    registry = ToolRegistry()
    app.state.runtime = AthenaRuntime(model=model)
    app.state.agent = Agent(model=model, tool_registry=registry)
    app.state.document_loader = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    response = await client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_healthy"] is True


@pytest.mark.asyncio
async def test_chat_returns_response(client):
    response = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"]
    assert data["model"]
    assert data["trace_id"]
    assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_chat_empty_messages_rejected(client):
    # Pydantic's min_length=1 on messages field returns 422 before the route runs
    response = await client.post("/v1/chat", json={"messages": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_run_returns_result(client):
    response = await client.post(
        "/v1/agents/run",
        json={"task": "Say hello", "tools": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["final_answer"]
    assert isinstance(data["steps"], list)
    assert data["total_iterations"] >= 1
    assert data["status"] in ("completed", "failed")


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text(client):
    # Trigger a request so at least one metric exists
    await client.get("/health")
    response = await client.get("/metrics")
    assert response.status_code == 200
    # prometheus_client always emits process metrics even with no custom ones
    assert "# HELP" in response.text


@pytest.mark.asyncio
async def test_chat_increments_request_counter(client):
    from prometheus_client import REGISTRY

    def _count_chat_successes() -> float:
        # Sample names include _total suffix; search sample.name not metric.name
        total = 0.0
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if (
                    "athena_requests" in sample.name
                    and sample.labels.get("endpoint") == "/v1/chat"
                    and sample.labels.get("status") == "success"
                ):
                    total += sample.value
        return total

    before = _count_chat_successes()

    await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "count me"}]},
    )

    after = _count_chat_successes()
    assert after == before + 1


@pytest.mark.asyncio
async def test_document_ingest_returns_503_when_loader_not_configured(client):
    response = await client.post(
        "/v1/documents/ingest",
        json={"content": "hello", "document_id": "doc1"},
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_response_has_required_fields(client):
    response = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "Test"}]},
    )
    assert response.status_code == 200
    data = response.json()
    for field in ("content", "model", "trace_id", "usage", "request_id"):
        assert field in data, f"missing field: {field}"
