"""Unit tests for gateway API-key auth and CORS lockdown.

WHY A SEPARATE FILE FROM test_gateway.py:
These tests need to vary app.state.gateway_api_key and the CORS origins
list per test, which the shared `client` fixture in test_gateway.py (one
fixed app per test, no auth/CORS configured) doesn't support.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from athenai.agents.agent import Agent
from athenai.gateway.app import create_app
from athenai.models.mock import MockModel
from athenai.runtime.pipeline import AthenaRuntime
from athenai.tools.registry import ToolRegistry


def _make_app(gateway_api_key: str = "", cors_allowed_origins: list[str] | None = None):
    app = create_app(cors_allowed_origins=cors_allowed_origins or [])
    model = MockModel()
    app.state.runtime = AthenaRuntime(model=model)
    app.state.agent = Agent(model=model, tool_registry=ToolRegistry())
    app.state.document_loader = None
    app.state.gateway_api_key = gateway_api_key
    return app


@pytest.mark.asyncio
async def test_protected_route_works_without_key_when_none_configured() -> None:
    """Backward compat / zero-config demo mode: no key set -> no auth enforced."""
    app = _make_app(gateway_api_key="")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_protected_route_rejects_missing_key_when_configured() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_rejects_wrong_key_when_configured() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_accepts_correct_key_when_configured() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "secret-123"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_run_is_protected() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/agents/run", json={"task": "hi"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_document_ingest_is_protected() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/documents/ingest", json={"content": "x", "document_id": "d1"}
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_document_search_is_protected() -> None:
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/documents/search", json={"query": "x"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_health_is_never_protected() -> None:
    """Liveness/readiness/metrics stay open -- probes and scrapers don't carry API keys."""
    app = _make_app(gateway_api_key="secret-123")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        health = await ac.get("/health")
        ready = await ac.get("/ready")
        metrics = await ac.get("/metrics")
    assert health.status_code == 200
    assert ready.status_code == 200
    assert metrics.status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_default_blocks_all_cross_origin() -> None:
    """No origins configured -> secure-by-default, matches the HTTPTool
    allowlist pattern (ATHENA_ALLOWED_DOMAINS) elsewhere in this codebase."""
    app = _make_app(cors_allowed_origins=[])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_cors_allows_configured_origin() -> None:
    app = _make_app(cors_allowed_origins=["https://allowed.example.com"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health", headers={"Origin": "https://allowed.example.com"})
    assert response.headers.get("access-control-allow-origin") == "https://allowed.example.com"


@pytest.mark.asyncio
async def test_cors_rejects_unconfigured_origin() -> None:
    app = _make_app(cors_allowed_origins=["https://allowed.example.com"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers
