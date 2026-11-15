"""Per-client inbound rate limiting on /v1/* routes."""

from __future__ import annotations

import httpx

from athenai.agents.agent import Agent
from athenai.gateway.app import create_app
from athenai.gateway.ratelimit import ClientRateLimiter
from athenai.models.mock import MockModel
from athenai.runtime.pipeline import AthenaRuntime
from athenai.tools.registry import ToolRegistry

_BODY = {"messages": [{"role": "user", "content": "hi"}], "session_id": "s"}


def _app(capacity: int):  # type: ignore[no-untyped-def]
    app = create_app()
    model = MockModel()
    app.state.runtime = AthenaRuntime(model=model)
    app.state.agent = Agent(model=model, tool_registry=ToolRegistry())
    app.state.gateway_api_key = ""
    app.state.rate_limiter = ClientRateLimiter(capacity=capacity, refill_per_s=0.0001)
    return app


def _client(app):  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_requests_over_capacity_get_429_with_retry_after() -> None:
    async with _client(_app(2)) as c:
        assert (await c.post("/v1/chat", json=_BODY)).status_code == 200
        assert (await c.post("/v1/chat", json=_BODY)).status_code == 200
        r = await c.post("/v1/chat", json=_BODY)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


async def test_buckets_are_per_client() -> None:
    async with _client(_app(1)) as c:
        a = {"x-api-key": "a"}
        assert (await c.post("/v1/chat", json=_BODY, headers=a)).status_code == 200
        assert (await c.post("/v1/chat", json=_BODY, headers=a)).status_code == 429
        assert (await c.post("/v1/chat", json=_BODY, headers={"x-api-key": "b"})).status_code == 200


async def test_probes_are_never_limited() -> None:
    async with _client(_app(1)) as c:
        for _ in range(5):
            assert (await c.get("/health")).status_code == 200


async def test_capacity_zero_disables_limit() -> None:
    async with _client(_app(0)) as c:
        for _ in range(5):
            assert (await c.post("/v1/chat", json=_BODY)).status_code == 200


async def test_client_limiter_evicts_oldest_past_max() -> None:
    lim = ClientRateLimiter(capacity=1, refill_per_s=1.0, max_clients=2)
    for key in ("a", "b", "c"):
        await lim.acquire(key)
    assert list(lim._buckets) == ["b", "c"]
