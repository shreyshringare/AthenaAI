"""Demo mode: controllable mock backend, scripted tool turns, /debug routes."""

from __future__ import annotations

import httpx
import pytest

from athenai.core.config import AthenaConfig
from athenai.core.exceptions import ModelUnavailableError
from athenai.gateway.app import _build_demo_model, create_app
from athenai.models.base import ModelRequest
from athenai.models.demo import DemoMockModel


def _req(*messages: dict[str, str]) -> ModelRequest:
    return ModelRequest(messages=list(messages), model_name="x")


async def test_failing_backend_raises_unavailable() -> None:
    m = DemoMockModel("m")
    m.set_failing(True)
    with pytest.raises(ModelUnavailableError):
        await m.generate(_req({"role": "user", "content": "hi"}))
    assert await m.health_check() is False


async def test_plain_prompt_echoes() -> None:
    m = DemoMockModel("m")
    r = await m.generate(_req({"role": "user", "content": "hi"}))
    assert r.content == "[m] Echo: hi"


async def test_calc_task_emits_one_tool_call_per_expression() -> None:
    m = DemoMockModel("m")
    r = await m.generate(_req({"role": "user", "content": "calc: 1+1; 2*3"}))
    lines = r.content.splitlines()
    assert len(lines) == 2
    assert all(line.startswith("TOOL_CALL:") for line in lines)


async def test_calc_task_summarises_once_tool_results_present() -> None:
    m = DemoMockModel("m")
    r = await m.generate(
        _req(
            {"role": "user", "content": "calc: 1+1"},
            {"role": "tool", "content": "2", "name": "calculator"},
        )
    )
    assert r.content == "Results: 2"


async def test_agent_runs_parallel_calculator_calls_end_to_end() -> None:
    from athenai.agents.agent import Agent
    from athenai.tools.calculator import CalculatorTool
    from athenai.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = Agent(model=DemoMockModel("m"), tool_registry=registry)
    result = await agent.run(task="calc: 12*7; 2**10")
    assert result.status.value == "completed"
    assert len(result.steps[0].tool_calls) == 2
    assert "84" in result.final_answer and "1024" in result.final_answer


def _demo_app():  # type: ignore[no-untyped-def]
    app = create_app()
    _build_demo_model(AthenaConfig(), app)
    return app


def _client(app):  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_debug_routes_404_when_demo_mode_off() -> None:
    async with _client(create_app()) as c:
        assert (await c.get("/debug/status")).status_code == 404


async def test_debug_fail_and_reset_roundtrip() -> None:
    async with _client(_demo_app()) as c:
        s = (await c.post("/debug/fail", json={"role": "fast-mock", "failing": True})).json()
        assert s["fast-mock"]["failing"] is True
        s = (await c.post("/debug/reset")).json()
        assert s["fast-mock"] == {"failing": False, "circuit": "CLOSED"}
        assert (await c.post("/debug/fail", json={"role": "nope"})).status_code == 404


async def test_failover_trips_circuit_and_uses_other_backend() -> None:
    from athenai.models.base import ModelRequest

    app = _demo_app()
    routed = app.state.routed_model
    app.state.demo_models["fast-mock"].set_failing(True)
    resp = await routed.generate(
        ModelRequest(messages=[{"role": "user", "content": "short"}], model_name="default")
    )
    assert resp.model_name == "smart-mock"
    assert routed.circuit_states()["fast-mock"] == "OPEN"
    routed.reset_circuits()
    assert routed.circuit_states()["fast-mock"] == "CLOSED"
