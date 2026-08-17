"""P8 gate tests — agent state machine and executor."""

from __future__ import annotations

from typing import ClassVar

import pytest

from athenai.agents.executor import AgentExecutor
from athenai.agents.state import AgentStatus, AgentStep, transition
from athenai.models.mock import MockModel
from athenai.tools.calculator import CalculatorTool
from athenai.tools.registry import ToolRegistry

# ── State machine ─────────────────────────────────────────────────────────────


def test_valid_transition_created_to_running():
    result = transition(AgentStatus.CREATED, AgentStatus.RUNNING)
    assert result == AgentStatus.RUNNING


def test_valid_transition_running_to_awaiting_tool():
    result = transition(AgentStatus.RUNNING, AgentStatus.AWAITING_TOOL)
    assert result == AgentStatus.AWAITING_TOOL


def test_valid_transition_running_to_completed():
    result = transition(AgentStatus.RUNNING, AgentStatus.COMPLETED)
    assert result == AgentStatus.COMPLETED


def test_illegal_transition_created_to_completed_raises():
    with pytest.raises(ValueError, match="illegal transition"):
        transition(AgentStatus.CREATED, AgentStatus.COMPLETED)


def test_illegal_transition_completed_to_running_raises():
    with pytest.raises(ValueError, match="illegal transition"):
        transition(AgentStatus.COMPLETED, AgentStatus.RUNNING)


def test_terminal_states_have_no_outgoing_transitions():
    for terminal in (AgentStatus.COMPLETED, AgentStatus.FAILED):
        for target in AgentStatus:
            with pytest.raises(ValueError):
                transition(terminal, target)


# ── AgentStep ─────────────────────────────────────────────────────────────────


def test_agent_step_defaults():
    step = AgentStep(iteration=0, model_response="hello")
    assert step.tool_calls == []
    assert step.tool_results == []


# ── AgentExecutor (no tool calls) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executor_completes_without_tools():
    executor = AgentExecutor(
        model=MockModel(),
        tool_registry=ToolRegistry(),
    )
    result = await executor.run("Say hello")

    assert result.status == AgentStatus.COMPLETED
    assert result.total_iterations == 1
    assert len(result.steps) == 1
    assert result.final_answer  # non-empty


@pytest.mark.asyncio
async def test_executor_tool_call_via_text_protocol():
    """Executor parses TOOL_CALL: directives when model embeds them in response."""

    class ToolCallingModel:
        name = "fake-tool-caller"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            # First call: emit a tool call directive
            if len(request.messages) == 1:
                return ModelResponse(
                    content='TOOL_CALL: {"name": "calculator", "arguments": {"expression": "6*7"}}',
                    model_name=self.name,
                    input_tokens=10,
                    output_tokens=10,
                )
            # Second call: final answer after tool result is in history
            return ModelResponse(
                content="The answer is 42.",
                model_name=self.name,
                input_tokens=10,
                output_tokens=5,
            )

        async def health_check(self):
            return True

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    executor = AgentExecutor(model=ToolCallingModel(), tool_registry=registry)
    result = await executor.run("What is 6 * 7?")

    assert result.status == AgentStatus.COMPLETED
    assert result.total_iterations == 2
    assert len(result.steps) == 2
    tool_step = result.steps[0]
    assert len(tool_step.tool_calls) == 1
    assert tool_step.tool_calls[0]["name"] == "calculator"
    assert len(tool_step.tool_results) == 1


@pytest.mark.asyncio
async def test_executor_stops_at_max_iterations():
    class LoopingModel:
        name = "looper"

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            return ModelResponse(
                content='TOOL_CALL: {"name": "calculator", "arguments": {"expression": "1+1"}}',
                model_name=self.name,
                input_tokens=5,
                output_tokens=5,
            )

        async def health_check(self):
            return True

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    executor = AgentExecutor(model=LoopingModel(), tool_registry=registry, max_iterations=3)
    result = await executor.run("Loop forever")

    assert result.status == AgentStatus.FAILED
    assert result.total_iterations == 3


@pytest.mark.asyncio
async def test_executor_handles_tool_error_gracefully():
    class ErrorTool:
        name = "error_tool"
        description = "Always fails"
        input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

        async def execute(self, arguments):
            raise RuntimeError("tool exploded")

    class OneCallModel:
        name = "once"
        _calls = 0

        async def generate(self, request):
            from athenai.models.base import ModelResponse

            self.__class__._calls += 1
            if self.__class__._calls == 1:
                return ModelResponse(
                    content='TOOL_CALL: {"name": "error_tool", "arguments": {}}',
                    model_name=self.name,
                    input_tokens=5,
                    output_tokens=5,
                )
            return ModelResponse(
                content="Recovered.",
                model_name=self.name,
                input_tokens=5,
                output_tokens=5,
            )

        async def health_check(self):
            return True

    registry = ToolRegistry()
    registry.register(ErrorTool())

    executor = AgentExecutor(model=OneCallModel(), tool_registry=registry)
    result = await executor.run("Try the error tool")

    # Error should be captured in tool_results, not crash the executor
    error_step = result.steps[0]
    assert any("error" in tr for tr in error_step.tool_results)
    assert result.status == AgentStatus.COMPLETED


# ── System prompt ─────────────────────────────────────────────────────────────


def test_system_prompt_with_no_tools():
    executor = AgentExecutor(model=MockModel(), tool_registry=ToolRegistry())
    prompt = executor._build_system_prompt()
    assert "helpful" in prompt.lower()


def test_system_prompt_with_tools_includes_schema():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = AgentExecutor(model=MockModel(), tool_registry=registry)
    prompt = executor._build_system_prompt()
    assert "calculator" in prompt
    assert "TOOL_CALL" in prompt
