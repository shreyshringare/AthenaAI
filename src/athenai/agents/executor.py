"""
AgentExecutor — the core agentic loop.

HOW IT FITS IN THE SYSTEM:
Constructed by Agent (agent.py), which just forwards run() calls to it.
Pulls tool contracts from a ToolRegistry (athenai.tools.registry) to build
its system prompt, calls model.generate() (athenai.models.base.ModelRequest)
each iteration, and dispatches tool calls back through the same registry.
Yields AgentResult/AgentStep/AgentStatus (state.py) as its trace.

HOW THE LOOP WORKS:
1. Build a system prompt that lists all registered tools.
2. Call the model with the current message history.
3. Inspect the response for structured tool_calls in response.metadata
   (currently unused by any adapter) or TOOL_CALL directives (text
   protocol — the path every adapter actually takes today).
4. If tool calls found: execute them in parallel via asyncio.gather,
   append results to message history, continue to next iteration.
5. If no tool calls: treat response as final answer, return AgentResult.
6. Hard cap at max_iterations prevents runaway loops.

WHY PARALLEL TOOL EXECUTION:
Independent tool calls (e.g. "look up A and calculate B") should not
serialise. asyncio.gather runs them concurrently; individual failures are
caught and surfaced per-call, not globally.

WHY TEXT PROTOCOL (TOOL_CALL: {...}):
The Anthropic tool_use block format requires the SDK to encode tool schemas
into the API request and decode content blocks. The text protocol works with
any model (including MockModel) without SDK dependency, making the loop
testable without API keys. The structured response.metadata["tool_calls"]
path below is checked first but is currently dead in practice: no adapter
(CloudModel included — see models/cloud.py) populates it yet, so every
model falls through to the text parser.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from athenai.agents.state import AgentResult, AgentStatus, AgentStep, transition
from athenai.models.base import ModelRequest
from athenai.tools.registry import ToolRegistry

_MAX_ITERATIONS = 10
_TOOL_CALL_PREFIX = "TOOL_CALL:"


class AgentExecutor:
    """Runs the model-tool loop until a final answer or max_iterations."""

    def __init__(
        self,
        model: Any,
        tool_registry: ToolRegistry,
        max_iterations: int = _MAX_ITERATIONS,
    ) -> None:
        self._model = model
        self._registry = tool_registry
        self._max_iterations = max_iterations

    async def run(self, task: str, user_id: str = "default") -> AgentResult:
        """Run the model-tool loop to completion for a single task.

        user_id is accepted for future per-user context/memory scoping but
        is not currently used inside the loop.

        Normal exit: the model responds with no tool calls, which is taken
        as the final answer (status COMPLETED).

        Iteration cap exit: if max_iterations is reached without a final
        answer, status is FAILED and the *last model response seen* is
        returned as final_answer — callers should check status rather than
        assume a non-empty final_answer means success.
        """
        status = AgentStatus.CREATED
        steps: list[AgentStep] = []
        messages: list[dict[str, str]] = [{"role": "user", "content": task}]
        system = self._build_system_prompt()

        status = transition(status, AgentStatus.RUNNING)

        for iteration in range(self._max_iterations):
            request = ModelRequest(
                messages=messages,
                model_name="agent",
                system=system,
            )
            response = await self._model.generate(request)

            # Prefer structured tool calls from model metadata (CloudModel path)
            tool_calls: list[dict[str, Any]] = list(response.metadata.get("tool_calls", []))

            # Fallback: parse text-protocol tool calls from response content
            if not tool_calls:
                tool_calls = self._parse_tool_calls(response.content)

            step = AgentStep(
                iteration=iteration,
                model_response=response.content,
                tool_calls=tool_calls,
            )

            if not tool_calls:
                steps.append(step)
                status = transition(status, AgentStatus.COMPLETED)
                return AgentResult(
                    final_answer=response.content,
                    steps=steps,
                    total_iterations=iteration + 1,
                    status=status,
                )

            # Transition to tool-execution sub-state
            status = transition(status, AgentStatus.AWAITING_TOOL)

            results = await asyncio.gather(
                *[self._execute_tool(tc) for tc in tool_calls],
                return_exceptions=True,
            )

            tool_results: list[dict[str, Any]] = []
            for tc, result in zip(tool_calls, results, strict=True):
                if isinstance(result, Exception):
                    payload = json.dumps({"error": str(result)})
                    tool_results.append({"name": tc["name"], "error": str(result)})
                else:
                    payload = json.dumps(result) if not isinstance(result, str) else result
                    tool_results.append({"name": tc["name"], "result": result})
                messages.append({
                    "role": "tool",
                    "content": payload,
                    "name": tc["name"],
                })

            step.tool_results = tool_results
            steps.append(step)

            # Add the assistant turn so the model knows what it said
            messages.append({"role": "assistant", "content": response.content})

            # Return to running for next iteration
            status = transition(status, AgentStatus.RUNNING)

        # Hard cap reached — treat last response as best available answer
        status = transition(status, AgentStatus.FAILED)
        last_response = steps[-1].model_response if steps else "max iterations reached"
        return AgentResult(
            final_answer=last_response,
            steps=steps,
            total_iterations=self._max_iterations,
            status=status,
        )

    def _parse_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Extract TOOL_CALL: {...} directives from model response text.

        Lines that start with the prefix but fail to parse as a well-formed
        {"name", "arguments"} JSON object are silently skipped rather than
        raising — a malformed line from the model should not crash the loop
        when the rest of the response may still be usable.
        """
        calls: list[dict[str, Any]] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith(_TOOL_CALL_PREFIX):
                continue
            json_str = stripped[len(_TOOL_CALL_PREFIX):].strip()
            try:
                call = json.loads(json_str)
                if isinstance(call, dict) and "name" in call and "arguments" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                pass
        return calls

    async def _execute_tool(self, tool_call: dict[str, Any]) -> Any:
        tool = self._registry.get(tool_call["name"])
        return await tool.execute(tool_call["arguments"])

    def _build_system_prompt(self) -> str:
        """Build the system prompt, including the text tool-call protocol.

        Falls back to a plain "helpful assistant" prompt (no TOOL_CALL
        instructions) when the registry has no tools, so an agent run with
        an empty registry doesn't confuse the model with a protocol it can
        never actually use.
        """
        schemas = self._registry.get_schemas()
        if not schemas:
            return "You are a helpful AI assistant."
        tools_json = json.dumps(schemas, indent=2)
        return (
            "You are a helpful AI assistant with access to the following tools:\n\n"
            f"{tools_json}\n\n"
            "To use a tool, include EXACTLY this format (one tool call per line):\n"
            'TOOL_CALL: {"name": "tool_name", "arguments": {...}}\n\n'
            "When you have computed the final answer, respond with plain text only "
            "(no TOOL_CALL prefix). Execute multiple tools on separate lines when needed."
        )
