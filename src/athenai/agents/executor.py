"""
AgentExecutor — the core agentic loop.

HOW IT FITS IN THE SYSTEM:
Constructed by Agent (agent.py), which just forwards run() calls to it.
Pulls tool contracts from a ToolRegistry (athenai.tools.registry) to build
its system prompt, calls model.generate() (athenai.models.base.ModelRequest)
each iteration, and dispatches tool calls back through the same registry —
each call is schema-validated via ToolValidator (athenai.tools.validator)
before tool.execute() runs, and permission-checked against `allowed_tools`
when one is configured. Yields AgentResult/AgentStep/AgentStatus (state.py)
as its trace.

WHY allowed_tools IS A SINGLE SET, NOT A PER-USER LOOKUP:
There's no multi-tenant identity system in this gateway today — just one
shared ATHENA_GATEWAY_API_KEY (see athenai.gateway.auth). `user_id` is a
free-form string the caller supplies, used here only for the error
message's audit trail, not to differentiate permissions per caller. A real
per-user permission store is future work once real user accounts exist;
this is the honest shape for what the system can actually authenticate
today.

HOW THE LOOP WORKS:
1. Build a system prompt that lists all registered tools; also pass the
   same schemas via ModelRequest.metadata["tools"] every iteration.
2. Call the model with the current message history.
3. Inspect the response for structured tool_calls in response.metadata
   (CloudModel populates this from Anthropic's native tool_use blocks —
   see models/cloud.py) or TOOL_CALL directives (text protocol — the
   fallback path MockModel/LocalModel and any plain-text response use).
4. If tool calls found: execute them in parallel via asyncio.gather,
   append results to message history, continue to next iteration.
5. If no tool calls: treat response as final answer, return AgentResult.
6. Hard cap at max_iterations prevents runaway loops.

WHY PARALLEL TOOL EXECUTION:
Independent tool calls (e.g. "look up A and calculate B") should not
serialise. asyncio.gather runs them concurrently; individual failures are
caught and surfaced per-call, not globally.

WHY TEXT PROTOCOL (TOOL_CALL: {...}) STAYS, EVEN THOUGH CloudModel NOW
SUPPORTS NATIVE tool_use:
MockModel and LocalModel have no structured tool-use mechanism to speak
of, so the text protocol remains the only way they can signal a tool call
-- and it's harmless to also instruct Claude this way, since the
structured response.metadata["tool_calls"] path is checked first; a
populated list always wins and _parse_tool_calls() on the text content
never runs. No double-execution risk.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from athenai.agents.state import AgentResult, AgentStatus, AgentStep, transition
from athenai.models.base import ModelRequest
from athenai.tools.registry import ToolRegistry
from athenai.tools.validator import ToolValidator

_MAX_ITERATIONS = 10
_TOOL_CALL_PREFIX = "TOOL_CALL:"


class AgentExecutor:
    """Runs the model-tool loop until a final answer or max_iterations."""

    def __init__(
        self,
        model: Any,
        tool_registry: ToolRegistry,
        max_iterations: int = _MAX_ITERATIONS,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self._model = model
        self._registry = tool_registry
        self._max_iterations = max_iterations
        self._validator = ToolValidator()
        self._allowed_tools = allowed_tools

    async def run(self, task: str, user_id: str = "default") -> AgentResult:
        """Run the model-tool loop to completion for a single task.

        user_id is forwarded to _execute_tool for the permission check's
        audit trail when allowed_tools is configured (see class docstring
        for why that's not a per-user lookup).

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
        schemas = self._registry.get_schemas()
        system = self._build_system_prompt(schemas)

        status = transition(status, AgentStatus.RUNNING)

        for iteration in range(self._max_iterations):
            request = ModelRequest(
                messages=messages,
                model_name="agent",
                system=system,
                metadata={"tools": schemas},
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
                *[self._execute_tool(tc, user_id) for tc in tool_calls],
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

    async def _execute_tool(self, tool_call: dict[str, Any], user_id: str) -> Any:
        """Look up, schema-validate, permission-check, then execute a single
        tool call.

        Schema runs before permission (matches ToolValidator.validate()'s
        own documented order): it's a pure in-process check, so failing
        fast on a malformed call gives a clearer error than "permission
        denied for a call that would have failed anyway" -- even though
        the permission check here is also in-process (no I/O), keeping
        the same order avoids two different orderings to reason about.

        Permission is only checked when allowed_tools was configured at
        construction time -- None means unrestricted (today's default,
        unchanged), matching every other optional security feature in
        this gateway (auth, RAG, HTTPTool's domain allowlist) degrading to
        permissive rather than crashing when unconfigured.
        """
        tool = self._registry.get(tool_call["name"])
        self._validator.validate_schema(tool, tool_call["arguments"])
        if self._allowed_tools is not None:
            self._validator.validate_permission(tool, user_id, self._allowed_tools)
        return await tool.execute(tool_call["arguments"])

    def _build_system_prompt(self, schemas: list[dict[str, Any]] | None = None) -> str:
        """Build the system prompt, including the text tool-call protocol.

        Falls back to a plain "helpful assistant" prompt (no TOOL_CALL
        instructions) when the registry has no tools, so an agent run with
        an empty registry doesn't confuse the model with a protocol it can
        never actually use.

        schemas defaults to a fresh registry lookup when not passed (kept
        optional so existing direct callers/tests don't break) -- run()
        passes the one it already computed to avoid calling get_schemas()
        twice per iteration.
        """
        if schemas is None:
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
