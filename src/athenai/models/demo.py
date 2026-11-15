"""
DemoMockModel — a MockModel that can be broken on demand and can drive tools.

HOW IT FITS IN THE SYSTEM:
Only used when ATHENA_DEMO_MODE is set (see gateway/app.py::_build_demo_model).
It lets a live demo show resilience features without real providers or API
keys:

  - `set_failing(True)` makes generate()/stream() raise ModelUnavailableError,
    which is what a down provider looks like to RoutedModel — so retries,
    circuit-breaker trips and fallback to the other backend can be triggered
    from the /debug routes (gateway/debug.py) while recording.
  - A task of the form "calc: 12*7; 2**10" makes the first model turn emit one
    TOOL_CALL line per expression (so AgentExecutor runs the calculator calls
    in parallel). Once tool results are in the transcript, the next turn
    returns a final answer summarising them. Real models do this themselves;
    the plain MockModel just echoes, which can't show tool use.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from athenai.core.exceptions import ModelUnavailableError
from athenai.models.base import ModelRequest, ModelResponse
from athenai.models.mock import MockModel

_CALC_PREFIX = "calc:"


class DemoMockModel(MockModel):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.failing = False

    def set_failing(self, failing: bool) -> None:
        self.failing = failing

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if self.failing:
            raise ModelUnavailableError(f"{self.name} is down (demo fault injection)")

        scripted = self._scripted_tool_turn(request)
        if scripted is None:
            return await super().generate(request)

        input_tokens = sum(len(m.get("content", "")) // 4 for m in request.messages)
        return ModelResponse(
            content=scripted,
            model_name=self.name,
            input_tokens=max(1, input_tokens),
            output_tokens=max(1, len(scripted) // 4),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        if self.failing:
            raise ModelUnavailableError(f"{self.name} is down (demo fault injection)")
        async for token in super().stream(request):
            yield token

    async def health_check(self) -> bool:
        return not self.failing

    @staticmethod
    def _scripted_tool_turn(request: ModelRequest) -> str | None:
        """Return scripted agent output, or None for a plain echo turn."""
        user_task = next(
            (
                m["content"]
                for m in request.messages
                if m.get("role") == "user" and m.get("content", "").lower().startswith(_CALC_PREFIX)
            ),
            None,
        )
        if user_task is None:
            return None

        tool_results = [m["content"] for m in request.messages if m.get("role") == "tool"]
        if tool_results:
            return "Results: " + ", ".join(tool_results)

        expressions = [e.strip() for e in user_task[len(_CALC_PREFIX):].split(";") if e.strip()]
        return "\n".join(
            "TOOL_CALL: "
            + json.dumps({"name": "calculator", "arguments": {"expression": expr}})
            for expr in expressions
        )
