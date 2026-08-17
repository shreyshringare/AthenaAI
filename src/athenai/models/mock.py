"""
WHY MockModel:
Deterministic, zero API keys, instant — all tests run in CI without credentials.
Echo-back makes assertions trivial: expected output is derivable from input.
stream() yields characters with a 10ms sleep to simulate realistic token flow
without relying on network timing.

WHY NOT REAL MODEL IN TESTS:
Real model calls add latency, cost, flakiness from rate limits, and non-determinism.
The mock lets gate tests focus on control flow, not model output quality.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from athenai.models.base import ModelRequest, ModelResponse


class MockModel:
    """Deterministic echo model — no API key required."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name
        self._healthy = True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        last_user_msg = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        content = f"[{self.name}] Echo: {last_user_msg}"
        input_tokens = sum(len(m.get("content", "")) // 4 for m in request.messages)
        output_tokens = len(content) // 4

        return ModelResponse(
            content=content,
            model_name=self.name,
            input_tokens=max(1, input_tokens),
            output_tokens=max(1, output_tokens),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        response = await self.generate(request)
        for char in response.content:
            await asyncio.sleep(0.01)
            yield char

    async def health_check(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy
