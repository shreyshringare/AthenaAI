"""
WHY asyncio.gather FOR MEMORY + RAG:
Memory retrieval (DB query) and RAG retrieval (vector search) have no data
dependency on each other. Parallel execution cuts context build latency from
(memory_ms + rag_ms) to max(memory_ms, rag_ms). At scale this saves hundreds
of milliseconds per request.

WHY NOT SEQUENTIAL:
Sequential retrieval is the naive path. Two 200ms queries take 400ms
sequentially vs 200ms in parallel — 2x faster with zero code complexity cost.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from athenai.context.budget import TokenBudgetManager
from athenai.context.packing import ContextItem, ContextPacker, PackedContext


@dataclass
class BuiltContext:
    packed: PackedContext
    total_tokens: int
    memory_items: list[str] = field(default_factory=list)
    rag_items: list[str] = field(default_factory=list)
    system_prompt: str = ""


class ContextEngine:
    def __init__(self, budget_manager: TokenBudgetManager) -> None:
        self._budget = budget_manager
        self._packer = ContextPacker()

    async def build(
        self,
        system_prompt: str,
        conversation: list[str],
        memory_fn: Callable[[], Awaitable[list[str]]],
        rag_fn: Callable[[], Awaitable[list[str]]],
        tool_results: list[str] | None = None,
    ) -> BuiltContext:
        # Parallel retrieval — core design decision
        memory_items, rag_items = await asyncio.gather(memory_fn(), rag_fn())

        def _count(text: str) -> int:
            return max(1, len(text) // 4)

        buckets: dict[str, list[ContextItem]] = {
            "system": [ContextItem("system", system_prompt, _count(system_prompt))],
            "conversation": [
                ContextItem("conversation", msg, _count(msg)) for msg in conversation
            ],
            "memory": [
                ContextItem("memory", mem, _count(mem)) for mem in memory_items
            ],
            "rag": [
                ContextItem("rag", chunk, _count(chunk)) for chunk in rag_items
            ],
        }
        if tool_results:
            buckets["tools"] = [
                ContextItem("tools", t, _count(t)) for t in tool_results
            ]

        packed = self._packer.pack(buckets, self._budget.total_budget)

        return BuiltContext(
            packed=packed,
            total_tokens=packed.total_tokens,
            memory_items=memory_items,
            rag_items=rag_items,
            system_prompt=system_prompt,
        )
