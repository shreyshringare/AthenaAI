"""
ContextEngine — orchestrates context assembly for a single agent turn.

HOW IT FITS IN THE SYSTEM:
This is the entry point callers (e.g. the agent executor) use to turn a
system prompt, conversation history, and pluggable memory/RAG retrieval
functions into a single BuiltContext ready to send to the model. It fetches
memory and RAG content concurrently, wraps every piece of content as a
ContextItem with a rough token estimate, and hands the whole set to
ContextPacker (packing.py) to lay out within TokenBudgetManager's
(budget.py) total token ceiling — dropping lowest-priority buckets first if
everything doesn't fit.

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
    """Result of ContextEngine.build().

    `packed` holds the final, budget-fitted items (and any dropped_buckets);
    `total_tokens` mirrors `packed.total_tokens` for convenience. memory_items
    and rag_items are the *unfiltered* retrieval results (before packing may
    have dropped some) — kept around so callers can log or inspect what was
    retrieved even if it didn't make it into the final prompt.
    """

    packed: PackedContext
    total_tokens: int
    memory_items: list[str] = field(default_factory=list)
    rag_items: list[str] = field(default_factory=list)
    system_prompt: str = ""


class ContextEngine:
    """Builds a token-budgeted prompt context from multiple content sources."""

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
        """Fetch memory/RAG concurrently, then pack everything into budget.

        memory_fn/rag_fn are injected as zero-arg async callables (rather
        than this engine owning a memory/RAG client directly) so it stays
        decoupled from retrieval implementations — callers close over
        whatever query/embedding context they need.
        """
        # Parallel retrieval — core design decision
        memory_items, rag_items = await asyncio.gather(memory_fn(), rag_fn())

        def _count(text: str) -> int:
            # Rough 4-chars-per-token heuristic — good enough for budgeting
            # decisions; not a substitute for a real tokenizer.
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
