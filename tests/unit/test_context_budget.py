"""P4 gate: context engine and token budget."""

from __future__ import annotations

import asyncio
import time

import pytest

from athenai.context.budget import TokenBudgetManager
from athenai.context.engine import BuiltContext, ContextEngine
from athenai.context.packing import ContextItem, ContextPacker
from athenai.core.exceptions import ContextOverflowError


# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------


def _budget(total: int = 10000) -> TokenBudgetManager:
    return TokenBudgetManager({
        "system": 1000,
        "conversation": 4000,
        "memory": 1000,
        "rag": 3000,
        "tools": 500,
        "total": total,
    })


def test_allocate_within_budget() -> None:
    bm = _budget()
    bm.allocate("system", 500)
    bm.allocate("conversation", 1000)
    bm.allocate("memory", 500)
    bm.allocate("rag", 500)
    assert bm.total_used == 2500


def test_allocate_past_bucket_ceiling_raises() -> None:
    bm = _budget()
    with pytest.raises(ContextOverflowError):
        bm.allocate("system", 1001)  # bucket limit is 1000


def test_allocate_past_global_ceiling_raises() -> None:
    bm = _budget(total=100)
    with pytest.raises(ContextOverflowError):
        bm.allocate("conversation", 101)


def test_unknown_bucket_raises() -> None:
    bm = _budget()
    with pytest.raises(ContextOverflowError):
        bm.allocate("unknown_bucket", 10)


# ---------------------------------------------------------------------------
# ContextPacker
# ---------------------------------------------------------------------------


def test_packer_truncates_rag_not_system() -> None:
    packer = ContextPacker()
    buckets = {
        "system": [ContextItem("system", "sys", 100)],
        "conversation": [ContextItem("conversation", "conv", 200)],
        "rag": [ContextItem("rag", "rag1", 800)],
    }
    packed = packer.pack(buckets, token_ceiling=400)
    bucket_names = {item.bucket for item in packed.items}
    assert "system" in bucket_names
    assert "conversation" in bucket_names
    assert "rag" not in bucket_names
    assert "rag" in packed.dropped_buckets


def test_packer_total_within_ceiling() -> None:
    packer = ContextPacker()
    buckets = {
        "system": [ContextItem("system", "x" * 50, 50)],
        "rag": [ContextItem("rag", "y" * 50, 50)],
    }
    packed = packer.pack(buckets, token_ceiling=90)
    assert packed.total_tokens <= 90


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_engine_build_returns_built_context() -> None:
    bm = _budget()
    engine = ContextEngine(bm)

    async def mem_fn() -> list[str]:
        return ["memory fact 1", "memory fact 2"]

    async def rag_fn() -> list[str]:
        return ["rag chunk 1", "rag chunk 2"]

    ctx = await engine.build(
        system_prompt="You are helpful.",
        conversation=["user: hello", "assistant: hi"],
        memory_fn=mem_fn,
        rag_fn=rag_fn,
    )

    assert isinstance(ctx, BuiltContext)
    assert ctx.total_tokens > 0
    assert ctx.system_prompt == "You are helpful."
    assert len(ctx.memory_items) == 2
    assert len(ctx.rag_items) == 2


@pytest.mark.asyncio
async def test_context_engine_parallel_retrieval() -> None:
    """Memory (100ms) + RAG (200ms) must complete in < 250ms, not 300ms."""
    bm = _budget()
    engine = ContextEngine(bm)

    async def slow_mem() -> list[str]:
        await asyncio.sleep(0.1)
        return ["mem"]

    async def slow_rag() -> list[str]:
        await asyncio.sleep(0.2)
        return ["rag"]

    start = time.perf_counter()
    await engine.build(
        system_prompt="sys",
        conversation=[],
        memory_fn=slow_mem,
        rag_fn=slow_rag,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 0.25, f"Parallel retrieval took {elapsed:.3f}s — expected < 0.25s"
