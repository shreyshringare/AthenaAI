"""
Priority-ordered context packing. Truncates lowest-priority buckets first on overflow.

HOW IT FITS IN THE SYSTEM:
ContextEngine (engine.py) is the sole caller: it groups content into named
buckets (system/conversation/memory/rag/tools), then calls
ContextPacker.pack() with a total token ceiling (from TokenBudgetManager,
budget.py). This module owns the layout/eviction decision only — it does not
estimate tokens or fetch content itself.

Priority order (highest to lowest):
  system > conversation > memory > rag > tools

WHY THIS ORDER:
System prompt defines model behaviour — never drop it.
Conversation history is the user's direct context — high value.
Memory is retrieved facts — valuable but compressible.
RAG chunks are retrieved documents — most numerous, easiest to drop.
Tool results are already consumed by the model — lowest residual value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIORITY_ORDER = ["system", "conversation", "memory", "rag", "tools"]


@dataclass
class ContextItem:
    """One unit of content destined for the prompt.

    `priority` is not meant to be set by callers — pack() overwrites it with
    the item's rank in PRIORITY_ORDER (or a lowest-priority sentinel for
    unknown buckets) as it processes the item.
    """

    bucket: str
    content: str
    token_count: int
    priority: int = 0


@dataclass
class PackedContext:
    """Output of ContextPacker.pack().

    `dropped_buckets` lists the *bucket name* of every item that didn't fit
    (not the item itself/content) — enough to log/alert on what was
    sacrificed without holding onto the dropped content.
    """

    items: list[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    dropped_buckets: list[str] = field(default_factory=list)


class ContextPacker:
    """
    Inserts items in priority order. Truncates lowest-priority bucket
    first when total token budget is exceeded.
    """

    def pack(
        self,
        buckets: dict[str, list[ContextItem]],
        token_ceiling: int,
    ) -> PackedContext:
        """Greedily includes items in priority order up to `token_ceiling`.

        Eviction is whole-item, not partial: an item that doesn't fit is
        dropped entirely rather than truncated mid-content, so every
        included item is guaranteed intact. Mutates each input item's
        `priority` field in place as a side effect of ordering.
        """
        # Flatten in priority order
        ordered: list[ContextItem] = []
        for bucket_name in PRIORITY_ORDER:
            items = buckets.get(bucket_name, [])
            for item in items:
                item.priority = PRIORITY_ORDER.index(bucket_name)
                ordered.append(item)

        # Include any buckets not in PRIORITY_ORDER at the end
        for bucket_name, items in buckets.items():
            if bucket_name not in PRIORITY_ORDER:
                for item in items:
                    item.priority = len(PRIORITY_ORDER)
                    ordered.append(item)

        result = PackedContext()
        for item in ordered:
            if result.total_tokens + item.token_count <= token_ceiling:
                result.items.append(item)
                result.total_tokens += item.token_count
            else:
                result.dropped_buckets.append(item.bucket)

        return result
