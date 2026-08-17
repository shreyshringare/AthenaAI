"""
Priority-ordered context packing. Truncates lowest-priority buckets first on overflow.

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
    bucket: str
    content: str
    token_count: int
    priority: int = 0


@dataclass
class PackedContext:
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
