"""
WHY HARD TOKEN CEILING:
Silent truncation is a worse failure mode than a loud error. If context silently
drops memory or RAG chunks, the model hallucinates without the caller knowing why.
A hard ceiling (ContextOverflowError) forces the caller to make an explicit
decision about what to sacrifice — system prompt? RAG? conversation?

WHY NOT SOFT TRUNCATION:
Soft truncation hides bugs in token estimation. When the token count exceeds the
limit by 10%, you want to know — not silently receive a degraded response.
"""

from __future__ import annotations

from athenai.core.exceptions import ContextOverflowError


class TokenBudgetManager:
    """Per-bucket token allocation with hard ceiling enforcement."""

    def __init__(self, allocations: dict[str, int]) -> None:
        self._allocations = dict(allocations)
        self._used: dict[str, int] = dict.fromkeys(allocations, 0)
        self._total = allocations.get("total", sum(
            v for k, v in allocations.items() if k != "total"
        ))

    @property
    def total_budget(self) -> int:
        return self._total

    @property
    def total_used(self) -> int:
        return sum(v for k, v in self._used.items() if k != "total")

    def allocate(self, key: str, tokens: int) -> int:
        """
        Request `tokens` for `key`. Returns actually allocated tokens.
        Raises ContextOverflowError if the bucket ceiling is exceeded.
        """
        bucket_limit = self._allocations.get(key, 0)
        if bucket_limit == 0:
            raise ContextOverflowError(f"No budget allocated for bucket {key!r}")

        current = self._used.get(key, 0)
        if current + tokens > bucket_limit:
            raise ContextOverflowError(
                f"Bucket {key!r} overflow: {current + tokens} > {bucket_limit} tokens"
            )

        total_after = self.total_used + tokens
        if total_after > self._total:
            raise ContextOverflowError(
                f"Global token ceiling exceeded: {total_after} > {self._total}"
            )

        self._used[key] = current + tokens
        return tokens

    def remaining(self, key: str) -> int:
        limit = self._allocations.get(key, 0)
        used = self._used.get(key, 0)
        return max(0, limit - used)

    def reset(self) -> None:
        self._used = dict.fromkeys(self._allocations, 0)
