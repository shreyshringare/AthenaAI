"""
SummaryMemory: Compresses old conversation history via a model call.

WHY SUMMARY COMPRESSION:
Unlimited raw conversation history grows unboundedly. After threshold (default 20)
messages, older messages are compressed into a single summary string. This keeps
the context window usage bounded while preserving semantic continuity.

WHY MODEL CALL (NOT EXTRACTIVE):
Extractive summarisation (keep top-N sentences) loses cross-message context.
A generative summary preserves the semantic thread — what was decided, what
constraints were established, what the user is trying to accomplish.
"""

from __future__ import annotations

from typing import Any, Protocol

from athenai.memory.base import MemoryEntry, MemoryType
from athenai.memory.conversation import ConversationMemory


class SummaryModel(Protocol):
    """Minimal interface needed to generate a summary."""

    async def generate(self, request: Any) -> Any: ...


_SUMMARY_PROMPT = (
    "You are a conversation summariser. Given the following messages, "
    "produce a concise summary (3-5 sentences) capturing the key decisions, "
    "facts established, and current goal. Be specific, not generic.\n\n"
    "Messages:\n{messages}\n\nSummary:"
)


class SummaryMemory:
    def __init__(
        self,
        conversation: ConversationMemory,
        model: SummaryModel,
        max_raw_messages: int = 20,
        keep_recent: int = 5,
    ) -> None:
        self._conv = conversation
        self._model = model
        self._max_raw = max_raw_messages
        self._keep_recent = keep_recent
        self._summaries: dict[str, str] = {}

    async def get_recent(self, session_id: str) -> list[MemoryEntry]:
        """Return summary (if exists) + latest keep_recent raw messages."""
        count = await self._conv.count(session_id)

        if count > self._max_raw:
            await self._compress(session_id)

        recent = await self._conv.get_recent(session_id, self._keep_recent)
        result: list[MemoryEntry] = []

        summary_text = self._summaries.get(session_id)
        if summary_text:
            result.append(MemoryEntry(
                id=f"summary-{session_id}",
                session_id=session_id,
                memory_type=MemoryType.SUMMARY,
                content=f"[Summary] {summary_text}",
            ))

        result.extend(recent)
        return result

    async def _compress(self, session_id: str) -> None:
        all_messages = await self._conv.get_recent(session_id, n=self._max_raw)
        older = all_messages[:-self._keep_recent]
        if not older:
            return

        messages_text = "\n".join(e.content for e in older)
        prompt = _SUMMARY_PROMPT.format(messages=messages_text)

        from athenai.models.base import ModelRequest
        request = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            model_name="summary",
            max_tokens=512,
            temperature=0.3,
        )
        response = await self._model.generate(request)
        self._summaries[session_id] = response.content
