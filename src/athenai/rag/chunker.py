"""SlidingWindowChunker — overlapping token-approximate text chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from athenai.rag.parser import ParsedDocument

# 1 token ≈ 4 characters for English prose (GPT-family approximation).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SlidingWindowChunker:
    """Splits a ParsedDocument into overlapping chunks by approximate token count.

    WHY SLIDING WINDOW (NOT SENTENCE SPLIT):
    Sentence-boundary splitting loses cross-sentence context that's often
    critical for retrieval — a question about "the result" may only make sense
    with the preceding sentence. Overlap preserves that context at the cost of
    mild redundancy, which is acceptable given pgvector's sub-millisecond ANN.

    WHY CHARACTER PROXY (NOT TIKTOKEN):
    Adding tiktoken as a hard dependency just for chunking adds ~15 MB and a
    network fetch on first run. The 4 chars/token approximation is accurate
    within ±10% for English prose and is sufficient for chunk-size budgeting.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        chunk_chars = self._chunk_size * _CHARS_PER_TOKEN
        overlap_chars = self._overlap * _CHARS_PER_TOKEN
        step = chunk_chars - overlap_chars

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            # Extend to next whitespace boundary to avoid mid-word cuts.
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document.document_id,
                        content=chunk_text,
                        chunk_index=index,
                        start_char=start,
                        end_char=end,
                        metadata=dict(document.metadata),
                    )
                )
                index += 1

            if end >= len(text):
                break
            start += step

        return chunks
