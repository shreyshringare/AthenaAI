"""Document parser — converts raw content into a ParsedDocument.

HOW IT FITS IN THE SYSTEM:
First stage of the ingest pipeline (see athenai/rag/__init__.py). Called by
DocumentLoader.ingest() before chunking. Its only job is to validate and
normalise raw text into the ParsedDocument shape that SlidingWindowChunker
expects — it does not chunk, embed, or store anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    """Normalised document ready for chunking. `content` is whitespace-collapsed."""

    document_id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Parses plain-text (and Markdown) content into ParsedDocument.

    WHY NO PDF/DOCX HERE:
    Binary format parsing adds heavy deps (pypdf, python-docx). Callers that
    need binary parsing should extract text upstream and pass it as plain text.
    The parser's job is normalisation, not format detection.
    """

    def parse(
        self,
        content: str,
        document_id: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """Normalise `content` into a ParsedDocument.

        Collapses all whitespace (including newlines) to single spaces via
        `" ".join(content.split())` — this destroys Markdown structure (headers,
        line breaks, list formatting) by design, since chunking downstream
        works on a flat character stream, not a document tree.

        Raises ValueError if `content` is empty or whitespace-only; callers
        should not attempt to ingest a document with no extractable text.
        """
        if not content or not content.strip():
            raise ValueError(f"document {document_id!r} has empty content")
        normalised = " ".join(content.split())
        return ParsedDocument(
            document_id=document_id,
            content=normalised,
            source=source,
            metadata=metadata or {},
        )
