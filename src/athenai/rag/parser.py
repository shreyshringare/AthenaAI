"""Document parser — converts raw content into a ParsedDocument."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
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
        if not content or not content.strip():
            raise ValueError(f"document {document_id!r} has empty content")
        normalised = " ".join(content.split())
        return ParsedDocument(
            document_id=document_id,
            content=normalised,
            source=source,
            metadata=metadata or {},
        )
