"""DocumentLoader — idempotent ingest pipeline: parse → chunk → embed → store."""

from __future__ import annotations

from typing import Any

from athenai.rag.chunker import SlidingWindowChunker
from athenai.rag.embedder import CloudEmbedder, MockEmbedder
from athenai.rag.parser import DocumentParser
from athenai.rag.retriever import PgVectorRetriever

_AnyEmbedder = CloudEmbedder | MockEmbedder


class DocumentLoader:
    """Orchestrates the full ingest pipeline.

    WHY IDEMPOTENT ON document_id:
    Re-ingesting the same document would duplicate chunks in the vector store,
    inflating retrieval results with identical content. Idempotency lets callers
    call ingest() safely on restart or retry without first checking state.
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: SlidingWindowChunker,
        embedder: _AnyEmbedder,
        retriever: PgVectorRetriever,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._retriever = retriever

    async def ingest(
        self,
        content: str,
        document_id: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Parse, chunk, embed, and store. Returns chunk count. Skips if already stored."""
        existing = await self._retriever.count_by_document(document_id)
        if existing > 0:
            return existing

        doc = self._parser.parse(content, document_id, source=source, metadata=metadata)
        chunks = self._chunker.chunk(doc)
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed(texts)
        await self._retriever.store_chunks(chunks, embeddings)
        return len(chunks)

    async def reingest(
        self,
        content: str,
        document_id: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Force re-ingest: delete existing chunks then ingest fresh."""
        await self._retriever.delete_document(document_id)
        return await self.ingest(content, document_id, source=source, metadata=metadata)
