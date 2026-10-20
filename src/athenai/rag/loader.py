"""DocumentLoader — idempotent ingest pipeline: parse → chunk → embed → store.
Also the query-side entry point: embed → retrieve → rerank.

HOW IT FITS IN THE SYSTEM:
This is the orchestrator for both halves of the RAG pipeline (see
athenai/rag/__init__.py for the full pipeline map). It is constructed once in
athenai.gateway.app._maybe_add_document_loader and stored on
app.state.document_loader. athenai.gateway.routes calls .ingest() for
/v1/documents/ingest and .search() for /v1/documents/search. It depends on,
but does not implement, each pipeline stage — parser, chunker, embedder,
retriever, and reranker are injected so tests can substitute MockEmbedder or
an in-memory retriever without touching this class.
"""

from __future__ import annotations

from typing import Any

from athenai.context.ranking import RelevanceRanker
from athenai.rag.chunker import SlidingWindowChunker
from athenai.rag.embedder import CloudEmbedder, MockEmbedder
from athenai.rag.parser import DocumentParser
from athenai.rag.reranker import CosineReranker
from athenai.rag.retriever import PgVectorRetriever, RetrievedChunk

_AnyEmbedder = CloudEmbedder | MockEmbedder


class DocumentLoader:
    """Orchestrates the full ingest pipeline, plus query-side search.

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
        reranker: CosineReranker | None = None,
        relevance_ranker: RelevanceRanker | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker or CosineReranker()
        self._relevance_ranker = relevance_ranker or RelevanceRanker()

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

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed `query` and return the top_k most relevant stored chunks.

        Retrieves 2x top_k candidates from the ANN search, re-sorts with
        CosineReranker (cheap, trusts the SQL-computed score), then does
        the authoritative final re-score + truncate with RelevanceRanker,
        which independently recomputes cosine similarity from each
        candidate's own embedding rather than trusting the SQL/ANN score --
        a cross-check against any approximation error in the SQL-side
        index, not just a relabeling of the same order.
        """
        [embedding] = await self._embedder.embed([query])
        candidates = await self._retriever.search(
            embedding, k=top_k * 2, metadata_filter=metadata_filter
        )
        reranked = self._reranker.rerank(embedding, candidates)

        ranked = self._relevance_ranker.rank(
            embedding,
            [
                {
                    "content": c.content,
                    "embedding": c.embedding,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                }
                for c in reranked
            ],
            top_k=top_k,
        )
        return [
            RetrievedChunk(
                chunk_id=r.metadata["chunk_id"],
                document_id=r.metadata["document_id"],
                content=r.content,
                chunk_index=r.metadata["chunk_index"],
                score=r.score,
            )
            for r in ranked
        ]
