"""P6 gate (integration): RAG pipeline end-to-end (requires PostgreSQL + pgvector)."""

from __future__ import annotations

import asyncpg
import pytest

from athenai.rag.chunker import SlidingWindowChunker
from athenai.rag.embedder import MockEmbedder
from athenai.rag.loader import DocumentLoader
from athenai.rag.parser import DocumentParser
from athenai.rag.reranker import CosineReranker
from athenai.rag.retriever import PgVectorRetriever

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def retriever(db_pool: asyncpg.Pool) -> PgVectorRetriever:
    r = PgVectorRetriever(db_pool)
    await r._ensure_schema()
    return r


@pytest.fixture
def loader(retriever: PgVectorRetriever) -> DocumentLoader:
    return DocumentLoader(
        parser=DocumentParser(),
        chunker=SlidingWindowChunker(chunk_size=50, overlap=5),
        embedder=MockEmbedder(),
        retriever=retriever,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_and_search(loader: DocumentLoader, retriever: PgVectorRetriever) -> None:
    content = (
        "Python is a high-level programming language known for its readability. "
        "It supports multiple programming paradigms including procedural, "
        "object-oriented, and functional programming. "
        "Python's extensive standard library makes it very versatile."
    )
    chunk_count = await loader.ingest(content, document_id="doc-python")
    assert chunk_count >= 1

    embedder = MockEmbedder()
    query_emb = (await embedder.embed(["Python programming language"]))[0]
    results = await retriever.search(query_emb, k=3)
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)


@pytest.mark.asyncio
async def test_ingest_idempotent(loader: DocumentLoader, retriever: PgVectorRetriever) -> None:
    content = "Machine learning is a subset of artificial intelligence."
    first = await loader.ingest(content, document_id="doc-ml")
    second = await loader.ingest(content, document_id="doc-ml")
    assert first == second
    count = await retriever.count_by_document("doc-ml")
    assert count == first


@pytest.mark.asyncio
async def test_reingest_replaces_chunks(
    loader: DocumentLoader, retriever: PgVectorRetriever
) -> None:
    content_v1 = "The cat sat on the mat."
    content_v2 = "The dog ran in the park. " * 10
    await loader.ingest(content_v1, document_id="doc-update")
    count_v1 = await retriever.count_by_document("doc-update")

    await loader.reingest(content_v2, document_id="doc-update")
    count_v2 = await retriever.count_by_document("doc-update")

    assert count_v2 > count_v1


@pytest.mark.asyncio
async def test_search_top_result_relevant(
    loader: DocumentLoader, retriever: PgVectorRetriever
) -> None:
    docs = {
        "doc-a": "Neural networks are computational models inspired by the brain.",
        "doc-b": "SQL databases store data in relational tables with rows and columns.",
        "doc-c": "Deep learning uses many layers of neural networks to learn representations.",
    }
    for doc_id, content in docs.items():
        await loader.ingest(content, document_id=doc_id)

    embedder = MockEmbedder()
    query_emb = (await embedder.embed(["neural network deep learning"]))[0]
    results = await retriever.search(query_emb, k=3)

    assert len(results) >= 1
    top = results[0]
    assert top.document_id in ("doc-a", "doc-c")


@pytest.mark.asyncio
async def test_delete_document(loader: DocumentLoader, retriever: PgVectorRetriever) -> None:
    await loader.ingest("Some content to delete.", document_id="doc-del")
    assert await retriever.count_by_document("doc-del") >= 1

    await retriever.delete_document("doc-del")
    assert await retriever.count_by_document("doc-del") == 0


@pytest.mark.asyncio
async def test_reranker_sorts_by_score(retriever: PgVectorRetriever) -> None:
    loader = DocumentLoader(
        parser=DocumentParser(),
        chunker=SlidingWindowChunker(chunk_size=50, overlap=5),
        embedder=MockEmbedder(),
        retriever=retriever,
    )
    for i in range(5):
        await loader.ingest(f"Document about topic {i} " * 10, document_id=f"rerank-doc-{i}")

    embedder = MockEmbedder()
    query_emb = (await embedder.embed(["topic"]))[0]
    results = await retriever.search(query_emb, k=5)

    reranker = CosineReranker()
    reranked = reranker.rerank(query_emb, results, top_k=3)
    assert len(reranked) == 3
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)
