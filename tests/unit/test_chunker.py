"""P6 gate (unit): SlidingWindowChunker."""

from __future__ import annotations

import pytest

from athenai.rag.chunker import SlidingWindowChunker
from athenai.rag.parser import DocumentParser, ParsedDocument


def _doc(text: str, doc_id: str = "doc-1") -> ParsedDocument:
    return ParsedDocument(document_id=doc_id, content=text, source="test")


def test_single_chunk_small_doc() -> None:
    chunker = SlidingWindowChunker(chunk_size=512, overlap=64)
    doc = _doc("Hello world. This is a short document.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0].content
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == "doc-1"


def test_multiple_chunks_large_doc() -> None:
    chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
    # 10 tokens * 4 chars = 40 chars per chunk, step = 8 tokens * 4 = 32 chars
    text = "a " * 200  # 400 chars — should produce multiple chunks
    doc = _doc(text)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1


def test_overlap_produces_shared_content() -> None:
    chunker = SlidingWindowChunker(chunk_size=20, overlap=5)
    # 20 tokens * 4 = 80 chars per chunk, overlap = 5 * 4 = 20 chars
    text = ("word " * 100).strip()
    doc = _doc(text)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    # overlap: end of chunk[0] should overlap with start of chunk[1]
    end_of_first = chunks[0].content[-20:]
    start_of_second = chunks[1].content[:20]
    # at least some words appear in both
    words_first = set(end_of_first.split())
    words_second = set(start_of_second.split())
    assert words_first & words_second  # non-empty intersection


def test_chunk_ids_unique() -> None:
    chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
    text = "token " * 150
    doc = _doc(text)
    chunks = chunker.chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_doc_returns_no_chunks() -> None:
    chunker = SlidingWindowChunker(chunk_size=512, overlap=64)
    doc = ParsedDocument(document_id="empty", content="", source="test")
    chunks = chunker.chunk(doc)
    assert chunks == []


def test_chunk_index_sequential() -> None:
    chunker = SlidingWindowChunker(chunk_size=10, overlap=2)
    text = "word " * 200
    doc = _doc(text)
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=10, overlap=10)


def test_metadata_propagated_to_chunks() -> None:
    chunker = SlidingWindowChunker(chunk_size=512, overlap=64)
    doc = ParsedDocument(
        document_id="d1",
        content="Some content here",
        source="test",
        metadata={"source": "wiki", "lang": "en"},
    )
    chunks = chunker.chunk(doc)
    assert chunks[0].metadata["source"] == "wiki"
    assert chunks[0].metadata["lang"] == "en"


def test_parser_normalises_whitespace() -> None:
    parser = DocumentParser()
    doc = parser.parse("  hello   world\n\nfoo  ", "d1")
    assert "  " not in doc.content
    assert doc.content == "hello world foo"


def test_parser_rejects_empty_content() -> None:
    parser = DocumentParser()
    with pytest.raises(ValueError):
        parser.parse("   ", "d1")
