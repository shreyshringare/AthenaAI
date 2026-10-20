"""Unit tests for gateway/app.py's startup wiring helpers (not routes --
see test_gateway.py for those).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from athenai.core.config import AthenaConfig
from athenai.gateway.app import _maybe_add_document_loader


@pytest.mark.asyncio
async def test_document_loader_disabled_when_embedder_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the bug found 2026-10-20: CloudEmbedder requires
    api_key with no default, but nothing previously read one -- RAG silently
    stayed disabled whenever ATHENA_EMBEDDER_URL was set without a key.
    Asserts the gate catches this *before* attempting any connection (no DB
    or network needed for this test to be meaningful)."""
    monkeypatch.setenv("ATHENA_EMBEDDER_URL", "http://embedder.example.com")
    monkeypatch.delenv("ATHENA_EMBEDDER_API_KEY", raising=False)
    app = FastAPI()
    config = AthenaConfig(db_url="postgresql://user:pass@localhost/db")

    await _maybe_add_document_loader(app, config, {})

    assert app.state.document_loader is None


@pytest.mark.asyncio
async def test_document_loader_disabled_when_db_url_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATHENA_EMBEDDER_URL", "http://embedder.example.com")
    monkeypatch.setenv("ATHENA_EMBEDDER_API_KEY", "fake-key")
    app = FastAPI()
    config = AthenaConfig(db_url="")

    await _maybe_add_document_loader(app, config, {})

    assert app.state.document_loader is None
