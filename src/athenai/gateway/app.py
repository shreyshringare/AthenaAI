"""
FastAPI application factory.

COMPONENT INIT ORDER (lifespan):
1. Logging — everything else logs, so configure first.
2. Model — MockModel by default; CloudModel when ANTHROPIC_API_KEY set.
3. Tool registry — register all built-in tools.
4. Agent — wraps model + registry.
5. Runtime — wraps model for single-turn chat.
6. (Optional) DocumentLoader — only if DB + embedder URLs are configured.

WHY LIFESPAN OVER on_event:
FastAPI's @app.on_event("startup") is deprecated. The lifespan context
manager gives explicit startup/shutdown symmetry and is compatible with
pytest's async test client (httpx.AsyncClient with app=).

WHY MOCK MODEL DEFAULT:
The server starts and serves traffic without any API key. This is critical
for Docker Compose demos, CI smoke tests, and onboarding. Real AI calls
are unlocked by setting ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from athenai.agents.agent import Agent
from athenai.gateway.routes import router
from athenai.models.mock import MockModel
from athenai.observability.logger import configure_logging, get_logger
from athenai.runtime.pipeline import AthenaRuntime
from athenai.tools.calculator import CalculatorTool
from athenai.tools.http import HTTPTool
from athenai.tools.registry import ToolRegistry

_logger = get_logger(__name__)


def _build_model() -> object:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        try:
            from athenai.models.cloud import CloudModel

            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
            _logger.info("model.init", backend="anthropic", model=model_name)
            return CloudModel(api_key=api_key, model_name=model_name)
        except Exception as exc:
            _logger.warning("model.fallback_to_mock", reason=str(exc))
    else:
        _logger.info("model.init", backend="mock", reason="ANTHROPIC_API_KEY not set")
    return MockModel()


def _build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    allowed_domains_raw = os.environ.get("ATHENA_ALLOWED_DOMAINS", "")
    if allowed_domains_raw.strip():
        allowed = [d.strip() for d in allowed_domains_raw.split(",") if d.strip()]
        registry.register(HTTPTool(allowed_domains=allowed))

    _logger.info("tools.registered", tools=registry.list_tools())
    return registry


async def _maybe_add_document_loader(app: FastAPI, model: object) -> None:
    db_url = os.environ.get("ATHENA_DB_URL", "").strip()
    embedder_url = os.environ.get("ATHENA_EMBEDDER_URL", "").strip()

    if not db_url or not embedder_url:
        _logger.info("rag.disabled", reason="ATHENA_DB_URL or ATHENA_EMBEDDER_URL not set")
        app.state.document_loader = None
        return

    try:
        import asyncpg

        from athenai.rag.chunker import SlidingWindowChunker
        from athenai.rag.embedder import CloudEmbedder
        from athenai.rag.loader import DocumentLoader
        from athenai.rag.parser import DocumentParser
        from athenai.rag.retriever import PgVectorRetriever

        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        embedder = CloudEmbedder(base_url=embedder_url)
        retriever = PgVectorRetriever(pool=pool)
        loader = DocumentLoader(
            parser=DocumentParser(),
            chunker=SlidingWindowChunker(),
            embedder=embedder,
            retriever=retriever,
        )
        app.state.document_loader = loader
        _logger.info("rag.enabled", db_url=db_url)
    except Exception as exc:
        _logger.warning("rag.init_failed", reason=str(exc))
        app.state.document_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    _logger.info("athena.starting")

    model = _build_model()
    registry = _build_tool_registry()

    app.state.runtime = AthenaRuntime(
        model=model,
        max_concurrent=int(os.environ.get("ATHENA_MAX_CONCURRENT", "20")),
    )
    app.state.agent = Agent(
        model=model,
        tool_registry=registry,
        max_iterations=int(os.environ.get("ATHENA_MAX_AGENT_ITER", "10")),
    )

    await _maybe_add_document_loader(app, model)

    _logger.info("athena.ready")
    yield
    _logger.info("athena.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AthenaAI",
        description=(
            "Production-grade modular AI runtime with autonomous agents, "
            "RAG pipeline, tool use, resilience, and observability."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


# Module-level app instance used by uvicorn
app = create_app()
