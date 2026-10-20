"""
FastAPI application factory.

COMPONENT INIT ORDER (lifespan):
1. Logging — everything else logs, so configure first.
2. AthenaConfig — load once from env, threaded into core.lifecycle.lifespan()
   so every component below reads from one validated settings object instead
   of calling os.environ directly.
3. Model — MockModel by default; CloudModel when anthropic_api_key is set.
4. Tool registry — register all built-in tools.
5. Agent — wraps model + registry.
6. Runtime — wraps model for single-turn chat; gets a ConversationMemory if
   db_url is reachable, so /v1/chat persists per-session history.
7. (Optional) DocumentLoader — only if DB URL, embedder URL, and embedder
   API key are all configured.

WHY LIFESPAN OVER on_event:
FastAPI's @app.on_event("startup") is deprecated. The lifespan context
manager gives explicit startup/shutdown symmetry and is compatible with
pytest's async test client (httpx.AsyncClient with app=).

WHY MOCK MODEL DEFAULT:
The server starts and serves traffic without any API key. This is critical
for Docker Compose demos, CI smoke tests, and onboarding. Real AI calls
are unlocked by setting ANTHROPIC_API_KEY.

WHY core.lifecycle.lifespan() NOW:
Previously this module built every component ad hoc, reading os.environ
directly, with no shutdown handling beyond a log line. AthenaConfig +
core.lifecycle.lifespan() were built (P0) specifically for this — a single
validated settings object plus ordered, failure-tolerant component shutdown
(every registered component's close()/aclose() is awaited in reverse order).
This function now registers each component into that registry as it's built,
so shutdown is no longer a no-op.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from athenai.agents.agent import Agent
from athenai.core.config import AthenaConfig
from athenai.core.lifecycle import lifespan as core_lifespan
from athenai.gateway.routes import router
from athenai.models.mock import MockModel
from athenai.observability.logger import configure_logging, get_logger
from athenai.runtime.pipeline import AthenaRuntime
from athenai.tools.calculator import CalculatorTool
from athenai.tools.http import HTTPTool
from athenai.tools.registry import ToolRegistry

_logger = get_logger(__name__)


def _build_model(config: AthenaConfig) -> object:
    """Pick the model backend for this process.

    Falls back to MockModel (not a raised error) if config.anthropic_api_key
    is set but CloudModel construction fails — a bad key or missing
    dependency should degrade the server to mock mode, not crash startup.

    model_name still comes from the raw ANTHROPIC_MODEL env var, not
    AthenaConfig: config.model_registry models routing *roles* (fast/smart/
    etc., for athenai.routing, which this module doesn't use), not "the
    concrete Anthropic model id to construct CloudModel with" — forcing
    that mapping would be inventing a meaning config.py's author didn't
    give it.
    """
    api_key = config.anthropic_api_key.strip()
    if api_key:
        try:
            from athenai.models.cloud import CloudModel

            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
            _logger.info("model.init", backend="anthropic", model=model_name)
            return CloudModel(api_key=api_key, model_name=model_name)
        except Exception as exc:
            _logger.warning("model.fallback_to_mock", reason=str(exc))
    else:
        _logger.info("model.init", backend="mock", reason="anthropic_api_key not set")
    return MockModel()


def _build_tool_registry() -> ToolRegistry:
    """Register built-in tools. HTTPTool is only added if ATHENA_ALLOWED_DOMAINS
    is set, since an HTTP tool with no domain allowlist would let the agent
    fetch arbitrary URLs — safer to omit the tool than register it wide open.

    Stays on the raw env var rather than config.allowed_http_domains: that
    field expects a JSON-array env value under ATHENA_ALLOWED_HTTP_DOMAINS
    (env_prefix + field name), not the comma-separated ATHENA_ALLOWED_DOMAINS
    this module (and its deployment docs) have always used — switching would
    silently stop parsing existing deployments' config.
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    allowed_domains_raw = os.environ.get("ATHENA_ALLOWED_DOMAINS", "")
    if allowed_domains_raw.strip():
        allowed = [d.strip() for d in allowed_domains_raw.split(",") if d.strip()]
        registry.register(HTTPTool(allowed_domains=allowed))

    _logger.info("tools.registered", tools=registry.list_tools())
    return registry


async def _maybe_add_conversation_memory(config: AthenaConfig) -> object | None:
    """Build a ConversationMemory if db_url is reachable, else return None.

    Mirrors _maybe_add_document_loader's degrade-not-crash pattern: a down
    Postgres at startup should not prevent /v1/chat from serving traffic in
    mock mode, it should just mean chat is stateless (today's behavior)
    rather than session-persistent.
    """
    try:
        from athenai.memory.conversation import ConversationMemory

        memory = await ConversationMemory.create(config.db_url)
        _logger.info("memory.enabled", db_url=config.db_url)
        return memory
    except Exception as exc:
        _logger.info("memory.disabled", reason=str(exc))
        return None


async def _maybe_add_document_loader(
    app: FastAPI, config: AthenaConfig, registry: dict[str, object]
) -> None:
    """Wire up the RAG document loader, or leave app.state.document_loader as
    None if unconfigured or unreachable.

    Both the "not configured" and "failed to connect" paths set the same
    None state rather than raising — RAG is an optional subsystem, and a
    down Postgres/embedder at startup should not prevent the rest of the
    gateway (chat, agents) from serving traffic. The /v1/documents/ingest
    and /v1/documents/search routes check for this None and return 503
    rather than crashing.

    ATHENA_EMBEDDER_API_KEY gates alongside ATHENA_DB_URL/ATHENA_EMBEDDER_URL
    rather than being read inside the try block: CloudEmbedder.__init__
    requires api_key with no default, so constructing it without one is a
    startup-time bug, not a reachability failure -- gating here makes a
    missing key degrade to "rag.disabled" (expected, like the other two
    unset-config cases) instead of surfacing as "rag.init_failed" noise.
    """
    db_url = config.db_url.strip()
    embedder_url = os.environ.get("ATHENA_EMBEDDER_URL", "").strip()
    embedder_api_key = os.environ.get("ATHENA_EMBEDDER_API_KEY", "").strip()

    if not db_url or not embedder_url or not embedder_api_key:
        _logger.info(
            "rag.disabled",
            reason=(
                "ATHENA_DB_URL, ATHENA_EMBEDDER_URL, or ATHENA_EMBEDDER_API_KEY not set"
            ),
        )
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
        embedder = CloudEmbedder(api_key=embedder_api_key, base_url=embedder_url)
        retriever = PgVectorRetriever(pool=pool)
        loader = DocumentLoader(
            parser=DocumentParser(),
            chunker=SlidingWindowChunker(),
            embedder=embedder,
            retriever=retriever,
        )
        app.state.document_loader = loader
        registry["rag_retriever"] = retriever
        _logger.info("rag.enabled", db_url=db_url)
    except Exception as exc:
        _logger.warning("rag.init_failed", reason=str(exc))
        app.state.document_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook (see module docstring for init order). Builds one
    model instance and shares it between `app.state.runtime` (single-turn
    chat) and `app.state.agent` (multi-step tool use) so both entry points
    hit the same backend and rate limits.

    Wraps the whole body in core.lifecycle.lifespan(config): every component
    with a close()/aclose() (today: ConversationMemory, and the asyncpg pool
    underneath the RAG document loader) gets registered into it as it's
    built, so shutdown actually tears them down in reverse order instead of
    the previous no-op "log and exit".
    """
    configure_logging()
    _logger.info("athena.starting")

    config = AthenaConfig()

    app.state.gateway_api_key = config.gateway_api_key
    if not config.gateway_api_key:
        _logger.warning(
            "auth.disabled",
            reason="ATHENA_GATEWAY_API_KEY not set -- all endpoints are unauthenticated",
        )
    else:
        _logger.info("auth.enabled")

    async with core_lifespan(config) as registry:
        model = _build_model(config)
        tool_registry = _build_tool_registry()
        memory = await _maybe_add_conversation_memory(config)
        if memory is not None:
            registry["conversation_memory"] = memory

        app.state.runtime = AthenaRuntime(
            model=model,
            max_concurrent=int(os.environ.get("ATHENA_MAX_CONCURRENT", "20")),
            memory=memory,
        )
        app.state.agent = Agent(
            model=model,
            tool_registry=tool_registry,
            max_iterations=int(os.environ.get("ATHENA_MAX_AGENT_ITER", "10")),
        )

        await _maybe_add_document_loader(app, config, registry)

        _logger.info("athena.ready")
        yield
        _logger.info("athena.shutdown")


def create_app(cors_allowed_origins: list[str] | None = None) -> FastAPI:
    """Build the FastAPI app instance. Call once; the module-level `app`
    below is what uvicorn actually serves.

    cors_allowed_origins defaults to parsing ATHENA_CORS_ALLOWED_ORIGINS
    (comma-separated, same convention as ATHENA_ALLOWED_DOMAINS elsewhere
    in this module) when not passed explicitly -- tests pass a list
    directly instead of touching process env vars. An empty list (the
    default when unset) means no cross-origin requests are allowed at all:
    secure-by-default, matching the HTTPTool allowlist's "safer to omit
    than register wide open" reasoning. The previous allow_origins=["*"]
    combined with allow_credentials=True let any website call this API
    using a logged-in user's session/credentials.
    """
    if cors_allowed_origins is None:
        raw = os.environ.get("ATHENA_CORS_ALLOWED_ORIGINS", "")
        cors_allowed_origins = [o.strip() for o in raw.split(",") if o.strip()]

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
        allow_origins=cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


# Module-level app instance used by uvicorn
app = create_app()
