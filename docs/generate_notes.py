"""
Generate phase-wise AthenaAI study notes as a PDF.
Run: python docs/generate_notes.py
Output: docs/AthenaAI_Notes.pdf
"""

import sys
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    print("reportlab not found. Run: pip install reportlab")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontSize=28,
    textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=6,
)
SUBTITLE = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=13,
    textColor=colors.HexColor("#4a4e69"),
    spaceAfter=20,
    alignment=1,
)
PHASE_HEADING = ParagraphStyle(
    "PhaseHeading",
    parent=styles["Heading1"],
    fontSize=18,
    textColor=colors.HexColor("#ffffff"),
    backColor=colors.HexColor("#16213e"),
    borderPad=8,
    spaceAfter=10,
    spaceBefore=20,
    leftIndent=-10,
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=13,
    textColor=colors.HexColor("#0f3460"),
    spaceBefore=10,
    spaceAfter=4,
)
H3 = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontSize=11,
    textColor=colors.HexColor("#533483"),
    spaceBefore=6,
    spaceAfter=2,
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#2d2d2d"),
    spaceAfter=4,
)
CODE = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=8,
    backColor=colors.HexColor("#f4f4f4"),
    borderColor=colors.HexColor("#cccccc"),
    borderWidth=1,
    borderPad=6,
    leading=11,
    fontName="Courier",
    spaceAfter=6,
    spaceBefore=4,
)
WHY = ParagraphStyle(
    "Why",
    parent=styles["Normal"],
    fontSize=10,
    textColor=colors.HexColor("#1a5276"),
    backColor=colors.HexColor("#eaf4fb"),
    borderColor=colors.HexColor("#5dade2"),
    borderWidth=1,
    borderPad=6,
    leading=14,
    spaceAfter=8,
    spaceBefore=4,
)
GATE = ParagraphStyle(
    "Gate",
    parent=styles["Normal"],
    fontSize=10,
    textColor=colors.HexColor("#1e8449"),
    backColor=colors.HexColor("#eafaf1"),
    borderColor=colors.HexColor("#58d68d"),
    borderWidth=1,
    borderPad=6,
    leading=14,
    spaceAfter=8,
)
RULE = ParagraphStyle(
    "Rule",
    parent=styles["Normal"],
    fontSize=10,
    textColor=colors.HexColor("#922b21"),
    backColor=colors.HexColor("#fdedec"),
    borderColor=colors.HexColor("#e74c3c"),
    borderWidth=1,
    borderPad=6,
    leading=14,
    spaceAfter=6,
)

def sp(n=1):
    return Spacer(1, n * 0.3 * cm)

def hr():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=6)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def cover_page():
    return [
        sp(4),
        Paragraph("AthenaAI", TITLE),
        Paragraph("Modular AI Runtime — Phase-by-Phase Study Notes", SUBTITLE),
        sp(1),
        hr(),
        sp(1),
        Paragraph(
            "15 phases (P0–P14) · Python 3.12+ · uv monorepo · FastAPI · asyncpg · pgvector · Redis · OpenTelemetry",
            SUBTITLE,
        ),
        sp(2),
        Paragraph(
            "<b>Architecture layers:</b><br/>"
            "1. <b>AthenaRuntime</b> — central engine (model selection, RAG, memory, tools, agents, resilience, observability)<br/>"
            "2. <b>AthenaGateway</b> — FastAPI public surface (auth, rate limiting, SSE streaming)<br/>"
            "3. <b>AthenaEval</b> — evaluation harness (correctness, latency, cost, retrieval quality)",
            BODY,
        ),
        sp(1),
        Paragraph(
            "<b>Single entry point principle:</b><br/>"
            "<font name='Courier'>response = await runtime.execute(request)</font><br/>"
            "Caller never knows if a model was selected, docs retrieved, tools called, retries happened, or traces recorded.",
            WHY,
        ),
        PageBreak(),
    ]


def operating_loop_page():
    return [
        Paragraph("Operating Loop (Every Phase)", PHASE_HEADING),
        sp(1),
        Paragraph(
            "Every phase follows this exact sequence — no skipping:",
            BODY,
        ),
        sp(0.5),
        Table(
            [
                ["Step", "Action", "Command / Check"],
                ["1. PLAN", "List files to create + why", "—"],
                ["2. IMPLEMENT", "Create files, smallest dep first", "—"],
                ["3. LINT/TYPE", "Run ruff + mypy", "uv run mypy src/ &amp;&amp; uv run ruff check src/"],
                ["4. TEST", "Run gate tests", "uv run pytest tests/&lt;module&gt;/ -v"],
                ["5. GATE", "Verify gate condition", "All tests green"],
                ["6. COMMIT", "Git commit", "git add -A &amp;&amp; git commit -m 'feat(scope): ...'"],
                ["7. REPORT", "Print phase complete", "—"],
                ["8. LOOP", "Start next phase", "—"],
            ],
            colWidths=[2.5 * cm, 5 * cm, 8 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(1),
        Paragraph(
            "<b>Non-negotiable rules enforced every file:</b>",
            H2,
        ),
        Paragraph("Protocol over ABC — ALL interfaces in core/protocols.py as typing.Protocol", RULE),
        Paragraph("No raw dict between components — use @dataclass(frozen=True)", RULE),
        Paragraph("No isinstance chains — use match statement on tagged types", RULE),
        Paragraph("No threading.Thread for I/O — async def + await everywhere", RULE),
        Paragraph("No eval() in CalculatorTool — use restricted AST walk", RULE),
        Paragraph("No silent truncation — raise ContextOverflowError", RULE),
        Paragraph("No recursive DFS in AgentPlanner — explicit stack only", RULE),
        Paragraph("No TODO stubs — implement fully or skip the file", RULE),
        PageBreak(),
    ]


def p0_page():
    return [
        Paragraph("P0 — Core Types + Project Scaffold", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Project installs, types resolve, mock runtime executes.", BODY),
        sp(0.5),
        Paragraph("Files Created", H2),
        Table(
            [
                ["File", "Purpose"],
                ["pyproject.toml", "uv project root — all 20+ deps declared"],
                ["src/athenai/__init__.py", "__version__ = '0.1.0'"],
                ["core/exceptions.py", "8 custom exceptions: ContextOverflowError, ModelUnavailableError, ToolDeniedError, ToolTimeoutError, PolicyViolationError, EmbeddingError, CircuitOpenError, RateLimitError"],
                ["core/types.py", "6 frozen dataclasses: Message, TokenUsage, AIRequest, AIResponse, TraceSpan, RoutingContext"],
                ["core/protocols.py", "7 Protocols: Model, StreamingModel, MemoryStore, Retriever, Tool, CacheBackend, EmbedderProtocol"],
                ["core/config.py", "AthenaConfig(BaseSettings) — env-driven, ATHENA_ prefix"],
                ["core/lifecycle.py", "@asynccontextmanager lifespan() — component registry as dict[str, Any]"],
            ],
            colWidths=[5 * cm, 10.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(0.5),
        Paragraph("Key Design Decisions", H2),
        Paragraph(
            "<b>WHY Protocol over ABC:</b> Structural subtyping — MockModel satisfies Model without inheriting from it. "
            "Decouples interface from implementation, eliminates circular imports, no MRO overhead on hot path.",
            WHY,
        ),
        Paragraph(
            "<b>WHY frozen=True dataclasses:</b> Requests flow through routing, context, model, observability pipeline. "
            "Immutability prevents mid-pipeline mutation and makes types safe as dict keys for caching.",
            WHY,
        ),
        Paragraph(
            "<b>WHY asynccontextmanager for lifecycle:</b> Single function owns startup/shutdown order explicitly. "
            "No scattered try/finally blocks. Cleanup guaranteed even if startup fails mid-way.",
            WHY,
        ),
        Paragraph(
            "<b>WHY ContextOverflowError (not silent truncation):</b> Silent drops cause hallucination with no "
            "observable cause. Hard ceiling forces caller to make an explicit decision about what to sacrifice.",
            WHY,
        ),
        Paragraph("Gate Tests (tests/unit/test_types.py)", H2),
        Paragraph(
            "<b>Gate:</b> Instantiate every @dataclass(frozen=True) — no exception. "
            "Assert frozen: assigning to field raises FrozenInstanceError. "
            "AIRequest with empty messages list raises ValueError. "
            "Import all Protocols — no ImportError. "
            "AthenaConfig loads from environment variables correctly.",
            GATE,
        ),
        Paragraph("<font name='Courier'>uv run pytest tests/unit/test_types.py -v</font>", CODE),
        PageBreak(),
    ]


def p1_page():
    return [
        Paragraph("P1 — Model Layer", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Mock and cloud model adapters working, registry config-driven.", BODY),
        sp(0.5),
        Paragraph("Files Created", H2),
        Table(
            [
                ["File", "Key Implementation"],
                ["models/base.py", "ModelRequest, ModelResponse frozen dataclasses"],
                ["models/mock.py", "MockModel: echo prompt back, deterministic token counts, stream() yields chars with 10ms sleep"],
                ["models/cloud.py", "CloudModel: Anthropic API via httpx.AsyncClient. 429 → RateLimitError, 5xx → ModelUnavailableError"],
                ["models/local.py", "LocalModel: Ollama /api/generate via httpx"],
                ["models/registry.py", "ModelRegistry: config-driven, get(role), list_roles(), health_check(role)"],
            ],
            colWidths=[4 * cm, 11.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(0.5),
        Paragraph("Key Design Decisions", H2),
        Paragraph(
            "<b>WHY MockModel:</b> Deterministic, no API key needed. Tests run in CI without credentials. "
            "Echo-back makes assertions trivial — expected output is derivable from input.",
            WHY,
        ),
        Paragraph(
            "<b>WHY httpx over Anthropic SDK:</b> httpx.AsyncClient is a standard async HTTP client with "
            "predictable timeout/retry behaviour. SDK hides HTTP details that matter for resilience (status codes, "
            "connection reuse, timeout propagation). respx can mock it cleanly in tests.",
            WHY,
        ),
        Paragraph("Gate Tests (tests/unit/test_models.py)", H2),
        Paragraph(
            "MockModel.generate() returns ModelResponse with non-empty content. "
            "MockModel.stream() yields at least 3 tokens. "
            "ModelRegistry.get('fast') returns configured model. "
            "ModelRegistry.get('nonexistent') raises KeyError. "
            "CloudModel + mock 401 → ModelUnavailableError. "
            "CloudModel + mock 429 → RateLimitError. "
            "Registry round-trip: register mock as 'fast' + 'reasoning', get both.",
            GATE,
        ),
        Paragraph("<font name='Courier'>uv run pytest tests/unit/test_models.py -v</font>", CODE),
        PageBreak(),
    ]


def p2_page():
    return [
        Paragraph("P2 — Resilience", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Retry, timeout, circuit breaker, rate limiter all working and testable.", BODY),
        sp(0.5),
        Paragraph("Files Created", H2),
        Table(
            [
                ["File", "Key Implementation"],
                ["resilience/retry.py", "RetryPolicy(frozen=True): max_attempts, base_delay_s, max_delay_s, jitter: bool. async with_retry(fn, policy) — exponential backoff + full jitter. Reraises last exception after exhaustion."],
                ["resilience/timeout.py", "async with_timeout(fn, seconds) — asyncio.wait_for wrapper, raises ToolTimeoutError on asyncio.TimeoutError"],
                ["resilience/circuit_breaker.py", "CircuitBreaker: asyncio.Lock for CAS state transitions. States: CLOSED/OPEN/HALF_OPEN. failure_threshold, cooldown_s, half_open_probe_count."],
                ["resilience/rate_limiter.py", "TokenBucketRateLimiter: asyncio.Lock guards token count + last_refill atomically. async acquire() raises RateLimitError when bucket empty."],
            ],
            colWidths=[4.5 * cm, 11 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(0.5),
        Paragraph("Key Design Decisions", H2),
        Paragraph(
            "<b>WHY asyncio.Lock for CAS in CircuitBreaker:</b> Without a lock, two concurrent tasks could both read "
            "CLOSED, both count failures, and both attempt the CLOSED→OPEN transition independently. The lock makes "
            "read-modify-write atomic, so exactly one transition fires regardless of concurrency level.",
            WHY,
        ),
        Paragraph(
            "<b>WHY one Lock covers both fields in TokenBucketRateLimiter:</b> token_count and last_refill must be "
            "updated together — if another coroutine reads between the refill and the decrement, it sees an "
            "inconsistent state. One lock covers both atomically.",
            WHY,
        ),
        Paragraph(
            "<b>WHY full jitter on retry backoff:</b> Thundering herd — without jitter, all N failing clients retry "
            "at the same instant after backoff, causing correlated load spikes. Full jitter spreads retries randomly "
            "across [0, delay] so the aggregate load stays flat.",
            WHY,
        ),
        Paragraph("Gate Tests", H2),
        Paragraph(
            "<b>Circuit breaker:</b> 2 failures → OPEN. Sleep cooldown → HALF_OPEN. 1 success → CLOSED. "
            "20 concurrent tasks all calling record_failure() → exactly one CLOSED→OPEN transition (test with counter).",
            GATE,
        ),
        Paragraph(
            "<b>Rate limiter:</b> capacity=5: 5 acquire() succeed, 6th raises RateLimitError. "
            "Drain, sleep 0.1s, 5 more succeed (refilled). 10 concurrent tasks with capacity=3 → exactly 3 succeed.",
            GATE,
        ),
        Paragraph(
            "<font name='Courier'>uv run pytest tests/unit/test_circuit_breaker.py tests/unit/test_rate_limiter.py -v</font>",
            CODE,
        ),
        PageBreak(),
    ]


def p3_page():
    return [
        Paragraph("P3 — Model Router", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Cost/latency/quality-aware router selecting models by RoutingContext.", BODY),
        sp(0.5),
        Paragraph("Files Created", H2),
        Table(
            [
                ["File", "Key Implementation"],
                ["routing/policies.py", "RoutingPolicy(frozen=True): quality_weight, cost_weight, latency_weight, max_cost_usd, max_latency_ms. default() classmethod."],
                ["routing/scorer.py", "ModelScorer: weighted score = quality*w_q + (1/cost)*w_c + (1/latency)*w_l from ModelMetadata"],
                ["routing/router.py", "ModelRouter.select(context, policy) → RoutingDecision(frozen=True). Classifies LOW/MEDIUM/HIGH complexity from token estimate. Skips circuit-OPEN models."],
            ],
            colWidths=[4 * cm, 11.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(0.5),
        Paragraph("Key Design Decisions", H2),
        Paragraph(
            "<b>WHY weighted scoring over hard cutoff routing:</b> Hard cutoffs (if latency &lt; X: use fast model) "
            "require manual tuning per deployment. Weighted scoring lets the routing policy express tradeoffs "
            "declaratively — cost-sensitive configs increase cost_weight, quality-critical configs increase "
            "quality_weight — without changing routing code.",
            WHY,
        ),
        Paragraph(
            "<b>Complexity classification:</b> token_estimate &lt; 500 = LOW, &lt; 2000 = MEDIUM, else HIGH. "
            "LOW routes to 'fast' model, HIGH routes to 'reasoning' model regardless of other weights.",
            BODY,
        ),
        Paragraph("Gate Tests (tests/unit/test_router.py)", H2),
        Paragraph(
            "'Summarise this sentence' → LOW → fast model. "
            "'Design a distributed payment system' → HIGH → reasoning. "
            "'Translate to French' (quality low, cost high) → fast model. "
            "All models circuit-OPEN → ModelUnavailableError. "
            "estimated_cost_usd positive and &lt; max_cost_usd. "
            "Custom weights → scorer picks different model.",
            GATE,
        ),
        Paragraph("<font name='Courier'>uv run pytest tests/unit/test_router.py -v</font>", CODE),
        PageBreak(),
    ]


def p4_page():
    return [
        Paragraph("P4 — Context Engine + Token Budget", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Token-budgeted context assembly, parallel memory+RAG retrieval.", BODY),
        sp(0.5),
        Paragraph("Files Created", H2),
        Table(
            [
                ["File", "Key Implementation"],
                ["context/budget.py", "TokenBudgetManager: allocations dict[str, int]. allocate(key, tokens) returns actually allocated, raises ContextOverflowError if over hard ceiling."],
                ["context/ranking.py", "RelevanceRanker: cosine similarity via numpy between query embedding and chunk embeddings. Returns top-k sorted by score."],
                ["context/packing.py", "ContextPacker: inserts in priority order system > conversation > memory > rag > tools. Truncates lowest-priority bucket first on overflow."],
                ["context/engine.py", "ContextEngine.build() — asyncio.gather(memory, rag) parallel retrieval. Calls packer. Returns BuiltContext."],
            ],
            colWidths=[4 * cm, 11.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(0.5),
        Paragraph("Key Design Decisions", H2),
        Paragraph(
            "<b>WHY asyncio.gather for memory+RAG:</b> Memory retrieval (DB query) and RAG retrieval (vector search) "
            "have no data dependency — they can run in parallel. If memory takes 100ms and RAG takes 200ms, "
            "sequential takes 300ms, parallel takes 200ms. At scale this matters significantly.",
            WHY,
        ),
        Paragraph(
            "<b>WHY hard token ceiling (not soft):</b> Silent truncation hides bugs in token estimation and causes "
            "non-deterministic model behaviour. A hard ceiling (ContextOverflowError) forces the caller to decide "
            "what to drop — system prompt? RAG chunks? conversation history?",
            WHY,
        ),
        Paragraph(
            "<b>Truncation priority (lowest first):</b> tools &lt; rag &lt; memory &lt; conversation &lt; system. "
            "System prompt is never truncated — it defines model behaviour. RAG chunks are cheapest to drop.",
            BODY,
        ),
        Paragraph("Gate Tests (tests/unit/test_context_budget.py)", H2),
        Paragraph(
            "4 buckets within total → no exception, sum equals total. "
            "Allocate past ceiling → ContextOverflowError. "
            "Packer: 3 buckets, last overflows → truncate rag bucket, not system/conversation. "
            "ContextEngine.build() → BuiltContext has all expected fields, total &lt;= ceiling. "
            "Parallel test: memory 100ms + RAG 200ms → build() completes in &lt;250ms (not 300ms).",
            GATE,
        ),
        Paragraph("<font name='Courier'>uv run pytest tests/unit/test_context_budget.py -v</font>", CODE),
        PageBreak(),
    ]


def p5_to_p8_page():
    return [
        Paragraph("P5 — Memory", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Four memory layers, conversation persists across calls.", BODY),
        Table(
            [
                ["Layer", "Backend", "Key Feature"],
                ["WorkingMemory (base.py)", "In-process", "MemoryEntry(frozen=True), MemoryType enum"],
                ["AgentState (working.py)", "In-process mutable", "Holds task, steps, tool_results, status"],
                ["ConversationMemory (conversation.py)", "asyncpg (PostgreSQL)", "add_message(), get_recent(n), clear(session_id)"],
                ["SummaryMemory (summary.py)", "PostgreSQL + model call", "Compresses older msgs when > 20 raw. Returns summary + latest 5 raw."],
                ["SemanticMemory (semantic.py)", "pgvector", "store(fact), search(query_embedding, k) with cosine <=> operator"],
            ],
            colWidths=[4.5 * cm, 3.5 * cm, 7.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Paragraph(
            "<b>WHY PostgreSQL for ConversationMemory (not Redis):</b> Conversation history needs durability across "
            "server restarts. Redis is ephemeral by default. SemanticMemory needs pgvector which is PostgreSQL-native. "
            "Using one DB for both simplifies ops.",
            WHY,
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/integration/test_memory.py -v (needs test DB)",
            GATE,
        ),
        hr(),
        Paragraph("P6 — RAG Pipeline", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Document ingestion + retrieval end-to-end.", BODY),
        Table(
            [
                ["File", "Key Detail"],
                ["rag/parser.py", "PDF via pypdf, TXT/MD as-is. Returns ParsedDocument(text, source, metadata)."],
                ["rag/chunker.py", "SlidingWindowChunker: chunk_size + overlap in tokens. Overlap prevents split-sentence retrieval misses."],
                ["rag/embedder.py", "CloudEmbedder: Anthropic embeddings API via httpx. embed(texts: list[str]) → list[list[float]]. Batch-aware."],
                ["rag/retriever.py", "PgVectorRetriever: cosine search on chunks table. search(query_embedding, k, metadata_filter)."],
                ["rag/reranker.py", "PassthroughReranker (no-op) + CrossEncoderReranker stub. Reranker optional/swappable."],
                ["rag/loader.py", "DocumentLoader.ingest(): parse → chunk → embed → store. Idempotent on document_id."],
            ],
            colWidths=[4 * cm, 11.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/unit/test_chunker.py tests/integration/test_rag_pipeline.py -v",
            GATE,
        ),
        hr(),
        Paragraph("P7 — Tool System", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Tool registry, schema validation, permission check, three tools working.", BODY),
        Paragraph(
            "<b>Validation order:</b> JSON Schema first, then permission check.<br/>"
            "<b>WHY schema before permission:</b> Schema validation is stateless/cheap. Permission check may require "
            "a DB lookup. Fail fast on the cheapest check.",
            WHY,
        ),
        Paragraph(
            "<b>CalculatorTool:</b> Uses ast module — walks AST nodes, allows only Add/Sub/Mul/Div/Pow/BinOp/Num. "
            "Any other node (Import, Call, Attribute) → ToolDeniedError. NEVER eval().<br/>"
            "<b>SQLTool:</b> Read-only asyncpg pool. Rejects non-SELECT. Parameterised queries only.<br/>"
            "<b>HTTPTool:</b> Allowlist-only domains. httpx.AsyncClient 10s timeout.",
            BODY,
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/unit/test_tool_validator.py tests/integration/test_tool_execution.py -v",
            GATE,
        ),
        hr(),
        Paragraph("P8 — Agent Runtime", PHASE_HEADING),
        Paragraph("<b>Goal:</b> Stateful agent with state machine, iterative DAG planner, tool loop.", BODY),
        Table(
            [
                ["Component", "Key Detail"],
                ["AgentStatus (state.py)", "Enum: CREATED/PLANNING/WAITING_TOOL/EXECUTING_TOOL/COMPLETED/FAILED/CANCELLED. transition() validates legal transitions, raises ValueError on illegal."],
                ["AgentPlanner (planner.py)", "plan(task) → ExecutionDAG. topological_sort() uses iterative DFS with explicit stack — no recursion."],
                ["AgentExecutor (executor.py)", "match on AgentStatus to dispatch. max_iterations=10 hard cap. asyncio.gather for parallel independent DAG nodes."],
                ["Agent (agent.py)", "run(task, context, tools) → AgentResult. Composes planner + executor + working memory. Writes AgentRun to DB."],
            ],
            colWidths=[4 * cm, 11.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Paragraph(
            "<b>WHY iterative DFS:</b> Python default recursion limit is 1000. A large DAG with 100+ tool nodes "
            "would hit it. Explicit stack is O(n) space, no stack overflow risk, and easier to add cycle detection.",
            WHY,
        ),
        Paragraph(
            "<b>WHY state machine over while-True loop:</b> Explicit state transitions make illegal paths "
            "compile-checkable. A COMPLETED agent cannot transition to PLANNING — the code enforces it, "
            "not just the comments.",
            WHY,
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/unit/test_agent_state.py tests/integration/test_agent_run.py -v",
            GATE,
        ),
        PageBreak(),
    ]


def p9_to_p14_page():
    return [
        Paragraph("P9 — API Gateway + Full Pipeline", PHASE_HEADING),
        Paragraph("<b>Goal:</b> FastAPI endpoints wired to AthenaRuntime, streaming working.", BODY),
        Table(
            [
                ["Endpoint", "Method", "Description"],
                ["/v1/chat", "POST", "Sync chat → AIResponse JSON"],
                ["/v1/chat/stream", "POST", "SSE streaming — at least 3 events"],
                ["/v1/agents/run", "POST", "Agent task execution"],
                ["/v1/documents", "POST/GET", "Document ingestion (202 + job_id) / list"],
                ["/v1/runs/{run_id}", "GET", "Agent run status (404 if unknown)"],
                ["/health", "GET", "{'status': 'ok'}"],
                ["/ready", "GET", "200 or 503 if DB unreachable"],
                ["/metrics", "GET", "Prometheus text format"],
            ],
            colWidths=[4 * cm, 2.5 * cm, 9 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        Paragraph(
            "<b>BoundedExecutor:</b> asyncio.Semaphore(max_concurrent_model_calls) wraps ALL model calls. "
            "Prevents unbounded gather() from spawning unlimited concurrent API calls.",
            WHY,
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/integration/test_chat_endpoint.py -v",
            GATE,
        ),
        hr(),
        Paragraph("P10 — Observability", PHASE_HEADING),
        Paragraph(
            "<b>Prometheus metrics:</b> requests_total{model,status}, "
            "model_latency_seconds{model} (histogram, buckets: 0.1/0.5/1/2/5/10s), "
            "tokens_total{model,type}, cache_hits_total, tool_calls_total{tool,status}, "
            "rag_chunks_retrieved_total, agent_iterations_total",
            BODY,
        ),
        Paragraph(
            "<b>Trace structure:</b> Every request gets trace_id. Spans: gateway → routing → context_build → model_call. "
            "Each span has name, start_ms, end_ms, attributes dict. Stored in DB for replay.",
            BODY,
        ),
        Paragraph(
            "<b>WHY structured trace vs log tailing:</b> A single AI request involves routing decisions, parallel "
            "retrieval, tool dispatch — filtering thousands of log lines for one request_id is impractical. "
            "A structured span tree lets you replay the exact execution path instantly.",
            WHY,
        ),
        hr(),
        Paragraph("P11 — Failure Scenarios", PHASE_HEADING),
        Paragraph(
            "<b>Model timeout:</b> Primary sleeps 30s, timeout=2s → completes in &lt;3s with fallback. "
            "Trace shows model_timeout, circuit_open, fallback_model_used.<br/>"
            "<b>Circuit recovery:</b> 5 failures → OPEN → cooldown → HALF_OPEN → probe → CLOSED.<br/>"
            "<b>Context overflow:</b> 120% ceiling → ContextOverflowError → 400 (not 500). RAG trimmed first.<br/>"
            "<b>Tool denied:</b> User without 'sql' permission + SQL tool → ToolDeniedError → 403. Metric +1.",
            BODY,
        ),
        Paragraph(
            "<b>Gate:</b> uv run pytest tests/failure/ -v (no new source files — exercises resilience layer)",
            GATE,
        ),
        hr(),
        Paragraph("P12 — Benchmarks", PHASE_HEADING),
        Paragraph(
            "<b>bench_context_engine.py:</b> Sequential vs asyncio.gather retrieval, 100 iterations, mean + p95.<br/>"
            "<b>bench_concurrent_requests.py:</b> 10/100/1000 concurrent execute() calls, p50/p95/p99, req/s, error rate.",
            BODY,
        ),
        Paragraph(
            "<b>Gate criteria:</b> Parallel retrieval &gt;= 30% faster than sequential. "
            "p99 MockModel at 1000 concurrent &lt; 500ms. Zero errors at 10 concurrent. Error rate &lt; 1% at 1000.",
            GATE,
        ),
        hr(),
        Paragraph("P13 — Evaluation Harness", PHASE_HEADING),
        Paragraph(
            "<b>EvalDataset:</b> Loads JSONL with question/expected_answer/documents. Validates on load.<br/>"
            "<b>EvalMetrics:</b> correctness (exact + token overlap), latency_ms, cost_usd, hallucination_score.<br/>"
            "<b>EvalRunner:</b> async batch execute against runtime, concurrency=5. EvalReport.to_markdown() → comparison table.",
            BODY,
        ),
        hr(),
        Paragraph("P14 — README + Docker", PHASE_HEADING),
        Paragraph(
            "<b>docker-compose.yml services:</b> athena-api (:8000), postgres (:5432 + pgvector), "
            "redis (:6379), prometheus (:9090), grafana (:3000).<br/>"
            "<b>Dockerfile:</b> Multi-stage — build stage installs with uv, runtime stage is slim Python 3.12.<br/>"
            "<b>Gate:</b> git clone → make install → make up → curl POST /v1/chat → 200 with content, trace_id, model, usage.",
            BODY,
        ),
        PageBreak(),
    ]


def interview_cheatsheet():
    return [
        Paragraph("Interview Cheat Sheet — Design Decisions", PHASE_HEADING),
        sp(0.5),
        Table(
            [
                ["Pattern", "Where Used", "Why (2-line answer)"],
                ["Protocol over ABC", "core/protocols.py", "Structural subtyping — MockModel satisfies Model without inheriting. Decouples interface from implementation, no circular imports."],
                ["asyncio.gather", "context/engine.py", "Memory + RAG have no data dependency. Parallel cuts latency from 300ms to 200ms at the retrievals stage."],
                ["@dataclass(frozen=True)", "All value types", "Requests flow through entire pipeline. Immutability prevents mid-pipeline mutation; types safe as cache keys."],
                ["Iterative DFS", "agents/planner.py", "Python recursion limit is 1000. Large DAGs hit it. Explicit stack is O(n) space, no stack overflow."],
                ["Token budget hard ceiling", "context/budget.py", "Silent truncation causes hallucination with no observable cause. Hard ceiling forces explicit decision."],
                ["CAS via asyncio.Lock", "resilience/circuit_breaker.py", "TOCTOU — two concurrent tasks could both read CLOSED and both fire OPEN transition. Lock makes it atomic."],
                ["Semaphore on model calls", "runtime/executor.py", "Unbounded asyncio.gather() with 1000 concurrent requests → 1000 API calls. Semaphore caps concurrency."],
                ["Schema before permission check", "tools/validator.py", "Schema validation is stateless/cheap. Permission check may need DB lookup. Fail fast on cheapest check."],
                ["Single execute() entry point", "runtime/runtime.py", "Caller decoupled from all runtime internals. Cache, retry, fallback transparent. Testable in isolation."],
                ["PassthroughReranker", "rag/reranker.py", "Reranking is expensive. Default no-op keeps latency low. Cross-encoder swappable without changing retriever."],
                ["ConversationMemory in PostgreSQL", "memory/conversation.py", "Durability across restarts. Redis is ephemeral. pgvector already requires PostgreSQL — one DB, simpler ops."],
                ["structlog bound context", "observability/logger.py", "Every log line carries trace_id, request_id, user_id automatically. No manual log.info('trace: %s', trace_id) everywhere."],
            ],
            colWidths=[3.5 * cm, 3.5 * cm, 8.5 * cm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        sp(1),
        hr(),
        Paragraph("Request Lifecycle (Request → AIResponse)", H2),
        Paragraph(
            "1. <b>RequestParser</b> — validate, assign request_id, trace_id<br/>"
            "2. <b>PolicyCheck</b> — content policy, user permissions<br/>"
            "3. <b>IntentAnalysis</b> — task classification, complexity estimate<br/>"
            "4. <b>ModelRouter</b> — select model(s), estimate cost + latency<br/>"
            "5. <b>ContextEngine</b> — asyncio.gather(memory, rag) in parallel<br/>"
            "6. <b>AgentRuntime</b> — plan → tool call loop → final answer<br/>"
            "7. <b>LLM Request</b> — via selected Model adapter<br/>"
            "8. <b>Observability</b> — record trace, tokens, cost, latency<br/>"
            "9. <b>AIResponse</b> — content, model, usage, trace_id",
            BODY,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent
    output_path = output_dir / "AthenaAI_Notes.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    story += cover_page()
    story += operating_loop_page()
    story += p0_page()
    story += p1_page()
    story += p2_page()
    story += p3_page()
    story += p4_page()
    story += p5_to_p8_page()
    story += p9_to_p14_page()
    story += interview_cheatsheet()

    doc.build(story)
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    main()
