# AthenaAI — Modular AI Runtime

> A production-grade Python runtime for building reliable, observable, tool-using
> AI applications — autonomous agents, RAG pipeline, intelligent model routing,
> token-budgeted context management, persistent memory, typed tool execution,
> structured logging, and Prometheus metrics. Ships as a FastAPI service with
> a single `docker compose up`.

---

## Quick Demo

```bash
# Clone and start (no API key needed — runs in mock mode)
git clone https://github.com/shreyshringare/AthenaAI.git
cd AthenaAI
docker compose up --build -d

# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'

# Autonomous agent with calculator tool
curl -X POST http://localhost:8000/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Calculate compound interest on $1000 at 5% for 3 years","tools":["calculator"]}'

# Streaming (Server-Sent Events)
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}]}'

# Prometheus metrics
curl http://localhost:8000/metrics

# Grafana dashboards → http://localhost:3000  (admin / athena)
```

Enable real Claude AI by setting your API key:

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build -d
```

Or run locally without Docker:

```bash
uv pip install -e ".[dev]"
make dev     # starts uvicorn on :8000
make demo    # runs the full demo sequence
```

---

## Architecture

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway (P9)                     │
│  /v1/chat  /v1/chat/stream  /v1/agents/run  /health  /metrics  │
└───────────────────┬────────────────────────┬────────────────────┘
                    │                        │
          ┌─────────▼──────────┐   ┌────────▼───────────┐
          │  AthenaRuntime (P9)│   │    Agent (P8)       │
          │  semaphore-bounded │   │  state machine      │
          │  model call        │   │  tool loop          │
          └─────────┬──────────┘   └────────┬────────────┘
                    │                        │
          ┌─────────▼──────────────────────▼──────────────┐
          │               Model Router (P3)                │
          │   policy scoring → circuit-aware selection     │
          └──────┬──────────────────────────┬─────────────┘
                 │                          │
        ┌────────▼──────┐         ┌────────▼──────────┐
        │  MockModel    │         │   CloudModel       │
        │  (no API key) │         │   (Anthropic)      │
        └───────────────┘         └───────────────────-┘
                 │
     ┌───────────┼───────────────────────┐
     │           │                       │
┌────▼────┐ ┌───▼──────────┐   ┌────────▼───────┐
│ Context │ │ Memory (P5)  │   │  RAG (P6)       │
│ Engine  │ │ conversation │   │  parser→chunker │
│  (P4)   │ │ semantic     │   │  →embedder      │
│  token  │ │ pgvector     │   │  →pgvector      │
│  budget │ │ summary      │   │  →reranker      │
└─────────┘ └──────────────┘   └────────────────┘
                 │
         ┌───────▼────────┐
         │ Tool System(P7)│
         │ calculator     │
         │ http (SSRF-    │
         │  protected)    │
         │ sql (readonly) │
         └────────────────┘
                 │
     ┌───────────▼───────────┐
     │   Observability (P10) │
     │  structlog + Prometheus│
     └───────────────────────┘
```

---

## Phase Status

| Phase | Scope | Status | Tests |
|-------|-------|--------|-------|
| P0 | Core types, protocols, config, lifecycle | ✅ Complete | 15 |
| P1 | Model layer (Mock, Cloud, Local, Registry) | ✅ Complete | 9 |
| P2 | Resilience (retry, timeout, circuit breaker, rate limiter) | ✅ Complete | 10 |
| P3 | Model router (scoring, policies, complexity routing) | ✅ Complete | 8 |
| P4 | Context engine + token budget manager | ✅ Complete | 8 |
| P5 | Memory layer (conversation, summary, semantic, working) | ✅ Complete | 11 |
| P6 | RAG pipeline (parser, chunker, embedder, retriever, reranker) | ✅ Complete | 16 |
| P7 | Tool system (validator, calculator, SQL, HTTP, registry) | ✅ Complete | 33 |
| P8 | Agent runtime (state machine, executor, parallel tools) | ✅ Complete | 13 |
| P9 | FastAPI gateway + streaming SSE + full pipeline | ✅ Complete | 9 |
| P10 | Observability (structlog, Prometheus metrics) | ✅ Complete | — |
| P14 | Docker Compose + Makefile + Prometheus + Grafana | ✅ Complete | — |

**Total: 112 unit tests passing.**

---

## What's Implemented

### P0 — Core types
Frozen dataclasses for `Message`, `AIRequest`, `AIResponse`, `TokenUsage`,
`TraceSpan`, `RoutingContext`. Structural `Protocol` interfaces for `Model`,
`Tool`, `MemoryStore`, `Retriever`, `CacheBackend`, `EmbedderProtocol`.
`AthenaConfig` via `pydantic-settings` with env-var override.

### P1 — Model layer
- `MockModel` — deterministic echo, zero API keys, streaming support
- `CloudModel` — async httpx, retries, 429 → `RateLimitError`, 5xx → `ModelUnavailableError`
- `LocalModel` — Ollama `/api/generate`
- `ModelRegistry` — config-driven, O(1) role lookup, health check

### P2 — Resilience
- `RetryPolicy` — exponential backoff + full jitter
- `CircuitBreaker` — CLOSED/OPEN/HALF_OPEN, `asyncio.Lock` for CAS transitions
- `TokenBucketRateLimiter` — per-user token buckets
- `with_timeout` — asyncio task timeout wrapper

### P3 — Model router
Weighted quality/cost/latency scoring. Complexity classification (LOW/MEDIUM/HIGH).
Open circuits skipped automatically. Returns frozen `RoutingDecision`.

### P4 — Context engine
`TokenBudgetManager` with hard ceilings per bucket (system/conversation/memory/rag/tools).
Raises `ContextOverflowError` — never silent truncation. `ContextEngine.build()` calls
memory and retriever in parallel via `asyncio.gather`.

### P5 — Memory layer
- `ConversationMemory` — asyncpg-backed message store
- `SummaryMemory` — compresses when > 20 raw messages
- `SemanticMemory` — pgvector cosine search (`<=>` operator)
- `WorkingMemory` — in-process dict for scratchpad state

### P6 — RAG pipeline
`DocumentParser` → `SlidingWindowChunker` (token-aware, sliding overlap) →
`CloudEmbedder` (batch-aware, OpenAI-compatible) → `PgVectorRetriever`
(pgvector `<=>` cosine search) → `CosineReranker`. `DocumentLoader.ingest()`
is idempotent on `document_id`.

### P7 — Tool system
- `ToolValidator` — JSON Schema validation first, permission check second
- `CalculatorTool` — AST walk only, never `eval()`
- `SQLTool` — read-only asyncpg transaction, rejects non-SELECT at string + DB level
- `HTTPTool` — domain allowlist with subdomain support, SSRF protection
- `ToolRegistry` — O(1) name lookup, schema export for model system prompt

### P8 — Agent runtime
`AgentStatus` state machine with legal-transition validation. `AgentExecutor`
runs the model-tool loop with hard `max_iterations` cap. Parallel tool execution
via `asyncio.gather`. Text-protocol tool call parsing (`TOOL_CALL: {json}`).
Structured tool call metadata path for real Anthropic API responses.

### P9 — FastAPI gateway
- `POST /v1/chat` — single-turn chat with usage stats
- `POST /v1/chat/stream` — Server-Sent Events streaming
- `POST /v1/agents/run` — autonomous multi-step agent
- `POST /v1/documents/ingest` — RAG document ingestion
- `GET /health` — liveness probe
- `GET /ready` — readiness probe (model health check)
- `GET /metrics` — Prometheus text format
- `GET /docs` — interactive Swagger UI

### P10 — Observability
`structlog` with ISO timestamps, bound context, JSON/console renderer switch
via `LOG_FORMAT=json`. Prometheus counters and histograms:
`athena_requests_total`, `athena_model_latency_seconds`,
`athena_tokens_total`, `athena_tool_calls_total`,
`athena_agent_iterations_total`, `athena_rag_chunks_retrieved_total`.

### P14 — Docker
Multi-stage `Dockerfile` (builder → runtime, non-root user, healthcheck).
`docker-compose.yml` with `api + postgres(pgvector) + prometheus + grafana`.
`migrations/init.sql` creates all tables and vector indexes on first start.
`Makefile` with `install / lint / type / test / up / down / demo` targets.

---

## Dev Setup

```bash
# Install (requires Python 3.12+, uv)
uv pip install -e ".[dev]"

# Lint + type check
make lint
make type

# Unit tests (no database needed)
make test-unit

# Integration tests (requires postgres on port 5433)
make test-integration

# Run API locally
make dev

# Full demo sequence
make demo
```

---

## API Reference

### `POST /v1/chat`

```json
{
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "model": "default",
  "user_id": "alice",
  "session_id": "session-1"
}
```

Response:

```json
{
  "content": "4",
  "model": "mock",
  "trace_id": "...",
  "request_id": "...",
  "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
}
```

### `POST /v1/agents/run`

```json
{
  "task": "Calculate the area of a circle with radius 7",
  "tools": ["calculator"],
  "max_iterations": 10
}
```

Response includes `final_answer`, `steps[]` (each with `model_response`,
`tool_calls`, `tool_results`), `total_iterations`, `status`.

### `POST /v1/chat/stream`

Same request body as `/v1/chat`. Response is Server-Sent Events:

```
data: Hello
data:  there
data: !
data: [DONE]
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| HTTP framework | FastAPI + uvicorn |
| Async runtime | asyncio (no threads) |
| Database | PostgreSQL + pgvector |
| Async DB driver | asyncpg |
| HTTP client | httpx (async) |
| Config | pydantic-settings |
| Logging | structlog |
| Metrics | prometheus-client |
| Containerisation | Docker multi-stage + Compose |
| Tests | pytest-asyncio |
| Linting | ruff |
| Type checking | mypy (strict) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Enable real Claude AI calls |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Model to use |
| `ATHENA_DB_URL` | — | PostgreSQL DSN for memory + RAG |
| `ATHENA_EMBEDDER_URL` | — | Embeddings API base URL |
| `ATHENA_ALLOWED_DOMAINS` | — | Comma-separated domains for HTTPTool |
| `ATHENA_MAX_CONCURRENT` | `20` | Max concurrent model calls |
| `ATHENA_MAX_AGENT_ITER` | `10` | Max agent loop iterations |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | — | Set to `json` for production logs |
