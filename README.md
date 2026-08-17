# AthenaAI — Modular AI Runtime

> The infrastructure layer that sits under an AI product — model routing, autonomous agents, RAG pipeline, resilience, observability, and a production API. Built from scratch instead of wrapping LangChain to understand every layer.

**`docker compose up`** starts the full stack. No API key required to demo.

---

## What this is

Most AI projects call an LLM API and call it done. AthenaAI is the layer *underneath* that — the platform an AI engineering team builds internally so product teams can ship AI features without re-solving routing, retries, context management, tool security, and observability every time.

Companies like Stripe, Shopify, and every serious AI team have something like this internally. They don't use LangChain in production — too opaque, too hard to debug, not built for their reliability requirements. They build their own. This is that thing.

```
Product team calls:  POST /v1/chat  or  POST /v1/agents/run
                              ↓
AthenaAI handles:    routing → context → model → tools → memory → observability
```

---

## Quick demo

```bash
git clone https://github.com/shreyshringare/AthenaAI && cd AthenaAI
docker compose up --build -d
```

```bash
# Chat
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'

# Autonomous agent with tool use — shows the full agentic loop
curl -X POST http://localhost:8000/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Calculate compound interest on $1000 at 5% for 3 years","tools":["calculator"]}'

# Streaming (Server-Sent Events)
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain asyncio in one sentence"}]}'

# Prometheus metrics
curl http://localhost:8000/metrics

# Interactive API docs
open http://localhost:8000/docs
```

**Runs in mock mode by default** (no API key, deterministic output, full pipeline active). Enable real Claude:

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up -d
```

---

## Architecture

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway (P9)                        │
│  POST /v1/chat  ·  POST /v1/chat/stream  ·  POST /v1/agents/run│
│  POST /v1/documents/ingest  ·  GET /health  ·  GET /metrics    │
└────────────────────┬──────────────────────┬─────────────────────┘
                     │                      │
           ┌─────────▼────────┐   ┌─────────▼─────────┐
           │  AthenaRuntime   │   │   AgentExecutor    │
           │  semaphore-bound │   │   state machine    │
           │  model call      │   │   parallel tools   │
           └─────────┬────────┘   └─────────┬──────────┘
                     │                      │
           ┌─────────▼──────────────────────▼──────────┐
           │              Model Router (P3)             │
           │  quality/cost/latency scoring              │
           │  circuit-aware selection · complexity cls  │
           └──────┬────────────────────────────────────-┘
                  │
       ┌──────────▼──────────┐
       │     Model Layer     │
       │  Mock · Cloud · Local│
       └──────────┬──────────┘
                  │
    ┌─────────────┼──────────────────────────┐
    │             │                          │
┌───▼────┐  ┌─────▼──────────┐  ┌───────────▼──────┐
│Context │  │  Memory (P5)   │  │   RAG (P6)        │
│Engine  │  │  conversation  │  │   parser→chunker  │
│(P4)    │  │  summary       │  │   →embedder       │
│token   │  │  semantic      │  │   →pgvector       │
│budget  │  │  pgvector      │  │   →reranker       │
└────────┘  └────────────────┘  └──────────────────┘
                  │
          ┌───────▼────────┐
          │  Tools (P7)    │
          │  calculator    │
          │  http (SSRF)   │
          │  sql (readonly)│
          └───────┬────────┘
                  │
          ┌───────▼────────────────┐
          │   Observability (P10)  │
          │   structlog · Prometheus│
          └────────────────────────┘
```

---

## Why this is hard to build

Six engineering decisions that aren't obvious until you've hit the problem:

| Problem | Decision | Why |
|---------|----------|-----|
| Calculator security | AST walk, never `eval()` | `eval()` executes arbitrary Python — one user input becomes code execution |
| Interface coupling | `Protocol` over `ABC` | Structural subtyping — `MockModel` satisfies `Model` without inheriting it, enabling test doubles with zero boilerplate |
| Context retrieval latency | `asyncio.gather(memory, rag)` | Memory and retrieval are independent — run them in parallel, pay the cost of the slower one instead of both |
| Model flakiness | Circuit breaker (CLOSED→OPEN→HALF_OPEN) | Open circuits are skipped at routing time, not mid-call — fail fast before the model is contacted |
| Pipeline correctness | Frozen dataclasses end-to-end | `AIRequest`, `AIResponse`, `TraceSpan` can't be mutated mid-pipeline, safe to cache, safe across async boundaries |
| SQL injection + privilege | String check + `readonly=True` transaction | String-level checks can be bypassed by obfuscation. DB-level readonly enforcement is the second, unbypassable layer |

---

## Phase status

| Phase | Scope | Tests |
|-------|-------|-------|
| P0 | Core types, protocols, config, lifecycle | 15 |
| P1 | Model layer — Mock, Cloud (Anthropic), Local (Ollama), Registry | 9 |
| P2 | Resilience — exponential-backoff retry, timeout, circuit breaker, rate limiter | 10 |
| P3 | Model router — quality/cost/latency scoring, complexity classification | 8 |
| P4 | Context engine — token budget per bucket, priority packing, parallel retrieval | 8 |
| P5 | Memory — conversation (asyncpg), summary (auto-compress), semantic (pgvector), working | 11 |
| P6 | RAG — parser, sliding-window chunker, batch embedder, pgvector retriever, reranker | 16 |
| P7 | Tool system — validator, calculator (AST), SQL (readonly), HTTP (SSRF-safe), registry | 33 |
| P8 | Agent runtime — state machine, executor, parallel tool dispatch, max-iteration cap | 13 |
| P9 | FastAPI gateway — chat, SSE streaming, agents, document ingest, health, metrics | 9 |
| P10 | Observability — structlog structured logging, Prometheus metrics | — |
| P14 | Docker Compose — api + postgres/pgvector + Prometheus + Grafana + Makefile | — |

**Total: 112 unit tests. All passing.**

---

## What you can build on top

| Use case | How |
|----------|-----|
| Document Q&A chatbot | Ingest PDFs via `/v1/documents/ingest`, query via `/v1/chat` |
| Autonomous data analyst | POST task to `/v1/agents/run` with SQL tool pointing at read-only DB |
| Internal knowledge base | RAG over company docs, semantic search over pgvector |
| Multi-model A/B testing | Register two models in ModelRegistry, score via router policies |
| Compliance-safe AI | SQLTool enforces readonly, HTTPTool enforces domain allowlist, all queries logged |

---

## Dev setup

```bash
# Install
uv pip install -e ".[dev]"

# Run locally (no Docker)
make dev           # uvicorn on :8000

# Test
make test-unit     # 112 unit tests, no DB needed
make test          # full suite including integration

# Lint + types
make lint
make type

# Full stack
make up            # docker compose up --build -d
make demo          # runs curl smoke test sequence
make down
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat` | Single-turn chat. Returns `content`, `model`, `trace_id`, `usage`. |
| `POST` | `/v1/chat/stream` | Same as `/v1/chat` but Server-Sent Events — tokens stream as they arrive. |
| `POST` | `/v1/agents/run` | Autonomous agent. Returns `final_answer`, `steps[]`, `tool_calls[]`, `status`. |
| `POST` | `/v1/documents/ingest` | RAG ingest. Idempotent on `document_id`. Returns `chunks_stored`. |
| `GET`  | `/health` | Liveness probe. Always 200 if process is up. |
| `GET`  | `/ready` | Readiness probe. Checks model health. |
| `GET`  | `/metrics` | Prometheus text format. |
| `GET`  | `/docs` | Interactive Swagger UI. |

### `/v1/agents/run` response shape

```json
{
  "status": "completed",
  "total_iterations": 2,
  "final_answer": "After 3 years: $1,157.63",
  "steps": [
    {
      "iteration": 0,
      "model_response": "TOOL_CALL: {\"name\": \"calculator\", \"arguments\": {\"expression\": \"1000*1.05**3\"}}",
      "tool_calls": [{"name": "calculator", "arguments": {"expression": "1000*1.05**3"}}],
      "tool_results": [{"name": "calculator", "result": 1157.625}]
    },
    {
      "iteration": 1,
      "model_response": "After 3 years: $1,157.63",
      "tool_calls": [],
      "tool_results": []
    }
  ]
}
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Enable real Claude AI calls. Without it: mock mode. |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Model ID |
| `ATHENA_DB_URL` | — | PostgreSQL DSN for memory + RAG |
| `ATHENA_EMBEDDER_URL` | — | Embeddings API base URL |
| `ATHENA_ALLOWED_DOMAINS` | — | Comma-separated domains HTTPTool can reach |
| `ATHENA_MAX_CONCURRENT` | `20` | Semaphore cap on concurrent model calls |
| `ATHENA_MAX_AGENT_ITER` | `10` | Hard cap on agent loop iterations |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | — | Set to `json` for production log aggregators |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| HTTP framework | FastAPI + uvicorn |
| Async runtime | asyncio (no threads anywhere) |
| Database | PostgreSQL 16 + pgvector |
| Async DB driver | asyncpg |
| HTTP client | httpx (async) |
| Config | pydantic-settings (env-driven) |
| Logging | structlog (JSON or coloured console) |
| Metrics | prometheus-client |
| Monitoring | Prometheus + Grafana |
| Container | Docker multi-stage build |
| Tests | pytest + pytest-asyncio |
| Linting | ruff |
| Type checking | mypy strict |
