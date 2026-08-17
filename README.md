# AthenaAI — Modular AI Runtime

> A modular Python runtime for building reliable, observable, tool-using
> AI applications — with pluggable models, intelligent routing, token-budgeted
> context management, persistent memory, RAG, typed tool execution, stateful
> agents, and end-to-end execution tracing.

---

## Project Status

| Phase | Scope | Status |
|-------|-------|--------|
| P0 | Core types, protocols, config, lifecycle | ✅ Complete |
| P1 | Model layer (Mock, Cloud, Local, Registry) | ✅ Complete |
| P2 | Resilience (retry, timeout, circuit breaker, rate limiter) | ✅ Complete |
| P3 | Model router (scoring, policies, complexity routing) | ✅ Complete |
| P4 | Context engine + token budget manager | ✅ Complete |
| P5 | Memory layer (conversation, summary, semantic, working) | ✅ Complete |
| P6 | RAG pipeline | 🔜 Upcoming |
| P7 | Tool system | 🔜 Upcoming |
| P8 | Agent runtime | 🔜 Upcoming |
| P9 | API gateway + full pipeline | 🔜 Upcoming |
| P10 | Observability (metrics, tracing, cost) | 🔜 Upcoming |
| P11 | Failure scenarios | 🔜 Upcoming |
| P12 | Benchmarks | 🔜 Upcoming |
| P13 | Evaluation harness | 🔜 Upcoming |
| P14 | Docker + README | 🔜 Upcoming |

---

## Architecture

```
athenai/
├── core/           # Types, protocols, config, lifecycle, exceptions
├── models/         # MockModel, CloudModel (httpx), LocalModel (Ollama), ModelRegistry
├── resilience/     # RetryPolicy, CircuitBreaker, TokenBucketRateLimiter, with_timeout
├── routing/        # ModelRouter, RoutingPolicy, ModelScorer — complexity-aware routing
├── context/        # TokenBudgetManager, ContextPacker, ContextEngine, RelevanceRanker
└── memory/         # ConversationMemory (PG), SummaryMemory, SemanticMemory (pgvector), AgentState
```

---

## What's Implemented

### P0 — Core Foundation
- **Frozen dataclasses**: `Message`, `AIRequest`, `AIResponse`, `TokenUsage`, `TraceSpan`, `RoutingContext` — all immutable value types
- **Protocols** (`typing.Protocol`): `Model`, `StreamingModel`, `MemoryStore`, `Retriever`, `Tool`, `CacheBackend`, `EmbedderProtocol` — structural subtyping, no ABC
- **`AthenaConfig`** (`pydantic-settings`): env-driven with `ATHENA_` prefix, typed `model_registry` and `token_budget` dicts
- **`lifespan()`**: async context manager for component startup/shutdown

### P1 — Model Layer
- **`MockModel`**: deterministic echo, streaming char-by-char, `set_healthy()` for tests
- **`CloudModel`**: `httpx.AsyncClient`, maps 429→`RateLimitError`, 401/403/5xx→`ModelUnavailableError`
- **`LocalModel`**: Ollama `/api/generate` via httpx
- **`ModelRegistry`**: config-driven, `get(role)` raises `KeyError from None` on miss

### P2 — Resilience
- **`with_retry[T]`**: exponential backoff + full jitter (`uniform(0, min(base*2^n, max))`)
- **`with_timeout[T]`**: wraps `asyncio.wait_for`, raises `ToolTimeoutError`
- **`CircuitBreaker`**: CLOSED/OPEN/HALF_OPEN state machine, `asyncio.Lock` for CAS transitions
- **`TokenBucketRateLimiter`**: single lock covers refill + decrement atomically

### P3 — Model Router
- **`ModelScorer`**: weighted quality/cost/latency scoring per `RoutingPolicy`
- **`ModelRouter.select()`**: LOW (<500 tokens) → `fast`, HIGH (>2000) → `reasoning`, MEDIUM → weighted score
- **`RoutingPolicy`**: frozen dataclass with `default()`, `cost_optimized()`, `quality_optimized()` classmethods
- Circuit-open models skipped from candidates

### P4 — Context Engine
- **`TokenBudgetManager`**: per-bucket + global ceiling, raises `ContextOverflowError` (never silent truncation)
- **`ContextPacker`**: priority order `system > conversation > memory > rag > tools`, drops lowest-priority on overflow
- **`ContextEngine.build()`**: `asyncio.gather(memory_fn(), rag_fn())` — parallel retrieval
- **`RelevanceRanker`**: numpy cosine similarity for item ranking

### P5 — Memory Layer
- **`ConversationMemory`** (`asyncpg`): PostgreSQL-backed message store, `get_recent(n)` returns chronological order, `clear(session_id)`
- **`SummaryMemory`**: when message count exceeds threshold, calls model to compress older messages into a `SUMMARY` entry; returns `[summary] + keep_recent raw`
- **`SemanticMemory`** (`pgvector`): stores 1536-dim embeddings, cosine similarity search via `<=>` operator, `store()`/`search(k)`/`delete()`
- **`AgentState`**: mutable in-process working memory, `add_step()`, `add_tool_result()`, `snapshot()`/`from_snapshot()` round-trip

---

## Development Setup

**Requirements**: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (for integration tests)

```bash
git clone https://github.com/shreyshringare/AthenaAI.git
cd AthenaAI
uv sync --dev
uv run pip install -e .
```

**Run unit tests:**
```bash
uv run pytest tests/unit/ -v
```

**Run P5 integration tests** (requires PostgreSQL + pgvector):
```bash
docker run -d --name athenai-test-pg \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=athenai_test \
  -p 5433:5432 pgvector/pgvector:pg16

uv run pytest tests/integration/test_memory.py -v
```

**Lint + type check:**
```bash
uv run ruff check src/ tests/
uv run mypy src/
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Python | 3.12+ with PEP 695 generics |
| Package manager | uv (monorepo) |
| HTTP client | httpx (async) |
| Config | pydantic-settings |
| Database | PostgreSQL via asyncpg |
| Vector search | pgvector (`<=>` cosine) |
| Linting | ruff |
| Type checking | mypy (strict) |
| Testing | pytest-asyncio |

---

*(Full API documentation and Docker Compose setup added in Phase P14.)*
