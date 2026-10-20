# Further Things — net-new work, deferred

Scope captured 2026-10-20 while wiring existing-but-dormant subsystems into the
live request path. These items are **not wiring** — the code they'd need
doesn't exist yet, so each is a real feature to design and build, not a
connection to make. Deliberately deferred out of the wiring pass so that work
stayed reviewable and low-risk.

## 1. Full `ContextEngine` integration (Option A, deferred)

Insert `ContextEngine.build()` into `AgentExecutor` (and `AthenaRuntime`) so
every agent/chat turn goes through `memory_fn` + `rag_fn` + `RelevanceRanker`
+ token-budget packing — the architecture the `context/` package's own
docstrings describe as intended, but which has no live caller today.

**Why deferred:** biggest blast radius of anything on this list. Requires:
- A serialization scheme between `ContextEngine`'s `list[str]` content model
  and `AgentExecutor`'s `list[dict[str, str]]` message history — doesn't
  exist, has to be designed.
- Every `/v1/chat` and `/v1/agents/run` call gaining a hard Postgres
  dependency (today only `/v1/documents/ingest` needs it).
- The RAG query route (#2 below) also being live, since `RelevanceRanker`
  has nothing to rank without it.

We chose Option B instead for now: `ConversationMemory` wired directly into
`AthenaRuntime`, bypassing `ContextEngine`/`budget`/`packing`/`ranking`
entirely. Revisit this once the RAG query route exists and there's a second
real consumer that justifies the shared pipeline.

## 2. RAG query-side gateway route

`PgVectorRetriever.search()` + `CosineReranker` work standalone (P6 gate,
tested) but no HTTP endpoint exposes them — only `/v1/documents/ingest`
is wired. Needs:
- New route, e.g. `POST /v1/documents/search` in `gateway/routes.py`.
- Request/response schemas in `gateway/schemas.py` (query text, top_k,
  returned chunks + scores).
- Decide embedding path for the query text itself (reuse `CloudEmbedder`
  already built for ingest).

## 3. Permission / `allowed_tools` model

`ToolValidator.validate_permission()` exists and is unit-tested, but nothing
in the system tracks per-user tool permissions — no schema field, no config,
no storage. Needs designing from scratch:
- Where does `allowed_tools` come from per request? (static config? per-user
  DB row? request field?)
- Plumbing from `gateway/routes.py` → `Agent.run()` → `AgentExecutor` →
  `_execute_tool()`.
- Decide default behavior when unset (allow all vs. deny all).

## 4. `CloudModel` tool-call parsing

`CloudModel.generate()` never populates `ModelResponse.metadata["tool_calls"]`,
so the "structured tool_calls" path in `AgentExecutor` (checked first, before
falling back to the `TOOL_CALL:` text parser) is dead code whenever
`CloudModel` is the backend. Needs new logic to read Anthropic's `tool_use`
content blocks off the real API response and translate them into the
`{"name", "arguments"}` shape the executor expects.

## 5. `CacheBackend` implementation

`athenai.core.protocols.CacheBackend` (Protocol: `get`/`set`) is defined but
has zero implementations anywhere in `src/`. This is also why the
`cache_hits_total` Prometheus counter (`observability/metrics.py`) can never
fire — there is no cache to hit. Needs:
- A concrete implementation (in-process dict for dev, Redis-backed for
  prod — `AthenaConfig.redis_url` already exists and is unused).
- A decision on what gets cached (model responses keyed on `AIRequest`'s
  `__hash__`? embeddings? RAG search results?) and where it's consulted.
