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

## 2. RAG query-side gateway route — DONE (2026-10-20)

`POST /v1/documents/search` now exists (`gateway/routes.py`), backed by
`DocumentLoader.search()` (`rag/loader.py`), which embeds the query, calls
`PgVectorRetriever.search()` for 2x `top_k` candidates, and reranks via
`CosineReranker` down to `top_k`. Tested: `tests/integration/test_rag_pipeline.py`
(`test_loader_search_returns_relevant_chunk`, `test_loader_search_respects_top_k`)
against real Postgres, plus `tests/unit/test_gateway.py` for the route's
503-when-unconfigured and happy-path-with-a-stub-loader cases.

### CloudEmbedder construction bug — FIXED (2026-10-20)

`gateway/app.py`'s `_maybe_add_document_loader` called
`CloudEmbedder(base_url=embedder_url)` without `api_key` (required, no
default) and no env var read one anywhere — RAG silently stayed disabled
the moment `ATHENA_EMBEDDER_URL` was set. Added `ATHENA_EMBEDDER_API_KEY`,
gated alongside the other two RAG env vars (missing key → `rag.disabled`,
same as missing DB/embedder URL, not a crash). Verified live: booted the
real gateway with a fake embedder URL + key, confirmed `rag.enabled` fires
(construction succeeds) and `/v1/documents/ingest` fails with a 500
*network* error against the fake URL, not a construction-time crash.
Regression tests in `tests/unit/test_gateway_app.py`.

## 3. Permission / `allowed_tools` model — DONE (2026-10-20)

`AgentExecutor` now accepts an `allowed_tools: set[str] | None` at
construction, checked via `ToolValidator.validate_permission()` right after
schema validation and before `tool.execute()`. Wired from
`ATHENA_ALLOWED_TOOLS` (comma-separated) in `gateway/app.py` →
`Agent(allowed_tools=...)`. `None` (unset) means unrestricted — matches
every other optional security feature in this gateway degrading to
permissive, not a silent lockout.

**Deliberately not per-user**: there's no multi-tenant identity system in
this gateway, just one shared `ATHENA_GATEWAY_API_KEY` — so `allowed_tools`
is a single gateway-wide set, and `user_id` is only used in the permission
error's audit-trail message, not to differentiate access. Revisit once real
per-user accounts exist (`user_id` would need to come from verified identity,
not a client-supplied free-form string in the request body as it does today).

Tested: `tests/unit/test_agent.py` (`test_executor_denies_tool_not_in_allowed_tools`,
`test_executor_allows_tool_in_allowed_tools`,
`test_executor_allows_all_tools_when_allowed_tools_unset`). Verified live:
booted the real gateway with `ATHENA_ALLOWED_TOOLS` set, confirmed
`tool_permissions.enabled` logs the configured set and the agent still runs
end to end.

## 4. `CloudModel` tool-call parsing — DONE (2026-10-20)

Turned out to need both sides, not just response parsing (the original framing
here undersold it): `AgentExecutor.run()` now passes `metadata={"tools": schemas}`
on every `ModelRequest` — `ToolRegistry.get_schemas()`'s shape already matches
Anthropic's tool-schema format exactly. `CloudModel.generate()` forwards that
as the request's `tools` field (omitted when empty) and, on the response side,
walks every `content` block: `text` blocks join into `.content`, `tool_use`
blocks become `metadata["tool_calls"]` in the `{"name", "arguments"}` shape
`AgentExecutor` already expects. Without the request side, Claude would never
emit native `tool_use` blocks regardless of how well the response is parsed —
the API only offers tool use when the request declares what tools exist.

The `TOOL_CALL:` text-protocol system-prompt instructions stay in place
unconditionally (harmless for MockModel/LocalModel, and `AgentExecutor`
checks `metadata["tool_calls"]` first so a populated list always wins —
no double-execution risk even if Claude also emits text matching the old
protocol).

Tested: `tests/unit/test_models.py` (tool_use parsing, mixed text+tool_use,
tools-field forwarding, omitted-when-empty) and
`tests/unit/test_agent.py::test_executor_uses_structured_tool_calls_and_forwards_schemas`
proving the executor-level wiring end to end. Not tested against the real
Anthropic API (no `base_url` override exists via env var in `app.py`, and
that would mean calling a real paid endpoint) — respx-mocked `httpx` calls
are the same level of verification this class has always used.

## 5. `CacheBackend` implementation

`athenai.core.protocols.CacheBackend` (Protocol: `get`/`set`) is defined but
has zero implementations anywhere in `src/`. This is also why the
`cache_hits_total` Prometheus counter (`observability/metrics.py`) can never
fire — there is no cache to hit. Needs:
- A concrete implementation (in-process dict for dev, Redis-backed for
  prod — `AthenaConfig.redis_url` already exists and is unused).
- A decision on what gets cached (model responses keyed on `AIRequest`'s
  `__hash__`? embeddings? RAG search results?) and where it's consulted.
