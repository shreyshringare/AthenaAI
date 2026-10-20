# Further Things — net-new work, deferred

Scope captured 2026-10-20 while wiring existing-but-dormant subsystems into the
live request path. These items are **not wiring** — the code they'd need
doesn't exist yet, so each is a real feature to design and build, not a
connection to make. Deliberately deferred out of the wiring pass so that work
stayed reviewable and low-risk.

## 1. Full `ContextEngine` integration — DONE for `AthenaRuntime` only (2026-10-20)

`/v1/chat` and `/v1/chat/stream` now route through `ContextEngine.build()`:
`ConversationMemory` history + RAG (via `DocumentLoader.search()`, which
itself does embed → retrieve → `CosineReranker`) + the current turn, all
token-budgeted by `ContextPacker`, dropping RAG first then memory under
pressure (system > conversation > memory > rag, per `ContextPacker`'s own
priority order).

**What this needed, that "just wire it" undersold:**
- A `\x00`-delimited role-encoding scheme (`_encode_role_content`/
  `_decode_role_content` in `runtime/pipeline.py`) — `ContextEngine`'s
  buckets are plain `list[str]`, no side-channel for role, so reusing
  `ConversationMemory`'s own `"{role}: {content}"` text format would have
  been ambiguous once `ContextPacker` flattens everything to bare strings
  (no way to tell "a message that starts with the text 'user: '" apart
  from "a message whose role is user").
- A chronological reassembly step: `ContextPacker` groups by bucket in
  *priority* order (conversation > memory, since conversation is dropped
  last) — which puts the *current* turn before *older history* if you
  trust `packed.items`'s sequence directly. Reordering to memory-then-
  conversation at reconstruction time was necessary; priority order and
  presentation order are different concerns `ContextEngine` doesn't
  separate.
- Folding `system` + `rag` buckets into `ModelRequest.system` rather than
  message-list entries, since retrieved document chunks aren't "turns."

**Found in the process:** `TokenBudgetManager.allocate()` and its
per-bucket `_allocations` enforcement are never actually called by
`ContextEngine` — only `.total_budget` (a single flat ceiling) is read.
A `token_budget` config dict's per-bucket numbers (system/conversation/
memory/rag/tools) are currently decorative; only `"total"` has any
effect. Documented in `context/budget.py`, not fixed (real feature, bigger
than this task, and the single-ceiling behavior is itself defensible —
just not what the per-bucket-looking config implies).

**Deliberately NOT done: `AgentExecutor` (the tool-use loop).** Scoped to
`AthenaRuntime` only — `AgentExecutor`'s message shape is structurally
different (tool-call protocol instructions baked into the system prompt,
tool results injected mid-loop, not at turn start), and doing both at once
would double the risk for one pass. Revisit as its own scoped task.

**Also deliberately NOT wired: `RelevanceRanker`.** Even with the RAG query
route live, `RelevanceRanker.rank()` expects raw candidate dicts carrying
an `"embedding"` key — but `PgVectorRetriever.search()` computes cosine
similarity in SQL and returns `RetrievedChunk` with a `score`, never the
embedding itself. `RelevanceRanker` is structurally built for a different
retrieval shape (embeddings fetched application-side) than how
`PgVectorRetriever`/`DocumentLoader` actually work today. `CosineReranker`
(already live, re-sorting the SQL-computed scores) fills the equivalent
role for this data shape — wiring `RelevanceRanker` alongside it would be
redundant, not complementary, unless `PgVectorRetriever.search()` is
changed to also return embeddings (a real schema/bandwidth decision, out
of scope here).

Tested: `tests/unit/test_runtime_context_engine.py` (encode/decode round
trip, chronological ordering, rag-drops-before-memory under a tight
budget, graceful no-op when `document_loader`/`context_engine` unset),
`tests/integration/test_runtime_context_engine.py` (real `ConversationMemory`
round trip through real Postgres). Verified live: booted the real gateway,
ran a two-turn chat, confirmed `input_tokens` grew between turns
(memory/system context actually flowing through, not just unit-tested).

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

## 5. `CacheBackend` implementation — DONE (2026-10-20), scoped to embeddings only

Two implementations now exist: `athenai.cache.in_memory.InMemoryCache`
(process-local, zero dependencies) and `athenai.cache.redis_cache.RedisCache`
(wraps `redis.asyncio`, JSON-serialised values). `athenai.rag.embedder
.CachingEmbedder` wraps either behind `EmbedderProtocol`, keyed on
`sha256(text)` + model name. Wired into `gateway/app.py`'s
`_build_embedding_cache`: pings Redis at startup, falls back to
`InMemoryCache` if unreachable — same degrade-to-permissive pattern as
every other optional infra dependency here. `docker-compose.yml` gained a
`redis` service (host port `6380`, same "probably already taken on 6379"
reasoning as Postgres's `5433` remap).

**Deliberately NOT caching chat/agent responses** — this was the original
framing in this doc, and it's actually wrong, not just low-ROI:
`embed()` is a deterministic function of `(text, model)` (no sampling), so
caching it is always correct. Chat completions are NOT — `ModelRequest
.temperature` defaults to `0.7`, real sampling, so a cached response can be
a different answer than the model would legitimately give on a repeat
call. Caching that would be a correctness bug, not a performance win.

Tested: `tests/unit/test_cache.py` (`InMemoryCache` + `CachingEmbedder`
logic, including cross-model cache isolation), `tests/integration
/test_redis_cache.py` (`RedisCache` against real Redis, including a real
`EXPIRE`-based TTL test, not an in-process timer). Verified live: booted
the real gateway with Redis configured, confirmed `embedding_cache.enabled
backend=redis`, and ran `CachingEmbedder` + real `RedisCache` together
against the live container outside the test suite to confirm the second
identical call was a genuine cache hit (`inner.calls == 1`).

## 6. `AgentExecutor` (tool-use loop) `ContextEngine` integration — DONE (2026-10-20)

`AgentExecutor` now accepts the same `context_engine`/`document_loader`/
`rag_top_k` as `AthenaRuntime`, wired from `gateway/app.py` into `Agent`.
Two real differences from `AthenaRuntime`'s integration, both because
there's no cross-call session concept for agent runs (no `session_id`
anywhere in `Agent`/`AgentExecutor`/the `/v1/agents/run` route):
- `memory_fn` always returns `[]` — nothing to fetch. `ContextEngine` is
  used here for RAG-augmenting the task and budgeting the transcript
  *within* one run, not persisted history.
- The `"conversation"` bucket carries the *entire* accumulated transcript
  (task + every tool result + assistant turn so far), not just "the new
  turn" — that transcript is what's actually at risk of exceeding budget
  as `max_iterations` grows it.

`encode_role_content`/`decode_role_content` moved to a shared
`athenai.context.encoding` module (was private to `runtime/pipeline.py`)
and gained an optional `name` field — tool-result messages carry a `name`
identifying which tool produced the result, and that needs to survive
the round trip same as role/content.

**Found and fixed while writing this, retroactively applied to
`AthenaRuntime` too**: `ContextPacker.pack()` is a greedy *forward* scan
within a bucket — it keeps the first items that fit and skips (doesn't
stop at) the rest. A chronologically-ordered bucket (oldest-first, how
both `ConversationMemory.get_recent()` and an append-built transcript
naturally come) fed to it as-is means a tight budget keeps old content
and drops recent content — backwards for conversation truncation. Fixed
by reversing to newest-first before `build()` and reversing back on
reconstruction, in both `AthenaRuntime` and `AgentExecutor`. Caught by a
hand-computed regression test (`test_context_engine_drops_oldest_memory_first_not_newest`,
`test_context_engine_drops_oldest_transcript_first_not_newest`) that
fails under the old ordering and passes under the fix — not a vague
"fewer items survived" assertion.

Tested: `tests/unit/test_executor_context_engine.py` (RAG query uses the
task text, tool `name` round-trips, oldest-drops-first under a
hand-computed tight budget, graceful no-op when unset). Verified live:
booted the real gateway, ran `/v1/agents/run`, confirmed 200 + completed
status through the new request-building path.

## 7. `RelevanceRanker` / embedding-aware retrieval

`RelevanceRanker.rank()` expects candidate chunks carrying their own
`"embedding"` — but `PgVectorRetriever.search()` computes cosine similarity
in SQL and only returns the score, never the vector. Wiring `RelevanceRanker`
for real means either changing `PgVectorRetriever.search()` to also select
and return each row's `embedding` column (bandwidth/schema decision), or
accepting it stays unused in favor of `CosineReranker` (already live),
which fills the equivalent role for the data shape `PgVectorRetriever`
actually produces.
