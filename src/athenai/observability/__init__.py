"""
athenai.observability — structured logging and metrics for the runtime.

HOW IT FITS IN THE SYSTEM:
Two independent pieces live here:
  - `athenai.observability.logger` (structlog) — `athenai.gateway.app`'s
    lifespan calls `configure_logging()` once at process startup, then
    `get_logger(__name__)` is used to get a named, structured logger
    (currently used in `athenai.gateway.app`; intended for use from any
    module that needs leveled, queryable logs).
  - `athenai.observability.metrics` (prometheus-client) — module-level
    Counter/Histogram objects. `requests_total` and
    `agent_iterations_total` are incremented directly from
    `athenai.gateway.routes`; the remaining counters/histograms
    (`model_latency_seconds`, `tokens_total`, `tool_calls_total`,
    `rag_chunks_retrieved`, `cache_hits_total`) are defined for
    instrumentation points that emit them but are not yet wired into every
    call site. The `/metrics` route in `athenai.gateway.routes` scrapes the
    global prometheus-client registry, so these counters do not need to be
    threaded through call signatures.

Neither submodule has a dedicated test gate (see project memory) — treat
changes here as low-risk infrastructure, not behavior-critical code paths.
"""

from __future__ import annotations
