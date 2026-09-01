"""
Prometheus metrics for the AthenaAI runtime.

HOW IT FITS IN THE SYSTEM:
Each metric below is a module-level Counter/Histogram registered against
prometheus-client's global default registry at import time. Callers that
want to record a metric import the specific object they need (e.g.
`from athenai.observability.metrics import requests_total`) and call
`.inc()` / `.labels(...).inc()` / `.observe()` directly — there is no
central metrics service to go through. `athenai.gateway.routes` currently
increments `requests_total` (per endpoint/status) and
`agent_iterations_total`; the other metrics defined here
(`model_latency_seconds`, `tokens_total`, `tool_calls_total`,
`rag_chunks_retrieved`, `cache_hits_total`) are declared for call sites
that have not been wired up yet. The `/metrics` route
(`athenai.gateway.routes.metrics`) calls prometheus_client.generate_latest()
to serialise whatever is in the registry — including these — for scraping.

WHY PROMETHEUS:
Pull-based metrics fit cloud deployments naturally — the scraper controls
the collection interval, and the process needs no knowledge of the monitoring
backend. Counter/Histogram is sufficient: counters track totals and rates;
histograms track latency distributions (p50, p95, p99).

Counters never decrease. Histograms bucket at boundaries relevant to LLM
latency: 0.1s (fast), 0.5s (ok), 1s (slow), 5s (very slow), 30s (timeout).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

requests_total = Counter(
    "athena_requests_total",
    "HTTP requests processed by the gateway",
    ["endpoint", "status"],
)

model_latency_seconds = Histogram(
    "athena_model_latency_seconds",
    "End-to-end model call latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

tokens_total = Counter(
    "athena_tokens_total",
    "Total LLM tokens consumed",
    ["model", "type"],  # type: input | output
)

tool_calls_total = Counter(
    "athena_tool_calls_total",
    "Tool executions",
    ["tool", "status"],  # status: success | error
)

agent_iterations_total = Counter(
    "athena_agent_iterations_total",
    "Total agent loop iterations across all runs",
)

rag_chunks_retrieved = Counter(
    "athena_rag_chunks_retrieved_total",
    "RAG chunks returned by vector search",
)

cache_hits_total = Counter(
    "athena_cache_hits_total",
    "Semantic cache hits (requests served without model call)",
)
