"""
Prometheus metrics for the AthenaAI runtime.

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
