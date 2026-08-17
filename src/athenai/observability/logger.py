"""
Structured logging via structlog.

WHY STRUCTLOG:
Standard logging emits unstructured strings — grep-able but not queryable.
structlog emits structured key-value pairs that log aggregators (Datadog,
Loki, CloudWatch Insights) can index and filter without regex parsing.
Bound contexts (trace_id, request_id, user_id) flow automatically through
async call chains when using structlog.contextvars.

WHY TWO RENDERERS:
ConsoleRenderer → coloured, human-readable output for local dev.
JSONRenderer   → machine-readable lines for production log aggregators.
Switch via LOG_FORMAT=json environment variable.
"""

from __future__ import annotations

import logging
import os

import structlog


def configure_logging() -> None:
    """Set up structlog for the process. Call once at startup."""
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    use_json = os.environ.get("LOG_FORMAT", "").lower() == "json"

    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog logger. Use module __name__ as the name."""
    return structlog.get_logger(name)
