"""
WHY asynccontextmanager FOR LIFECYCLE:
Resource acquisition and release must be paired. Using @asynccontextmanager
on a single function makes the pairing explicit and compile-checkable — the
yield separates startup from shutdown, and the async context manager protocol
guarantees cleanup runs even if startup fails mid-way.

WHY NOT BARE try/finally:
Scattered try/finally blocks across modules make the shutdown order unclear
and easy to break when adding new resources. A single lifecycle function owns
the order explicitly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from athenai.core.config import AthenaConfig

logger = logging.getLogger(__name__)

# Component registry: name → instance. Populated during startup.
ComponentRegistry = dict[str, Any]


@asynccontextmanager
async def lifespan(config: AthenaConfig) -> AsyncGenerator[ComponentRegistry, None]:
    """
    Startup/shutdown context manager for the AthenaAI runtime.

    Usage:
        async with lifespan(config) as registry:
            runtime = registry["runtime"]
            ...
    """
    registry: ComponentRegistry = {}

    logger.info("AthenaAI starting up", extra={"environment": config.environment})

    try:
        # Each phase will register its components here.
        # P1 adds: registry["model_registry"]
        # P4 adds: registry["context_engine"]
        # P5 adds: registry["memory"]
        # P6 adds: registry["retriever"]
        # P9 adds: registry["runtime"]
        registry["config"] = config

        yield registry

    finally:
        logger.info("AthenaAI shutting down")

        # Shutdown in reverse dependency order.
        for name in reversed(list(registry.keys())):
            component = registry[name]
            close = getattr(component, "close", None) or getattr(component, "aclose", None)
            if close is not None:
                try:
                    await close()
                    logger.debug("Closed component", extra={"component": name})
                except Exception as exc:
                    logger.error(
                        "Error closing component",
                        extra={"component": name, "error": str(exc)},
                    )
