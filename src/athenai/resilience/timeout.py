"""Timeout wrapper that converts asyncio.TimeoutError to ToolTimeoutError."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from athenai.core.exceptions import ToolTimeoutError


async def with_timeout[T](fn: Callable[[], Awaitable[T]], seconds: float) -> T:
    try:
        return await asyncio.wait_for(fn(), timeout=seconds)
    except TimeoutError as exc:
        raise ToolTimeoutError(f"Operation timed out after {seconds}s") from exc
