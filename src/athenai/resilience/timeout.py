"""
Timeout wrapper that converts asyncio.TimeoutError to ToolTimeoutError.

HOW IT FITS IN THE SYSTEM:
Bounds how long a single tool/model call is allowed to run, so one slow
dependency can't stall the whole agent loop indefinitely. Raises the
project's own ToolTimeoutError (rather than the stdlib TimeoutError) so
callers can catch it alongside other athenai.core.exceptions.AthenaError
subclasses — e.g. as part of the `retryable` set passed to
athenai.resilience.retry.with_retry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from athenai.core.exceptions import ToolTimeoutError


async def with_timeout[T](fn: Callable[[], Awaitable[T]], seconds: float) -> T:
    """Run fn() with a wall-clock deadline of `seconds`.

    On timeout, asyncio.wait_for cancels the underlying task, but fn() may
    have already caused side effects (e.g. a partially-sent HTTP request)
    before cancellation — this only bounds how long the caller waits, it
    does not guarantee the operation had no effect.
    """
    try:
        return await asyncio.wait_for(fn(), timeout=seconds)
    except TimeoutError as exc:
        raise ToolTimeoutError(f"Operation timed out after {seconds}s") from exc
