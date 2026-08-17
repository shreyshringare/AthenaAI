"""
WHY asyncio.Lock FOR CAS (Compare-And-Swap):
Without a lock, two concurrent tasks could both read the CLOSED state, both
increment the failure counter past the threshold, and both attempt the
CLOSED→OPEN transition. The lock makes read-modify-write atomic so exactly
one transition fires regardless of concurrency level (TOCTOU prevention).

WHY HALF_OPEN STATE:
After cooldown expires, immediately reopening to full traffic risks slamming
a recovering service. HALF_OPEN allows one probe request — if it succeeds,
transition to CLOSED; if it fails, reopen the circuit for another cooldown.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from athenai.core.exceptions import CircuitOpenError


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        half_open_probe_count: int = 1,
        window_s: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.half_open_probe_count = half_open_probe_count
        self.window_s = window_s

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._probe_successes = 0
        self._lock = asyncio.Lock()

        # Track transitions for concurrency tests
        self._transition_count = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return False
        if self._state == CircuitState.OPEN:
            if self._opened_at and (time.monotonic() - self._opened_at) >= self.cooldown_s:
                return False  # Cooldown elapsed — caller should re-check state
            return True
        return False  # HALF_OPEN: allow probe

    async def _maybe_transition_to_half_open(self) -> None:
        async with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._opened_at is not None
                and (time.monotonic() - self._opened_at) >= self.cooldown_s
            ):
                self._state = CircuitState.HALF_OPEN
                self._probe_successes = 0

    async def record_failure(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._transition_count += 1
                return

            if self._state == CircuitState.OPEN:
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                if self._state == CircuitState.CLOSED:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    self._transition_count += 1

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._probe_successes += 1
                if self._probe_successes >= self.half_open_probe_count:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._opened_at = None
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def call(self, fn: object) -> object:
        """Execute fn respecting circuit state."""
        await self._maybe_transition_to_half_open()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker is OPEN")

        try:
            result = await fn()  # type: ignore[operator]
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    @property
    def closed_to_open_transitions(self) -> int:
        return self._transition_count
