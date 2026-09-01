"""
Circuit breaker — stops sending calls to a dependency that's already failing.

HOW IT FITS IN THE SYSTEM:
Wraps calls to flaky external services (LLM provider APIs, tool backends).
athenai.routing.router.ModelRouter holds one CircuitBreaker per model role
and consults breaker.state / breaker.is_open() when selecting a model, so a
model that's erroring gets skipped in favour of the next-best available one
without any manual intervention. Callers can also route a single async call
through breaker.call(fn) to get the same protection directly.

STATE MACHINE: CLOSED -> OPEN -> HALF_OPEN -> (CLOSED | OPEN)
  CLOSED:    normal operation; failures accumulate towards failure_threshold.
  OPEN:      calls are rejected immediately (CircuitOpenError) without
             touching the dependency, for cooldown_s seconds.
  HALF_OPEN: one probe call is allowed through; success closes the circuit,
             failure reopens it for another full cooldown.

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
    """The three states in the circuit breaker state machine (see module docstring)."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Per-dependency failure tracker that trips OPEN after repeated failures.

    Not thread-safe across event loops (uses asyncio.Lock), but safe for any
    number of concurrent coroutines on the same loop. One instance should be
    shared per logical dependency (e.g. per model role), not per call.
    """

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
        """Whether calls should currently be rejected.

        Read-only: does NOT perform the OPEN->HALF_OPEN transition even when
        the cooldown has elapsed — it just reports "cooldown elapsed" by
        returning False so the caller can proceed. The actual state flip only
        happens inside the locked _maybe_transition_to_half_open(), invoked
        from call(). Checking this method alone will not advance the state.
        """
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
        """Report a failed call.

        A single failure while HALF_OPEN reopens the circuit immediately —
        failure_threshold only applies to the initial CLOSED->OPEN trip, not
        to the probe. A failure while already OPEN is a no-op (state is
        unaffected; the cooldown timer is not reset).
        """
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
        """Report a successful call.

        Requires half_open_probe_count consecutive successes while HALF_OPEN
        before closing the circuit (default 1). While CLOSED, a success just
        resets the failure counter so isolated errors don't accumulate
        towards the threshold.
        """
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
