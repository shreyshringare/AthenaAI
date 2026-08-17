"""P2 gate: circuit breaker."""

from __future__ import annotations

import asyncio

import pytest

from athenai.core.exceptions import CircuitOpenError
from athenai.resilience.circuit_breaker import CircuitBreaker, CircuitState


def _cb(threshold: int = 2, cooldown: float = 0.3) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, cooldown_s=cooldown)


@pytest.mark.asyncio
async def test_two_failures_opens_circuit() -> None:
    cb = _cb(threshold=2)
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_is_open_after_threshold() -> None:
    cb = _cb(threshold=2)
    await cb.record_failure()
    await cb.record_failure()
    assert cb.is_open() is True


@pytest.mark.asyncio
async def test_cooldown_transitions_to_half_open() -> None:
    cb = _cb(threshold=2, cooldown=0.1)
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    await asyncio.sleep(0.15)
    # is_open() signals cooldown elapsed; _maybe_transition needs explicit trigger
    assert cb.is_open() is False  # cooldown elapsed — let probe through
    await cb._maybe_transition_to_half_open()
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_success_in_half_open_closes_circuit() -> None:
    cb = _cb(threshold=2, cooldown=0.1)
    await cb.record_failure()
    await cb.record_failure()
    await asyncio.sleep(0.15)
    await cb._maybe_transition_to_half_open()
    assert cb.state == CircuitState.HALF_OPEN
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_concurrent_failures_exactly_one_transition() -> None:
    cb = CircuitBreaker(failure_threshold=20, cooldown_s=60.0)

    async def fail() -> None:
        await cb.record_failure()

    await asyncio.gather(*[fail() for _ in range(20)])
    assert cb.state == CircuitState.OPEN
    assert cb.closed_to_open_transitions == 1


@pytest.mark.asyncio
async def test_open_circuit_call_raises() -> None:
    cb = _cb(threshold=1)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await cb.call(lambda: asyncio.sleep(0))
