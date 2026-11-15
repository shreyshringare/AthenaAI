"""RoutedModel: tier routing, circuit-breaker skipping, retry, fallback, limits."""

from __future__ import annotations

import time

import pytest

from athenai.core.exceptions import ModelUnavailableError
from athenai.models.base import ModelRequest, ModelResponse
from athenai.models.mock import MockModel
from athenai.models.routed import Backend, RoutedModel
from athenai.resilience.circuit_breaker import CircuitBreaker, CircuitState
from athenai.resilience.rate_limiter import TokenBucketRateLimiter
from athenai.resilience.retry import RetryPolicy
from athenai.routing.policies import RoutingPolicy
from athenai.routing.scorer import ModelMetadata

_FAST_RETRY = RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter=False)


class _Stub:
    """Model that records calls and optionally fails."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self.model_name = name
        self.fail = fail
        self.calls = 0
        self.seen_model_names: list[str] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.seen_model_names.append(request.model_name)
        if self.fail:
            raise ModelUnavailableError(f"{self.model_name} down")
        return ModelResponse(
            content=f"from {self.model_name}",
            model_name=self.model_name,
            input_tokens=1,
            output_tokens=1,
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return not self.fail


def _backend(
    role: str,
    model: object,
    quality: float,
    cost: float,
    latency: float,
    limiter: TokenBucketRateLimiter | None = None,
    threshold: int = 2,
) -> Backend:
    return Backend(
        metadata=ModelMetadata(role, quality, cost, latency),
        model=model,
        breaker=CircuitBreaker(failure_threshold=threshold, cooldown_s=60.0),
        limiter=limiter,
    )


def _req(chars: int) -> ModelRequest:
    return ModelRequest(messages=[{"role": "user", "content": "x" * chars}], model_name="default")


def _pair() -> tuple[_Stub, _Stub, RoutedModel]:
    cheap = _Stub("cheap")
    smart = _Stub("smart")
    model = RoutedModel(
        [
            _backend("cheap", cheap, 0.5, 0.001, 150),
            _backend("smart", smart, 0.95, 0.08, 3000),
        ],
        policy=RoutingPolicy.default(),
        retry_policy=_FAST_RETRY,
    )
    return cheap, smart, model


async def test_short_prompt_routes_to_cheap_fast_model() -> None:
    _, smart, model = _pair()
    resp = await model.generate(_req(100))  # ~25 tokens -> LOW tier
    assert resp.content == "from cheap"
    assert smart.calls == 0


async def test_long_prompt_routes_to_high_quality_model() -> None:
    cheap, _, model = _pair()
    resp = await model.generate(_req(12_000))  # ~3000 tokens -> HIGH tier
    assert resp.content == "from smart"
    assert cheap.calls == 0


async def test_backend_receives_its_own_model_id_not_the_role() -> None:
    cheap, _, model = _pair()
    await model.generate(_req(100))
    assert cheap.seen_model_names == ["cheap"]


async def test_open_circuit_is_skipped_at_routing_time() -> None:
    cheap, _, model = _pair()
    breaker = model._backends["cheap"].breaker
    breaker._state = CircuitState.OPEN
    breaker._opened_at = time.monotonic()
    resp = await model.generate(_req(100))
    assert resp.content == "from smart"
    assert cheap.calls == 0


async def test_failures_trip_breaker_and_fall_back() -> None:
    cheap, _, model = _pair()
    cheap.fail = True
    resp = await model.generate(_req(100))
    assert resp.content == "from smart"
    assert cheap.calls == 2  # retried once (max_attempts=2) before falling over
    assert model._backends["cheap"].breaker.state == CircuitState.OPEN
    # Next request never even touches the tripped model.
    await model.generate(_req(100))
    assert cheap.calls == 2


async def test_transient_failure_is_retried_on_same_model() -> None:
    cheap, smart, model = _pair()
    original = cheap.generate
    state = {"n": 0}

    async def flaky(req: ModelRequest) -> ModelResponse:
        state["n"] += 1
        if state["n"] == 1:
            raise ModelUnavailableError("blip")
        return await original(req)

    cheap.generate = flaky  # type: ignore[method-assign]
    resp = await model.generate(_req(100))
    assert resp.content == "from cheap"
    assert smart.calls == 0


async def test_all_backends_failing_raises_last_error() -> None:
    cheap, smart, model = _pair()
    cheap.fail = smart.fail = True
    with pytest.raises(ModelUnavailableError):
        await model.generate(_req(100))


async def test_outbound_rate_limit_falls_back_to_other_model() -> None:
    cheap = _Stub("cheap")
    smart = _Stub("smart")
    limiter = TokenBucketRateLimiter(capacity=1, refill_rate=0.0001)
    model = RoutedModel(
        [
            _backend("cheap", cheap, 0.5, 0.001, 150, limiter=limiter),
            _backend("smart", smart, 0.95, 0.08, 3000),
        ],
        retry_policy=_FAST_RETRY,
    )
    assert (await model.generate(_req(100))).content == "from cheap"
    assert (await model.generate(_req(100))).content == "from smart"


async def test_stream_fails_over_before_first_token() -> None:
    class _DeadStream(_Stub):
        async def stream(self, request: ModelRequest):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise ModelUnavailableError("dead")
            yield ""  # pragma: no cover

    dead = _DeadStream("dead", fail=True)
    model = RoutedModel(
        [
            _backend("dead", dead, 0.9, 0.001, 100),
            _backend("mock", MockModel(), 0.5, 0.01, 200),
        ],
        retry_policy=_FAST_RETRY,
    )
    tokens = [t async for t in model.stream(_req(40))]
    assert tokens
    assert dead.calls == 2  # retried, then fell over to the mock backend


async def test_health_check_true_if_any_backend_healthy() -> None:
    cheap, smart, model = _pair()
    cheap.fail = True
    assert await model.health_check() is True
    smart.fail = True
    assert await model.health_check() is False


def test_requires_a_backend() -> None:
    with pytest.raises(ValueError):
        RoutedModel([])
