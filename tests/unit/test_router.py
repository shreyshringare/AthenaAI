"""P3 gate: model router."""

from __future__ import annotations

import pytest

from athenai.core.exceptions import ModelUnavailableError
from athenai.core.types import RoutingContext
from athenai.resilience.circuit_breaker import CircuitBreaker, CircuitState
from athenai.routing.policies import RoutingPolicy
from athenai.routing.router import ModelRouter
from athenai.routing.scorer import ModelMetadata


def _ctx(tokens: int, description: str = "test") -> RoutingContext:
    return RoutingContext(
        request_id="r1",
        user_id="u1",
        estimated_input_tokens=tokens,
        task_description=description,
    )


def _catalog() -> list[ModelMetadata]:
    return [
        ModelMetadata(role="fast", quality_score=0.6, cost_per_1k_tokens=0.003, avg_latency_ms=200),
        ModelMetadata(
            role="default", quality_score=0.75, cost_per_1k_tokens=0.015, avg_latency_ms=800
        ),
        ModelMetadata(
            role="reasoning", quality_score=0.95, cost_per_1k_tokens=0.075, avg_latency_ms=3000
        ),
    ]


def _router(breakers: dict | None = None) -> ModelRouter:
    return ModelRouter(model_catalog=_catalog(), circuit_breakers=breakers)


def test_low_complexity_selects_fast() -> None:
    router = _router()
    decision = router.select(_ctx(100, "Summarise this sentence"))
    assert decision.complexity == "LOW"
    assert decision.selected_role == "fast"


def test_high_complexity_selects_reasoning() -> None:
    router = _router()
    decision = router.select(_ctx(5000, "Design a distributed payment system"))
    assert decision.complexity == "HIGH"
    assert decision.selected_role == "reasoning"


def test_cost_optimized_policy_prefers_fast() -> None:
    router = _router()
    policy = RoutingPolicy.cost_optimized()
    # Even MEDIUM complexity should prefer fast under cost-optimized policy if fast available
    decision = router.select(_ctx(100, "Translate to French"), policy=policy)
    assert decision.selected_role == "fast"


def test_all_circuits_open_raises() -> None:
    breakers = {}
    for role in ("fast", "default", "reasoning"):
        cb = CircuitBreaker(failure_threshold=1, cooldown_s=60.0)
        cb._state = CircuitState.OPEN
        cb._opened_at = 0.0  # very old, but cooldown is 60s
        breakers[role] = cb

    router = ModelRouter(
        model_catalog=_catalog(),
        circuit_breakers=breakers,
    )
    with pytest.raises(ModelUnavailableError):
        router.select(_ctx(100))


def test_estimated_cost_is_positive() -> None:
    router = _router()
    decision = router.select(_ctx(500))
    assert decision.estimated_cost_usd > 0


def test_estimated_cost_under_policy_max() -> None:
    router = _router()
    policy = RoutingPolicy(max_cost_usd=1.0)
    decision = router.select(_ctx(500), policy=policy)
    assert decision.estimated_cost_usd < policy.max_cost_usd


def test_routing_decision_is_frozen() -> None:
    import dataclasses
    router = _router()
    decision = router.select(_ctx(100))
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        decision.selected_role = "other"  # type: ignore[misc]


def test_quality_policy_can_pick_different_model() -> None:
    router = _router()
    cost_decision = router.select(_ctx(100), policy=RoutingPolicy.cost_optimized())
    quality_decision = router.select(_ctx(5000), policy=RoutingPolicy.quality_optimized())
    # Quality-optimized HIGH complexity must pick reasoning
    assert quality_decision.selected_role == "reasoning"
    # Cost-optimized LOW complexity must pick fast
    assert cost_decision.selected_role == "fast"


def test_tier_tilts_weights_toward_quality_for_high() -> None:
    base = RoutingPolicy.default()
    high = base.for_complexity("HIGH")
    low = base.for_complexity("LOW")
    assert high.quality_weight > base.quality_weight > low.quality_weight
    assert low.cost_weight > base.cost_weight > high.cost_weight
    assert abs(high.quality_weight + high.cost_weight + high.latency_weight - 1.0) < 1e-9


def test_exclude_skips_roles_for_fallback() -> None:
    router = _router()
    first = router.select(_ctx(100)).selected_role
    second = router.select(_ctx(100), exclude=frozenset({first})).selected_role
    assert second != first


def test_exclude_everything_raises() -> None:
    router = _router()
    with pytest.raises(ModelUnavailableError):
        router.select(_ctx(100), exclude=frozenset({"fast", "default", "reasoning"}))
