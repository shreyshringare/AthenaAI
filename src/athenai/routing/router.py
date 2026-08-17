"""ModelRouter: selects the best available model for a given routing context."""

from __future__ import annotations

from dataclasses import dataclass, field

from athenai.core.exceptions import ModelUnavailableError
from athenai.core.types import RoutingContext
from athenai.resilience.circuit_breaker import CircuitBreaker, CircuitState
from athenai.routing.policies import RoutingPolicy
from athenai.routing.scorer import ModelMetadata, ModelScorer


@dataclass(frozen=True)
class RoutingDecision:
    selected_role: str
    reason: str
    estimated_cost_usd: float
    complexity: str
    metadata: dict[str, object] = field(default_factory=dict)


_COMPLEXITY_THRESHOLDS = {
    "LOW": 500,
    "MEDIUM": 2000,
}

_ROLE_FOR_COMPLEXITY = {
    "LOW": "fast",
    "MEDIUM": "default",
    "HIGH": "reasoning",
}


def _classify_complexity(tokens: int) -> str:
    if tokens < _COMPLEXITY_THRESHOLDS["LOW"]:
        return "LOW"
    if tokens < _COMPLEXITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


class ModelRouter:
    """
    WHY CIRCUIT-AWARE ROUTING:
    A model that's hitting errors or timeouts should not keep receiving traffic.
    The router checks circuit breaker state and skips OPEN circuits, falling back
    to the next-best available model automatically — no manual intervention needed.
    """

    def __init__(
        self,
        model_catalog: list[ModelMetadata],
        circuit_breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        self._catalog = model_catalog
        self._breakers = circuit_breakers or {}
        self._scorer = ModelScorer()

    def _is_circuit_open(self, role: str) -> bool:
        breaker = self._breakers.get(role)
        if breaker is None:
            return False
        return breaker.state == CircuitState.OPEN and breaker.is_open()

    def select(
        self,
        context: RoutingContext,
        policy: RoutingPolicy | None = None,
    ) -> RoutingDecision:
        policy = policy or RoutingPolicy.default()
        complexity = _classify_complexity(context.estimated_input_tokens)
        preferred_role = _ROLE_FOR_COMPLEXITY[complexity]

        available = [
            m for m in self._catalog
            if m.is_available and not self._is_circuit_open(m.role)
        ]

        if not available:
            raise ModelUnavailableError(
                "All models are unavailable (circuit open or marked unhealthy)"
            )

        # For LOW/HIGH complexity, force the matched role if available.
        # For MEDIUM, use weighted scoring across all available models.
        preferred = next((m for m in available if m.role == preferred_role), None)
        if preferred and complexity != "MEDIUM":
            best = preferred
        else:
            ranked = self._scorer.rank(available, policy)
            best = ranked[0]
        estimated_cost = (context.estimated_input_tokens / 1000.0) * best.cost_per_1k_tokens

        return RoutingDecision(
            selected_role=best.role,
            reason=f"complexity={complexity}, policy=default, circuit=CLOSED",
            estimated_cost_usd=max(0.0001, estimated_cost),
            complexity=complexity,
        )
