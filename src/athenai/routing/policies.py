"""
Routing policies — declarative weight declarations for model selection.

HOW IT FITS IN THE SYSTEM:
A RoutingPolicy is passed into athenai.routing.router.ModelRouter.select(),
which forwards it to athenai.routing.scorer.ModelScorer to rank candidate
models. The policy carries no logic itself — it's pure configuration data
that the scorer interprets (see scorer.py for how the weights combine).
"""

from __future__ import annotations

from dataclasses import dataclass

# (quality, cost, latency) multipliers applied per complexity tier.
_TIER_MULTIPLIERS: dict[str, tuple[float, float, float]] = {
    "LOW": (0.5, 1.5, 1.5),
    "MEDIUM": (1.0, 1.0, 1.0),
    "HIGH": (2.0, 0.5, 0.5),
}


@dataclass(frozen=True)
class RoutingPolicy:
    """
    WHY DECLARATIVE WEIGHTS:
    Routing decisions depend on deployment context — a cost-sensitive batch
    pipeline wants different weights than a latency-critical user-facing API.
    Declarative weights separate the policy (config) from the mechanism (scorer),
    so environment changes require only a config change, never a code change.
    """

    quality_weight: float = 0.4
    cost_weight: float = 0.3
    latency_weight: float = 0.3
    # NOTE: max_cost_usd / max_latency_ms are advisory budget hints, not
    # enforced caps — ModelScorer.score() factors cost/latency into the
    # weighted score but does not reject models that exceed these values.
    max_cost_usd: float = 0.10
    max_latency_ms: float = 5000.0

    def for_complexity(self, complexity: str) -> RoutingPolicy:
        """Skew this policy's weights toward what the request tier needs.

        LOW (short prompt): cost and latency dominate, quality matters less.
        HIGH (long/complex prompt): quality dominates. MEDIUM: unchanged.
        Weights are multiplied then renormalised to sum to 1.0, so the
        caller's base policy still sets the overall tradeoff and the tier
        only tilts it.
        """
        qm, cm, lm = _TIER_MULTIPLIERS.get(complexity, (1.0, 1.0, 1.0))
        q, c, lat = self.quality_weight * qm, self.cost_weight * cm, self.latency_weight * lm
        total = (q + c + lat) or 1.0
        return RoutingPolicy(
            quality_weight=q / total,
            cost_weight=c / total,
            latency_weight=lat / total,
            max_cost_usd=self.max_cost_usd,
            max_latency_ms=self.max_latency_ms,
        )

    @classmethod
    def default(cls) -> RoutingPolicy:
        return cls()

    @classmethod
    def cost_optimized(cls) -> RoutingPolicy:
        return cls(quality_weight=0.2, cost_weight=0.6, latency_weight=0.2)

    @classmethod
    def quality_optimized(cls) -> RoutingPolicy:
        return cls(quality_weight=0.7, cost_weight=0.1, latency_weight=0.2)
