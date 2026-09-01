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

    @classmethod
    def default(cls) -> RoutingPolicy:
        return cls()

    @classmethod
    def cost_optimized(cls) -> RoutingPolicy:
        return cls(quality_weight=0.2, cost_weight=0.6, latency_weight=0.2)

    @classmethod
    def quality_optimized(cls) -> RoutingPolicy:
        return cls(quality_weight=0.7, cost_weight=0.1, latency_weight=0.2)
