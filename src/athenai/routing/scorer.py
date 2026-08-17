"""
WHY WEIGHTED SCORING OVER HARD CUTOFFS:
Hard cutoffs (if latency < X: use fast model) require manual tuning per
deployment and break when model latencies change. Weighted scoring lets the
policy express tradeoffs declaratively — increasing cost_weight automatically
prefers cheaper models without touching routing logic.

WHY NORMALISED INVERSE FOR COST/LATENCY:
Higher cost and latency are worse. Taking (1/value) * weight converts them
to "higher score = better" so the scorer uses a single max() selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from athenai.routing.policies import RoutingPolicy


@dataclass(frozen=True)
class ModelMetadata:
    role: str
    quality_score: float       # 0.0-1.0 (higher = better)
    cost_per_1k_tokens: float  # USD (lower = cheaper)
    avg_latency_ms: float      # milliseconds (lower = faster)
    is_available: bool = True


class ModelScorer:
    """Computes a weighted score for a model given a routing policy."""

    def score(self, model: ModelMetadata, policy: RoutingPolicy) -> float:
        if not model.is_available:
            return -1.0

        if model.cost_per_1k_tokens > 0:
            cost_score = 1.0 / model.cost_per_1k_tokens
        else:
            cost_score = 1.0

        if model.avg_latency_ms > 0:
            latency_score = 1.0 / model.avg_latency_ms
        else:
            latency_score = 1.0

        # Normalise each dimension to [0, 1] approximately by capping
        cost_norm = min(cost_score / 100.0, 1.0)
        latency_norm = min(latency_score / 0.01, 1.0)

        return (
            model.quality_score * policy.quality_weight
            + cost_norm * policy.cost_weight
            + latency_norm * policy.latency_weight
        )

    def rank(
        self,
        models: list[ModelMetadata],
        policy: RoutingPolicy,
    ) -> list[ModelMetadata]:
        return sorted(
            [m for m in models if m.is_available],
            key=lambda m: self.score(m, policy),
            reverse=True,
        )
