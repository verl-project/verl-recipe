"""Shared schema for routing-aware replay utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FisherMaskConfig:
    """Configuration for Fisher-weighted replay mask construction.

    Args:
        target_budget: Optional number of experts to preserve. When set, the
            highest-scoring experts are selected and threshold fields are not
            used for hard selection.
        tau: Center point used for threshold or soft-mask interpolation.
        theta_high: Normalized score above which an expert is always preserved.
        theta_low: Normalized score below which an expert is always released.
        soft_mask_temperature: If positive, experts between ``theta_low`` and
            ``theta_high`` receive a sigmoid soft mask around ``tau``.
        policy_name: Human-readable policy identifier stored in results.
    """

    target_budget: int | None = None
    tau: float = 0.5
    theta_high: float = 0.7
    theta_low: float = 0.2
    soft_mask_temperature: float = 0.0
    policy_name: str = "fisher_weighted"

    def __post_init__(self) -> None:
        if self.target_budget is not None and self.target_budget < 0:
            raise ValueError("target_budget must be non-negative or None")
        if self.theta_low > self.theta_high:
            raise ValueError("theta_low must be <= theta_high")
        for field_name in ("tau", "theta_high", "theta_low"):
            value = getattr(self, field_name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.soft_mask_temperature < 0.0:
            raise ValueError("soft_mask_temperature must be non-negative")


@dataclass(frozen=True)
class ReplayMaskResult:
    """Result returned by replay mask builders."""

    policy_name: str
    mask: tuple[float, ...]
    scores: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_experts(self) -> int:
        """Number of experts covered by the mask."""

        return len(self.mask)

    @property
    def effective_budget(self) -> int:
        """Number of experts with a non-zero replay mask."""

        return sum(1 for value in self.mask if value > 0.0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "policy_name": self.policy_name,
            "mask": list(self.mask),
            "scores": list(self.scores),
            "num_experts": self.num_experts,
            "effective_budget": self.effective_budget,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RoutingReplayDiagnostics:
    """Compact diagnostics for a replay mask."""

    policy_name: str
    num_experts: int
    effective_replay_budget: int
    hard_expert_count: int
    soft_expert_count: int
    preserved_count: int
    released_count: int
    mask_mean: float
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "policy_name": self.policy_name,
            "num_experts": self.num_experts,
            "effective_replay_budget": self.effective_replay_budget,
            "hard_expert_count": self.hard_expert_count,
            "soft_expert_count": self.soft_expert_count,
            "preserved_count": self.preserved_count,
            "released_count": self.released_count,
            "mask_mean": self.mask_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "metadata": dict(self.metadata),
        }
