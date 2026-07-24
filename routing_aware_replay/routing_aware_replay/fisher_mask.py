"""Fisher-weighted replay mask utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable

from routing_aware_replay.schema import FisherMaskConfig, ReplayMaskResult


def _as_float_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _minmax_normalize(scores: tuple[float, ...]) -> tuple[float, ...]:
    min_score = min(scores)
    max_score = max(scores)
    width = max_score - min_score
    if width == 0.0:
        return tuple(1.0 if max_score > 0.0 else 0.0 for _ in scores)
    return tuple((score - min_score) / width for score in scores)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _top_budget_mask(scores: tuple[float, ...], budget: int) -> tuple[float, ...]:
    num_experts = len(scores)
    clamped_budget = min(max(budget, 0), num_experts)
    ranked_indices = sorted(range(num_experts), key=lambda index: (-scores[index], index))
    selected = set(ranked_indices[:clamped_budget])
    return tuple(1.0 if index in selected else 0.0 for index in range(num_experts))


def compute_fisher_weighted_replay_mask(
    fisher_scores: Iterable[float],
    config: FisherMaskConfig | None = None,
) -> ReplayMaskResult:
    """Build an expert replay mask from Fisher/proxy importance scores.

    Args:
        fisher_scores: Per-expert Fisher trace or Fisher-like importance proxy.
            Larger values indicate experts whose routing behavior should be
            preserved more strongly.
        config: Optional mask configuration.

    Returns:
        ReplayMaskResult with one mask value per expert.

    Raises:
        ValueError: If scores are empty or contain non-finite values.
    """

    cfg = config or FisherMaskConfig()
    raw_scores = _as_float_tuple(fisher_scores, "fisher_scores")
    normalized_scores = _minmax_normalize(raw_scores)

    if cfg.target_budget is not None:
        mask = _top_budget_mask(normalized_scores, cfg.target_budget)
        selection_mode = "top_budget"
    else:
        mask_values: list[float] = []
        for score in normalized_scores:
            if score >= cfg.theta_high:
                mask_values.append(1.0)
            elif score <= cfg.theta_low:
                mask_values.append(0.0)
            elif cfg.soft_mask_temperature > 0.0:
                mask_values.append(_sigmoid((score - cfg.tau) / cfg.soft_mask_temperature))
            else:
                mask_values.append(1.0 if score >= cfg.tau else 0.0)
        mask = tuple(mask_values)
        selection_mode = "threshold"

    return ReplayMaskResult(
        policy_name=cfg.policy_name,
        mask=mask,
        scores=normalized_scores,
        metadata={
            "selection_mode": selection_mode,
            "target_budget": cfg.target_budget,
            "tau": cfg.tau,
            "theta_high": cfg.theta_high,
            "theta_low": cfg.theta_low,
            "soft_mask_temperature": cfg.soft_mask_temperature,
        },
    )
