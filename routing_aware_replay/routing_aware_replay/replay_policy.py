"""Budget-matched replay policy baselines."""

from __future__ import annotations

import random

from routing_aware_replay.schema import ReplayMaskResult


def _validate_budget(num_experts: int, budget: int) -> int:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return min(budget, num_experts)


def _mask_from_indices(num_experts: int, selected: set[int]) -> tuple[float, ...]:
    return tuple(1.0 if index in selected else 0.0 for index in range(num_experts))


def make_budget_matched_mask(
    num_experts: int,
    budget: int,
    policy: str,
    seed: int = 0,
) -> ReplayMaskResult:
    """Create a replay mask with the same budget under a control policy.

    Args:
        num_experts: Number of experts covered by the mask.
        budget: Number of experts to preserve.
        policy: ``"uniform"`` or ``"random"``.
        seed: Random seed used only for the random control.

    Returns:
        ReplayMaskResult with exactly ``min(budget, num_experts)`` preserved
        experts.

    Raises:
        ValueError: If the policy or dimensions are invalid.
    """

    clamped_budget = _validate_budget(num_experts, budget)
    if policy == "uniform":
        selected = (
            {int(index * num_experts / clamped_budget) for index in range(clamped_budget)} if clamped_budget else set()
        )
    elif policy == "random":
        rng = random.Random(seed)
        selected = set(rng.sample(range(num_experts), clamped_budget))
    else:
        raise ValueError("policy must be 'uniform' or 'random'")

    return ReplayMaskResult(
        policy_name=f"{policy}_budget_matched",
        mask=_mask_from_indices(num_experts, selected),
        metadata={
            "budget": clamped_budget,
            "requested_budget": budget,
            "seed": seed if policy == "random" else None,
        },
    )
