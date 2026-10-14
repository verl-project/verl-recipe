"""Boxed exact-match reward used by the HACPO reference implementation."""

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Score the last boxed answer independently of the benchmark identifier."""

    del data_source, extra_info, kwargs
    from verl.utils.reward_score import math_reward

    return float(math_reward.compute_score(solution_str, ground_truth))
