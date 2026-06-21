"""Diagnostics for routing-aware replay masks."""

from __future__ import annotations

from routing_aware_replay.schema import ReplayMaskResult, RoutingReplayDiagnostics


def summarize_replay_mask(result: ReplayMaskResult) -> RoutingReplayDiagnostics:
    """Summarize replay mask behavior in a compact diagnostics object.

    Args:
        result: Replay mask result produced by a replay policy.

    Returns:
        RoutingReplayDiagnostics with budget and score statistics.
    """

    mask = result.mask
    if not mask:
        raise ValueError("result.mask must not be empty")

    hard_count = sum(1 for value in mask if value >= 1.0)
    soft_count = sum(1 for value in mask if 0.0 < value < 1.0)
    preserved_count = sum(1 for value in mask if value > 0.0)
    released_count = sum(1 for value in mask if value == 0.0)
    mask_mean = sum(mask) / len(mask)

    score_min = None
    score_max = None
    score_mean = None
    if result.scores:
        score_min = min(result.scores)
        score_max = max(result.scores)
        score_mean = sum(result.scores) / len(result.scores)

    return RoutingReplayDiagnostics(
        policy_name=result.policy_name,
        num_experts=len(mask),
        effective_replay_budget=preserved_count,
        hard_expert_count=hard_count,
        soft_expert_count=soft_count,
        preserved_count=preserved_count,
        released_count=released_count,
        mask_mean=mask_mean,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        metadata=dict(result.metadata),
    )
