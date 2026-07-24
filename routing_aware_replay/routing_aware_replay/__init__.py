"""Routing-aware replay utilities for MoE RL recipes."""

from routing_aware_replay.diagnostics import summarize_replay_mask
from routing_aware_replay.fisher_mask import compute_fisher_weighted_replay_mask
from routing_aware_replay.replay_policy import make_budget_matched_mask
from routing_aware_replay.schema import (
    FisherMaskConfig,
    ReplayMaskResult,
    RoutingReplayDiagnostics,
)

__all__ = [
    "FisherMaskConfig",
    "ReplayMaskResult",
    "RoutingReplayDiagnostics",
    "compute_fisher_weighted_replay_mask",
    "make_budget_matched_mask",
    "summarize_replay_mask",
]
