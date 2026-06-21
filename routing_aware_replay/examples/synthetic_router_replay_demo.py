"""Synthetic demo for routing-aware replay masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing_aware_replay import (
    FisherMaskConfig,
    compute_fisher_weighted_replay_mask,
    make_budget_matched_mask,
    summarize_replay_mask,
)


def main() -> None:
    """Run a small CPU-only routing replay comparison."""

    fisher_scores = [0.05, 0.12, 0.81, 0.33, 1.20, 0.09, 0.74, 0.44]
    fisher_result = compute_fisher_weighted_replay_mask(
        fisher_scores,
        FisherMaskConfig(target_budget=3),
    )
    uniform_result = make_budget_matched_mask(
        num_experts=len(fisher_scores),
        budget=fisher_result.effective_budget,
        policy="uniform",
    )
    random_result = make_budget_matched_mask(
        num_experts=len(fisher_scores),
        budget=fisher_result.effective_budget,
        policy="random",
        seed=7,
    )

    payload = {
        "fisher_weighted": summarize_replay_mask(fisher_result).as_dict(),
        "uniform_budget_matched": summarize_replay_mask(uniform_result).as_dict(),
        "random_budget_matched": summarize_replay_mask(random_result).as_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
