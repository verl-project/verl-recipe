"""Tests for Fisher-weighted replay masks."""

from __future__ import annotations

import unittest

from routing_aware_replay import (
    FisherMaskConfig,
    compute_fisher_weighted_replay_mask,
)


class FisherMaskTest(unittest.TestCase):
    def test_target_budget_selects_highest_scores(self) -> None:
        result = compute_fisher_weighted_replay_mask(
            [0.1, 0.9, 0.4, 1.2],
            FisherMaskConfig(target_budget=2),
        )

        self.assertEqual(result.mask, (0.0, 1.0, 0.0, 1.0))
        self.assertEqual(result.effective_budget, 2)
        self.assertEqual(result.num_experts, 4)

    def test_fixed_scores_are_deterministic(self) -> None:
        scores = [0.2, 0.5, 0.5, 0.1]
        config = FisherMaskConfig(target_budget=2)

        first = compute_fisher_weighted_replay_mask(scores, config)
        second = compute_fisher_weighted_replay_mask(scores, config)

        self.assertEqual(first.mask, second.mask)
        self.assertEqual(first.mask, (0.0, 1.0, 1.0, 0.0))

    def test_threshold_mode_produces_soft_values(self) -> None:
        result = compute_fisher_weighted_replay_mask(
            [0.0, 0.4, 0.8, 1.0],
            FisherMaskConfig(
                theta_low=0.1,
                theta_high=0.9,
                tau=0.5,
                soft_mask_temperature=0.2,
            ),
        )

        self.assertEqual(result.mask[0], 0.0)
        self.assertEqual(result.mask[-1], 1.0)
        self.assertGreater(result.mask[1], 0.0)
        self.assertLess(result.mask[1], 1.0)

    def test_invalid_scores_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_fisher_weighted_replay_mask([])


if __name__ == "__main__":
    unittest.main()
