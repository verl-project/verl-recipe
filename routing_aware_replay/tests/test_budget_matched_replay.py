"""Tests for budget-matched replay controls."""

from __future__ import annotations

import unittest

from routing_aware_replay import make_budget_matched_mask


class BudgetMatchedReplayTest(unittest.TestCase):
    def test_uniform_mask_matches_budget(self) -> None:
        result = make_budget_matched_mask(num_experts=8, budget=3, policy="uniform")

        self.assertEqual(result.effective_budget, 3)
        self.assertEqual(result.mask, (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    def test_random_mask_is_deterministic_with_seed(self) -> None:
        first = make_budget_matched_mask(
            num_experts=8,
            budget=3,
            policy="random",
            seed=11,
        )
        second = make_budget_matched_mask(
            num_experts=8,
            budget=3,
            policy="random",
            seed=11,
        )

        self.assertEqual(first.mask, second.mask)
        self.assertEqual(first.effective_budget, 3)

    def test_budget_is_clamped_to_num_experts(self) -> None:
        result = make_budget_matched_mask(num_experts=4, budget=10, policy="uniform")

        self.assertEqual(result.effective_budget, 4)
        self.assertEqual(result.mask, (1.0, 1.0, 1.0, 1.0))

    def test_invalid_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_budget_matched_mask(num_experts=4, budget=1, policy="unknown")


if __name__ == "__main__":
    unittest.main()
