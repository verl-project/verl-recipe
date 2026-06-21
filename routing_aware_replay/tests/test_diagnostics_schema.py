"""Tests for routing replay diagnostics."""

from __future__ import annotations

import json
import unittest

from routing_aware_replay import (
    FisherMaskConfig,
    compute_fisher_weighted_replay_mask,
    summarize_replay_mask,
)


class DiagnosticsSchemaTest(unittest.TestCase):
    def test_diagnostics_are_json_serializable(self) -> None:
        result = compute_fisher_weighted_replay_mask(
            [0.1, 0.3, 0.9, 0.2],
            FisherMaskConfig(target_budget=2),
        )
        diagnostics = summarize_replay_mask(result).as_dict()

        encoded = json.dumps(diagnostics, sort_keys=True)
        self.assertIn("effective_replay_budget", encoded)
        self.assertEqual(diagnostics["effective_replay_budget"], 2)
        self.assertEqual(diagnostics["preserved_count"], 2)
        self.assertEqual(diagnostics["released_count"], 2)
        self.assertEqual(diagnostics["policy_name"], "fisher_weighted")


if __name__ == "__main__":
    unittest.main()
