# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Dataset Generator.

Supports simple test cases (for quick validation), loaded from JSON files.

Data format is VERL-compatible:
- prompt: Minimal chat messages (satisfies framework's ``raw_prompt`` requirement).
          The *real* system/instance templates are applied at runtime by
          SWE-Agent via ``swe_agent_config.yaml``.
- reward_model: Evaluation configuration
- extra_info: Contains problem_statement, repo_content, expected_patch,
              and data-affine overrides (sandbox_overrides / agent_overrides).
- agent_name: "swe_agent"

Simple test cases are stored in separate JSON files for easy editing:
  - simple_cases_train.json  (8 easy + 12 medium cases)
  - simple_cases_val.json    (3 easy + 5 medium validation cases)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

_PREPARE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_prompt(problem_statement: str) -> list[dict[str, str]]:
    """Minimal prompt satisfying VERL's ``raw_prompt`` requirement."""
    return [{"role": "user", "content": problem_statement}]


def _load_simple_cases(filename: str) -> list[dict[str, Any]]:
    """Load simple test cases from a JSON file next to this script."""
    path = _PREPARE_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Simple test data
# ---------------------------------------------------------------------------


def generate_simple_test_data(
    num_samples: int,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Generate simple test data for quick validation.

    Train and validation use completely disjoint problem pools.
    Cases are loaded from JSON files in the same directory.

    Repos are pre-baked into the Docker image by ``bake_simple_repos.sh``
    at ``/<split>_<case_idx>/``.  Each sample uses preexisting repo mode
    so no repo creation happens at rollout time.
    """
    if split == "train":
        pool = _load_simple_cases("simple_cases_train.json")
        repo_prefix = "train"
    else:
        pool = _load_simple_cases("simple_cases_val.json")
        repo_prefix = "val"

    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        case = pool[idx % len(pool)]
        case_idx = idx % len(pool)
        repo_name = f"{repo_prefix}_{case_idx}"

        rows.append(
            {
                "prompt": _make_minimal_prompt(case["problem_statement"]),
                "data_source": "swe_agent_simple",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "swe_agent",
                    "ground_truth": case["expected_patch"],
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "expected_patch": case["expected_patch"],
                    "problem_statement": case["problem_statement"],
                    "sandbox_overrides": {
                        "use_preexisting_repo": True,
                        "preexisting_repo_name": repo_name,
                        "preexisting_repo_reset": False,
                    },
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE Agent Dataset Generator")
    parser.add_argument("--mode", choices=["simple"], default="simple", help="Data generation mode")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--output_dir", default="data/swe_agent")
    parser.add_argument("--agent_name", default="swe_agent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating simple test data...")
    train_df = generate_simple_test_data(args.train_size, "train", args.agent_name)
    test_df = generate_simple_test_data(args.test_size, "test", args.agent_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print("\nDataset generation completed!")
    print(f"Train: {len(train_df)} samples -> {train_path}")
    print(f"Test:  {len(test_df)} samples -> {test_path}")


if __name__ == "__main__":
    main()
