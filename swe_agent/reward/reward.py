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
SWE-Agent Reward Function for VERL.

Provides two components:
1. Patch comparison helpers: normalize_patch, compare_patches_simple
2. compute_score: VERL-compatible reward function that reads the patch from
   extra_info["patch"] (populated by the agent loop) instead of solution_str.

Usage:
  Configure in the training script:
    custom_reward_function.path=recipe/swe_agent/reward/reward.py
    custom_reward_function.name=compute_score
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch comparison helpers
# ---------------------------------------------------------------------------


def normalize_patch(patch: str) -> str:
    """Normalize a patch string for comparison.

    Strips whitespace, removes blank lines and git index headers.

    Args:
        patch: Raw patch string.

    Returns:
        Normalized patch string.
    """
    if not patch:
        return ""

    lines = [line.rstrip() for line in patch.strip().split("\n")]
    normalized_lines = []
    for line in lines:
        # Skip index lines (commit hashes)
        if line.startswith("index "):
            continue
        # Skip empty lines
        if not line.strip():
            continue
        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def _extract_changed_files(patch: str) -> set[str]:
    """Extract set of changed files from a patch.

    Args:
        patch: Patch string.

    Returns:
        Set of file paths.
    """
    if not patch:
        return set()

    # Match diff --git a/path b/path
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    return {match[1] for match in matches}  # Use the b/ path


def compare_patches_simple(generated: str, expected: str) -> float:
    """Simple comparison of two patches.

    Scoring:
    - 1.0: Exact match (after normalization)
    - 0.5: All changed files match
    - 0.2 + 0.3 * overlap: Partial file overlap
    - 0.1: A patch was generated but no file overlap
    - 0.0: No patch generated

    Args:
        generated: Generated patch.
        expected: Expected patch.

    Returns:
        Similarity score (0.0 - 1.0).
    """
    if not generated:
        return 0.0

    gen_normalized = normalize_patch(generated)
    exp_normalized = normalize_patch(expected)

    # Exact match
    if gen_normalized == exp_normalized:
        return 1.0

    # Compare changed files
    gen_files = _extract_changed_files(generated)
    exp_files = _extract_changed_files(expected)

    if not exp_files:
        return 0.1 if gen_files else 0.0

    file_overlap = len(gen_files & exp_files) / len(exp_files)

    if file_overlap == 1.0:
        return 0.5
    elif file_overlap > 0:
        return 0.2 + 0.3 * file_overlap
    else:
        return 0.1


# ---------------------------------------------------------------------------
# VERL-compatible compute_score entry point
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    """Custom reward function for SWE-agent.

    For SWE-agent data sources, the actual patch is stored in
    extra_info["patch"] (populated by the agent loop), not in solution_str
    (which is the decoded model response tokens).

    For non-SWE-agent data sources, falls back to the default VERL
    compute_score.

    Args:
        data_source: Dataset identifier (e.g. "swe_agent_simple").
        solution_str: Decoded model response (NOT the patch for SWE-agent).
        ground_truth: Expected answer / patch.
        extra_info: Extra fields from agent loop, including "patch".

    Returns:
        Reward score as float.
    """
    if data_source in ("swe_agent_simple", "swe_agent", "swe_bench", "swe_bench_lite", "swe_bench_verified"):
        # Extract the actual patch from extra_info
        generated_patch = None
        if extra_info is not None:
            generated_patch = extra_info.get("patch", None)

        if generated_patch is None:
            logger.debug("SWE-agent reward: no 'patch' in extra_info; score=0.0")
            return 0.0

        # Extract expected patch from ground_truth
        # ground_truth may be a dict with different key names depending on the data source:
        #   - simple test data uses "ground_truth" key
        #   - SWE-bench data uses "gold_patch" key (set in prepare_data.py)
        if isinstance(ground_truth, dict):
            expected_patch = ground_truth.get("gold_patch") or ground_truth.get("ground_truth") or ""
        else:
            expected_patch = ground_truth or ""

        score = compare_patches_simple(generated_patch, expected_patch)
        logger.info(
            f"SWE-agent reward: score={score:.2f}, "
            f"generated_patch_len={len(generated_patch) if generated_patch else 0}, "
            f"expected_patch_len={len(expected_patch)}"
        )
        return score

    else:
        # Fallback to default VERL compute_score for non-SWE-agent data
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
