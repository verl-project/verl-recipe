# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
SWE-Agent Custom Reward Function for VERL

This module provides a custom `compute_score` function that properly handles
the SWE-agent reward computation. The key difference from the default VERL
reward function is:

  - The default reward function receives `solution_str` as the decoded model
    response tokens (natural language text), which is NOT the git diff patch.
  - For SWE-agent, the actual patch is extracted by the agent loop and passed
    via `extra_info["patch"]`. This custom reward function reads the patch
    from `extra_info` instead of relying on `solution_str`.

Usage:
  Configure in the training script:
    custom_reward_function.path=recipe/swe_agent/reward/compute_score.py
    custom_reward_function.name=compute_score
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_patch(patch: str) -> str:
    """Normalize a patch string for comparison."""
    if not patch:
        return ""
    lines = [line.rstrip() for line in patch.strip().split("\n")]
    lines = [line for line in lines if line and not line.startswith("index ")]
    return "\n".join(lines)


def _extract_changed_files(patch: str) -> List[str]:
    """Extract changed file paths from a patch."""
    if not patch:
        return []
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    return [match[1] for match in matches]


def _compare_patches(generated: str, expected: str) -> float:
    """Compare two patches and return a similarity score.

    Returns:
        1.0 - exact match (normalized)
        0.5 - same files modified
        0.2~0.5 - partial file overlap
        0.1 - generated a patch but wrong files
        0.0 - no patch generated
    """
    if not generated:
        return 0.0

    gen_normalized = _normalize_patch(generated)
    exp_normalized = _normalize_patch(expected)

    if gen_normalized == exp_normalized:
        return 1.0

    gen_files = set(_extract_changed_files(generated))
    exp_files = set(_extract_changed_files(expected))

    if not exp_files:
        return 0.1 if gen_files else 0.0

    file_overlap = len(gen_files & exp_files) / len(exp_files)

    if file_overlap == 1.0:
        return 0.5
    elif file_overlap > 0:
        return 0.2 + 0.3 * file_overlap
    else:
        return 0.1


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
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
    if data_source in ("swe_agent_simple", "swe_agent", "swe_bench"):
        # Extract the actual patch from extra_info
        generated_patch = None
        if extra_info is not None:
            generated_patch = extra_info.get("patch", None)

        if generated_patch is None:
            logger.debug(
                "SWE-agent reward: no 'patch' in extra_info; score=0.0"
            )
            return 0.0

        # Extract expected patch from ground_truth
        if isinstance(ground_truth, dict):
            expected_patch = ground_truth.get("ground_truth", "")
        else:
            expected_patch = ground_truth or ""

        score = _compare_patches(generated_patch, expected_patch)
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
