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

Reward structure (for swe_agent data sources):
  1.0       — exact patch match
  0.10-0.85 — partial patch match (file overlap + line similarity)
  0.05      — patch generated but wrong files / no patch but edited correct file
  0.03      — no patch, but ran tests or python verification
  0.02      — no patch, but model made edits (str_replace_editor on wrong file)
  0.01      — no patch, but model explored code (cat/ls used)
  0.0       — no patch and no meaningful tool usage / 0 turns (timeout)
 -0.05      — long and fruitless (>=10 turns, no patch, no editor)
 -0.1       — premature submit without any tool usage (1-2 turns)
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch comparison helpers
# ---------------------------------------------------------------------------


def normalize_patch(patch: str) -> str:
    """Normalize a patch string for comparison."""
    if not patch:
        return ""
    lines = [line.rstrip() for line in patch.strip().split("\n")]
    normalized_lines = []
    for line in lines:
        if line.startswith("index "):
            continue
        if not line.strip():
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _extract_changed_files(patch: str) -> set[str]:
    """Extract set of changed files from a patch."""
    if not patch:
        return set()
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    return {match[1] for match in matches}


def _extract_changed_lines(patch: str) -> set[str]:
    """Extract the set of added/removed content lines from a patch."""
    lines: set[str] = set()
    if not patch:
        return lines
    for raw in patch.split("\n"):
        stripped = raw.strip()
        if stripped.startswith(("+++", "---", "@@", "diff ", "index ", "similarity", "rename", "new file", "deleted")):
            continue
        if stripped.startswith(("+", "-")):
            lines.add(stripped[1:].strip())
    return lines


def compare_patches(generated: str, expected: str) -> float:
    """Fine-grained patch comparison with line-level similarity.

    Scoring:
    - 0.0:  no patch generated
    - 0.05: patch generated but wrong files
    - 0.10 - 0.85: partial match (file overlap + line similarity)
    - 1.0:  exact match (after normalization)
    """
    if not generated:
        return 0.0

    gen_normalized = normalize_patch(generated)
    exp_normalized = normalize_patch(expected)

    if gen_normalized == exp_normalized:
        return 1.0

    gen_files = _extract_changed_files(generated)
    exp_files = _extract_changed_files(expected)

    if not exp_files:
        return 0.05 if gen_files else 0.0

    file_overlap = len(gen_files & exp_files) / len(exp_files)

    if file_overlap == 0:
        return 0.05

    gen_lines = _extract_changed_lines(generated)
    exp_lines = _extract_changed_lines(expected)
    if exp_lines:
        line_sim = len(gen_lines & exp_lines) / len(exp_lines)
    else:
        line_sim = 0.0

    combined = 0.4 * file_overlap + 0.6 * line_sim
    score = 0.10 + combined * 0.75
    return min(score, 0.85)


def _detect_tool_usage(solution_str: str) -> dict[str, bool]:
    """Detect which SWE-Agent tools the model used from the decoded response."""
    text = solution_str or ""
    return {
        "used_editor": "str_replace_editor" in text or "str_replace" in text,
        "used_cat": "cat " in text,
        "used_ls": "ls " in text or "ls\n" in text,
        "used_submit": "submit" in text,
        "used_python": "python " in text or "python3 " in text,
        "used_test": "pytest" in text or "unittest" in text,
    }


def _targeted_correct_file(solution_str: str, expected_patch: str) -> bool:
    """Check if the model interacted with the correct file(s) from the expected patch."""
    target_files = _extract_changed_files(expected_patch)
    if not target_files:
        return False
    text = solution_str or ""
    return any(f in text for f in target_files)


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
    """Custom reward function for SWE-agent with tool-use shaping."""
    if data_source not in ("swe_agent_simple", "swe_agent"):
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    generated_patch = None
    num_turns = 0
    alignment_failed = False
    alignment_failure_reason = ""
    alignment_failure_details = ""
    if extra_info is not None:
        generated_patch = extra_info.get("patch", None)
        num_turns = int(extra_info.get("num_turns", 0) or 0)
        alignment_failed = bool(extra_info.get("alignment_failed", False))
        alignment_failure_reason = str(extra_info.get("alignment_failure_reason", "") or "")
        alignment_failure_details = str(extra_info.get("alignment_failure_details", "") or "")

    if isinstance(ground_truth, dict):
        expected_patch = ground_truth.get("gold_patch") or ground_truth.get("ground_truth") or ""
    else:
        expected_patch = ground_truth or ""

    # Timeout: agent loop never started (Docker/swerex connection failure)
    if num_turns == 0:
        logger.info("SWE-agent reward: score=0.00 (0 turns / timeout)")
        return 0.0

    if alignment_failed:
        logger.info(
            "SWE-agent reward: score=0.00 (alignment failed), "
            f"turns={num_turns}, reason={alignment_failure_reason or 'unknown'}, "
            f"details={alignment_failure_details or 'n/a'}"
        )
        return 0.0

    tools = _detect_tool_usage(solution_str)
    hit_correct_file = _targeted_correct_file(solution_str, expected_patch)

    # Patch was generated — use patch comparison
    if generated_patch:
        score = compare_patches(generated_patch, expected_patch)
        logger.info(
            f"SWE-agent reward: score={score:.2f}, patch_len={len(generated_patch)}, "
            f"turns={num_turns}, correct_file={hit_correct_file}, tools={tools}"
        )
        return score

    # No patch — shaped reward based on tool usage (graduated)
    if tools["used_editor"] and hit_correct_file:
        score = 0.05
    elif tools["used_python"] or tools["used_test"]:
        score = 0.03
    elif tools["used_editor"]:
        score = 0.02
    elif (tools["used_cat"] or tools["used_ls"]) and hit_correct_file:
        score = 0.02
    elif tools["used_cat"] or tools["used_ls"]:
        score = 0.01
    elif num_turns <= 2:
        score = -0.1
    else:
        score = 0.0

    # Long and fruitless: many turns but never even tried editing
    if num_turns >= 10 and not tools["used_editor"] and score >= 0.0:
        score = -0.05

    logger.info(
        f"SWE-agent reward: score={score:.2f} (no patch), turns={num_turns}, "
        f"correct_file={hit_correct_file}, tools={tools}"
    )
    return score
