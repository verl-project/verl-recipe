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
SWE Agent Reward Function

Computes reward based on patch correctness:
1. Simple mode: Compare generated patch with expected_patch
2. SWE-bench mode: Run test cases to verify patch

Reward levels:
- Fully correct: 1.0
- Partially correct: 0.5
- Generated valid patch but incorrect: 0.1
- Failed to generate patch: 0.0
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_patch(patch: str) -> str:
    """Normalize a patch string for comparison.
    
    Args:
        patch: Raw patch string.
        
    Returns:
        Normalized patch string.
    """
    if not patch:
        return ""
    
    # Remove blank lines
    lines = [line.rstrip() for line in patch.strip().split("\n")]
    lines = [line for line in lines if line]
    
    # Remove commit hash from git diff header
    normalized_lines = []
    for line in lines:
        # Skip index lines
        if line.startswith("index "):
            continue
        # Skip empty lines
        if not line.strip():
            continue
        normalized_lines.append(line)
    
    return "\n".join(normalized_lines)


def extract_changed_files(patch: str) -> List[str]:
    """Extract list of changed files from a patch.
    
    Args:
        patch: Patch string.
        
    Returns:
        List of file paths.
    """
    if not patch:
        return []
    
    files = []
    # Match diff --git a/path b/path
    pattern = r"diff --git a/(.+?) b/(.+)"
    matches = re.findall(pattern, patch)
    for match in matches:
        files.append(match[1])  # Use the b/ path
    
    return files


def compare_patches_simple(generated: str, expected: str) -> float:
    """Simple comparison of two patches.
    
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
    gen_files = set(extract_changed_files(generated))
    exp_files = set(extract_changed_files(expected))
    
    if not exp_files:
        return 0.1 if gen_files else 0.0
    
    # File overlap ratio
    file_overlap = len(gen_files & exp_files) / len(exp_files)
    
    if file_overlap == 1.0:
        # All files match, give partial score
        return 0.5
    elif file_overlap > 0:
        # Partial file match
        return 0.2 + 0.3 * file_overlap
    else:
        # No file match, but a patch was generated
        return 0.1


def compute_swe_agent_reward(
    generated_patch: Optional[str],
    reward_model_config: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute SWE Agent reward.
    
    Args:
        generated_patch: Generated patch.
        reward_model_config: Reward model configuration.
        extra_info: Additional information.
        
    Returns:
        Reward score (0.0 - 1.0).
    """
    style = reward_model_config.get("style", "swe_agent")
    
    if style == "swe_agent":
        # Simple mode: directly compare patches
        expected_patch = reward_model_config.get("ground_truth", "")
        return compare_patches_simple(generated_patch, expected_patch)
    
    elif style == "swe_bench":
        # SWE-bench mode: requires running tests
        # Using simple comparison for now; actual test execution can be integrated later
        gold_patch = reward_model_config.get("gold_patch", "")
        
        if not generated_patch:
            return 0.0
        
        # Simple comparison
        score = compare_patches_simple(generated_patch, gold_patch)
        
        # TODO: Integrate SWE-bench test execution
        # This requires:
        # 1. Clone repo at specified commit
        # 2. Apply generated patch
        # 3. Run test cases
        # 4. Compute reward based on test results
        
        return score
    
    else:
        logger.warning(f"Unknown reward style: {style}, using default comparison")
        ground_truth = reward_model_config.get("ground_truth", "")
        return compare_patches_simple(generated_patch, ground_truth)


def compute_batch_rewards(
    generated_patches: List[Optional[str]],
    reward_model_configs: List[Dict[str, Any]],
    extra_infos: Optional[List[Dict[str, Any]]] = None,
) -> List[float]:
    """Compute rewards in batch.
    
    Args:
        generated_patches: List of generated patches.
        reward_model_configs: List of reward model configurations.
        extra_infos: List of additional information.
        
    Returns:
        List of reward scores.
    """
    if extra_infos is None:
        extra_infos = [None] * len(generated_patches)
    
    rewards = []
    for patch, config, info in zip(generated_patches, reward_model_configs, extra_infos):
        reward = compute_swe_agent_reward(patch, config, info)
        rewards.append(reward)
    
    return rewards


# Test code
if __name__ == "__main__":
    # Test simple comparison
    expected = """diff --git a/1.txt b/2.txt
similarity index 100%
rename from 1.txt
rename to 2.txt"""
    
    generated_correct = expected
    generated_partial = """diff --git a/1.txt b/2.txt
--- a/1.txt
+++ b/2.txt"""
    generated_wrong = """diff --git a/other.txt b/other.txt
--- a/other.txt
+++ b/other.txt"""
    
    config = {"style": "swe_agent", "ground_truth": expected}
    
    print("Testing SWE Agent Reward Function:")
    print(f"Correct patch: {compute_swe_agent_reward(generated_correct, config)}")  # Should be 1.0
    print(f"Partial patch: {compute_swe_agent_reward(generated_partial, config)}")  # Should be ~0.5
    print(f"Wrong patch: {compute_swe_agent_reward(generated_wrong, config)}")  # Should be ~0.1
    print(f"Empty patch: {compute_swe_agent_reward(None, config)}")  # Should be 0.0
