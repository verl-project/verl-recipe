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
"""Offline tests: no GPU, no browser, no NeMo Gym server.

pytest recipe/nemo_gym/browser/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from recipe.nemo_gym.browser.group_stats import (
    MAX_INVALID_FRACTION,
    MIN_VALID_PER_GROUP,
    compute_grpo_env_aware_advantage,
    split_valid,
)
from recipe.nemo_gym.browser.judge import render_transcript
from recipe.nemo_gym.browser.prepare_webvoyager_data import to_rollout_input


def _rewards(values: list[float]) -> torch.Tensor:
    rewards = torch.zeros(len(values), 4)
    rewards[:, -1] = torch.tensor(values)
    return rewards


def _mask(n: int) -> torch.Tensor:
    return torch.ones(n, 4, dtype=torch.int64)


def test_invalid_samples_leave_the_baseline_alone():
    """Four environment failures must not drag the group mean below the real one."""
    scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    index = np.array(["g"] * 8)
    invalid = np.array([False] * 4 + [True] * 4)

    with_flags, _ = compute_grpo_env_aware_advantage(
        _rewards(scores), _mask(8), index, non_tensor_batch={"env_invalid": invalid}
    )
    without_flags, _ = compute_grpo_env_aware_advantage(_rewards(scores), _mask(8), index)

    # Valid-only baseline is 0.75, so a correct sample keeps a modest positive
    # advantage; treating the failures as real zeros inflates it.
    assert with_flags[0, 0] < without_flags[0, 0]
    # Flagged samples carry no gradient.
    assert torch.equal(with_flags[4:], torch.zeros(4, 4))


def test_group_is_dropped_when_too_few_valid_samples_remain():
    index = np.array(["g"] * 4)
    invalid = np.array([False, True, True, True])
    advantages, _ = compute_grpo_env_aware_advantage(
        _rewards([1.0, 0.0, 0.0, 0.0]), _mask(4), index, non_tensor_batch={"env_invalid": invalid}
    )
    assert torch.equal(advantages, torch.zeros(4, 4))


def test_split_valid_thresholds():
    index = np.array(["a"] * 8 + ["b"] * 8)
    invalid = np.array([False] * 8 + [True] * 3 + [False] * 5)
    valid_counts, dropped = split_valid(index, invalid)
    assert valid_counts["a"] == 8
    assert "a" not in dropped
    # 3/8 invalid exceeds the fraction ceiling.
    assert (3 / 8) > MAX_INVALID_FRACTION
    assert "b" in dropped
    assert valid_counts["b"] >= MIN_VALID_PER_GROUP


def test_without_flags_behaviour_matches_stock_grpo():
    """No flags must not silently mean 'everything is invalid'."""
    index = np.array(["g"] * 4)
    advantages, _ = compute_grpo_env_aware_advantage(_rewards([1.0, 1.0, 0.0, 0.0]), _mask(4), index)
    assert advantages[0, 0] > 0
    assert advantages[2, 0] < 0


def test_rollout_input_carries_task_fields():
    row = to_rollout_input({"question": "find the paper", "start_url": "https://arxiv.org", "task_id": "wv-1"})
    assert row["initial_url"] == "https://arxiv.org"
    assert row["question"] == "find the paper"
    assert row["task_id"] == "wv-1"
    assert row["responses_create_params"]["input"][-1]["content"] == "find the paper"
    assert "verifier_metadata" not in row


def test_rollout_input_requires_question_and_url():
    with pytest.raises(ValueError):
        to_rollout_input({"question": "no url"})


def test_transcript_truncates_but_keeps_both_ends():
    events = [{"action": '{"action": "observe"}', "observation": "x" * 5000} for _ in range(4)]
    rendered = render_transcript(events, budget=2000)
    assert len(rendered) <= 2200
    assert "truncated" in rendered
    assert rendered.startswith("CALL")
