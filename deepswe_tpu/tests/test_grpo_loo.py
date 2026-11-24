# Copyright 2026 Google LLC
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
"""Unit tests for compute_grpo_loo_outcome_advantage in deepswe_tpu."""

import math

import numpy as np
import torch

from deepswe_tpu.deepswe import compute_grpo_loo_outcome_advantage
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY


def test_grpo_loo_registered() -> None:
    assert ADV_ESTIMATOR_REGISTRY["grpo_loo"] is compute_grpo_loo_outcome_advantage
    assert ADV_ESTIMATOR_REGISTRY["grpo-loo"] is compute_grpo_loo_outcome_advantage


def test_grpo_loo_matches_leave_one_out_mean_and_std() -> None:
    rewards = torch.zeros((16, 4), dtype=torch.float32)
    rewards[0, -1] = 1.0  # 1/16 pass rate
    mask = torch.tensor([[1, 1, 1, 0]] * 16, dtype=torch.float32)
    index = np.array(["q0"] * 16, dtype=object)

    adv, ret = compute_grpo_loo_outcome_advantage(rewards, mask, index)
    assert torch.equal(adv, ret)
    assert torch.all(adv[:, -1] == 0.0)

    # For row 0 (r=1), remaining 15 rows are all 0 -> loo_var = 0 -> zeroed by noise floor.
    assert adv[0, 0].item() == 0.0

    # With 2/16 pass rate, every row has positive LOO variance.
    rewards[1, -1] = 1.0
    adv2, _ = compute_grpo_loo_outcome_advantage(rewards, mask, index)
    other = torch.tensor([1.0] + [0.0] * 14)
    expected_pos = (1.0 - other.mean().item()) / (other.std(unbiased=True).item() + 1e-6)
    assert math.isclose(adv2[0, 0].item(), expected_pos, rel_tol=1e-5)
