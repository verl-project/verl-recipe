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
"""GRPO group statistics that ignore environment failures.

GRPO scores a sample against its group mean. An environment outage produces
`reward=0` without a policy cause, which lowers that mean and inflates the
advantage of whatever survived — the batch then trains away from correct
behaviour. Excluding flagged samples from the baseline keeps the comparison
between trajectories that actually ran.

Registered as the `grpo_env_aware` advantage estimator, so selecting it is a
config change and the stock GRPO path is untouched.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est
from verl.workers.config import AlgoConfig

# A group whose baseline is computed from too few valid samples describes the
# outage, not the task, so the whole group is dropped instead.
MIN_VALID_PER_GROUP = 2
MAX_INVALID_FRACTION = 0.25


def split_valid(index: np.ndarray, env_invalid: np.ndarray) -> tuple[dict, set]:
    """Return per-group valid scores placeholder and the groups to drop entirely."""
    totals: dict = defaultdict(int)
    valid_counts: dict = defaultdict(int)
    for i, group in enumerate(index):
        totals[group] += 1
        if not env_invalid[i]:
            valid_counts[group] += 1

    dropped = {
        group
        for group, total in totals.items()
        if valid_counts[group] < MIN_VALID_PER_GROUP or (total - valid_counts[group]) / total > MAX_INVALID_FRACTION
    }
    return valid_counts, dropped


@register_adv_est("grpo_env_aware")
def compute_grpo_env_aware_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: AlgoConfig | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO outcome advantage with environment-invalid samples out of the baseline.

    `env_invalid` arrives per sample from the agent loop's `extra_fields`. When
    it is absent the estimator degrades to stock GRPO behaviour rather than
    guessing: an all-zero response mask is also what a policy that emits a stop
    token first produces, and that zero is a legitimate outcome.
    """
    scores = token_level_rewards.sum(dim=-1)
    non_tensor_batch = kwargs.get("non_tensor_batch") or {}
    raw_flags = non_tensor_batch.get("env_invalid")
    env_invalid = (
        np.array([bool(flag) for flag in raw_flags], dtype=bool)
        if raw_flags is not None
        else np.zeros(len(scores), dtype=bool)
    )

    _, dropped = split_valid(index, env_invalid)

    id2score: dict = defaultdict(list)
    with torch.no_grad():
        for i in range(len(scores)):
            if not env_invalid[i] and index[i] not in dropped:
                id2score[index[i]].append(scores[i])

        id2mean, id2std = {}, {}
        for group, group_scores in id2score.items():
            stacked = torch.stack(group_scores)
            id2mean[group] = torch.mean(stacked)
            id2std[group] = torch.std(stacked) if len(group_scores) > 1 else torch.tensor(1.0)

        for i in range(len(scores)):
            if env_invalid[i] or index[i] in dropped:
                scores[i] = torch.tensor(0.0, device=scores.device)
                continue
            mean = id2mean[index[i]]
            std = id2std[index[i]]
            scores[i] = (scores[i] - mean) / (std + 1e-6)

        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages
