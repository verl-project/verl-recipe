# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recipe-local advantage estimators for the DR.Kernel recipe.

This module keeps the **TRLOO** (Turn-level REINFORCE Leave-One-Out) advantage
estimator self-contained inside the recipe instead of relying on a built-in
``verl`` enum entry. It registers under the string name ``"trloo"`` via
``register_adv_est`` (see ``verl/trainer/ppo/core_algos.py``), so setting
``algorithm.adv_estimator=trloo`` resolves to ``compute_trloo_outcome_advantage``
against stock upstream ``verl`` — no core patch required.

Importing this module is what performs the registration. The recipe entry points
(``main.py``, ``main_validate.py``) and the trainer actor module
(``trainer/drkernel_async_trainer.py``) import it so the estimator is registered
in whichever process computes advantages.

Precedent for recipe-local registration: ``verl/trainer/diffusion/diffusion_algos.py``.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("trloo")
def compute_trloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage for Turn-level REINFORCE Leave-One-Out (TRLOO).

    From DR.Kernel (https://github.com/hkust-nlp/KernelGYM). Each row is a
    full multi-turn trajectory; assistant-turn token spans are inferred from
    `response_mask` (contiguous runs of 1 = one assistant turn, runs of 0 =
    tool/user observation tokens). Per-turn rewards are the sum of
    `token_level_rewards` within the corresponding span. Returns are
    γ-discounted across turns within a trajectory; advantages are the LOO
    deviation across trajectories that share the same prompt and turn index.

    When no row has more than one assistant turn (e.g. a kernel-RL setup
    where the multi-turn loop is enabled but `tool_config_path` is null,
    so the model writes one response and exits), TRLOO degenerates to
    per-prompt LOO across the single turn=0 group — mathematically
    equivalent to plain RLOO. We allow this without erroring, mirroring
    DR.Kernel's `compute_multi_turn_rloo_outcome_advantage` behavior.

    Args:
        token_level_rewards: shape (bs, response_length).
        response_mask: shape (bs, response_length); 1 = LLM-generated token,
            0 = tool/user observation token (the boundary marker between
            assistant turns).
        index: shape (bs,) ndarray of prompt UIDs used for grouping.
        config: algorithm config; only `gamma` is read (defaults to 1.0).

    Returns:
        advantages: shape (bs, response_length).
        returns: shape (bs, response_length).
    """
    gamma = float(getattr(config, "gamma", 1.0)) if config is not None else 1.0
    bs, response_length = token_level_rewards.shape

    # Step 1: per-row assistant-turn spans from response_mask runs of 1.
    spans_per_row: list[list[tuple[int, int]]] = []
    for i in range(bs):
        spans: list[tuple[int, int]] = []
        in_span = False
        start = 0
        row = response_mask[i].tolist()
        for j, m in enumerate(row):
            if m == 1 and not in_span:
                start = j
                in_span = True
            elif m == 0 and in_span:
                spans.append((start, j))
                in_span = False
        if in_span:
            spans.append((start, response_length))
        spans_per_row.append(spans)

    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)

    with torch.no_grad():
        # Step 2: per-turn reward = sum of token rewards within the turn's span.
        rewards_per_row: list[list[torch.Tensor]] = []
        for i, spans in enumerate(spans_per_row):
            rewards_per_row.append([token_level_rewards[i, s:e].sum() for (s, e) in spans])

        # Step 3: γ-discounted returns within each row, last-turn-first.
        returns_per_row: list[list[torch.Tensor]] = []
        for r in rewards_per_row:
            num_turns = len(r)
            if num_turns == 0:
                returns_per_row.append([])
                continue
            ret = [None] * num_turns
            ret[num_turns - 1] = r[num_turns - 1]
            for t in range(num_turns - 2, -1, -1):
                ret[t] = r[t] + gamma * ret[t + 1]
            returns_per_row.append(ret)

        # Step 4: group (row_idx, turn_idx) by (prompt_uid, turn_idx); LOO baseline.
        group_to_members: dict[tuple, list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for i, ret_list in enumerate(returns_per_row):
            for t, ret in enumerate(ret_list):
                group_to_members[(index[i], t)].append((i, ret))

        # Step 5: spread per-turn advantage and return back to tokens within each span.
        for (_, t), members in group_to_members.items():
            n = len(members)
            if n == 1:
                # No LOO baseline available; pass-through return as advantage.
                row_idx, ret = members[0]
                s, e = spans_per_row[row_idx][t]
                advantages[row_idx, s:e] = ret
                returns[row_idx, s:e] = ret
                continue
            total = sum(r for (_, r) in members)
            for row_idx, ret in members:
                loo_baseline = (total - ret) / (n - 1)
                s, e = spans_per_row[row_idx][t]
                advantages[row_idx, s:e] = ret - loo_baseline
                returns[row_idx, s:e] = ret

    return advantages, returns
