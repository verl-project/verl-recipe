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
SDPO (Self-Distillation Policy Optimization) Recipe.

This recipe implements SDPO for training language models using self-distillation
to provide dense per-token credit assignment.

Reference: https://arxiv.org/abs/2601.20802
"""

from dataclasses import dataclass
from typing import Optional

from .sdpo_advantage import compute_sdpo_loss, update_teacher_ema
from .sdpo_ray_trainer import RaySDPOTrainer


@dataclass
class SDPOConfig:
    """Configuration for SDPO self-distillation.

    Args:
        full_logit_distillation: Whether to use full-logit KL distillation.
        alpha: KL interpolation. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        ema_update_rate: EMA update rate for teacher weights (0.0 = no EMA, use ref directly).
        is_clip: Clip value for importance sampling ratio; None disables IS clipping.
        distillation_topk: If set, use top-k logits for distillation.
        distillation_add_tail: Whether to add tail bucket for top-k distillation.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0  # 0.0 = forward KL (teacher → student)
    ema_update_rate: float = 0.05
    is_clip: Optional[float] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True


__all__ = [
    "SDPOConfig",
    "RaySDPOTrainer",
    "compute_sdpo_loss",
    "update_teacher_ema",
]
