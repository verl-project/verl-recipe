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
Self-Distillation Policy Optimization (SDPO) loss computation.

SDPO uses the model conditioned on the ground truth answer as a "teacher" to provide
dense per-token credit assignment for policy optimization. The teacher is an EMA
of the reference policy and training policy weights.

Reference: https://arxiv.org/abs/2601.20802
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss


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


def compute_sdpo_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: SDPOConfig,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    sdpo_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute SDPO self-distillation loss.

    This loss encourages the student (current policy) to match the teacher
    (EMA of reference + training policy conditioned on correct answer).

    Args:
        student_log_probs: Log probs from student for selected tokens. Shape: (batch, seq_len)
        teacher_log_probs: Log probs from teacher for selected tokens. Shape: (batch, seq_len)
        response_mask: Mask for valid response tokens. Shape: (batch, seq_len)
        config: SDPO configuration.
        old_log_probs: Log probs from old policy (for IS clipping). Shape: (batch, seq_len)
        student_all_log_probs: Full vocab log probs from student. Shape: (batch, seq_len, vocab)
        teacher_all_log_probs: Full vocab log probs from teacher. Shape: (batch, seq_len, vocab)
        student_topk_log_probs: Top-k log probs from student. Shape: (batch, seq_len, k)
        teacher_topk_log_probs: Top-k log probs from teacher. Shape: (batch, seq_len, k)
        sdpo_mask: Additional mask for SDPO (e.g., only failed samples). Shape: (batch,)
        loss_agg_mode: Loss aggregation mode.
        rollout_is_weights: Importance sampling weights from rollout correction.

    Returns:
        loss: Scalar loss value.
        metrics: Dictionary of metrics.
    """
    metrics = {}

    loss_mask = response_mask
    if sdpo_mask is not None:
        loss_mask = loss_mask * sdpo_mask.unsqueeze(1)

    if config.full_logit_distillation:
        use_topk = config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("Top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                """Add tail bucket for probability mass not in top-k."""
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                """Renormalize top-k log probs to sum to 1."""
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("Full logit distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        # Compute KL divergence based on alpha
        if config.alpha == 0.0:
            # Forward KL: KL(teacher || student)
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif config.alpha == 1.0:
            # Reverse KL: KL(student || teacher)
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            # Generalized Jensen-Shannon Divergence
            alpha = torch.tensor(
                config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_distill_log_probs + torch.log(1 - alpha),
                    teacher_distill_log_probs + torch.log(alpha)
                ]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

        per_token_loss = kl_loss.sum(-1)  # Sum over vocab dimension
    else:
        # Non-full-logit distillation: use reverse KL on selected tokens
        assert config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        # REINFORCE-style gradient: detach the log_ratio, gradient only through student_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    # Apply IS clipping if configured
    if config.is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for IS clipping.")

        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=config.is_clip)
        per_token_loss = per_token_loss * ratio

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )

    # Metrics
    with torch.no_grad():
        mean_log_ratio = ((student_log_probs - teacher_log_probs) * response_mask).sum() / response_mask.sum().clamp(min=1)
        metrics["sdpo/teacher_student_logp_diff"] = mean_log_ratio.item()

    return loss, metrics


def update_teacher_ema(
    teacher_params,
    student_params,
    ema_rate: float,
) -> None:
    """
    Update teacher parameters using EMA of student parameters.

    teacher = (1 - ema_rate) * teacher + ema_rate * student

    Args:
        teacher_params: Iterator over teacher model parameters.
        student_params: Iterator over student model parameters.
        ema_rate: EMA update rate.
    """
    if ema_rate == 0.0:
        return

    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_params, student_params):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - ema_rate).add_(student_data, alpha=ema_rate)
