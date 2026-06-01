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
GRPO algorithm for diffusion models (DANCE-GRPO).

This module provides pure functions for:
- GRPO group-based advantage normalization
- SDE log-probability computation (matching rollout math exactly)
- PPO-clip policy loss for diffusion trajectories
- SDE-corrected Euler step helper

All functions are stateless and operate on plain tensors, making them easy to
test in isolation and reuse across different worker implementations.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "compute_grpo_advantages",
    "compute_sde_log_prob",
    "compute_diffusion_ppo_loss",
    "euler_sde_step_mean",
]


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


def compute_grpo_advantages(
    rewards: torch.Tensor,
    grpo_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO (Group Relative Policy Optimisation) advantages.

    For each group of *grpo_size* consecutive samples (produced by repeating a
    single prompt), normalise the rewards to zero mean and unit variance.

    Args:
        rewards: Float tensor of shape ``[B]`` with per-sample scalar rewards.
            Must satisfy ``B % grpo_size == 0``.
        grpo_size: Number of samples per prompt group.

    Returns:
        advantages: Float tensor of shape ``[B]`` with normalised advantages.
        metrics: Dict of diagnostic scalars (``avg_group_reward_std``,
            ``min_group_reward_std``).

    Raises:
        ValueError: If ``B`` is not divisible by *grpo_size*.
    """
    batch_size = rewards.shape[0]
    if batch_size % grpo_size != 0:
        raise ValueError(f"Batch size {batch_size} is not divisible by grpo_size {grpo_size}")

    advantages = torch.zeros_like(rewards)
    group_reward_stds: list[float] = []

    group_indices = torch.chunk(torch.arange(batch_size, device=rewards.device), batch_size // grpo_size)
    for group_idx in group_indices:
        group_rewards = rewards[group_idx].float()
        grp_mean = group_rewards.mean()
        grp_std = group_rewards.std().clamp_min(1e-8)
        advantages[group_idx] = (group_rewards - grp_mean) / grp_std
        group_reward_stds.append(group_rewards.std().item())

    metrics = {
        "avg_group_reward_std": sum(group_reward_stds) / max(len(group_reward_stds), 1),
        "min_group_reward_std": min(group_reward_stds) if group_reward_stds else 0.0,
    }
    return advantages, metrics


# ---------------------------------------------------------------------------
# SDE helpers
# ---------------------------------------------------------------------------


def euler_sde_step_mean(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    t_val: float,
    t_next_val: float,
    eta: float,
) -> torch.Tensor:
    """Compute the SDE-corrected Euler step mean (DanceGRPO formulation).

    Matches ``GRPOMockScheduler.compute_dance_grpo_step`` exactly (when
    ``sde_solver=True``).

    Args:
        latents: Noisy latents ``x_t`` of any shape.
        model_output: Velocity prediction ``v`` from the DiT.
        t_val: Current time ``t`` (scalar float, 0–1).
        t_next_val: Next time ``t_next`` (scalar float, 0–1).
        eta: SDE noise scale.

    Returns:
        SDE-corrected mean ``mu`` of the next-step distribution.
    """
    dt = t_next_val - t_val
    latents_f = latents.to(torch.float32)
    model_output_f = model_output.to(torch.float32)

    # Euler mean: x_t + dt * v
    prev_sample_mean = latents_f + dt * model_output_f

    # Score-based SDE correction
    x_hat = latents_f + (1.0 - t_val) * model_output_f
    score_estimate = -(latents_f - t_val * x_hat) / ((1.0 - t_val) ** 2 + 1e-12)
    prev_sample_mean = prev_sample_mean + 0.5 * (eta**2) * score_estimate * dt

    return prev_sample_mean


def compute_sde_std(
    dt: float,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute the SDE noise standard deviation for a single step.

    Args:
        dt: Absolute time difference ``|t_next - t|``.
        eta: SDE noise scale.
        device: Target device.

    Returns:
        Scalar tensor ``std_dev_t = eta * sqrt(|dt| + 1e-12)``.
    """
    return eta * torch.sqrt(torch.abs(torch.tensor(dt, device=device, dtype=torch.float32)) + 1e-12)


def compute_sde_log_prob(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    next_latents: torch.Tensor,
    t_val: float,
    t_next_val: float,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian log-probability of the SDE transition.

    Recomputes the log-probability of the observed next latents under the
    current policy, using the same SDE math as ``GRPOMockScheduler``.

    Args:
        latents: Current noisy latents ``x_t``, shape ``[B, C, H, W]``.
        model_output: Velocity prediction ``v``, shape ``[B, C, H, W]``.
        next_latents: Next noisy latents ``x_{t+1}``, shape ``[B, C, H, W]``.
        t_val: Current time ``t`` (float, 0–1).
        t_next_val: Next time ``t_next`` (float, 0–1).
        eta: SDE noise scale.

    Returns:
        log_probs: Per-sample log-probabilities, shape ``[B]``.
        prev_sample_mean: SDE-corrected Euler mean, shape ``[B, C, H, W]``.
        std_dev_t: Scalar SDE noise standard deviation.
    """
    device = latents.device
    dt = abs(t_next_val - t_val)

    std_dev_t = compute_sde_std(dt, eta, device)
    prev_sample_mean = euler_sde_step_mean(latents, model_output, t_val, t_next_val, eta)

    diff = next_latents.to(torch.float32) - prev_sample_mean
    var_t = (std_dev_t**2).clamp_min(1e-20)
    log_std = torch.log(std_dev_t.clamp_min(1e-20))
    log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))

    log_probs = -(diff**2) / (2.0 * var_t) - log_std - 0.5 * log_two_pi
    log_probs = log_probs.mean(dim=tuple(range(1, log_probs.ndim)))  # [B]

    return log_probs, prev_sample_mean, std_dev_t


# ---------------------------------------------------------------------------
# PPO-clip loss
# ---------------------------------------------------------------------------


def compute_diffusion_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    adv_clip_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-clip policy loss for a single microbatch / timestep.

    Args:
        new_log_probs: Log-probs under the current policy, shape ``[B]``.
        old_log_probs: Log-probs recorded during rollout, shape ``[B]``.
        advantages: Per-sample advantages, shape ``[B]``.
        clip_range: PPO epsilon clip range.
        adv_clip_max: Maximum absolute advantage after clipping.

    Returns:
        loss: Scalar policy loss (positive; we minimise it).
        debug: Dict of diagnostic scalars.
    """
    clamped_adv = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = -clamped_adv * ratio
    clipped = -clamped_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.mean(torch.maximum(unclipped, clipped))

    with torch.no_grad():
        is_clipped = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
        debug = {
            "frac_clipped": is_clipped.float().mean().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "ratio_min": ratio.min().item(),
            "approx_kl": (0.5 * ((ratio - 1.0) ** 2)).mean().item(),
        }

    return loss, debug


# ---------------------------------------------------------------------------
# KL regulariser
# ---------------------------------------------------------------------------


def compute_mean_prediction_kl(
    prev_sample_mean: torch.Tensor,
    ref_mean: torch.Tensor,
    std_dev_t: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL between two Gaussians with same variance.

    ``KL(N(mu, sigma) || N(mu_ref, sigma)) = ||mu - mu_ref||^2 / (2 sigma^2)``

    Args:
        prev_sample_mean: Current policy mean, shape ``[B, C, H, W]``.
        ref_mean: Reference policy mean, shape ``[B, C, H, W]``.
        std_dev_t: SDE noise standard deviation (scalar tensor).

    Returns:
        Scalar KL loss.
    """
    var_t = (std_dev_t**2).clamp_min(1e-20)
    return ((prev_sample_mean - ref_mean.detach()) ** 2).mean() / (2.0 * var_t)
