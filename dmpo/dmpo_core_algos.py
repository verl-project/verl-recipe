# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, register_policy_loss
from verl.utils import as_torch_index
from verl.workers.config import ActorConfig

DMPO_POLICY_LOSS_MODES = frozenset(
    {
        "dmpo_zero",
        "dmpo",
        "dmpo_js",
        "grpo_dmpo_zero",
        "grpo_dmpo",
        "grpo_dmpo_js",
        "pure_dmpo",
    }
)


def _get_dmpo_params(config: Optional[DictConfig | ActorConfig]) -> tuple[float, float]:
    beta = 1.0
    temperature = 1.0 / 15.0
    if config is None:
        return beta, temperature

    policy_loss_cfg = getattr(config, "policy_loss", None)
    if policy_loss_cfg is None:
        return beta, temperature

    return float(policy_loss_cfg.get("dmpo_beta", beta)), float(policy_loss_cfg.get("dmpo_temperature", temperature))


def _group_softmax(
    logits: torch.Tensor, group_index: torch.Tensor, num_groups: int, eps: float = 1e-10
) -> torch.Tensor:
    group_max = torch.full((num_groups,), float("-inf"), device=logits.device, dtype=logits.dtype)
    group_max = group_max.scatter_reduce(0, group_index, logits, reduce="amax", include_self=False)

    exp_val = torch.exp(logits - group_max[group_index])
    group_sum = torch.zeros(num_groups, device=logits.device, dtype=logits.dtype)
    group_sum = group_sum.index_add(0, group_index, exp_val)
    return exp_val / (group_sum[group_index] + eps)


def _compute_dmpo_distribution_loss(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor | np.ndarray | None,
    temperature: float,
    divergence: str = "mse",
    filter_zero_signal: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if index is None:
        raise ValueError("DMPO policy loss requires uid group index in the training batch.")

    group_index = as_torch_index(index, device=log_prob.device)
    if group_index.numel() != log_prob.shape[0]:
        raise ValueError(
            f"DMPO group index length must match batch size: got {group_index.numel()} and {log_prob.shape[0]}."
        )
    if group_index.numel() == 0:
        return log_prob.sum() * 0.0, {"actor/dmpo_valid_samples": 0.0, "actor/dmpo_group_count": 0.0}

    num_groups = int(group_index.max().item()) + 1
    response_mask = response_mask.to(log_prob.dtype)
    seq_len = response_mask.sum(dim=-1).clamp(min=1)
    seq_log_prob = (log_prob * response_mask).sum(dim=-1) / seq_len
    seq_advantages = (advantages * response_mask).sum(dim=-1) / seq_len

    with torch.no_grad():
        target_dist = _group_softmax(seq_advantages / temperature, group_index, num_groups, eps=eps)
        active_groups = torch.zeros(num_groups, device=log_prob.device, dtype=torch.bool)
        active_groups[group_index] = True

        sample_weight = torch.ones_like(seq_advantages)
        if filter_zero_signal:
            group_min = torch.full((num_groups,), float("inf"), device=log_prob.device, dtype=seq_advantages.dtype)
            group_max = torch.full((num_groups,), float("-inf"), device=log_prob.device, dtype=seq_advantages.dtype)
            group_min = group_min.scatter_reduce(0, group_index, seq_advantages, reduce="amin", include_self=False)
            group_max = group_max.scatter_reduce(0, group_index, seq_advantages, reduce="amax", include_self=False)
            has_signal = (group_max - group_min) > 1e-5
            sample_weight = has_signal[group_index].to(seq_advantages.dtype)

    valid_samples = sample_weight.sum()
    if filter_zero_signal and valid_samples.item() <= 0:
        return seq_log_prob.sum() * 0.0, {
            "actor/dmpo_valid_samples": 0.0,
            "actor/dmpo_group_count": active_groups.sum().detach().item(),
        }

    model_dist = _group_softmax(seq_log_prob, group_index, num_groups, eps=eps)
    if divergence == "mse":
        loss_per_sample = (target_dist - model_dist).square()
        loss = (loss_per_sample * sample_weight).sum() / valid_samples.clamp(min=1)
    elif divergence == "js":
        target_safe = torch.clamp(target_dist, min=eps)
        model_safe = torch.clamp(model_dist, min=eps)
        mixture = 0.5 * (target_safe + model_safe)
        js_per_sample = 0.5 * (
            target_safe * (torch.log(target_safe) - torch.log(mixture))
            + model_safe * (torch.log(model_safe) - torch.log(mixture))
        )
        js_per_group = torch.zeros(num_groups, device=log_prob.device, dtype=log_prob.dtype)
        js_per_group = js_per_group.index_add(0, group_index, js_per_sample * sample_weight)
        group_weight = torch.zeros(num_groups, device=log_prob.device, dtype=log_prob.dtype)
        group_weight = group_weight.index_add(0, group_index, sample_weight)
        loss = js_per_group[group_weight > 0].mean()
    else:
        raise ValueError(f"Unsupported DMPO divergence: {divergence}.")

    metrics = {
        "actor/dmpo_valid_samples": valid_samples.detach().item(),
        "actor/dmpo_group_count": active_groups.sum().detach().item(),
    }
    return loss, metrics


def _dmpo_metrics(loss: torch.Tensor, beta: float, temperature: float, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **metrics,
        "actor/dmpo_loss": loss.detach().item(),
        "actor/dmpo_beta": beta,
        "actor/dmpo_temperature": temperature,
    }


@register_policy_loss("dmpo_zero")  # type: ignore[arg-type]
def compute_policy_loss_dmpo_zero(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the DMPO distribution-matching loss while ignoring zero-signal groups."""
    del old_log_prob, loss_agg_mode, rollout_is_weights
    beta, temperature = _get_dmpo_params(config)
    dmpo_loss, metrics = _compute_dmpo_distribution_loss(
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        index=index,
        temperature=temperature,
        filter_zero_signal=True,
    )
    return dmpo_loss, _dmpo_metrics(dmpo_loss, beta, temperature, metrics)


@register_policy_loss("dmpo")  # type: ignore[arg-type]
def compute_policy_loss_dmpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the DMPO distribution-matching loss with MSE divergence."""
    del old_log_prob, loss_agg_mode, rollout_is_weights
    beta, temperature = _get_dmpo_params(config)
    dmpo_loss, metrics = _compute_dmpo_distribution_loss(
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        index=index,
        temperature=temperature,
    )
    return dmpo_loss, _dmpo_metrics(dmpo_loss, beta, temperature, metrics)


@register_policy_loss("dmpo_js")  # type: ignore[arg-type]
def compute_policy_loss_dmpo_js(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the DMPO distribution-matching loss with Jensen-Shannon divergence."""
    del old_log_prob, loss_agg_mode, rollout_is_weights
    beta, temperature = _get_dmpo_params(config)
    dmpo_loss, metrics = _compute_dmpo_distribution_loss(
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        index=index,
        temperature=temperature,
        divergence="js",
    )
    return dmpo_loss, _dmpo_metrics(dmpo_loss, beta, temperature, metrics)


def _compute_grpo_dmpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config: Optional[DictConfig | ActorConfig],
    rollout_is_weights: torch.Tensor | None,
    index: torch.Tensor | np.ndarray | None,
    divergence: str = "mse",
    filter_zero_signal: bool = False,
    pure_dmpo: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    beta, temperature = _get_dmpo_params(config)
    dmpo_loss, dmpo_metrics = _compute_dmpo_distribution_loss(
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        index=index,
        temperature=temperature,
        divergence=divergence,
        filter_zero_signal=filter_zero_signal,
    )
    total_loss = beta * dmpo_loss if pure_dmpo else pg_loss + beta * dmpo_loss
    pg_metrics.update(_dmpo_metrics(dmpo_loss, beta, temperature, dmpo_metrics))
    return total_loss, pg_metrics


@register_policy_loss("grpo_dmpo_zero")  # type: ignore[arg-type]
def compute_policy_loss_grpo_dmpo_zero(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clipped GRPO plus DMPO over non-degenerate uid groups."""
    return _compute_grpo_dmpo_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode,
        config,
        rollout_is_weights,
        index,
        filter_zero_signal=True,
    )


@register_policy_loss("grpo_dmpo")  # type: ignore[arg-type]
def compute_policy_loss_grpo_dmpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clipped GRPO plus DMPO with MSE divergence."""
    return _compute_grpo_dmpo_loss(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights, index
    )


@register_policy_loss("grpo_dmpo_js")  # type: ignore[arg-type]
def compute_policy_loss_grpo_dmpo_js(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clipped GRPO plus DMPO with Jensen-Shannon divergence."""
    return _compute_grpo_dmpo_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode,
        config,
        rollout_is_weights,
        index,
        divergence="js",
    )


@register_policy_loss("pure_dmpo")  # type: ignore[arg-type]
def compute_policy_loss_pure_dmpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    index: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute beta-scaled DMPO while still reporting vanilla PPO diagnostics."""
    return _compute_grpo_dmpo_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode,
        config,
        rollout_is_weights,
        index,
        pure_dmpo=True,
    )
