"""HACPO equations from https://arxiv.org/abs/2603.02604."""

import math
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class HacpoLossConfig:
    """Hyperparameters in the paper's Appendix E, equations (56)--(57)."""

    self_clip_low: float
    self_clip_high: float
    cross_clip_delta: float = 0.2
    cross_clip_step: float = 0.0
    alpha: float = 1.0
    max_abs_log_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.self_clip_low < 0 or self.self_clip_high < 0:
            raise ValueError("self clipping radii must be non-negative")
        if not 0 <= self.cross_clip_delta <= 1:
            raise ValueError("cross_clip_delta must be in [0, 1]")
        if self.cross_clip_step < 0:
            raise ValueError("cross_clip_step must be non-negative")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_abs_log_ratio is not None and self.max_abs_log_ratio <= 0:
            raise ValueError("max_abs_log_ratio must be positive when set")

    def cross_clip_lower(self, minibatch_update_index: int) -> float:
        """Return ``1 - delta + m * delta_step`` capped at the upper bound."""

        if minibatch_update_index < 0:
            raise ValueError("minibatch_update_index must be non-negative")
        return min(1.0, 1.0 - self.cross_clip_delta + minibatch_update_index * self.cross_clip_step)


def _validate_vector(name: str, value: torch.Tensor, batch_size: int) -> None:
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(value.shape)}")


def _capability_tensor(
    agent_id: str,
    capabilities: Mapping[str, float | torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if agent_id not in capabilities:
        raise KeyError(f"missing capability for agent {agent_id!r}")
    value = torch.as_tensor(capabilities[agent_id], dtype=reference.dtype, device=reference.device)
    if value.numel() != 1 or not bool(torch.isfinite(value)) or value.item() <= 0:
        raise ValueError(f"capability for agent {agent_id!r} must be a finite positive scalar")
    return value.reshape(())


@torch.no_grad()
def compute_hacpo_advantages(
    rewards: torch.Tensor,
    prompt_ids: Sequence[Hashable],
    source_agent_ids: Sequence[str],
    *,
    learner_agent_id: str,
    capabilities: Mapping[str, float | torch.Tensor],
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
    expected_source_agents: Sequence[str] | None = None,
    source_reward_weights: Mapping[str, float | torch.Tensor] | None = None,
    require_equal_samples_per_source: bool = True,
    disable_weighting_for_uniform_rewards: bool = True,
    binary_baseline_guard: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the capability-aware advantage from equations (5)--(6)."""

    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got {tuple(rewards.shape)}")
    batch_size = rewards.shape[0]
    if len(prompt_ids) != batch_size or len(source_agent_ids) != batch_size:
        raise ValueError("rewards, prompt_ids, and source_agent_ids must have the same length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must all be finite")

    expected_sources = tuple(expected_source_agents) if expected_source_agents is not None else None
    if expected_sources is not None and len(set(expected_sources)) != len(expected_sources):
        raise ValueError("expected_source_agents contains duplicates")

    learner_capability = _capability_tensor(learner_agent_id, capabilities, rewards)
    effective_source_weights = None
    if source_reward_weights is not None:
        effective_source_weights = {
            source_id: _capability_tensor(source_id, source_reward_weights, rewards)
            for source_id in set(source_agent_ids)
        }
    grouped_indices: dict[Hashable, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        grouped_indices[prompt_id].append(index)

    advantages = torch.empty_like(rewards)
    baselines = torch.empty_like(rewards)
    for prompt_id, indices in grouped_indices.items():
        counts = Counter(source_agent_ids[index] for index in indices)
        if expected_sources is not None and set(counts) != set(expected_sources):
            raise ValueError(f"prompt {prompt_id!r} has sources {sorted(counts)}, expected {sorted(expected_sources)}")
        if require_equal_samples_per_source and len(set(counts.values())) > 1:
            raise ValueError(f"prompt {prompt_id!r} has unequal samples per source: {dict(counts)}")

        group_rewards = rewards[indices]
        if disable_weighting_for_uniform_rewards and torch.unique(group_rewards).numel() == 1:
            # Capability weights should not create a signal for a uniform-reward group.
            baseline = group_rewards.mean()
        else:
            weighted_rewards = []
            for index in indices:
                source_id = source_agent_ids[index]
                if effective_source_weights is None:
                    source_capability = _capability_tensor(source_id, capabilities, rewards)
                    # Equation (6): omega^(learner, source) = P_learner / P_source.
                    source_weight = learner_capability / source_capability
                else:
                    # Use the same clipped capability ratio as the policy loss.
                    source_weight = effective_source_weights[source_id]
                weighted_rewards.append(rewards[index] * source_weight)
            baseline = torch.stack(weighted_rewards).mean()

            if binary_baseline_guard and baseline > 1.0:
                learner_rewards = [rewards[index] for index in indices if source_agent_ids[index] == learner_agent_id]
                if not learner_rewards:
                    raise ValueError(
                        f"prompt {prompt_id!r} has no self samples for binary baseline guard "
                        f"of learner {learner_agent_id!r}"
                    )
                baseline = torch.stack(learner_rewards).mean()

        if normalize_by_std and len(indices) > 1:
            denominator = group_rewards.std(correction=1)
        else:
            denominator = rewards.new_tensor(1.0)
        denominator = denominator + epsilon if normalize_by_std else denominator

        group_advantages = (group_rewards - baseline) / denominator
        advantages[indices] = group_advantages
        baselines[indices] = baseline

    return advantages, baselines


def sequence_mean_log_prob(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Return per-response mean log-probability in its own tokenizer space."""

    if log_probs.shape != response_mask.shape or log_probs.ndim != 2:
        raise ValueError(
            "log_probs and response_mask must be rank-2 tensors with identical shapes, "
            f"got {tuple(log_probs.shape)} and {tuple(response_mask.shape)}"
        )
    mask = response_mask.to(dtype=log_probs.dtype)
    lengths = mask.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("every response must contain at least one probability-bearing token")
    return (log_probs * mask).sum(dim=-1) / lengths


def compute_hacpo_loss(
    *,
    target_log_probs: torch.Tensor,
    target_response_mask: torch.Tensor,
    source_mean_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    is_self: torch.Tensor,
    source_to_target_capability_ratio: torch.Tensor,
    minibatch_update_index: int,
    config: HacpoLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute HACPO loss with the sequence-level ratio from equation (59)."""

    batch_size = target_log_probs.shape[0]
    target_mean_log_probs = sequence_mean_log_prob(target_log_probs, target_response_mask)
    _validate_vector("source_mean_log_probs", source_mean_log_probs, batch_size)
    _validate_vector("advantages", advantages, batch_size)
    _validate_vector("is_self", is_self, batch_size)
    _validate_vector("source_to_target_capability_ratio", source_to_target_capability_ratio, batch_size)

    if not bool(torch.isfinite(source_mean_log_probs).all()):
        raise ValueError("source_mean_log_probs must all be finite")
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("advantages must all be finite")
    if not bool(torch.isfinite(source_to_target_capability_ratio).all()) or bool(
        (source_to_target_capability_ratio <= 0).any()
    ):
        raise ValueError("source_to_target_capability_ratio must contain finite positive values")

    self_mask = is_self.to(dtype=torch.bool, device=target_log_probs.device)
    source_mean_log_probs = source_mean_log_probs.to(
        dtype=target_mean_log_probs.dtype, device=target_mean_log_probs.device
    )
    advantages = advantages.to(dtype=target_mean_log_probs.dtype, device=target_mean_log_probs.device)
    capability_ratio = source_to_target_capability_ratio.to(
        dtype=target_mean_log_probs.dtype, device=target_mean_log_probs.device
    )

    log_ratio = target_mean_log_probs - source_mean_log_probs
    if config.max_abs_log_ratio is not None:
        log_ratio = log_ratio.clamp(min=-config.max_abs_log_ratio, max=config.max_abs_log_ratio)
    sequence_ratio = log_ratio.exp()

    self_clipped_ratio = sequence_ratio.clamp(
        min=1.0 - config.self_clip_low,
        max=1.0 + config.self_clip_high,
    )
    self_objective = torch.minimum(sequence_ratio * advantages, self_clipped_ratio * advantages)

    cross_lower = config.cross_clip_lower(minibatch_update_index)
    cross_clipped_ratio = sequence_ratio.clamp(min=cross_lower, max=1.0)
    # Equation (59) detaches the alpha modulation from autograd.
    cross_objective = (
        cross_clipped_ratio * cross_clipped_ratio.detach().pow(config.alpha) * capability_ratio * advantages
    )

    objective = torch.where(self_mask, self_objective, cross_objective)
    loss = -objective.mean()

    cross_mask = ~self_mask
    zero = loss.detach().new_zeros(())

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return value[mask].float().mean() if bool(mask.any()) else zero

    metrics = {
        "objective": objective.detach().mean(),
        "sequence_ratio": sequence_ratio.detach().mean(),
        "self_sequence_ratio": masked_mean(sequence_ratio.detach(), self_mask),
        "cross_sequence_ratio": masked_mean(sequence_ratio.detach(), cross_mask),
        "self_clip_fraction": masked_mean((sequence_ratio != self_clipped_ratio).detach(), self_mask),
        "cross_clip_lower_fraction": masked_mean((sequence_ratio < cross_lower).detach(), cross_mask),
        "cross_clip_upper_fraction": masked_mean((sequence_ratio > 1.0).detach(), cross_mask),
        "cross_clip_lower": loss.detach().new_tensor(cross_lower),
        # Reduce sums and counts before reconstructing masked metrics.
        "self_sequence_ratio_sum": sequence_ratio.detach()[self_mask].float().sum(),
        "cross_sequence_ratio_sum": sequence_ratio.detach()[cross_mask].float().sum(),
        "self_clip_count": (sequence_ratio != self_clipped_ratio).detach()[self_mask].float().sum(),
        "cross_clip_lower_count": (sequence_ratio < cross_lower).detach()[cross_mask].float().sum(),
        "cross_clip_upper_count": (sequence_ratio > 1.0).detach()[cross_mask].float().sum(),
        "cross_clip_lower_sum": loss.detach().new_tensor(cross_lower) * cross_mask.detach().float().sum(),
        "self_sample_count": self_mask.detach().float().sum(),
        "cross_sample_count": cross_mask.detach().float().sum(),
    }
    return loss, metrics


class CapabilityTracker:
    """Track rolling policy capability and produce bounded ratios."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        *,
        window_size: int = 5,
        metric: Literal["positive_rate", "reward_mean"] = "positive_rate",
        minimum: float = 1e-8,
        ratio_min: float = 0.1,
        ratio_max: float = 10.0,
        warmup_steps: int = 0,
    ) -> None:
        if not agent_ids or len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids must be a non-empty sequence of unique ids")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if metric not in ("positive_rate", "reward_mean"):
            raise ValueError(f"unsupported capability metric {metric!r}")
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("minimum must be finite and positive")
        if not 0 < ratio_min <= ratio_max:
            raise ValueError("capability ratio bounds must satisfy 0 < min <= max")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.agent_ids = tuple(agent_ids)
        self.window_size = window_size
        self.metric = metric
        self.minimum = minimum
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.warmup_steps = warmup_steps
        self.steps = 0
        self._history = {agent_id: deque(maxlen=window_size) for agent_id in self.agent_ids}

    def update(self, rewards: torch.Tensor, source_agent_ids: Sequence[str]) -> dict[str, float]:
        if rewards.ndim != 1 or rewards.shape[0] != len(source_agent_ids):
            raise ValueError("rewards must be a vector aligned with source_agent_ids")
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("rewards must all be finite")

        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
        for reward, source_id in zip(rewards, source_agent_ids, strict=True):
            if source_id not in self._history:
                raise ValueError(f"unknown source agent {source_id!r}")
            grouped[source_id].append(reward)
        missing = set(self.agent_ids) - set(grouped)
        if missing:
            raise ValueError(f"capability batch is missing agents: {sorted(missing)}")

        for agent_id in self.agent_ids:
            values = torch.stack(grouped[agent_id]).float()
            estimate = (values > 0).float().mean() if self.metric == "positive_rate" else values.mean()
            self._history[agent_id].append(max(estimate.item(), self.minimum))
        self.steps += 1
        return self.capabilities()

    def capabilities(self) -> dict[str, float]:
        result = {}
        for agent_id, history in self._history.items():
            if not history:
                result[agent_id] = 1.0
            else:
                result[agent_id] = max(sum(history) / len(history), self.minimum)
        return result

    def ratio(self, source: str, target: str) -> float:
        if source not in self._history or target not in self._history:
            raise KeyError(f"unknown capability edge {source!r} -> {target!r}")
        if self.steps <= self.warmup_steps:
            return 1.0
        capabilities = self.capabilities()
        raw_ratio = capabilities[source] / capabilities[target]
        return min(max(raw_ratio, self.ratio_min), self.ratio_max)

    def state_dict(self) -> dict:
        return {
            "steps": self.steps,
            "agent_ids": self.agent_ids,
            "window_size": self.window_size,
            "metric": self.metric,
            "minimum": self.minimum,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "warmup_steps": self.warmup_steps,
            "history": {agent_id: list(history) for agent_id, history in self._history.items()},
        }

    def load_state_dict(self, state: Mapping) -> None:
        compatibility_fields = (
            "agent_ids",
            "window_size",
            "metric",
            "minimum",
            "ratio_min",
            "ratio_max",
            "warmup_steps",
        )
        current = self.state_dict()
        for field_name in compatibility_fields:
            loaded = tuple(state[field_name]) if field_name == "agent_ids" else state[field_name]
            if loaded != current[field_name]:
                raise ValueError(
                    f"capability state {field_name} mismatch: checkpoint={loaded!r}, runtime={current[field_name]!r}"
                )

        history = state["history"]
        if set(history) != set(self.agent_ids):
            raise ValueError("capability checkpoint history has different agents")
        for agent_id, values in history.items():
            if len(values) > self.window_size or not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"invalid capability history for agent {agent_id!r}")
            self._history[agent_id].clear()
            self._history[agent_id].extend(float(value) for value in values)
        self.steps = int(state["steps"])
        if self.steps < 0:
            raise ValueError("capability checkpoint steps must be non-negative")
