"""Thin adapters between HACPO's pure functions and verl's V1 model engine."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import ray
import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.workers.utils.padding import no_padding_2_padding

from .hacpo_core_algos import HacpoLossConfig, compute_hacpo_loss
from .hacpo_trajectory import TrainingView

_HACPO_MASKED_METRIC_COMPONENTS = {
    "self_sequence_ratio": ("self_sequence_ratio_sum", "self_sample_count"),
    "cross_sequence_ratio": ("cross_sequence_ratio_sum", "cross_sample_count"),
    "self_clip_fraction": ("self_clip_count", "self_sample_count"),
    "cross_clip_lower_fraction": ("cross_clip_lower_count", "cross_sample_count"),
    "cross_clip_upper_fraction": ("cross_clip_upper_count", "cross_sample_count"),
    "cross_clip_lower": ("cross_clip_lower_sum", "cross_sample_count"),
}
_HACPO_MASKED_METRIC_NAMES = frozenset(_HACPO_MASKED_METRIC_COMPONENTS)
_HACPO_ADDITIVE_COMPONENT_NAMES = frozenset(
    name for components in _HACPO_MASKED_METRIC_COMPONENTS.values() for name in components
)


def finalize_hacpo_metrics(metrics: dict[str, Any], *, prefix: str = "actor/hacpo/") -> dict[str, Any]:
    """Reconstruct self/cross masked means after verl reduces minibatches."""

    finalized = dict(metrics)
    for public_name, (numerator_name, denominator_name) in _HACPO_MASKED_METRIC_COMPONENTS.items():
        numerator_key = f"{prefix}{numerator_name}"
        denominator_key = f"{prefix}{denominator_name}"
        if numerator_key not in finalized or denominator_key not in finalized:
            raise KeyError(f"missing HACPO metric components {numerator_key!r} or {denominator_key!r}")
        numerator = float(finalized[numerator_key])
        denominator = float(finalized[denominator_key])
        finalized[f"{prefix}{public_name}"] = numerator / denominator if denominator > 0 else 0.0

    for name in _HACPO_ADDITIVE_COMPONENT_NAMES:
        finalized.pop(f"{prefix}{name}", None)
    return finalized


@dataclass
class MinibatchUpdateCounter:
    """Recover the optimizer mini-batch index not exposed by V1."""

    update_id: str | None = None
    index: int = 0

    def next(self, update_id: str) -> int:
        if not update_id:
            raise ValueError("hacpo_update_id must be a non-empty string")
        if update_id != self.update_id:
            self.update_id = update_id
            self.index = 0
        current = self.index
        self.index += 1
        return current


class HacpoTrainingWorker(TrainingWorker):
    """Training worker that injects HACPO's stepwise clipping index."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hacpo_update_counter = MinibatchUpdateCounter()

    def train_batch(self, data):
        update_id = tu.get_non_tensor_data(data, "hacpo_update_id", None)
        update_index = self._hacpo_update_counter.next(update_id)
        tu.assign_non_tensor_data(data, "hacpo_update_index", update_index)
        return super().train_batch(data)


def apply_rollout_rank_offset(
    rollout: Any,
    rank_offset: int,
    *,
    job_id: str,
    local_world_size: int,
) -> None:
    """Apply one policy namespace to a verl hybrid rollout adapter."""

    if rank_offset < 0:
        raise ValueError("rank_offset must be non-negative")
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    previous_offset = getattr(rollout, "_hacpo_rank_offset", None)
    if previous_offset is not None:
        if previous_offset != rank_offset:
            raise RuntimeError(
                f"rollout already has HACPO rank offset {previous_offset}, cannot change it to {rank_offset}"
            )
        return

    rollout.replica_rank += rank_offset
    rollout._hacpo_rank_offset = rank_offset

    if hasattr(rollout, "zmq_handle"):
        local_rank = rollout.rollout_rank % local_world_size
        rollout.zmq_handle = (
            f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{rollout.replica_rank}-rank-{local_rank}.sock"
        )


class HacpoActorRolloutRefWorker(ActorRolloutRefWorker):
    """Standard V1 hybrid worker with the HACPO training worker installed."""

    actor_worker_cls = HacpoTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_hacpo_rollout_rank_offset(self, rank_offset: int) -> None:
        """Move this policy's rollout into a disjoint replica-rank range."""

        apply_rollout_rank_offset(
            self.rollout,
            rank_offset,
            job_id=str(ray.get_runtime_context().get_job_id()),
            local_world_size=int(os.environ["RAY_LOCAL_WORLD_SIZE"]),
        )


def build_model_tensordict(views: Sequence[TrainingView]) -> TensorDict:
    """Build a no-padding batch with a consistent jagged layout."""

    if not views:
        raise ValueError("views must be non-empty")
    rows: dict[str, list[torch.Tensor]] = {
        "prompts": [],
        "responses": [],
        "input_ids": [],
        "attention_mask": [],
        "position_ids": [],
        "response_mask": [],
        "loss_mask": [],
    }
    for view in views:
        prompts = torch.tensor(view.prompt_ids, dtype=torch.int64)
        responses = torch.tensor(view.response_ids, dtype=torch.int64)
        input_ids = torch.cat((prompts, responses))
        response_mask = torch.ones_like(responses, dtype=torch.bool)
        rows["prompts"].append(prompts)
        rows["responses"].append(responses)
        rows["input_ids"].append(input_ids)
        rows["attention_mask"].append(torch.ones_like(input_ids, dtype=torch.bool))
        rows["position_ids"].append(torch.arange(input_ids.numel(), dtype=torch.int64))
        rows["response_mask"].append(response_mask)
        rows["loss_mask"].append(response_mask)

    return TensorDict(
        {name: torch.nested.as_nested_tensor(values, layout=torch.jagged) for name, values in rows.items()},
        batch_size=[len(views)],
    )


def build_training_tensordict(
    views: Sequence[TrainingView],
    *,
    source_mean_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    is_self: torch.Tensor,
    source_to_target_capability_ratio: torch.Tensor,
    update_id: str,
) -> TensorDict:
    """Build the no-padding TensorDict consumed by V1's model engine."""

    batch_size = len(views)
    fields = {
        "source_mean_log_probs": source_mean_log_probs,
        "advantages": advantages,
        "is_self": is_self,
        "source_to_target_capability_ratio": source_to_target_capability_ratio,
    }
    for name, value in fields.items():
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(value.shape)}")
    if not update_id:
        raise ValueError("update_id must be non-empty")

    batch = build_model_tensordict(views)
    batch["hacpo_source_mean_log_probs"] = source_mean_log_probs.float()
    batch["hacpo_advantages"] = advantages.float()
    batch["hacpo_is_self"] = is_self.bool()
    batch["hacpo_capability_ratio"] = source_to_target_capability_ratio.float()
    tu.assign_non_tensor_data(batch, "hacpo_update_id", update_id)
    return batch


def hacpo_ppo_loss(
    config: Any,
    hacpo_config: HacpoLossConfig,
    model_output,
    data,
    dp_group=None,
):
    """verl engine loss wrapper for the HACPO objective and optional KL/entropy."""

    del dp_group
    target_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]
    config.global_batch_info["dp_size"] = dp_size
    config.global_batch_info["batch_num_tokens"] = batch_num_tokens
    config.global_batch_info["global_batch_size"] = global_batch_size
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    uses_global_normalization = (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    )
    metric_aggregation = AggregationType.SUM if uses_global_normalization else AggregationType.MEAN
    minibatch_update_index = tu.get_non_tensor_data(data, "hacpo_update_index", 0)

    selected_fields = [
        "response_mask",
        "hacpo_source_mean_log_probs",
        "hacpo_advantages",
        "hacpo_is_self",
        "hacpo_capability_ratio",
    ]
    if config.use_kl_loss:
        selected_fields.append("ref_log_prob")
    padded = data.select(*selected_fields).to_padded_tensor()
    response_mask = padded["response_mask"].bool()

    local_pg_loss, hacpo_metrics = compute_hacpo_loss(
        target_log_probs=target_log_probs,
        target_response_mask=response_mask,
        source_mean_log_probs=padded["hacpo_source_mean_log_probs"],
        advantages=padded["hacpo_advantages"],
        is_self=padded["hacpo_is_self"],
        source_to_target_capability_ratio=padded["hacpo_capability_ratio"],
        minibatch_update_index=minibatch_update_index,
        config=hacpo_config,
    )

    # Engine micro-batches accumulate gradients. Scale each local sequence mean
    # so their sum, followed by DDP's mean reduction, is the global sequence mean.
    local_batch_size = target_log_probs.shape[0]
    if global_batch_size is not None:
        pg_loss = local_pg_loss * local_batch_size / global_batch_size * dp_size
    else:
        pg_loss = local_pg_loss

    metrics = Metric.from_dict(
        {
            f"actor/hacpo/{name}": value
            for name, value in hacpo_metrics.items()
            if name not in _HACPO_MASKED_METRIC_NAMES and name not in _HACPO_ADDITIVE_COMPONENT_NAMES
        },
        aggregation=AggregationType.MEAN,
    )
    for name in _HACPO_ADDITIVE_COMPONENT_NAMES:
        metrics[f"actor/hacpo/{name}"] = Metric(
            value=hacpo_metrics[name],
            aggregation=AggregationType.SUM,
        )
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        ref_log_prob = padded["ref_log_prob"]
        self_mask = padded["hacpo_is_self"].bool()
        if bool(self_mask.any()):
            kld = kl_penalty(
                logprob=target_log_probs,
                ref_logprob=ref_log_prob,
                kl_penalty=config.kl_loss_type,
            )
            self_response_mask = response_mask & self_mask.unsqueeze(-1)
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=self_response_mask,
                loss_agg_mode=config.loss_agg_mode,
                **config.global_batch_info,
            )
        else:
            # update_actor expects a differentiable loss for every minibatch.
            kl_loss = target_log_probs.sum() * 0.0
        policy_loss += config.kl_loss_coef * kl_loss
        metrics["actor/kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["actor/kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
