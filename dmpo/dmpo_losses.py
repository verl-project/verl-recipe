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

from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding

from .dmpo_core_algos import DMPO_POLICY_LOSS_MODES


def dmpo_ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """PPO loss wrapper that passes the prompt uid group index to DMPO policy losses."""
    del dp_group
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    group_index = None
    if loss_mode in DMPO_POLICY_LOSS_MODES:
        group_index = tu.get(data, "uid", None)
        if group_index is None:
            raise ValueError("DMPO policy losses require uid in the training batch.")

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    policy_loss_kwargs = {
        "old_log_prob": old_log_prob,
        "log_prob": log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        "loss_agg_mode": config.loss_agg_mode,
        "config": config,
        "rollout_is_weights": rollout_is_weights,
    }
    if loss_mode in DMPO_POLICY_LOSS_MODES:
        policy_loss_kwargs["index"] = group_index
    pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kwargs)

    metrics.update(Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN))
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
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
