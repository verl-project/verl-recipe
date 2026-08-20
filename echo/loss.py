from __future__ import annotations

from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def echo_ppo_loss(
    config: ActorConfig,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
    *,
    aux_token_loss_coeff: float,
):
    policy_loss, metrics = ppo_loss(config, model_output, data, dp_group)
    if "aux_token_loss_mask" not in data:
        raise ValueError("ECHO loss requires aux_token_loss_mask in the rollout batch")

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    aux_data = data.select("aux_token_loss_mask").to_padded_tensor()
    aux_token_loss = agg_loss(
        loss_mat=-log_prob,
        loss_mask=aux_data["aux_token_loss_mask"].to(log_prob.dtype),
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    policy_loss += aux_token_loss_coeff * aux_token_loss

    metrics["actor/aux_token_loss"] = Metric(
        value=aux_token_loss,
        aggregation=metrics["actor/pg_loss"].aggregation,
    )
    metrics["actor/aux_token_loss_coeff"] = aux_token_loss_coeff
    return policy_loss, metrics
