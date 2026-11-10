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
"""SPPO (Self-Play Preference Optimization) on the current verl core.

This is a registry-based port of this recipe. The legacy implementation next to
it (``sppo_ray_trainer.py`` / ``sppo_worker.py`` / ``dp_actor.py`` / ``config.py``,
pinned to verl bcb6386 in ``REQUIRED_VERL.txt``) subclasses
``verl.workers.fsdp_workers.ActorRolloutRefWorker`` and
``verl.workers.actor.dp_actor.DataParallelPPOActor`` and calls
``verl.trainer.ppo.reward.compute_reward`` -- all three are gone from this core,
and the legacy worker path is not what hardware plugins (e.g. the Cambricon MLU
``fsdp_mlu`` engine) register against. So instead of porting subclasses, SPPO is
expressed as the two things it actually is:

* a batch-level ``softmean`` reward baseline  -> ``@register_adv_est("sppo")``
* a sequence-level squared-error policy loss  -> ``@register_policy_loss("sppo")``

No custom worker / actor / trainer / main is needed; run the stock entry point::

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=sppo \
        +algorithm.sppo_eta=1.0 \
        actor_rollout_ref.actor.policy_loss.loss_mode=sppo \
        actor_rollout_ref.model.external_lib='[recipe.sppo.sppo_core_algos]'

Importing this module performs the registration, so it has to be imported in both
the driver (which computes advantages) and every actor worker (which computes the
loss). Two mechanisms, either of which is sufficient on its own:

* ``actor_rollout_ref.model.external_lib`` -- config-driven, imported by
  ``HFModelConfig.__post_init__`` inside each worker;
* ``VERL_USE_EXTERNAL_MODULES=...,recipe.sppo.sppo_core_algos`` -- imported by
  ``verl/__init__.py``, so it covers the driver too.

The two halves communicate through ``advantages``: the estimator writes the
*sequence-level* target ``eta * (r_i - softmean_beta(r))`` on every response token
(the same broadcasting GRPO uses for its group-relative advantage), and the loss
reads it back with a masked mean, which recovers the scalar for that sequence.
``returns`` mirrors ``advantages`` -- SPPO has no critic
(``algorithm.adv_estimator != gae`` disables it).

Numerical equivalence with the legacy implementation
----------------------------------------------------
legacy trainer (``sppo_ray_trainer.py``, ``compute_advantage`` + ``fit``)::

    beta = config.algorithm.sppo_eta
    rewards           = token_level_rewards.sum(-1)              # (bs,)
    seq_level_rewards = rewards - softmean(rewards, beta)        # (bs,)

legacy actor (``dp_actor.py``, ``compute_sppo_loss``)::

    log_ratios     = (log_prob * mask).sum(1) - (old_log_prob * mask).sum(1)
    scaled_rewards = eta * seq_level_rewards
    loss_vec       = (log_ratios - scaled_rewards) ** 2
    loss           = masked_mean(loss_vec, mask.any(1))          # mean over sequences

Both are reproduced below, with two deliberate differences:

1. the sequence reward is masked (``token_level_rewards * response_mask``); the
   legacy sum is unmasked and only equal because rewards are zero off-response.
2. the mean is taken over the *global* mini-batch via ``agg_loss`` and
   ``config.global_batch_info`` instead of over the local micro-batch, so the
   objective is invariant to DP size and micro-batch splitting (this is what
   every loss in this core does). With equal-size micro-batches the two agree.

``eta`` is used both as the softmean temperature ``beta`` and as the reward scale,
matching the legacy trainer, which read both from ``algorithm.sppo_eta``.
"""

from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss
from verl.utils import torch_functional as verl_F
from verl.workers.config import ActorConfig

#: Name this recipe registers under, for both registries.
SPPO_NAME = "sppo"

#: Config key holding ``eta``, read from the ``algorithm`` node
#: (``+algorithm.sppo_eta=...`` on the command line).
SPPO_ETA_KEY = "sppo_eta"

#: Fallback for ``algorithm.sppo_eta``; the legacy ``config/sppo_trainer.yaml`` defaulted to 1.0.
SPPO_ETA_DEFAULT = 1.0

# SPPO's loss is defined per sequence, so its aggregation is fixed rather than
# taken from actor.loss_agg_mode: with "seq-mean-token-sum" and a per-sequence
# loss spread evenly over its own tokens, agg_loss returns exactly
# (sum_i loss_vec_i) / global_batch_size * dp_size, i.e. the mean over the
# sequences of the global mini-batch -- the legacy `masked_mean(loss_vec, mask.any(1))`.
_SPPO_LOSS_AGG_MODE = "seq-mean-token-sum"


def softmean(x: torch.Tensor, beta: float, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Compute ``SoftMean_beta(x) = (1/beta) * log((1/n) * sum_i exp(beta * x_i))``.

    Falls back to the arithmetic mean when ``beta == 0``. Unchanged from the
    legacy ``recipe/sppo/sppo_ray_trainer.py``.
    """
    if beta == 0.0:
        return x.mean(dim=dim, keepdim=keepdim)

    # cast beta to tensor on same device/dtype
    beta_t = x.new_tensor(beta)
    # numerically-stable logsumexp(beta * x)
    lse = torch.logsumexp(x * beta_t, dim=dim, keepdim=keepdim)
    log_n = x.new_tensor(x.size(dim)).log()

    return (lse - log_n) / beta_t


def get_sppo_eta(config: Optional[Any] = None) -> float:
    """Read ``eta`` from the algorithm config (``algorithm.sppo_eta``).

    ``config`` is the ``algorithm`` node handed to every advantage estimator by
    ``verl.trainer.ppo.ray_trainer.compute_advantage``. It stays an OmegaConf
    ``DictConfig`` on that path (``AlgoConfig`` is only ever used as a type hint),
    which is why ``sppo_eta`` can be added from the command line with hydra's
    ``+`` prefix instead of requiring a field on a frozen core dataclass.
    """
    eta = None
    if config is not None:
        eta = config.get(SPPO_ETA_KEY, None)
    if eta is None:
        eta = SPPO_ETA_DEFAULT
    return float(eta)


@register_adv_est(SPPO_NAME)
def compute_sppo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    config: Optional[AlgoConfig] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SPPO advantage: sequence reward minus the batch-level ``softmean`` baseline.

    The baseline is computed over the whole batch handed to
    ``compute_advantage`` (one training step), not per prompt -- that is the
    "self-play" part of SPPO and the reason ``rollout.n=1`` is a valid setting.
    ``index`` (the ``uid`` group ids GRPO needs) is accepted for dispatcher
    compatibility and deliberately unused.

    Args:
        token_level_rewards: ``(bs, response_length)`` rewards; only the last valid
            token of a response is normally non-zero.
        response_mask: ``(bs, response_length)`` mask of valid response tokens.
        index: unused, see above.
        config: the ``algorithm`` config node, source of ``sppo_eta``.

    Returns:
        ``(advantages, returns)``, both ``(bs, response_length)``, holding
        ``eta * (r_i - softmean_beta(r))`` on every response token and 0 elsewhere.
    """
    eta = get_sppo_eta(config)
    mask = response_mask.to(token_level_rewards.dtype)

    # (bs,) sequence-level reward, then the batch-level softmean baseline (beta = eta).
    seq_rewards = (token_level_rewards * mask).sum(dim=-1)
    baseline = softmean(seq_rewards, eta, keepdim=True)
    target = eta * (seq_rewards - baseline)

    # Broadcast the sequence-level target over response tokens so it survives the
    # response->nested conversion the v1 trainer applies to `advantages`, and so
    # the policy loss can recover it with a masked mean.
    advantages = target.unsqueeze(-1) * mask
    return advantages, advantages.clone()


@register_policy_loss(SPPO_NAME)
def compute_policy_loss_sppo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SPPO loss: squared error between the sequence log-ratio and the SPPO target.

    ``loss = mean_i (sum_t (log_prob - old_log_prob) - eta * (r_i - softmean(r)))^2``

    Args:
        old_log_prob: ``(bs, response_length)`` log-probs of the rollout policy.
        log_prob: ``(bs, response_length)`` log-probs of the current policy.
        advantages: ``(bs, response_length)``, as written by
            :func:`compute_sppo_outcome_advantage` -- one constant per row.
        response_mask: ``(bs, response_length)`` mask of valid response tokens.
        loss_agg_mode: ignored; SPPO aggregates per sequence (see
            ``_SPPO_LOSS_AGG_MODE``). Accepted because the engine passes it to
            every registered loss.
        config: ``ActorConfig``; supplies ``global_batch_info`` for the
            global-batch normalization.
        rollout_is_weights: unsupported -- SPPO's loss is sequence-level, so
            token-level importance-sampling weights have no defined meaning here.
            Only present when ``algorithm.rollout_correction`` is enabled.

    Returns:
        ``(pg_loss, pg_metrics)`` with the legacy metric names
        ``actor/log_ratio_mean`` and ``actor/preference_mean``.
    """
    assert config is not None, "SPPO's policy loss needs config.global_batch_info to normalize over the global batch"
    if rollout_is_weights is not None:
        raise ValueError(
            "policy_loss.loss_mode=sppo does not support rollout correction (token-level importance-sampling "
            "weights are undefined for a sequence-level loss). Disable algorithm.rollout_correction."
        )

    mask = response_mask.to(log_prob.dtype)
    seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
    seq_mask = (mask.sum(dim=-1) > 0).to(log_prob.dtype)

    # Sequence-level log-ratio and the target broadcast by the advantage estimator.
    log_ratios = ((log_prob - old_log_prob) * mask).sum(dim=-1)
    preference = (advantages * mask).sum(dim=-1) / seq_lengths
    loss_vec = (log_ratios - preference) ** 2

    # Spread each sequence's loss over its own tokens so agg_loss's per-sequence
    # token-sum gives loss_vec back, then average over the global mini-batch.
    loss_mat = (loss_vec / seq_lengths).unsqueeze(-1) * mask
    pg_loss = agg_loss(loss_mat=loss_mat, loss_mask=mask, loss_agg_mode=_SPPO_LOSS_AGG_MODE, **config.global_batch_info)

    pg_metrics = {
        "actor/log_ratio_mean": verl_F.masked_mean(log_ratios, seq_mask).detach().item(),
        "actor/preference_mean": verl_F.masked_mean(preference, seq_mask).detach().item(),
    }
    return pg_loss, pg_metrics
