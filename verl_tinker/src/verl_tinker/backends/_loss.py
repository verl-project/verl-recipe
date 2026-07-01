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

"""Loss branching for the colocated Tinker server engine.

The wire-side ``loss_fn`` parameter on /forward_backward picks the TD
shape (sft / topk_distill / rl) via ``__loss_mode__``. ColocatedBackend
binds this branching loss to its actor worker group via ``set_loss_fn``
once at init.
"""

from functools import partial

from omegaconf import DictConfig

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.losses import ppo_loss, sft_loss


__all__ = ["is_ref_in_actor", "make_branching_loss"]


def is_ref_in_actor(config: DictConfig) -> bool:
    """Whether ref policy can reuse actor weights (LoRA: disable adapters).

    Adapted from RayPPOTrainer.__init__() (verl/trainer/ppo/ray_trainer.py).
    """
    lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    return lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None


def make_branching_loss(config: DictConfig):
    """Return a single loss callable that picks ppo_loss / sft_loss /
    top-K weighted-CE at call time based on the TD's ``__loss_mode__``
    non-tensor field, and handles verl's dual-call invocation
    convention.

    Set ONCE at engine init via ``actor_rollout_wg.set_loss_fn`` so the
    wire-side ``loss_fn`` parameter on /forward_backward only changes
    what the translator puts in the TD, never what we tell verl to do
    with the actor.

    verl invokes ``loss_function`` in two patterns:
      (a) Final-loss path (every micro-batch):
          ``loss_function(model_output=…, data=…, dp_group=…)``
          → must return ``(scalar_loss, metrics_dict)``.
      (b) In-forward logit-processor (only when
          ``data["distillation_use_topk"]=True``,
          transformer_impl.py:1105):
          ``loss_function(student_logits=…, data=…)``
          → must return a dict of ``(1, total_nnz)`` tensors that the
          engine stashes into ``model_output``.

    ``ppo_loss`` / ``sft_loss`` only implement (a). The top-K branch
    implements both (a) and (b). When the engine *would* try to call
    ppo/sft as a logit processor (it won't, because their TDs leave
    ``distillation_use_topk=False``), we return ``{}`` as a defensive
    no-op."""
    import torch
    import torch.nn.functional as F

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric

    # ``sft_loss`` doesn't actually read ``config`` (it's accepted for
    # signature compat); ``ppo_loss`` reads ``clip_ratio /
    # loss_agg_mode / policy_loss / use_kl_loss / global_batch_info /
    # loss_scale_factor`` — all of which are top-level fields on
    # ActorConfig populated by ``omega_conf_to_dataclass``. Don't force
    # ``dataclass_type=ActorConfig`` — that triggers ``HFModelConfig``
    # validation on ``actor.model_config``, which is a placeholder
    # until the worker fills it in during init_model. Auto-resolving
    # via the DictConfig's ``_target_`` (same call verl's own
    # ActorRolloutRefWorker makes at engine_workers.py:545) skips
    # validation.
    actor_cfg = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    ppo = partial(ppo_loss, config=actor_cfg)
    sft = partial(sft_loss, config=actor_cfg)

    def topk_logit_processor(student_logits, data):
        """In-forward logit processor for top-K weighted CE.

        verl passes ``student_logits`` shape ``(1, total_nnz, V)`` and
        expects the returned dict's values to be ``(1, total_nnz)``
        (transformer_impl.py:1108 asserts ``v.shape == log_probs.shape``).
        """
        teacher_topk_log_probs = data["teacher_logprobs"]
        teacher_topk_ids = data["teacher_ids"]
        # data["teacher_logprobs"] is nested (B, j1=valid_len, K);
        # verl's path uses ``.values().unsqueeze(0)`` to flatten to
        # ``(1, total_nnz, K)`` matching student_logits' total_nnz axis.
        tlp = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, K)
        tids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, K)

        student_log_probs = F.log_softmax(student_logits, dim=-1)  # (1, total_nnz, V)
        student_topk = student_log_probs.gather(-1, tids)  # (1, total_nnz, K)

        teacher_weights = tlp.exp()  # (1, total_nnz, K) — 0 at invalid slots (log(0)→-10→exp~5e-5; tiny)
        # Weighted CE per position: -Σ_k w_k · log_student[k].
        distillation_losses = -(teacher_weights * student_topk).sum(-1)  # (1, total_nnz)
        teacher_mass = teacher_weights.sum(dim=-1)
        student_mass = student_topk.exp().sum(dim=-1)
        return {
            "distillation_losses": distillation_losses,
            "teacher_mass": teacher_mass,
            "student_mass": student_mass,
        }

    def topk_final_loss(model_output, data, dp_group=None):
        """Aggregate the per-position top-K weighted CE that the
        logit-processor stashed into model_output."""
        # model_output["distillation_losses"] is a nested (B, j1) tensor
        # written by the engine at transformer_impl.py:1114.
        distillation_losses = model_output["distillation_losses"]
        loss_mask = data["loss_mask"]
        # Same roll-by-(-1) trick verl's sft_loss uses to align with
        # the predicted-next-token positions.
        loss_flat = distillation_losses.values()
        mask_flat = torch.roll(loss_mask.values(), shifts=-1, dims=0)
        # token-mean aggregation, matching sft_loss's default. agg_loss
        # provides cross-DP normalisation via ``batch_num_tokens``.
        batch_num_tokens = data["batch_num_tokens"]
        dp_size = data["dp_size"]
        # Reuse agg_loss for consistency with other loss paths' DP scaling.
        loss = agg_loss(
            loss_mat=loss_flat.unsqueeze(0),
            loss_mask=mask_flat.unsqueeze(0),
            loss_agg_mode="token-mean",
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
        )
        metrics = {
            "distillation/loss": Metric(value=loss, aggregation=AggregationType.SUM),
        }
        return loss, metrics

    def branching_loss(model_output=None, data=None, dp_group=None, student_logits=None, data_format="thd"):
        mode = tu.get_non_tensor_data(data=data, key="__loss_mode__", default="ppo")

        # (b) In-forward logit-processor invocation. Only the topk_distill
        # path sets ``distillation_use_topk=True`` (the gate at
        # transformer_impl.py:1105) so any other mode reaching here is an
        # upstream contract break — fail loud so the bug surfaces at the
        # first wrong-mode invocation, not as a silently-zero distillation
        # term in some downstream metric.
        if student_logits is not None:
            if mode == "topk_distill":
                return topk_logit_processor(student_logits, data)
            raise AssertionError(
                f"branching_loss invoked as in-forward logit processor with mode={mode!r}; "
                "only TDs that set distillation_use_topk=True (topk_distill mode) should reach this branch."
            )

        # (a) Final-loss invocation.
        if mode == "sft":
            return sft(model_output=model_output, data=data, dp_group=dp_group)
        if mode == "topk_distill":
            return topk_final_loss(model_output, data, dp_group=dp_group)
        return ppo(model_output=model_output, data=data, dp_group=dp_group)

    return branching_loss
