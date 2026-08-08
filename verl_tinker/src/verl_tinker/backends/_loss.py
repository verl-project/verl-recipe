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

from copy import deepcopy
from dataclasses import replace

from omegaconf import DictConfig

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.losses import ppo_loss

__all__ = ["is_ref_in_actor", "make_branching_loss"]

# VERL clamps PPO's log-ratio to 20 before exponentiating, so the largest
# ratio that can reach the dual-clip branch is exp(20) ~= 4.85e8. Tinker PPO
# has no third/dual clip; a finite value above that bound disables it while
# satisfying VERL's ``clip_ratio_c > 1`` validation.
_TINKER_PPO_DUAL_CLIP_C = 1e9


def is_ref_in_actor(config: DictConfig) -> bool:
    """Whether ref policy can reuse actor weights (LoRA: disable adapters).

    Adapted from RayPPOTrainer.__init__() (verl/trainer/ppo/ray_trainer.py).
    """
    model_config = config.actor_rollout_ref.model
    lora_config = model_config.get("lora") or {}
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = model_config.get("lora_rank", 0)
    return lora_rank > 0 or model_config.get("lora_adapter_path") is not None


def make_branching_loss(config: DictConfig):
    """Return a single loss callable that picks PPO / weighted CE /
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

    PPO and weighted CE only implement (a). The top-K branch implements
    both (a) and (b)."""
    import torch
    import torch.nn.functional as F

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.metric import AggregationType, Metric

    # ``ppo_loss`` reads ``clip_ratio /
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

    def actor_config_for(spec: dict):
        """Build an isolated request config; ppo_loss mutates global_batch_info."""
        loss_name = spec["name"]
        if loss_name == "custom_from_config":
            # Preserve every startup setting while isolating mutable fields that
            # verl updates during ppo_loss (notably global_batch_info).
            return deepcopy(actor_cfg)

        loss_mode = {
            "ppo": "vanilla",
            # Tinker's ``logprobs`` are behavior/rollout-policy log probs and
            # are translated to ``old_log_probs``. That is already VERL's
            # bypass-mode input contract, so compute token-TIS in the loss.
            "importance_sampling": "bypass_mode",
        }.get(loss_name, loss_name)
        rollout_correction = replace(actor_cfg.policy_loss.rollout_correction)
        if loss_name == "importance_sampling":
            rollout_correction = replace(
                rollout_correction,
                rollout_is="token",
                rollout_is_threshold=2.0,
                rollout_is_batch_normalize=False,
                rollout_rs=None,
                rollout_rs_threshold=None,
                bypass_mode=True,
                loss_type="reinforce",
            )
        policy_loss = replace(
            actor_cfg.policy_loss,
            loss_mode=loss_mode,
            dro_beta=spec.get("dro_beta", actor_cfg.policy_loss.dro_beta),
            rollout_correction=rollout_correction,
        )
        overrides = {
            "policy_loss": policy_loss,
            "loss_agg_mode": "token-sum",
            "entropy_coeff": 0.0,
            "use_kl_loss": False,
            "global_batch_info": {},
        }
        if loss_name in {"ppo", "cispo"}:
            overrides["clip_ratio_low"] = spec["clip_ratio_low"]
            overrides["clip_ratio_high"] = spec["clip_ratio_high"]
        if loss_name == "ppo":
            overrides["clip_ratio_c"] = _TINKER_PPO_DUAL_CLIP_C
        return replace(actor_cfg, **overrides)

    def sft_final_loss(model_output, data, dp_group=None):
        """Tinker weighted cross entropy uses a global token-sum reduction."""
        pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        log_prob = model_output["log_probs"]
        if pad_mode == DatasetPadMode.NO_PADDING:
            loss_mat = -log_prob.values()
            loss_mask = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
        else:
            loss_mat = -log_prob
            loss_mask = data["response_mask"]
        loss = agg_loss(
            loss_mat=loss_mat.unsqueeze(0) if loss_mat.ndim == 1 else loss_mat,
            loss_mask=loss_mask.unsqueeze(0) if loss_mask.ndim == 1 else loss_mask,
            loss_agg_mode="token-sum",
            dp_size=data["dp_size"],
        )
        return loss, {"loss": Metric(value=loss, aggregation=AggregationType.SUM)}

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
        dp_size = data["dp_size"]
        loss = agg_loss(
            loss_mat=loss_flat.unsqueeze(0),
            loss_mask=mask_flat.unsqueeze(0),
            loss_agg_mode="token-sum",
            dp_size=dp_size,
        )
        metrics = {
            "loss": Metric(value=loss, aggregation=AggregationType.SUM),
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
            return sft_final_loss(model_output=model_output, data=data, dp_group=dp_group)
        if mode == "topk_distill":
            return topk_final_loss(model_output, data, dp_group=dp_group)
        # This metadata is created by tinker_ops after request validation. Its
        # absence is an internal server-plumbing bug, not invalid client input.
        spec = tu.get_non_tensor_data(data=data, key="__tinker_loss_spec__", default=None)
        if spec is None:
            raise RuntimeError("Internal RL TensorDict is missing normalized __tinker_loss_spec__ metadata")
        request_cfg = actor_config_for(spec)
        loss, metrics = ppo_loss(config=request_cfg, model_output=model_output, data=data, dp_group=dp_group)
        metrics["loss"] = Metric(value=loss, aggregation=AggregationType.SUM)
        return loss, metrics

    return branching_loss
