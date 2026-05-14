import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch

from .sdpo_advantage import compute_sdpo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)

def _init_teacher_module(self) -> None:
    """Initialize the teacher model after the main model is initialized."""
    if not self._is_actor:
        return

    self_distillation_cfg = self.config.actor.get("self_distillation", None)
    if self_distillation_cfg is None:
        # No self-distillation config
        raise ValueError("No self-distillation config found in the actor config.")

    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
    if teacher_regularization == "trust-region":
        self.actor.teacher_module = TrustRegionTeacher(
            ref_module=self.ref_module_fsdp,
            student_module=self.actor_module_fsdp,
            mix_coef=self_distillation_cfg.get("teacher_update_rate", 0.0),
        )
    else:
        self.actor.teacher_module = self.ref_module_fsdp

def _update_teacher(self) -> None:
    """Update the teacher model weights after each policy update."""
    self_distillation_cfg = getattr(self.config, "self_distillation", None)
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    if not self_distillation_cfg or loss_mode != "sdpo":
        return
    teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
    if teacher_regularization != "ema":
        return
    update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
    if update_rate == 0.0:
        return
    if self.teacher_module is None or self.teacher_module is self.actor_module:
        raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
    with torch.no_grad():
        for teacher_param, student_param in zip(
            self.teacher_module.parameters(),
            self.actor_module.parameters(),
            strict=True,
        ):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)


@GPUMemoryLogger(role="dp actor", logger=logger)
def update_policy(self, data: DataProto):
    """Policy update function which includes SDPO loss."""
    # make sure we are in training mode
    self.actor_module.train()

    temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
    pad_token_id = data.meta_info.get("pad_token_id", 0)
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

    self_distillation_enabled = loss_mode == "sdpo"
    self_distillation_cfg = getattr(self.config, "self_distillation", None)
    if self_distillation_enabled:
        self_distillation_required_keys = {
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "self_distillation_mask",
        }
        assert self_distillation_required_keys.issubset(set(data.batch.keys())), (
            f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"
        )

    select_keys = [
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "advantages",
    ]
    if self.use_prefix_grouper and "prompts" in data.batch.keys():
        select_keys.append("prompts")
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")
    if self_distillation_enabled:
        select_keys.extend(list(self_distillation_required_keys))
    # Include pre-computed IS weights if present in batch
    # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
    if "rollout_is_weights" in data.batch.keys():
        select_keys.append("rollout_is_weights")
    # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
    if "rollout_log_probs" in data.batch.keys():
        select_keys.append("rollout_log_probs")

    has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get("multi_modal_inputs"))
    non_tensor_select_keys = []
    if has_multi_modal_inputs:
        non_tensor_select_keys.append("multi_modal_inputs")
    if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
        non_tensor_select_keys.append("uid")

    data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

    # Split to make minibatch iterator for updating the actor
    # See PPO paper for details. https://arxiv.org/abs/1707.06347
    mini_batches = data.split(self.config.ppo_mini_batch_size)

    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

    metrics = {
        "actor/pg_loss": 0.0,
        "actor/kl_loss": 0.0,
    }
    did_update = False
    for _ in range(self.config.ppo_epochs):
        for batch_idx, mini_batch in enumerate(mini_batches):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                micro_batch_metrics = {}
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                response_mask = model_inputs["response_mask"]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                self_distillation_mask = (
                    model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                )
                if self_distillation_enabled:
                    assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                if self.config.use_dynamic_bsz:
                    loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                else:
                    loss_scale_factor = 1 / self.gradient_accumulation

                teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                if teacher_regularization == "trust-region" and self.use_fused_kernels:
                    raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                # all return: (bsz, response_length)
                return_all_logps = (
                    self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                )
                distill_topk = (
                    self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                )
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_all_logps=return_all_logps,
                    distill_topk=distill_topk,
                )
                log_prob = outputs["log_probs"]
                entropy = outputs["entropys"] if calculate_entropy else None
                student_all_logps = outputs.get("all_logps") if return_all_logps else None
                student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                # for fully_async_policy
                if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                    old_log_prob = model_inputs["old_log_probs"]
                else:
                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                # Extract pre-computed rollout correction weights if present
                # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                if self_distillation_enabled:
                    teacher_inputs = {
                        "responses": model_inputs["responses"],
                        "input_ids": model_inputs["teacher_input_ids"],
                        "attention_mask": model_inputs["teacher_attention_mask"],
                        "position_ids": model_inputs["teacher_position_ids"],
                    }
                    teacher_model = self.teacher_module or self.actor_module
                    if teacher_regularization == "trust-region" and (
                        self.teacher_module is None or self.teacher_module is self.actor_module
                    ):
                        raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                    with torch.no_grad():
                        teacher_outputs = self._forward_micro_batch(
                            teacher_inputs,
                            temperature=temperature,
                            calculate_entropy=False,
                            return_all_logps=return_all_logps,
                            distill_topk=distill_topk,
                            topk_indices=student_topk_indices,
                            module=teacher_model,
                        )
                    teacher_log_prob = teacher_outputs["log_probs"]
                    teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                    teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                    pg_loss, pg_metrics = compute_sdpo_loss(
                        student_log_probs=log_prob,
                        teacher_log_probs=teacher_log_prob,
                        response_mask=response_mask,
                        self_distillation_config=self_distillation_cfg,
                        old_log_probs=old_log_prob,
                        student_all_log_probs=student_all_logps,
                        teacher_all_log_probs=teacher_all_logps,
                        student_topk_log_probs=student_topk_logps,
                        teacher_topk_log_probs=teacher_topk_logps,
                        self_distillation_mask=self_distillation_mask,
                        loss_agg_mode=loss_agg_mode,
                        rollout_is_weights=rollout_is_weights,
                    )

                    pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                    micro_batch_metrics.update(pg_metrics)
                else:
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                    # Compute metrics using CURRENT policy π_θ vs π_rollout
                    # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    micro_batch_metrics.update(rollout_corr_metrics)

                policy_loss = pg_loss
                if calculate_entropy and entropy is not None:
                    entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                    if entropy_coeff != 0:
                        policy_loss -= entropy_agg * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                    micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * loss_scale_factor
                else:
                    loss = policy_loss * loss_scale_factor
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                append_to_dict(metrics, micro_batch_metrics)

            grad_norm = self._optimizer_step()
            if torch.isfinite(grad_norm).item():
                did_update = True
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
    self.actor_optimizer.zero_grad()
    if did_update:
        self._update_teacher()
    return metrics


def patch_dp_actor():
    """Patch the DPActor with teacher EMA"""
    from functools import wraps

    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    DataParallelPPOActor._update_teacher = _update_teacher
    DataParallelPPOActor.update_policy = update_policy

    # Add the teacher init as a separate method on the class
    ActorRolloutRefWorker._init_teacher_module = _init_teacher_module

    # Save the original init_model
    _original_init_model = ActorRolloutRefWorker.init_model

    @wraps(_original_init_model)
    def init_model_with_teacher(self):
        # Call the original init_model first
        _original_init_model(self)
        # Then add teacher initialization
        self._init_teacher_module()

    ActorRolloutRefWorker.init_model = init_model_with_teacher