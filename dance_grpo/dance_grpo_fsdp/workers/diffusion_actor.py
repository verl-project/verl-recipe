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
DiffusionActor – PPO actor for diffusion-model (DiT) training.

Extends :class:`~verl.workers.actor.BasePPOActor` with diffusion-specific
forward logic and a complete :meth:`update_policy` that runs the DanceGRPO
PPO-clip optimisation loop over denoising timesteps.

The heavy algorithm logic (advantage normalisation, SDE log-prob, PPO-clip
loss, KL regulariser) is delegated to :mod:`recipe.dance_grpo.algorithms.grpo_diffusion`.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from recipe.dance_grpo.algorithms.grpo_diffusion import (
    compute_diffusion_ppo_loss,
    compute_grpo_advantages,
    compute_mean_prediction_kl,
    compute_sde_log_prob,
    compute_sde_std,
    euler_sde_step_mean,
)
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.actor import BasePPOActor

__all__ = ["DiffusionActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DiffusionActor(BasePPOActor):
    """PPO actor for diffusion (DiT) models.

    Wraps a DiT module with an optimizer and lr_scheduler, and implements the
    full DanceGRPO training loop in :meth:`update_policy`.

    Args:
        config: Actor configuration node (``actor_rollout_ref.actor``).
        dit_module: The DiT model (FSDP2-wrapped, fp32 weights).
        optimizer: AdamW optimizer for *dit_module*.
        lr_scheduler: Learning-rate scheduler.
        ref_dit_module: Optional frozen reference DiT for KL regularisation.
    """

    def __init__(
        self,
        config,
        dit_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        ref_dit_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config)
        self.dit = dit_module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ref_dit = ref_dit_module
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(config.get("dtype", "bfloat16"))

    # ------------------------------------------------------------------
    # BasePPOActor abstract interface (stubs – diffusion does not use these)
    # ------------------------------------------------------------------

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:  # type: ignore[override]
        """Not used for diffusion actors; trajectories are recorded during rollout."""
        raise NotImplementedError("DiffusionActor does not support compute_log_prob")

    # ------------------------------------------------------------------
    # DiT forward helpers
    # ------------------------------------------------------------------

    def _dit_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        freqs_cis,
        neg_text_hidden_states: Optional[torch.Tensor] = None,
        neg_text_attention_mask: Optional[torch.Tensor] = None,
        img_hidden_states: Optional[torch.Tensor] = None,
        img_attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the DiT forward pass, with optional classifier-free guidance.

        When ``cfg_scale > 1`` and negative embeddings are provided, two
        forward passes are executed (cond + uncond) and the outputs are
        combined as ``uncond + cfg_scale * (cond - uncond)``.

        Returns:
            Velocity prediction tensor of the same shape as *latents*.
        """
        use_cfg = cfg_scale > 1.0 and neg_text_hidden_states is not None

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            if use_cfg:
                cond_out = self.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    ar_image_hidden_states=img_hidden_states,
                    ar_image_attention_mask=img_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(cond_out, (list, tuple)):
                    cond_out = cond_out[0]
                uncond_out = self.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=neg_text_hidden_states,
                    text_attention_mask=neg_text_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(uncond_out, (list, tuple)):
                    uncond_out = uncond_out[0]
                return uncond_out + cfg_scale * (cond_out - uncond_out)
            else:
                out = self.dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    ar_image_hidden_states=img_hidden_states,
                    ar_image_attention_mask=img_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(out, (list, tuple)):
                    out = out[0]
                return out

    def _ref_dit_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        freqs_cis,
        neg_text_hidden_states: Optional[torch.Tensor] = None,
        neg_text_attention_mask: Optional[torch.Tensor] = None,
        img_hidden_states: Optional[torch.Tensor] = None,
        img_attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Reference policy forward pass (no gradient)."""
        assert self.ref_dit is not None, "ref_dit_module was not provided"
        use_cfg = cfg_scale > 1.0 and neg_text_hidden_states is not None

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            if use_cfg:
                cond_ref = self.ref_dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    ar_image_hidden_states=img_hidden_states,
                    ar_image_attention_mask=img_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(cond_ref, (list, tuple)):
                    cond_ref = cond_ref[0]
                uncond_ref = self.ref_dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=neg_text_hidden_states,
                    text_attention_mask=neg_text_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(uncond_ref, (list, tuple)):
                    uncond_ref = uncond_ref[0]
                return uncond_ref + cfg_scale * (cond_ref - uncond_ref)
            else:
                ref_out = self.ref_dit(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    ar_image_hidden_states=img_hidden_states,
                    ar_image_attention_mask=img_attention_mask,
                    freqs_cis=freqs_cis,
                    return_dict=False,
                )
                if isinstance(ref_out, (list, tuple)):
                    ref_out = ref_out[0]
                return ref_out

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self, data: DataProto) -> DataProto:
        """Run one DanceGRPO update step over all selected denoising timesteps.

        Input ``data`` is expected to contain (produced by
        :class:`~recipe.dance_grpo.workers.diffusion_worker.DiffusionActorRolloutWorker`):

        Batch tensors:
          - ``all_latents``          ``[B, T+1, C, H, W]``
          - ``all_log_probs``        ``[B, T]``
          - ``rewards``              ``[B]``
          - ``text_hidden_states``   ``[B, L, D]``
          - ``text_attention_mask``  ``[B, L]``
          - ``sigma_schedule``       ``[B, T+1]``
          - ``text_seq_len``         ``[B]``
          - (optional) ``neg_*``, ``image_*`` conditioning tensors

        Returns:
            DataProto whose ``meta_info["metrics"]`` holds a dict of training
            diagnostics (losses, ratios, grad norms, etc.).
        """
        from mammothmoda2.model.mammothmoda2_dit import RotaryPosEmbedReal

        assert self.optimizer is not None, "No optimizer – this actor is read-only."

        context_manager = getattr(self, "_ulysses_sharding_manager", None) or nullcontext()

        with context_manager:
            # ------------------------------------------------------------------ #
            # 1.  Unpack rollout data                                             #
            # ------------------------------------------------------------------ #
            device = get_device_name()
            data = data.to(device)

            all_latents = data.batch["all_latents"].float()  # [B, T+1, C, H, W]
            old_log_probs = data.batch["all_log_probs"].float()  # [B, T]
            rewards = data.batch["rewards"].float()  # [B]

            text_hs = data.batch["text_hidden_states"]
            text_am = data.batch["text_attention_mask"]

            has_neg = "negative_text_hidden_states" in data.batch.keys()
            neg_hs = data.batch.get("negative_text_hidden_states", None)
            neg_am = data.batch.get("negative_text_attention_mask", None)

            has_img = "image_hidden_states" in data.batch.keys()
            img_hs = data.batch.get("image_hidden_states", None)
            img_am = data.batch.get("image_attention_mask", None)

            # Truncate padding back to the original (unpadded) conditioning length.
            text_actual_len = int(data.batch["text_seq_len"].max().item())
            text_hs = text_hs[:, -text_actual_len:, :]
            text_am = text_am[:, -text_actual_len:]
            if has_neg and "neg_seq_len" in data.batch.keys():
                neg_len = int(data.batch["neg_seq_len"].max().item())
                neg_hs = neg_hs[:, -neg_len:, :]
                neg_am = neg_am[:, -neg_len:]
            if has_img and "image_seq_len" in data.batch.keys():
                img_len = int(data.batch["image_seq_len"].max().item())
                img_hs = img_hs[:, -img_len:, :]
                img_am = img_am[:, -img_len:]

            # ------------------------------------------------------------------ #
            # 2.  Configuration                                                   #
            # ------------------------------------------------------------------ #
            actor_cfg = self.config
            clip_range = actor_cfg.ppo_clip_range
            adv_clip_max = actor_cfg.ppo_adv_clip_max
            max_grad_norm = actor_cfg.ppo_max_grad_norm
            kl_coeff = actor_cfg.ppo_kl_coeff
            micro_bs = actor_cfg.ppo_micro_batch_size_per_gpu
            grpo_size = actor_cfg.rollout_n
            train_cfg_scale = float(actor_cfg.get("train_cfg_scale", 1.0))

            # ------------------------------------------------------------------ #
            # 3.  Sigma schedule & timestep selection                             #
            # ------------------------------------------------------------------ #
            if "sigma_schedule" in data.batch.keys():
                sigma_schedule = data.batch["sigma_schedule"][0].float().to(device)
            else:
                num_steps_fallback = actor_cfg.sampling_steps
                sigma_schedule = torch.linspace(0, 1, num_steps_fallback + 1, device=device)

            num_steps = sigma_schedule.shape[0] - 1
            timestep_fraction = actor_cfg.timestep_fraction
            n_train_timesteps = max(1, int(num_steps * timestep_fraction))
            train_timestep_indices = random.sample(range(num_steps), n_train_timesteps)

            # ------------------------------------------------------------------ #
            # 4.  Rotary positional embeddings (frozen)                           #
            # ------------------------------------------------------------------ #
            freqs_cis = RotaryPosEmbedReal.get_freqs_real(
                self.dit.config.axes_dim_rope,
                self.dit.config.axes_lens,
                theta=10000,
            )

            # ------------------------------------------------------------------ #
            # 5.  GRPO advantages                                                 #
            # ------------------------------------------------------------------ #
            batch_size = all_latents.shape[0]
            advantages, adv_metrics = compute_grpo_advantages(rewards, grpo_size)
            avg_group_reward_std = adv_metrics["avg_group_reward_std"]
            min_group_reward_std = adv_metrics["min_group_reward_std"]

            # ------------------------------------------------------------------ #
            # 6.  Training loop                                                   #
            # ------------------------------------------------------------------ #
            self.dit.train()
            self.optimizer.zero_grad()

            total_loss = total_policy_loss = total_kl_loss = 0.0
            total_frac_clipped = total_new_lp = total_old_lp = 0.0
            total_lp_diff = total_approx_kl = total_ratio_mean = 0.0
            total_ratio_max = -float("inf")
            total_ratio_min = float("inf")
            total_std_dev_t = total_diff_abs = 0.0
            debug_count = 0

            n_microbatches_per_timestep = max(1, batch_size // micro_bs)
            grad_accum_denom = n_train_timesteps * n_microbatches_per_timestep

            for timestep_idx in train_timestep_indices:
                current_latents = all_latents[:, timestep_idx]  # [B, C, H, W]
                next_latents = all_latents[:, timestep_idx + 1]  # [B, C, H, W]

                t_val = sigma_schedule[timestep_idx]
                t_next_val = sigma_schedule[timestep_idx + 1]
                dt = (t_next_val - t_val).abs()
                eta = getattr(actor_cfg, "eta", 0.3)
                std_dev_t = compute_sde_std(dt.item(), eta, device)
                total_std_dev_t += std_dev_t.item()

                timestep_tensor = t_val.to(torch.bfloat16).expand(batch_size)

                mb_indices = torch.chunk(torch.arange(batch_size, device=device), n_microbatches_per_timestep)

                for mb_idx in mb_indices:
                    mb_current = current_latents[mb_idx].to(torch.bfloat16)
                    mb_next = next_latents[mb_idx]
                    mb_text = text_hs[mb_idx].to(device)
                    mb_text_mask = text_am[mb_idx].to(device)
                    mb_adv = advantages[mb_idx]
                    mb_old_lp = old_log_probs[mb_idx, timestep_idx]
                    mb_ts = timestep_tensor[mb_idx]

                    mb_neg_hs = neg_hs[mb_idx].to(device) if has_neg else None
                    mb_neg_am = neg_am[mb_idx].to(device) if has_neg else None
                    mb_img_hs = img_hs[mb_idx].to(device) if has_img else None
                    mb_img_am = img_am[mb_idx].to(device) if has_img else None

                    use_cfg = train_cfg_scale > 1.0 and has_neg and mb_neg_hs is not None

                    # Forward pass
                    model_output = self._dit_forward(
                        latents=mb_current,
                        timestep=mb_ts,
                        text_hidden_states=mb_text,
                        text_attention_mask=mb_text_mask,
                        freqs_cis=freqs_cis,
                        neg_text_hidden_states=mb_neg_hs if use_cfg else None,
                        neg_text_attention_mask=mb_neg_am if use_cfg else None,
                        img_hidden_states=mb_img_hs,
                        img_attention_mask=mb_img_am,
                        cfg_scale=train_cfg_scale if use_cfg else 1.0,
                    )

                    # Log-prob of rollout trajectory under current policy
                    new_log_probs, prev_sample_mean, _ = compute_sde_log_prob(
                        latents=mb_current,
                        model_output=model_output,
                        next_latents=mb_next,
                        t_val=t_val.item(),
                        t_next_val=t_next_val.item(),
                        eta=eta,
                    )

                    # PPO-clip loss
                    policy_loss, ppo_debug = compute_diffusion_ppo_loss(
                        new_log_probs=new_log_probs,
                        old_log_probs=mb_old_lp.to(device),
                        advantages=mb_adv,
                        clip_range=clip_range,
                        adv_clip_max=adv_clip_max,
                    )
                    policy_loss = policy_loss / grad_accum_denom
                    loss = policy_loss

                    # Optional KL regulariser (mean-prediction KL)
                    if kl_coeff > 0.0 and self.ref_dit is not None:
                        ref_output = self._ref_dit_forward(
                            latents=mb_current,
                            timestep=mb_ts,
                            text_hidden_states=mb_text,
                            text_attention_mask=mb_text_mask,
                            freqs_cis=freqs_cis,
                            neg_text_hidden_states=mb_neg_hs if use_cfg else None,
                            neg_text_attention_mask=mb_neg_am if use_cfg else None,
                            img_hidden_states=mb_img_hs,
                            img_attention_mask=mb_img_am,
                            cfg_scale=train_cfg_scale if use_cfg else 1.0,
                        )
                        ref_mean = euler_sde_step_mean(mb_current, ref_output, t_val.item(), t_next_val.item(), eta)
                        kl_loss = compute_mean_prediction_kl(prev_sample_mean, ref_mean, std_dev_t) / grad_accum_denom
                        loss = policy_loss + kl_coeff * kl_loss

                        kl_reduced = kl_loss.detach().clone()
                        dist.all_reduce(kl_reduced, op=dist.ReduceOp.AVG)
                        total_kl_loss += kl_reduced.item()

                    loss.backward()

                    # Track metrics
                    with torch.no_grad():
                        pl_reduced = policy_loss.detach().clone()
                        dist.all_reduce(pl_reduced, op=dist.ReduceOp.AVG)
                        total_policy_loss += pl_reduced.item()

                        l_reduced = loss.detach().clone()
                        dist.all_reduce(l_reduced, op=dist.ReduceOp.AVG)
                        total_loss += l_reduced.item()

                        total_frac_clipped += ppo_debug["frac_clipped"]
                        total_new_lp += new_log_probs.mean().item()
                        total_old_lp += mb_old_lp.mean().item()
                        total_lp_diff += (new_log_probs - mb_old_lp.to(device)).mean().item()
                        total_approx_kl += ppo_debug["approx_kl"]
                        total_ratio_mean += ppo_debug["ratio_mean"]
                        total_ratio_max = max(total_ratio_max, ppo_debug["ratio_max"])
                        total_ratio_min = min(total_ratio_min, ppo_debug["ratio_min"])
                        total_diff_abs += (mb_next.float() - prev_sample_mean).abs().mean().item()
                        debug_count += 1

            # ------------------------------------------------------------------ #
            # 7.  Gradient clip + optimiser step                                 #
            # ------------------------------------------------------------------ #
            grad_norm_raw = torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_grad_norm).item()
            grad_clip_ratio = grad_norm_raw / max(max_grad_norm, 1e-8)

            self.optimizer.step()
            self.lr_scheduler.step()

            if dist.is_initialized():
                dist.barrier()

            # ------------------------------------------------------------------ #
            # 8.  Build metrics dict                                              #
            # ------------------------------------------------------------------ #
            dc = max(debug_count, 1)
            metrics = {
                "train/total_loss": total_loss,
                "train/policy_loss": total_policy_loss,
                "train/kl_loss": total_kl_loss,
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/ratio_mean": total_ratio_mean / dc,
                "train/ratio_max": total_ratio_max,
                "train/ratio_min": total_ratio_min,
                "train/grad_norm": grad_norm_raw,
                "train/grad_clip_ratio": grad_clip_ratio,
                "train/avg_group_reward_std": avg_group_reward_std,
                "train/min_group_reward_std": min_group_reward_std,
                "train/avg_sde_std": total_std_dev_t / max(n_train_timesteps, 1),
                "train/avg_diff_abs": total_diff_abs / dc,
                "train/diff_over_noise_ratio": (total_diff_abs / dc)
                / max(total_std_dev_t / max(n_train_timesteps, 1), 1e-10),
                "train/frac_clipped": total_frac_clipped / dc,
                "train/new_log_prob_mean": total_new_lp / dc,
                "train/old_log_prob_mean": total_old_lp / dc,
                "train/log_prob_diff_mean": total_lp_diff / dc,
                "train/approx_kl": total_approx_kl / dc,
                "train/advantage_mean": advantages.mean().item(),
                "train/advantage_std": advantages.std().item(),
                "train/reward_mean": rewards.mean().item(),
                "train/reward_std": rewards.std().item(),
                "train/reward_max": rewards.max().item(),
                "train/reward_min": rewards.min().item(),
            }

        return DataProto(meta_info={"metrics": metrics})
