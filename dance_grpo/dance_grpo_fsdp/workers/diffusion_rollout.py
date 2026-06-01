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
Diffusion rollout for DanceGRPO.

``GRPOMockScheduler`` injects the SDE math into the built-in
``decode_diffusion_image`` pipeline while recording latent trajectories and
per-step log-probabilities.

``DiffusionRollout`` wraps the full MLLM→DiT→VAE pipeline and is the
single object responsible for generating rollout batches.
"""

from __future__ import annotations

import logging
import math
import os
import re
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

__all__ = ["GRPOMockScheduler", "DiffusionRollout"]


# ---------------------------------------------------------------------------
# SDE scheduler
# ---------------------------------------------------------------------------


class GRPOMockScheduler:
    """Mock scheduler that records SDE trajectories for GRPO training.

    Replaces ``FlowMatchEulerDiscreteScheduler`` during rollout via
    ``unittest.mock.patch`` so the existing ``decode_diffusion_image`` pipeline
    can be reused without modification.

    The timestep schedule mirrors the native ``FlowMatchEulerDiscreteScheduler``
    with ``dynamic_time_shift=True``:
    ``t' = t / (m + t*(1-m))``  where ``m = sqrt(num_tokens) / 40``.
    This concentrates steps near the low-noise end at higher resolutions.

    Args:
        sample_steps: Number of denoising steps.
        eta: SDE noise level (0 → pure ODE; larger → more stochastic).
        num_tokens: Optional latent spatial token count for resolution-adaptive
            time-warping.  Pass ``latent_h * latent_w`` from the rollout.
    """

    def __init__(
        self,
        sample_steps: int = 10,
        eta: float = 0.01,
        num_tokens: int | None = None,
    ) -> None:
        self.sample_steps = sample_steps
        self.eta = eta

        timesteps = torch.linspace(0, 1, sample_steps + 1)
        if num_tokens is not None:
            m = (num_tokens**0.5) / 40.0
            timesteps = timesteps / (m + timesteps * (1.0 - m))
        self.timesteps = timesteps

        self.all_latents: list[torch.Tensor] = []
        self.all_log_probs: list[torch.Tensor] = []
        self.current_step_idx: int = 0

    def set_timesteps(self, num_inference_steps: int, device):
        self.timesteps = self.timesteps.to(device)
        return self.timesteps[:-1]

    def compute_dance_grpo_step(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        t: float,
        t_next: float,
        eta: float,
        prev_sample: torch.Tensor | None = None,
        sde_solver: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One SDE step with log-probability computation.

        The computation is performed in float32 to ensure numerical consistency
        between the old log-probs recorded here and the new log-probs
        recomputed during actor update (when policy weights are unchanged the
        ratio must be exactly 1.0).

        Returns:
            prev_sample: Next noisy latent.
            log_prob: Per-sample scalar log-probability, shape ``[B]``.
        """
        dt = t_next - t
        latents = latents.to(torch.float32)
        model_output = model_output.to(torch.float32)

        x_hat = latents + (1.0 - t) * model_output
        prev_sample_mean = latents + dt * model_output

        score_estimate = -(latents - t * x_hat) / ((1.0 - t) ** 2 + 1e-12)

        std_dev_t = eta * torch.sqrt(torch.abs(torch.tensor(dt, device=latents.device, dtype=torch.float32)) + 1e-12)

        if sde_solver:
            prev_sample_mean = prev_sample_mean + 0.5 * (eta**2) * score_estimate * dt

        if prev_sample is None:
            noise = torch.randn_like(prev_sample_mean)
            prev_sample = prev_sample_mean + noise * std_dev_t

        diff = prev_sample.to(torch.float32) - prev_sample_mean
        var_t = (std_dev_t**2).clamp_min(1e-20)
        log_std = torch.log(std_dev_t.clamp_min(1e-20))
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=latents.device, dtype=torch.float32))
        log_prob = -(diff**2) / (2 * var_t) - log_std - 0.5 * log_two_pi
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob

    def step(self, model_output, timestep, sample, return_dict=False):
        # Record the input latent on the very first step.
        if self.current_step_idx == 0:
            self.all_latents.append(sample.clone())

        t = self.timesteps[self.current_step_idx].item()
        t_next = self.timesteps[self.current_step_idx + 1].item()

        prev_sample, log_prob = self.compute_dance_grpo_step(
            latents=sample,
            model_output=model_output,
            t=t,
            t_next=t_next,
            eta=self.eta,
            prev_sample=None,
        )

        self.all_latents.append(prev_sample.clone())
        self.all_log_probs.append(log_prob)
        self.current_step_idx += 1

        return (prev_sample,)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


class DiffusionRollout:
    """Full multimodal diffusion rollout (MLLM → DiT → VAE).

    Wraps the ``Mammothmoda2`` inference pipeline.  A single call to
    :meth:`generate_sequences` generates *n* images per prompt, records the
    full SDE trajectory (latents + log-probs) and the MLLM text conditioning,
    and returns everything packed into a :class:`~verl.DataProto`.

    Args:
        mllm_model: Frozen MLLM model for text/image conditioning.
        dit_model: DiT model for denoising.
        vae_model: VAE decoder.
        full_model: The complete Mammothmoda2Model (used for ``generate`` and
            ``decode_diffusion_image``).
        processor: Multimodal processor / tokenizer.
        config: Rollout-specific config node (``actor_rollout_ref.rollout``).
    """

    def __init__(
        self,
        mllm_model: nn.Module,
        dit_model: nn.Module,
        vae_model: nn.Module,
        full_model,
        processor,
        config,
    ) -> None:
        self.config = config
        self.mllm_model = mllm_model
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.full_model = full_model
        self.processor = processor

        # Set during ``fit()`` by the worker so sub-directories are structured.
        self._current_global_step: int = 0
        self._current_experiment_name: str = "dance_grpo"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_or_truncate_conditioning(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad (left) or truncate (right) conditioning tensors to *max_seq_len*."""
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states should be 3D [B, L, D], got {tuple(hidden_states.shape)}")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask should be 2D [B, L], got {tuple(attention_mask.shape)}")

        cur_len = hidden_states.shape[1]
        if cur_len == max_seq_len:
            return hidden_states.contiguous(), attention_mask.contiguous()

        if cur_len > max_seq_len:
            return hidden_states[:, -max_seq_len:, :].contiguous(), attention_mask[:, -max_seq_len:].contiguous()

        pad_len = max_seq_len - cur_len
        hs_pad = torch.zeros(
            hidden_states.shape[0],
            pad_len,
            hidden_states.shape[2],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        am_pad = torch.zeros(
            attention_mask.shape[0],
            pad_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return (
            torch.cat([hs_pad, hidden_states], dim=1).contiguous(),
            torch.cat([am_pad, attention_mask], dim=1).contiguous(),
        )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_minibatch(self, data: DataProto) -> DataProto:
        """Generate images for a single mini-batch of prompts.

        Runs MLLM → diffusion decode pipeline with the GRPO mock scheduler
        injected, then packs trajectory data and conditioning tensors into a
        :class:`~verl.DataProto`.

        Args:
            data: DataProto with ``non_tensor_batch["prompts"]`` (1-D array of
                prompt strings).  Expects all prompts to be identical (they
                are repeated *n* times at the caller level).

        Returns:
            DataProto containing:
              - ``batch["all_latents"]``  – ``[B, T+1, C, H, W]``
              - ``batch["all_log_probs"]`` – ``[B, T]``
              - ``batch["sigma_schedule"]`` – ``[B, T+1]``
              - ``batch["text_hidden_states"]`` + mask
              - optional neg / image conditioning tensors
              - ``non_tensor_batch["all_images"]`` – numpy uint8 HWC arrays
              - ``non_tensor_batch["image_paths"]`` – saved image paths
        """
        device = get_device_name()
        cfg = self.config

        h = int(getattr(cfg, "height", 512))
        w = int(getattr(cfg, "width", 512))
        vae_scale_factor = int(getattr(cfg, "vae_scale_factor", 16))
        num_inference_steps = int(getattr(cfg, "num_inference_steps", 40))
        eta = float(getattr(cfg, "eta", 0.3))
        conditioning_max_length = int(getattr(cfg, "prompt_length", 512))
        init_same_noise = bool(getattr(cfg, "init_same_noise", True))
        output_dir = str(getattr(cfg, "output_dir", "logs"))
        cfg_scale = float(getattr(cfg, "cfg_scale", 7.0))
        text_cfg_scale = float(getattr(cfg, "text_cfg_scale", 9.0))

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)

        prompts = data.non_tensor_batch["prompts"]
        rollout_n = len(prompts)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        from mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT
        from mammothmoda2.utils import decode_diffusion_image

        main_prompt = prompts[0]
        messages = [{"role": "user", "content": [{"type": "text", "text": str(main_prompt)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            num_images_per_prompt=1,
            cfg_scale=cfg_scale,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)

        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            generated_ids, attention_mask = self.full_model.generate(**inputs, cfg_scale=cfg_scale)

        generated_ids = torch.repeat_interleave(generated_ids, repeats=rollout_n, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, repeats=rollout_n, dim=0)
        for key, val in inputs.items():
            if val is not None and isinstance(val, torch.Tensor):
                inputs[key] = torch.repeat_interleave(val, repeats=rollout_n, dim=0)

        latent_h = 2 * h // vae_scale_factor
        latent_w = 2 * w // vae_scale_factor
        num_tokens = latent_h * latent_w
        grpo_scheduler = GRPOMockScheduler(sample_steps=num_inference_steps, eta=eta, num_tokens=num_tokens)

        # Capture conditioning tensors produced inside decode_diffusion_image.
        captured_conditioning: dict = {}
        import mammothmoda2.utils.t2i_utils as _t2i_utils

        _orig_encode_full_prompts = _t2i_utils.encode_full_prompts

        def _capturing_encode_full_prompts(*args, **kwargs):
            result = _orig_encode_full_prompts(*args, **kwargs)
            text_cond, text_cond_mask, image_cond, image_cond_mask, neg_cond, neg_mask = result
            captured_conditioning["text_hidden_states"] = text_cond.detach().cpu()
            captured_conditioning["text_attention_mask"] = text_cond_mask.detach().cpu()
            captured_conditioning["text_seq_len"] = text_cond.shape[1]
            if image_cond is not None:
                captured_conditioning["image_hidden_states"] = image_cond.detach().cpu()
                captured_conditioning["image_attention_mask"] = image_cond_mask.detach().cpu()
                captured_conditioning["image_seq_len"] = image_cond.shape[1]
            if neg_cond is not None:
                captured_conditioning["negative_text_hidden_states"] = neg_cond.detach().cpu()
                captured_conditioning["negative_text_attention_mask"] = neg_mask.detach().cpu()
                captured_conditioning["neg_seq_len"] = neg_cond.shape[1]
            return result

        step_dir = os.path.join(
            output_dir,
            self._current_experiment_name,
            f"step_{self._current_global_step:06d}",
        )
        os.makedirs(step_dir, exist_ok=True)

        with (
            patch(
                "mammothmoda2.utils.t2i_utils.FlowMatchEulerDiscreteScheduler",
                return_value=grpo_scheduler,
            ),
            patch(
                "mammothmoda2.utils.t2i_utils.retrieve_timesteps",
                return_value=(
                    grpo_scheduler.set_timesteps(num_inference_steps, device),
                    num_inference_steps,
                ),
            ),
            patch(
                "mammothmoda2.utils.t2i_utils.encode_full_prompts",
                side_effect=_capturing_encode_full_prompts,
            ),
        ):
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                imgs_list = decode_diffusion_image(
                    prompt_ids=inputs.input_ids,
                    generated_ids=generated_ids,
                    attention_mask=attention_mask,
                    negative_ids=inputs.get("negative_ids", None),
                    negative_mask=inputs.get("negative_mask", None),
                    model=self.full_model,
                    tokenizer=self.processor.tokenizer,
                    output_dir=output_dir,
                    num_images_per_prompt=rollout_n,
                    text_guidance_scale=text_cfg_scale,
                    vae_scale_factor=16,
                    cfg_range=(0.0, 1.0),
                    num_inference_steps=num_inference_steps,
                    height=h,
                    width=w,
                    device=device,
                    init_same_noise=init_same_noise,
                    return_img_only=True,
                )

        safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", str(main_prompt)).strip("_")[:32] or "prompt"
        image_paths: list[str] = []
        for idx, decoded_img in enumerate(imgs_list):
            fname = f"rank_{rank}_img_{idx}_{safe_prompt}.png"
            fpath = os.path.join(step_dir, fname)
            decoded_img.save(fpath)
            image_paths.append(fpath)

        # Pack trajectory
        all_latents = torch.stack(grpo_scheduler.all_latents, dim=1)  # [B, T+1, C, H, W]
        all_log_probs = torch.stack(grpo_scheduler.all_log_probs, dim=1)  # [B, T]
        sigma_schedule = grpo_scheduler.timesteps  # [T+1]
        sigma_schedule_tensor = sigma_schedule.unsqueeze(0).expand(rollout_n, -1).contiguous()

        # Pack conditioning with uniform sequence length for DataProto.concat
        text_hs = captured_conditioning["text_hidden_states"].expand(rollout_n, -1, -1).contiguous()
        text_am = captured_conditioning["text_attention_mask"].expand(rollout_n, -1).contiguous()
        text_seq_len = captured_conditioning["text_seq_len"]
        text_hs, text_am = self._pad_or_truncate_conditioning(text_hs, text_am, conditioning_max_length)

        conditioning_tensors: dict = {
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "sigma_schedule": sigma_schedule_tensor,
            "text_hidden_states": text_hs,
            "text_attention_mask": text_am,
            "text_seq_len": torch.tensor([text_seq_len] * rollout_n, dtype=torch.long),
        }

        if "negative_text_hidden_states" in captured_conditioning:
            neg_hs = captured_conditioning["negative_text_hidden_states"].expand(rollout_n, -1, -1).contiguous()
            neg_am = captured_conditioning["negative_text_attention_mask"].expand(rollout_n, -1).contiguous()
            neg_seq_len = captured_conditioning["neg_seq_len"]
            neg_hs, neg_am = self._pad_or_truncate_conditioning(neg_hs, neg_am, conditioning_max_length)
            conditioning_tensors["negative_text_hidden_states"] = neg_hs
            conditioning_tensors["negative_text_attention_mask"] = neg_am
            conditioning_tensors["neg_seq_len"] = torch.tensor([neg_seq_len] * rollout_n, dtype=torch.long)

        if "image_hidden_states" in captured_conditioning:
            img_hs = captured_conditioning["image_hidden_states"].expand(rollout_n, -1, -1).contiguous()
            img_am = captured_conditioning["image_attention_mask"].expand(rollout_n, -1).contiguous()
            img_seq_len = captured_conditioning["image_seq_len"]
            img_hs, img_am = self._pad_or_truncate_conditioning(img_hs, img_am, conditioning_max_length)
            conditioning_tensors["image_hidden_states"] = img_hs
            conditioning_tensors["image_attention_mask"] = img_am
            conditioning_tensors["image_seq_len"] = torch.tensor([img_seq_len] * rollout_n, dtype=torch.long)

        return DataProto.from_dict(
            tensors=conditioning_tensors,
            non_tensors={
                "all_images": np.array(imgs_list),
                "image_paths": np.array(image_paths, dtype=object),
            },
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate *n* image sequences for each prompt in *prompts*.

        Splits the batch into mini-batches of size ``config.n``, calls
        :meth:`_generate_minibatch` for each, and concatenates results.

        Args:
            prompts: DataProto where ``non_tensor_batch["prompts"]`` has been
                pre-repeated (length ``B = num_prompts * n``).

        Returns:
            DataProto with trajectory and conditioning data for all samples.
        """
        batch_size = len(prompts)
        n = self.config.n
        mini_batches = prompts.chunk(chunks=batch_size // n)
        outputs = [self._generate_minibatch(mb) for mb in mini_batches]
        return DataProto.concat(outputs)
