# Copyright 2026 The Wan Team and The HuggingFace Team. All rights reserved.
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

import html
import math
import os
from typing import List, Optional, Union

import ftfy
import imageio
import regex as re
import torch
from diffusers.utils.torch_utils import randn_tensor
from megatron.training.global_vars import get_args
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.text_encoder import Tokenizer
from torch import nn

do_classifier_free_guidance = False


def flux_step(
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
        log_term = -0.5 * eta ** 2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
                            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (
                            2 * (std_dev_t ** 2))
                    )
                    - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob  # prev_sample 加了噪声的，放到下一步，pred_original_sample 没有加噪声，直接生成视频
    else:
        return prev_sample_mean, pred_original_sample


class ModelingSoraModelInference():
    def __init__(self, actor_module: nn.Module, tokenizer: Tokenizer, scheduler: DiffusionModel):
        super().__init__()
        args = get_args()
        args = args.mm.model
        self.sora_model = actor_module
        self.predictor = self.sora_model.module.predictor
        self.vae_model = self.sora_model.module.ae.model.model
        self.text_encoder = self.sora_model.module.text_encoder.text_encoders
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.num_frames, self.height, self.width = args.pipeline_config.input_size
        self.vae_scale_factor_temporal = 2 ** sum(self.vae_model.config.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae_model.config.temperal_downsample)
        self.vae_scale_factor_spatial = getattr(args.pipeline_config, "vae_scale_factor_spatial",
                                                self.vae_scale_factor_spatial)
        self.generator = None if not hasattr(args.pipeline_config, "seed") else torch.Generator().manual_seed(
            args.pipeline_config.seed)
        self.expand_timesteps = getattr(args.pipeline_config, "expand_timesteps", False)
        self.model_type = args.predictor.model_type

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_noice_latents(self, dtype):
        device = 'npu'
        shape = (
            1, self.predictor.in_dim,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            self.height // self.vae_scale_factor_spatial,
            self.width // self.vae_scale_factor_spatial,
        )
        return self.prepare_latents(shape, generator=self.generator, device=device, dtype=dtype)

    @torch.no_grad()
    def generate(self,
                 p_index: int,
                 index: int,
                 prompt_embeds,
                 negative_prompt_embeds,
                 latents):
        device = 'npu'
        all_log_probs = []
        video_paths = []
        latents = latents.repeat(prompt_embeds.shape[0], 1, 1, 1, 1)
        all_latents = [latents]
        clip_features, vae_features = None, None
        first_frame_mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
        }

        # Denoising to get clean latents
        num_inference_steps = self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        guidance_scale = self.scheduler.guidance_scale
        self.scheduler.diffusion.set_timesteps(num_inference_steps)  # reset timesteps

        for i, t in enumerate(timesteps):
            latent_model_input, timestep = self.get_latents_timestep(first_frame_mask, latents, t, vae_features)

            curr_guidance_scale = guidance_scale[0] if isinstance(guidance_scale, (list, tuple)) else guidance_scale
            noise_pred = self.sora_model(
                latent_model_input, timestep, model_kwargs.get("prompt_embeds"), **model_kwargs
            )
            if do_classifier_free_guidance:
                noise_uncond = self.sora_model(
                    latent_model_input, timestep, model_kwargs.get("negative_prompt_embeds"), **model_kwargs
                )
                noise_pred = noise_uncond + curr_guidance_scale * (noise_pred - noise_uncond)
            # 计算log_prob
            latents, pred_original, log_prob = flux_step(noise_pred, latents, 0.3, sigmas, i, prev_sample=None,
                                                         grpo=True, sde_solver=True)
            all_log_probs.append(log_prob)
            all_latents.append(latents)
        # 增加一个维度进行视频拼接
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        # 分开解码视频并保存（只需要保存最后一次扩散的视频，即推理结果视频）
        for i in range(pred_original.shape[0]):
            video_latents = pred_original[i:i + 1].to(self.vae_model.dtype)
            latents_mean = (
                torch.tensor(self.vae_model.config.latents_mean)
                .view(1, self.vae_model.config.z_dim, 1, 1, 1)
                .to(video_latents.device, video_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_model.config.latents_std).view(
                1, self.vae_model.config.z_dim, 1, 1, 1
            ).to(video_latents.device, video_latents.dtype)
            video_latents = video_latents / latents_std + latents_mean
            video = self.decode_latents(video_latents)[0]
            # 保存视频
            save_path = './temp_result/'
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"video_{p_index}_{index}_{i}_rank{torch.distributed.get_rank()}.mp4")
            imageio.mimwrite(save_path, video, fps=16, quality=6)
            video_paths.append(save_path)
        return video_paths, all_latents, all_log_probs

    def decode_latents(self, latents, value_range=(-1, 1), normalize=True, **kwargs):
        video = self.vae_model.decode(latents, **kwargs)  # [b, c, t, h, w]
        video = video.sample
        if normalize:
            low, high = value_range
            video.clamp_(min=low, max=high)
            video.sub_(low).div_(max(high - low, 1e-5))
        # [b, c, t, h, w] --> [b, t, h, w, c]
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)
        return video

    def get_latents_timestep(self, first_frame_mask, latents, t, vae_features):
        if self.expand_timesteps:
            if self.model_type == "ti2v":
                latent_model_input = (1 - first_frame_mask) * vae_features + first_frame_mask * latents
                latent_model_input = latent_model_input.to(self.predictor.dtype).contiguous().clone()
            else:
                latent_model_input = latents.to(self.predictor.dtype).contiguous().clone()
            temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1).float()
        else:
            latent_model_input = latents.to(self.predictor.dtype).contiguous().clone()
            timestep = t.expand(latents.shape[0]).to(device=latents.device).float()
        return latent_model_input, timestep

    @torch.no_grad()
    def generate_test(self,
                      step: int,
                      prompt_embeds,
                      negative_prompt_embeds, save_path, src_latents):
        device = 'npu'
        clip_features, vae_features = None, None
        first_frame_mask = torch.ones(src_latents.shape, dtype=torch.float32, device=device)
        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
        }

        # 5. Denoising to get clean latents
        num_inference_steps = self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        guidance_scale = self.scheduler.guidance_scale
        self.scheduler.diffusion.set_timesteps(num_inference_steps)  # reset timesteps
        latents = src_latents.contiguous()
        for i, t in enumerate(timesteps):
            latent_model_input, timestep = self.get_latents_timestep(first_frame_mask, latents, t, vae_features)

            curr_guidance_scale = guidance_scale[0] if isinstance(guidance_scale, (list, tuple)) else guidance_scale
            noise_pred = self.sora_model(
                latent_model_input, timestep, model_kwargs.get("prompt_embeds"), **model_kwargs
            )
            if negative_prompt_embeds is not None:
                noise_uncond = self.sora_model(
                    latent_model_input, timestep, model_kwargs.get("negative_prompt_embeds"), **model_kwargs
                )
                noise_pred = noise_uncond + curr_guidance_scale * (noise_pred - noise_uncond)
            # 计算log_prob
            latents, pred_original, log_prob = flux_step(noise_pred, latents, 0.3, sigmas, i, prev_sample=None,
                                                         grpo=True, sde_solver=True)
        video_latents = pred_original.to(self.vae_model.dtype)
        latents_mean = (
            torch.tensor(self.vae_model.config.latents_mean)
            .view(1, self.vae_model.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae_model.config.latents_std).view(
            1, self.vae_model.config.z_dim, 1, 1, 1
        ).to(video_latents.device, video_latents.dtype)
        video_latents = video_latents / latents_std + latents_mean
        video = self.decode_latents(video_latents)[0]
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"test_video_step_{step}_rank{torch.distributed.get_rank()}.mp4")
        imageio.mimwrite(save_path, video, fps=16, quality=6)
        return save_path

    def encode_texts(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prompt_preprocess(self, prompt):

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return whitespace_clean(basic_clean(prompt))

    def _get_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            max_sequence_length: int = 226,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        prompt = [self.prompt_preprocess(u) for u in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        self.text_encoder = self.text_encoder.to(device)
        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds.to(self.predictor.dtype)


class ModelingSoraModelTrain():
    def __init__(self, actor_module: nn.Module, tokenizer: Tokenizer, scheduler: DiffusionModel):
        super().__init__()
        self.sora_model = actor_module
        self.predictor = self.sora_model.module.predictor
        self.vae_model = self.sora_model.module.ae.model.model
        self.text_encoder = self.sora_model.module.text_encoder.text_encoders
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.expand_timesteps = getattr(get_args().mm.model.pipeline_config, "expand_timesteps", False)

    def train(
            self,
            latents,
            pre_latents,
            idx,
            prompt_embeds,
            negative_prompt_embeds,
    ):
        device = 'npu'
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        t = timesteps[idx]
        latents = latents.to(device=device)
        pre_latents = pre_latents.to(device=device)

        guidance_scale = self.scheduler.guidance_scale
        curr_guidance_scale = guidance_scale[0] if isinstance(guidance_scale, (list, tuple)) else guidance_scale

        if self.expand_timesteps:
            first_frame_mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
            temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1).to(device=latents.device).float()
        else:
            timestep = t.expand(latents.shape[0]).to(device=latents.device).float()
        pred = self.sora_model(
            latents,
            timestep=timestep,
            prompt=prompt_embeds,
            video_mask=None,
            prompt_mask=None
        )
        if do_classifier_free_guidance:
            uncond_states = negative_prompt_embeds if negative_prompt_embeds is not None else torch.zeros_like(
                prompt_embeds)
            pred_u = self.sora_model(
                latents,
                timestep=timestep,
                prompt=uncond_states,
                video_mask=None,
                prompt_mask=None
            )
            pred = pred_u + curr_guidance_scale * (pred - pred_u)

        latents, pred_original, log_prob = flux_step(pred, latents, 0.3, sigmas, idx, prev_sample=pre_latents,
                                                     grpo=True, sde_solver=True)
        return log_prob
