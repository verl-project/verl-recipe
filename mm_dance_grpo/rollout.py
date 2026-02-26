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

import numpy as np
import torch
import torch.distributed
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.text_encoder import Tokenizer
from recipe.mm_dance_grpo.model.modeling_sora_model import ModelingSoraModelInference
from torch import nn
from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.config import ActorConfig

__all__ = ["HFRollout"]

NEGATIVE_PROMOPT_DEFAULT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


class HFRollout:
    def __init__(self, module: nn.Module, config: ActorConfig, scheduler: DiffusionModel, tokenizer: Tokenizer):
        self.config = config
        self.sora_rollout = ModelingSoraModelInference(module, tokenizer, scheduler)
        # 测试复用
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.src_latents = None

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto, p_index: int) -> DataProto:
        device = 'npu'
        prompt = prompts.non_tensor_batch['raw_prompt'][0]['content']
        prompt_embeds, negative_prompt_embeds = self.sora_rollout.encode_texts(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMOPT_DEFAULT,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=device,
        )
        prompt_embeds = torch.repeat_interleave(prompt_embeds, self.config.n, dim=0)
        negative_prompt_embeds = torch.repeat_interleave(negative_prompt_embeds, self.config.n, dim=0)

        imgs_list, all_latents_list, all_log_probs_list = [], [], []
        grpo_size = self.config.n

        num_chunks = grpo_size // self.config.micro_batch_size
        batch_indices = torch.chunk(torch.arange(grpo_size), num_chunks)

        # 一个prompt使用同一个噪声输入
        self.src_latents = self.sora_rollout.get_noice_latents(prompt_embeds.dtype)

        for index, chunk in enumerate(batch_indices):
            prompt_embeds_chunk = prompt_embeds[chunk]
            negative_prompt_embeds_chunk = negative_prompt_embeds[chunk]
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                imgs, all_latents, all_log_probs = self.sora_rollout.generate(
                    p_index,
                    index,
                    prompt_embeds_chunk,
                    negative_prompt_embeds_chunk,
                    self.src_latents.clone()
                )
            imgs_list += imgs
            all_latents_list.append(all_latents)
            all_log_probs_list.append(all_log_probs)
        all_latents_list, all_log_probs_list = torch.cat(all_latents_list), torch.cat(all_log_probs_list)
        batch = DataProto.from_dict(
            tensors={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "all_latents": all_latents_list,
                "all_log_probs": all_log_probs_list,
            },
            non_tensors={
                "all_imgs": np.array(imgs_list, dtype=object),
            },
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()
        return batch

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        file_index = prompts.non_tensor_batch['index']
        batch_size = len(file_index)
        batch_prompts = prompts.chunk(chunks=batch_size)
        output = [self._generate_minibatch(p[0], index) for index, p in enumerate(batch_prompts)]
        output = DataProto.concat(output)
        repeat_file_index = []
        for index in file_index:
            repeat_file_index.extend([index] * self.config.n)
        output.non_tensor_batch['file_index'] = repeat_file_index
        return output

    def release(self):
        pass

    def resume(self):
        pass

    def update_weights(self, data: DataProto) -> dict:
        pass

    @torch.no_grad()
    def generate_test(self, prompt: str, step: int, save_path: str) -> str:
        device = 'npu'
        if self.prompt_embeds is None and self.negative_prompt_embeds is None:
            self.prompt_embeds, self.negative_prompt_embeds = self.sora_rollout.encode_texts(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMOPT_DEFAULT,
                do_classifier_free_guidance=True,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=device,
            )
        return self.sora_rollout.generate_test(
            step,
            self.prompt_embeds,
            self.negative_prompt_embeds,
            save_path,
            self.src_latents.clone()
        )
