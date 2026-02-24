# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.text_encoder import Tokenizer
from recipe.mm_dance_grpo.model.modeling_sora_model import ModelingSoraModelTrain
from torch import nn
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        actor_module (nn.Module): Actor or ref module
        config (ActorConfig): Actor config
        scheduler (mindspeed_mm.models.diffusion, DiffusionModel): Actor scheduler. Defaults to None.
        tokenizer (mindspeed_mm.models.text_encoder, Tokenizer): Model tokenizer. Defaults to None.
    """

    def __init__(self, actor_module: nn.Module, config: ActorConfig, scheduler: DiffusionModel, tokenizer: Tokenizer):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_train = ModelingSoraModelTrain(actor_module, tokenizer, scheduler)
        self.config = config
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(config.get("dtype", "bfloat16"))

    def forward_micro_batch(
            self,
            latents,
            pre_latents,
            i,
            text_hidden_states=None,
            negative_text_hidden_states=None,
    ) -> torch.Tensor:
        """
        GRPO one step implementation for diffusion model training

        Args:
            latents: Current latents
            pre_latents: Previous latents
            i: Current step index
            text_hidden_states: Text embeddings for conditioning
            negative_text_hidden_states: Negative prompts for CFG

        Returns:
            log_prob: Log probability for PPO bookkeeping
        """
        # Forward pass through the actor module
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            log_probs = self.actor_train.train(
                latents,
                pre_latents,
                i,
                text_hidden_states,
                negative_text_hidden_states,
            )
        return log_probs

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        raise NotImplementedError("compute_log_prob is not implemented for DataParallelPPOActor")

    def update_policy(self, data: DataProto) -> dict:
        raise NotImplementedError("update_policy is not implemented for DataParallelPPOActor")
