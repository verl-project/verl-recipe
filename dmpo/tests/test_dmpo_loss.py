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

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from dmpo.config import DMPOActorConfig
from dmpo.dmpo_core_algos import DMPO_POLICY_LOSS_MODES
from dmpo.dmpo_losses import dmpo_ppo_loss
from verl.trainer.ppo.core_algos import get_policy_loss_fn


def _config():
    config = DMPOActorConfig(strategy="fsdp", rollout_n=2, ppo_micro_batch_size=2)
    object.__setattr__(config.policy_loss, "dmpo_beta", 2.0)
    object.__setattr__(config.policy_loss, "dmpo_temperature", 1.0)
    return config


def test_dmpo_policy_losses_backward():
    config = _config()
    old_log_prob = torch.zeros(4, 2)
    log_prob = torch.tensor([[-0.1, -0.2], [-1.0, -1.1], [-0.3, -0.4], [-0.8, -0.9]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    response_mask = torch.ones(4, 2, dtype=torch.bool)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    for name in DMPO_POLICY_LOSS_MODES:
        loss_fn = get_policy_loss_fn(name)
        loss, metrics = loss_fn(
            old_log_prob, log_prob, advantages, response_mask, "token-mean", config, None, index=index
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert "actor/dmpo_loss" in metrics

    loss.backward()
    assert log_prob.grad is not None
    assert torch.isfinite(log_prob.grad).all()


def test_dmpo_ppo_loss_passes_uid():
    config = _config()
    object.__setattr__(config.policy_loss, "loss_mode", "grpo_dmpo")
    bsz, prompt_len, response_len = 4, 1, 2
    uid = NonTensorStack.from_list([NonTensorData(item) for item in ["a", "a", "b", "b"]])
    data = TensorDict(
        {
            "prompts": torch.ones(bsz, prompt_len, dtype=torch.long),
            "responses": torch.ones(bsz, response_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, prompt_len + response_len, dtype=torch.long),
            "response_mask": torch.ones(bsz, response_len, dtype=torch.bool),
            "old_log_probs": torch.zeros(bsz, response_len),
            "advantages": torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
            "uid": uid,
            "dp_size": NonTensorData(1),
            "batch_num_tokens": NonTensorData(None),
            "global_batch_size": NonTensorData(None),
        },
        batch_size=[bsz],
    )
    flat_log_probs = torch.linspace(-1.0, -0.1, bsz * (prompt_len + response_len), requires_grad=True)

    loss, metrics = dmpo_ppo_loss(config, {"log_probs": flat_log_probs}, data)

    assert torch.isfinite(loss)
    assert "actor/dmpo_loss" in metrics
    loss.backward()
    assert flat_log_probs.grad is not None
