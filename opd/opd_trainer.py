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
Online Policy Distillation (OPD) Trainer.

Extends verl's RayPPOTrainer to add teacher knowledge distillation.
After each rollout, queries an external teacher model for per-token log
probabilities via ZMQ, and injects them as ref_log_prob. The existing
use_kl_loss mechanism then computes KL(student || teacher) as an
auxiliary training signal.

Architecture:
    - Zero modification to verl source code
    - Inherits RayPPOTrainer, overrides _compute_ref_log_prob
    - Teacher server: GKD recipe's ZMQ-based teacher (proxy + worker)
    - Serialization: torch.save/load (matching GKD teacher protocol)

Usage:
    See main_opd.py for the entry point and run_opd.sh for launch scripts.
"""

import io
import logging
import time
from typing import Optional

import torch
import zmq

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

logger = logging.getLogger(__name__)


def _serialize(data):
    """Serialize data using torch.save (matching GKD teacher protocol)."""
    buf = io.BytesIO()
    torch.save(data, buf)
    return buf.getbuffer()


def _deserialize(msg):
    """Deserialize data using torch.load (matching GKD teacher protocol)."""
    buf = io.BytesIO(msg)
    return torch.load(buf, weights_only=False)


class TeacherClient:
    """ZMQ client for querying teacher model log probabilities.

    Communicates with the GKD recipe's teacher server (proxy + worker)
    to obtain top-k log probabilities for generated sequences.

    The teacher returns top-k (token_id, log_prob) pairs per position.
    This client extracts the log probability of the actual next token
    at each position to produce per-token teacher log probs.

    Args:
        server_ip: Teacher proxy server IP address.
        server_port: Teacher proxy frontend port.
    """

    def __init__(self, server_ip: str = "127.0.0.1", server_port: int = 15555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{server_port}")
        logger.info(f"TeacherClient connected to {server_ip}:{server_port}")

    def get_teacher_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int,
    ) -> torch.Tensor:
        """Query teacher for per-token log probs on response tokens.

        Args:
            input_ids: (batch_size, seq_len) full sequences (prompt + response).
            attention_mask: (batch_size, seq_len) attention mask.
            response_length: Length of response portion.

        Returns:
            teacher_log_probs: (batch_size, response_length) log probability of
                each response token under the teacher model.
        """
        batch_size, seq_len = input_ids.shape
        prompt_length = seq_len - response_length
        teacher_log_probs = torch.zeros(batch_size, response_length, dtype=torch.float32)

        for i in range(batch_size):
            mask = attention_mask[i].bool()
            ids = input_ids[i][mask].tolist()
            if len(ids) < 2:
                continue

            # Query teacher for top-k log probs of the full sequence
            request = _serialize({
                "prompt_token_ids": [ids],
                "temperature": 0.0,
                "max_tokens": 1,
                "only_response": False,
            })
            self.socket.send(request)
            response = _deserialize(self.socket.recv())

            if response.get("status") != "ok" or len(response["teacher_topk_logprobs"]) == 0:
                logger.warning(f"Teacher error for sample {i}: {response.get('reason', 'empty')}")
                continue

            topk_logps = response["teacher_topk_logprobs"][0]  # (valid_len, k)
            topk_indices = response["teacher_topk_indices"][0]  # (valid_len, k)

            # Extract log prob of actual next token at each response position.
            #
            # Teacher logprob layout: row j of topk_logps gives P(token | ids[0:j+1]),
            # i.e. the distribution over ids[j+1]. So to get the teacher's prediction
            # for the t-th response token ids[actual_prompt_len + t], we read from
            # row (actual_prompt_len - 1 + t).
            actual_prompt_len = int(attention_mask[i][:prompt_length].sum().item())

            # Compute a per-sample fallback for tokens outside teacher's top-k.
            # Use the remaining probability mass spread uniformly over non-top-k tokens:
            #   fallback = log((1 - sum(top_k_probs)) / (vocab_size - k))
            k = topk_logps.shape[1] if topk_logps.dim() == 2 else 1
            vocab_size = 151936  # Qwen series vocab size; conservative default

            for t in range(response_length):
                src_idx = actual_prompt_len - 1 + t
                target_token_pos = actual_prompt_len + t
                if src_idx < 0 or src_idx >= topk_logps.shape[0] or target_token_pos >= len(ids):
                    break
                next_token = ids[target_token_pos]
                match = (topk_indices[src_idx] == next_token).nonzero(as_tuple=True)
                if len(match[0]) > 0:
                    teacher_log_probs[i, t] = topk_logps[src_idx, match[0][0]].item()
                else:
                    # Token not in teacher's top-k. Estimate its log-prob from the
                    # residual probability mass: P_residual = 1 - sum(top_k_probs).
                    topk_probs = topk_logps[src_idx].exp()
                    residual_mass = max(1.0 - topk_probs.sum().item(), 1e-10)
                    remaining_tokens = max(vocab_size - k, 1)
                    teacher_log_probs[i, t] = torch.tensor(
                        residual_mass / remaining_tokens
                    ).log().item()

        return teacher_log_probs


class OPDTrainer(RayPPOTrainer):
    """Online Policy Distillation trainer.

    Extends RayPPOTrainer by replacing the reference model log probability
    computation with queries to an external teacher model. This enables
    knowledge distillation from a larger teacher to a smaller student
    during on-policy RL training.

    The training objective becomes:
        L = L_GRPO + kl_loss_coef * KL(student || teacher)

    where L_GRPO is the standard GRPO policy gradient loss and the KL term
    encourages the student to match the teacher's token-level distribution.

    Args:
        config: verl training configuration.
        teacher_ip: Teacher server IP address.
        teacher_port: Teacher server port.
        kl_loss_coef: Coefficient for the teacher KL distillation loss.
            - 0.0: No teacher influence (pure GRPO baseline).
            - 0.001: Weak teacher guidance.
            - 0.01: Moderate teacher guidance (recommended starting point).
            - 0.1: Strong teacher guidance (may cause catastrophic forgetting).
        **kwargs: Additional arguments passed to RayPPOTrainer.
    """

    def __init__(
        self,
        config,
        teacher_ip: str = "127.0.0.1",
        teacher_port: int = 15555,
        kl_loss_coef: float = 0.01,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)

        # Enable reference policy path (teacher replaces ref model)
        self.use_reference_policy = True
        self.ref_in_actor = True  # Skip creating separate ref worker group

        # Enable KL loss in actor with teacher log probs
        self.config.actor_rollout_ref.actor.use_kl_loss = True
        self.config.actor_rollout_ref.actor.kl_loss_coef = kl_loss_coef
        self.config.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"

        self.teacher_ip = teacher_ip
        self.teacher_port = teacher_port
        self._teacher_client: Optional[TeacherClient] = None

        logger.info(f"OPDTrainer initialized: teacher={teacher_ip}:{teacher_port}, kl_loss_coef={kl_loss_coef}")

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Override: query teacher model instead of reference model.

        This method is called by the training loop after rollout generation
        and reward computation. It queries the external teacher for per-token
        log probabilities on the generated response tokens.

        Args:
            batch: Training batch containing input_ids, attention_mask, and responses.

        Returns:
            DataProto containing ref_log_prob tensor of shape (batch_size, response_length).
        """
        t0 = time.time()

        if self._teacher_client is None:
            self._teacher_client = TeacherClient(self.teacher_ip, self.teacher_port)

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        response_length = responses.shape[1]

        teacher_lp = self._teacher_client.get_teacher_log_probs(
            input_ids, attention_mask, response_length
        )

        elapsed = time.time() - t0
        logger.info(
            f"Teacher query: {input_ids.shape[0]} samples, "
            f"response_length={response_length}, took {elapsed:.2f}s"
        )

        return DataProto.from_dict({"ref_log_prob": teacher_lp})
