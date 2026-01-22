"""
GRPO Trainer with Atropos Integration

This module implements a GRPO trainer that uses Atropos environments
for computing advantages with token-level overrides.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, compute_response_mask

from .atropos_integration import AtroposConfig, AtroposTrainerClient, convert_scalar_or_token_advantages

logger = logging.getLogger(__name__)


class RayGRPOAtroposTrainer(RayPPOTrainer):
    """
    Ray-based GRPO trainer with Atropos integration.

    This trainer extends the standard PPO trainer to use GRPO with
    Atropos environment feedback for advantage computation.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Initialize Atropos integration
        trainer_cfg = getattr(config, "trainer", None)
        atropos_cfg = {}
        if trainer_cfg is not None:
            if hasattr(trainer_cfg, "get"):
                atropos_cfg = trainer_cfg.get("atropos", {}) or {}
            else:
                atropos_cfg = getattr(trainer_cfg, "atropos", {}) or {}

        atropos_config = AtroposConfig(
            api_url=atropos_cfg.get("api_url", "http://localhost:9001"),
            timeout=atropos_cfg.get("timeout", 30),
            retry_attempts=atropos_cfg.get("retry_attempts", 10),
            retry_delay=atropos_cfg.get("retry_delay", 0.5),
            max_wait_time=atropos_cfg.get("max_wait_time", 30.0),
            use_advantages=atropos_cfg.get("use_advantages", True),
            fallback_to_standard=atropos_cfg.get("fallback_to_grpo", True),
        )

        self.atropos_client = AtroposTrainerClient(atropos_config)
        self._register_with_atropos()
        self._patch_rollout_to_use_atropos()

        # Ensure we're using GRPO
        if config.algorithm.adv_estimator != "grpo":
            logger.warning(f"Overriding adv_estimator from {config.algorithm.adv_estimator} to grpo")
            config.algorithm.adv_estimator = "grpo"

        # GRPO doesn't use critic
        self.use_critic = False
        config.algorithm.use_critic = False

        logger.info("Initialized RayGRPOAtroposTrainer with Atropos integration")

    def _register_with_atropos(self) -> None:
        trainer_cfg = getattr(self.config, "trainer", None)
        project_name = (
            getattr(trainer_cfg, "project_name", "verl_atropos") if trainer_cfg is not None else "verl_atropos"
        )
        experiment_name = getattr(trainer_cfg, "experiment_name", "grpo") if trainer_cfg is not None else "grpo"

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        batch_size = int(self.config.data.train_batch_size) * max(1, rollout_n)
        max_token_len = int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)

        registration = {
            "wandb_group": project_name,
            "wandb_project": experiment_name,
            "batch_size": batch_size,
            "max_token_len": max_token_len,
            "checkpoint_dir": (
                getattr(trainer_cfg, "default_local_dir", "./checkpoints") if trainer_cfg else "./checkpoints"
            ),
            "save_checkpoint_interval": getattr(trainer_cfg, "save_freq", 0) if trainer_cfg else 0,
            "starting_step": 0,
            "num_steps": int(getattr(trainer_cfg, "total_training_steps", 0) or 0),
        }

        if not self.atropos_client.is_available():
            raise RuntimeError(f"Atropos API not reachable at {self.atropos_client.config.api_url}")

        self.atropos_client.register_trainer(registration)

    def _patch_rollout_to_use_atropos(self) -> None:
        def _atropos_generate_sequences(_batch: DataProto, **_kwargs) -> DataProto:
            return self._fetch_atropos_batch()

        # Patch async rollout manager (used by default)
        if hasattr(self, "async_rollout_manager"):
            self.async_rollout_manager.generate_sequences = _atropos_generate_sequences

    def _fetch_atropos_batch(self) -> DataProto:
        batch_groups = self.atropos_client.get_batch()
        if not batch_groups:
            raise RuntimeError("Atropos /batch returned empty batch.")

        return self._atropos_batch_to_dataproto(batch_groups)

    def _atropos_batch_to_dataproto(self, batch_groups: list[dict[str, Any]]) -> DataProto:
        sequences = []
        response_masks = []
        scores = []
        advantages = []
        uids = []

        for group_idx, group in enumerate(batch_groups):
            group_tokens = group.get("tokens", [])
            group_masks = group.get("masks", [])
            group_scores = group.get("scores", [])
            group_advantages = group.get("advantages", [])

            for i, tokens in enumerate(group_tokens):
                sequences.append(tokens)
                mask = group_masks[i] if i < len(group_masks) else None
                response_masks.append(mask)
                score = group_scores[i] if i < len(group_scores) else None
                scores.append(score)
                adv = group_advantages[i] if i < len(group_advantages) else None
                advantages.append(adv)
                uids.append(str(group_idx))

        if not sequences:
            raise RuntimeError("Atropos batch contained no sequences.")

        max_len = max(len(seq) for seq in sequences)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = []
        attention_mask = []
        response_mask = []
        token_level_scores = []
        token_level_advantages = []

        seq_lens = []
        for seq, mask, score, adv in zip(sequences, response_masks, scores, advantages, strict=False):
            seq_len = len(seq)
            seq_lens.append(seq_len)
            pad_len = max_len - seq_len

            seq_padded = seq + [pad_id] * pad_len
            input_ids.append(seq_padded)

            attention_mask.append([1] * seq_len + [0] * pad_len)

            if mask is None:
                resp_mask = [1] * seq_len + [0] * pad_len
            else:
                resp_mask = list(mask)
                if len(resp_mask) < seq_len:
                    resp_mask = resp_mask + [0] * (seq_len - len(resp_mask))
                resp_mask = resp_mask + [0] * (max_len - len(resp_mask))
            response_mask.append(resp_mask)

            # Build token-level scores aligned to response mask
            resp_len = int(sum(resp_mask[:seq_len]))
            score_vec = [0.0] * max_len
            if isinstance(score, (list, tuple)):
                if len(score) == seq_len:
                    score_vec[:seq_len] = list(score)
                elif len(score) == resp_len:
                    start = seq_len - resp_len
                    score_vec[start : start + resp_len] = list(score)
            elif score is not None and resp_len > 0:
                per_token = float(score) / resp_len
                start = seq_len - resp_len
                score_vec[start : start + resp_len] = [per_token] * resp_len
            token_level_scores.append(score_vec)

            # Build token-level advantages if provided
            if adv is None:
                token_level_advantages.append(None)
            elif isinstance(adv, (list, tuple)):
                if len(adv) == seq_len:
                    adv_vec = [0.0] * max_len
                    adv_vec[:seq_len] = list(adv)
                    token_level_advantages.append(adv_vec)
                elif len(adv) == resp_len:
                    start = seq_len - resp_len
                    adv_vec = [0.0] * max_len
                    adv_vec[start : start + resp_len] = list(adv)
                    token_level_advantages.append(adv_vec)
                else:
                    adv_vec = list(adv)
                    if len(adv_vec) < max_len:
                        adv_vec = adv_vec + [0.0] * (max_len - len(adv_vec))
                    token_level_advantages.append(adv_vec[:max_len])
            else:
                if resp_len > 0:
                    per_token = float(adv)
                    start = seq_len - resp_len
                    adv_vec = [0.0] * max_len
                    adv_vec[start : start + resp_len] = [per_token] * resp_len
                    token_level_advantages.append(adv_vec)
                else:
                    token_level_advantages.append([0.0] * max_len)

        device = torch.device("cpu")
        response_lengths = [int(sum(mask[:seq_len])) for mask, seq_len in zip(response_mask, seq_lens, strict=False)]
        max_resp_len = max(response_lengths) if response_lengths else 0
        responses = []
        for seq, seq_len, resp_len in zip(input_ids, seq_lens, response_lengths, strict=False):
            prompt_len = max(0, seq_len - resp_len)
            resp_tokens = seq[prompt_len:seq_len]
            resp_pad = max_resp_len - len(resp_tokens)
            responses.append(resp_tokens + [pad_id] * resp_pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
            "responses": torch.tensor(responses, dtype=torch.long, device=device),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32, device=device),
            "token_level_scores": torch.tensor(token_level_scores, dtype=torch.float32, device=device),
        }

        batch_proto = DataProto(batch=batch)
        batch_proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        batch_proto.meta_info["timing"] = {}

        # Attach token-level advantages if provided
        if any(a is not None for a in token_level_advantages):
            adv_tensor = torch.tensor(
                [a if a is not None else [0.0] * max_len for a in token_level_advantages],
                dtype=torch.float32,
                device=device,
            )
            adv_tensor = convert_scalar_or_token_advantages(adv_tensor, batch_proto.batch["response_mask"])
            batch_proto.batch["token_level_advantages"] = adv_tensor

        return batch_proto

    def _compute_advantages_grpo(
        self,
        batch: DataProto,
        scores: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """
        Compute GRPO advantages with Atropos environment overrides.

        This method overrides the standard advantage computation to integrate
        with Atropos environments for token-level advantages.
        """
        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            response_mask = compute_response_mask(batch)
            batch.batch["response_mask"] = response_mask

        advantages = batch.batch.get("token_level_advantages")
        if advantages is None:
            return None, {"fallback": True}

        advantages = convert_scalar_or_token_advantages(advantages, response_mask)
        return advantages, {"source": "atropos_batch"}

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        if "token_level_scores" in batch.batch:
            reward_tensor = batch.batch["token_level_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            if return_dict:
                return reward_tensor, {}
            return reward_tensor, {}

        return super()._compute_or_extract_reward(
            batch, reward_fn=reward_fn, return_dict=return_dict, sum_reward=sum_reward
        )
