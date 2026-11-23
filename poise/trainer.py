# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""POISE extensions to veRL's synchronous V1 trainer."""

from pathlib import Path

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import PPOTrainerSync

from .features import entropy_features
from .probe import ProbeBank, ProbeConfig, domain_for
from .workers import PoiseActorRolloutWorker


def validate_poise_config(config):
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("POISE requires the V1 synchronous trainer")
    if actor.strategy not in {"fsdp", "fsdp2"} or rollout.n != 2 or rollout.multi_turn.enable:
        raise ValueError("POISE supports FSDP/FSDP2, exactly two rollouts and single-turn generation")
    if actor.fsdp_config.ulysses_sequence_parallel_size != 1 or actor.ulysses_sequence_parallel_size != 1:
        raise ValueError("POISE currently requires sequence parallel size 1")
    if config.algorithm.adv_estimator != "poise" or config.critic.enable:
        raise ValueError("Use algorithm.adv_estimator=poise and critic.enable=false")
    if config.algorithm.use_kl_in_reward or actor.use_kl_loss:
        raise ValueError("This recipe implements the paper's correctness rewards without KL regularization")
    correction = config.algorithm.rollout_correction
    if correction and (correction.bypass_mode or correction.rollout_is or correction.rollout_rs):
        raise ValueError("POISE requires recomputed log probabilities without rollout correction")
    if config.algorithm.filter_groups.enable:
        raise ValueError("POISE does not use group filtering")
    if config.trainer.default_hdfs_dir or actor.checkpoint.get("async_save", False):
        raise ValueError("POISE checkpoints currently require synchronous saves to local/shared storage")
    if config.trainer.critic_warmup != 0 or config.trainer.v1.sync.get("parameter_sync_step", 1) != 1:
        raise ValueError("POISE requires one actor update per trainer step")
    if not config.actor_rollout_ref.hybrid_engine or config.distillation.enabled:
        raise ValueError("POISE requires colocated actor/rollout workers without distillation")
    return ProbeConfig(**OmegaConf.to_container(config.poise, resolve=True))


class PoiseTrainer(PPOTrainerSync):
    def __init__(self, config):
        self.probes = ProbeBank(validate_poise_config(config))
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(PoiseActorRolloutWorker)

    def _compute_old_log_prob(self, batch, metrics):
        batch.extra_info["poise_capture"] = {
            "layer": self.probes.config.layer,
            "pool_tokens": self.probes.config.pool_tokens,
            "think_end_ids": self.tokenizer.encode("</think>", add_special_tokens=False),
        }
        output = super()._compute_old_log_prob(batch, metrics)
        output.extra_info.pop("poise_capture", None)
        return output

    def _compute_advantage(self, batch, metrics):
        fields = [
            "uid",
            "data_source",
            "responses",
            "response_mask",
            "rm_scores",
            "entropy",
            "poise_prompt",
            "poise_response",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        active = [i for i, tag in enumerate(batch.tags) if not tag.get("is_padding", False)]
        uids, sources = list(data["uid"]), list(data["data_source"])
        masks = data["response_mask"].unbind()
        scalar_features = []
        for i in active:
            mask = masks[i].bool()
            scalar_features.append(
                entropy_features(
                    self.tokenizer,
                    data["responses"][i][mask].tolist(),
                    data["entropy"][i][mask].float().numpy(),
                )
            )
        values, probe_metrics = self.probes.advantages(
            uids=[uids[i] for i in active],
            domains=[domain_for(sources[i]) for i in active],
            rewards=np.asarray([float(data["rm_scores"][i].sum()) for i in active]),
            prompt=np.stack([data["poise_prompt"][i].float().numpy() for i in active]),
            response=np.stack([data["poise_response"][i].float().numpy() for i in active]),
            scalars=np.stack(scalar_features),
        )
        metrics.update(probe_metrics)
        per_row = dict(zip(active, values, strict=True))
        advantages = torch.nested.as_nested_tensor(
            [mask.float() * float(per_row.get(i, 0.0)) for i, mask in enumerate(masks)],
            layout=torch.jagged,
        )
        output = TensorDict({"advantages": advantages, "returns": advantages.clone()}, batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    def _update_actor(self, batch, metrics):
        output = super()._update_actor(batch, metrics)
        metrics.update(self.probes.commit())
        return output

    def _save_checkpoint(self):
        # Write probe state before upstream publishes latest_checkpointed_iteration.txt.
        directory = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        if self.probes.completed_steps != self.global_steps:
            raise RuntimeError("POISE and actor update counts differ")
        self.probes.save(directory)
        super()._save_checkpoint()

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps:
            directory = (
                Path(self.config.trainer.resume_from_path)
                if self.config.trainer.resume_mode == "resume_path"
                else Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            )
            self.probes.load(directory, expected_step=self.global_steps)
