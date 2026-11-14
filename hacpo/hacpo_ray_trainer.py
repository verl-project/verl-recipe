"""Shared-GPU, text-first HACPO trainer built on verl V1 model engines."""

import json
import logging
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.utils import need_reference_policy
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.workers.config import HFModelConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.replica import RolloutMode
from verl.workers.utils.padding import response_from_nested

from .hacpo_config import (
    AgentRuntimeSpec,
    build_agent_runtime_specs,
    build_loss_config,
    public_agent_summary,
    select_runtime_agent_specs,
    select_validation_agent_specs,
    validate_hacpo_config,
)
from .hacpo_core_algos import CapabilityTracker, compute_hacpo_advantages
from .hacpo_trajectory import (
    RolloutRecord,
    Trajectory,
    attach_source_log_probs,
    build_training_view,
    materialize_rollout_records,
)
from .hacpo_workers import (
    HacpoActorRolloutRefWorker,
    build_model_tensordict,
    build_training_tensordict,
    finalize_hacpo_metrics,
    hacpo_ppo_loss,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _validation_column(batch_dict: dict, name: str, batch_size: int, default: Any = None) -> list[Any]:
    """Normalize one collated validation column to a controller-side list."""

    if name not in batch_dict:
        return [default] * batch_size
    value = batch_dict[name]
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    if not isinstance(values, list):
        values = [values]
    if len(values) != batch_size:
        raise ValueError(f"validation column {name!r} has {len(values)} rows, expected {batch_size}")
    return values


@contextmanager
def _active_rollout_policy(checkpoint_manager, global_steps: int):
    """Keep one rollout policy resident for a bounded sequence of generations."""

    replicas_awake = False
    try:
        checkpoint_manager.update_weights(global_steps)
        replicas_awake = True
        yield
    finally:
        if replicas_awake:
            checkpoint_manager.sleep_replicas()


def _target_trajectories_self_first(
    trajectories: tuple[Trajectory, ...],
    *,
    target: str,
    sources: tuple[str, ...],
) -> tuple[Trajectory, ...]:
    """Place self trajectories before cross trajectories for scheduled clipping."""

    if target not in sources:
        raise ValueError(f"target {target!r} is missing its required self source")
    ordered_sources = (target, *(source for source in sources if source != target))
    by_source: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.source_agent_id in sources:
            by_source[trajectory.source_agent_id].append(trajectory)
    missing = [source for source in ordered_sources if not by_source[source]]
    if missing:
        raise ValueError(f"target {target!r} has no trajectories from sources {missing}")
    return tuple(trajectory for source in ordered_sources for trajectory in by_source[source])


def _rank_offset_replica_class(base_class, rank_offset: int):
    """Keep globally unique replica ids while slicing shared workers locally."""

    class RankOffsetReplica(base_class):
        async def init_hybrid(self, worker_group) -> None:
            self.rollout_mode = RolloutMode.HYBRID
            local_replica_rank = self.replica_rank - rank_offset
            if local_replica_rank < 0:
                raise RuntimeError(f"replica rank {self.replica_rank} is below HACPO offset {rank_offset}")
            start = self.world_size * local_replica_rank
            stop = start + self.world_size
            self.workers = worker_group.workers[start:stop]
            if len(self.workers) != self.world_size:
                raise RuntimeError(
                    f"HACPO replica {self.replica_rank} mapped to {len(self.workers)} workers, "
                    f"expected {self.world_size}"
                )
            await self.launch_servers()

    RankOffsetReplica.__name__ = f"HacpoRankOffset{rank_offset}{base_class.__name__}"
    return RankOffsetReplica


class HacpoLLMServerManager(LLMServerManager):
    """Give each policy a collision-free hybrid rollout replica-rank range."""

    def __init__(self, *args, rank_offset: int, **kwargs) -> None:
        super().__init__(*args, start_rank=rank_offset, **kwargs)
        self.rollout_replica_class = _rank_offset_replica_class(
            self.rollout_replica_class,
            rank_offset,
        )


class HacpoRewardLoopManager(RewardLoopManager):
    """Namespace reward workers so each policy can use its own tokenizer."""

    def __init__(self, *args, worker_name_prefix: str, **kwargs) -> None:
        self.worker_name_prefix = worker_name_prefix
        super().__init__(*args, **kwargs)

    def _init_reward_loop_workers(self) -> None:
        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for index in range(num_workers):
            node_id = node_ids[index % len(node_ids)]
            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    name=f"{self.worker_name_prefix}_reward_loop_worker_{index}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )


class HacpoTrainer(PPOTrainer):
    """Coordinate N policy runtimes through a tokenizer-neutral trajectory pool."""

    def __init__(self, config) -> None:
        self.agent_specs = build_agent_runtime_specs(config)
        validate_hacpo_config(config, self.agent_specs)
        self.loss_config = build_loss_config(config)
        self.specs_by_id = {spec.agent_id: spec for spec in self.agent_specs}
        self.agent_ids = tuple(spec.agent_id for spec in self.agent_specs)
        self.runtime_agent_specs = select_runtime_agent_specs(config, self.agent_specs)
        self.runtime_agent_ids = tuple(spec.agent_id for spec in self.runtime_agent_specs)
        capability = config.hacpo.capability
        self.capability_tracker = CapabilityTracker(
            self.agent_ids,
            window_size=int(capability.window_size),
            metric=str(capability.metric),
            minimum=float(capability.minimum),
            ratio_min=float(capability.ratio_min),
            ratio_max=float(capability.ratio_max),
            warmup_steps=int(capability.warmup_steps),
        )
        self.policy_wgs: dict[str, RayWorkerGroup] = {}
        self.llm_server_managers: dict[str, HacpoLLMServerManager] = {}
        self.checkpoint_managers: dict[str, CheckpointEngineManager] = {}
        self.reward_loop_managers: dict[str, HacpoRewardLoopManager] = {}
        self.tokenizers: dict[str, Any] = {}
        self.processors: dict[str, Any] = {}
        self.generation_eos_token_ids: dict[str, Any] = {}
        self.ref_in_actor: dict[str, bool] = {}
        self._last_trajectories: tuple[Trajectory, ...] = ()
        self._last_step_metrics: dict[str, float] = {}
        super().__init__(config)

    def _setup(self) -> None:
        self._init_policy_tokenizers()
        self._init_dataloader()
        self._set_optimizer_horizons()
        self._init_dump_executor()
        self._init_shared_policy_workers()

        for spec in self.runtime_agent_specs:
            self.reward_loop_managers[spec.agent_id] = HacpoRewardLoopManager(
                config=spec.config,
                rm_resource_pool=None,
                worker_name_prefix=spec.worker_key,
            )
        logger.info("HACPO rule-based reward loops initialized for %s", self.runtime_agent_ids)

        self._init_policy_engines_and_rollouts()
        self._load_checkpoint()

        # PPOTrainer utilities expect these single-policy aliases.
        primary_id = self.runtime_agent_specs[0].agent_id
        self.actor_rollout_wg = self.policy_wgs[primary_id]
        self.llm_server_manager = self.llm_server_managers[primary_id]
        self.checkpoint_manager = self.checkpoint_managers[primary_id]
        self.ref_policy_wg = self.actor_rollout_wg
        logger.info("all HACPO policy runtimes initialized and sleeping")

    def _init_policy_tokenizers(self) -> None:
        for spec in self.runtime_agent_specs:
            model_config: HFModelConfig = omega_conf_to_dataclass(spec.config.actor_rollout_ref.model)
            self.tokenizers[spec.agent_id] = model_config.tokenizer
            self.processors[spec.agent_id] = model_config.processor
            self.generation_eos_token_ids[spec.agent_id] = (
                None
                if model_config.generation_config is None
                else getattr(model_config.generation_config, "eos_token_id", None)
            )
        primary_id = self.runtime_agent_specs[0].agent_id
        self.tokenizer = self.tokenizers[primary_id]
        self.processor = self.processors[primary_id]

    def _set_optimizer_horizons(self) -> None:
        total_updates = self.total_training_steps * self.parameter_sync_step
        for spec in self.runtime_agent_specs:
            with open_dict(spec.config):
                spec.config.actor_rollout_ref.actor.optim.total_training_steps = total_updates

    def _init_shared_policy_workers(self) -> None:
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={
                "global_pool": [int(self.config.trainer.n_gpus_per_node)] * int(self.config.trainer.nnodes)
            },
            mapping={},
            max_colocate_count=3,
        )
        self.resource_pool_manager.create_resource_pool()
        resource_pool = self.resource_pool_manager.resource_pool_dict["global_pool"]

        remote_worker_cls = ray.remote(HacpoActorRolloutRefWorker)
        class_dict = {}
        for spec in self.runtime_agent_specs:
            role, ref_in_actor = self._policy_role(spec.config)
            self.ref_in_actor[spec.agent_id] = ref_in_actor
            class_dict[spec.worker_key] = RayClassWithInitArgs(
                cls=remote_worker_cls,
                config=spec.config.actor_rollout_ref,
                distillation_config=spec.config.get("distillation"),
                role=role,
            )

        wg_kwargs = {"device_name": self.config.trainer.device}
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )

        colocated_cls = create_colocated_worker_cls(class_dict=class_dict)
        colocated_wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=colocated_cls,
            **wg_kwargs,
        )
        spawned = colocated_wg.spawn(prefix_set=class_dict.keys())
        for spec in self.runtime_agent_specs:
            self.policy_wgs[spec.agent_id] = spawned[spec.worker_key]
        self._rollout_resource_pool = resource_pool

    @staticmethod
    def _policy_role(config) -> tuple[str, bool]:
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        role = "actor_rollout_ref" if need_reference_policy(config) and not ref_in_actor else "actor_rollout"
        return role, ref_in_actor

    @staticmethod
    def _rollout_replica_count(spec: AgentRuntimeSpec, world_size: int) -> int:
        rollout = spec.config.actor_rollout_ref.rollout
        replica_world_size = (
            int(rollout.tensor_model_parallel_size)
            * int(rollout.data_parallel_size)
            * int(rollout.pipeline_model_parallel_size)
        )
        if world_size % replica_world_size != 0:
            raise ValueError(
                f"agent {spec.agent_id!r} rollout footprint {replica_world_size} "
                f"does not divide worker world size {world_size}"
            )
        return world_size // replica_world_size

    def _init_policy_engines_and_rollouts(self) -> None:
        # Offload each rollout before allocating the next policy.
        rank_offset = 0
        for spec in self.runtime_agent_specs:
            agent_id = spec.agent_id
            wg = self.policy_wgs[agent_id]
            logger.info("initializing HACPO policy %s from %s", agent_id, spec.config.actor_rollout_ref.model.path)
            wg.init_model()
            wg.set_hacpo_rollout_rank_offset(rank_offset)

            actor_config = omega_conf_to_dataclass(spec.config.actor_rollout_ref.actor)
            wg.set_loss_fn(partial(hacpo_ppo_loss, actor_config, self.loss_config))

            llm_manager = HacpoLLMServerManager.create(
                config=spec.config,
                worker_group=wg,
                rollout_resource_pool=self._rollout_resource_pool,
                rank_offset=rank_offset,
            )
            checkpoint_config = omega_conf_to_dataclass(spec.config.actor_rollout_ref.rollout.checkpoint_engine)
            checkpoint_config.backend = "naive"
            checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_config,
                actor_wg=wg,
                replicas=llm_manager.get_replicas(),
            )
            checkpoint_manager.sleep_replicas()
            self.llm_server_managers[agent_id] = llm_manager
            self.checkpoint_managers[agent_id] = checkpoint_manager
            logger.info("HACPO policy %s initialized and offloaded", agent_id)
            rank_offset += self._rollout_replica_count(spec, wg.world_size)

    def get_llm_clients(self) -> dict[str, Any]:
        return {agent_id: manager.get_client() for agent_id, manager in self.llm_server_managers.items()}

    def get_reward_handles(self, agent_id: str):
        return self.reward_loop_managers[agent_id].reward_loop_worker_handles

    def fit(self, agent_loop_managers: dict[str, Any]):
        if set(agent_loop_managers) != set(self.runtime_agent_ids):
            raise ValueError(
                f"agent-loop managers {sorted(agent_loop_managers)} do not match "
                f"active policies {sorted(self.runtime_agent_ids)}"
            )
        self.agent_loop_managers = agent_loop_managers
        return super().fit(agent_loop_managers)

    def on_step_end(self):
        return

    def on_sample_end(self):
        return

    def _start_profiling(self) -> None:
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        if do_profile:
            for agent_id, wg in self.policy_wgs.items():
                wg.start_profile(role=f"hacpo_{agent_id}", profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile
        if do_profile:
            for wg in self.policy_wgs.values():
                wg.stop_profile()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        prompt_batch = self._next_train_batch()
        canonical_prompt_ids = [str(value) for value in tu.get(prompt_batch, "uid")]

        all_trajectories: list[Trajectory] = []
        rollout_metas: list[KVBatchMeta] = []
        for policy_index, spec in enumerate(self.agent_specs):
            with marked_timer(f"rollout/{spec.agent_id}", timing_raw, color="red"):
                trajectories, batch_meta, rollout_metrics = self._rollout_and_score_source(
                    spec=spec,
                    policy_index=policy_index,
                    prompt_batch=prompt_batch,
                    canonical_prompt_ids=canonical_prompt_ids,
                )
            all_trajectories.extend(trajectories)
            rollout_metas.append(batch_meta)
            metrics.update({f"agent/{spec.agent_id}/{key}": value for key, value in rollout_metrics.items()})

        trajectories = tuple(all_trajectories)
        rewards = torch.tensor([trajectory.reward for trajectory in trajectories], dtype=torch.float32)
        source_ids = [trajectory.source_agent_id for trajectory in trajectories]
        capabilities = self.capability_tracker.update(rewards, source_ids)

        step_metrics: dict[str, float] = {
            "training/hacpo/num_trajectories": float(len(trajectories)),
        }
        for agent_id in self.agent_ids:
            agent_rewards = rewards[torch.tensor([source == agent_id for source in source_ids])]
            step_metrics[f"agent/{agent_id}/capability"] = capabilities[agent_id]
            step_metrics[f"agent/{agent_id}/reward/mean"] = agent_rewards.mean().item()
            step_metrics[f"agent/{agent_id}/reward/positive_rate"] = (agent_rewards > 0).float().mean().item()

        for target in self.agent_ids:
            with marked_timer(f"update/{target}", timing_raw, color="blue"):
                target_metrics = self._update_target(target, trajectories, capabilities)
            step_metrics.update(target_metrics)

        self._last_trajectories = trajectories
        self._last_step_metrics = step_metrics
        metrics.update(step_metrics)
        return KVBatchMeta(
            partition_id="train",
            keys=[key for batch_meta in rollout_metas for key in batch_meta.keys],
            tags=[tag for batch_meta in rollout_metas for tag in batch_meta.tags],
        )

    @staticmethod
    def _replace_non_tensor_column(batch, name: str, values: list[Any]) -> None:
        batch.pop(name, None)
        tu.assign_non_tensor_stack(batch, name, values)

    def _policy_prompt_batch(
        self,
        *,
        prompt_batch,
        policy_index: int,
        canonical_prompt_ids: list[str],
        validate: bool,
    ):
        batch = prompt_batch.clone()
        policy_uids = [f"p{policy_index}-{uuid.uuid4().hex}" for _ in range(len(batch))]
        self._replace_non_tensor_column(batch, "uid", policy_uids)
        self._replace_non_tensor_column(batch, "hacpo_prompt_id", canonical_prompt_ids)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        if validate:
            tu.assign_non_tensor_data(batch, "validate", True)
        else:
            batch.pop("validate", None)
        return batch, policy_uids

    def _generate_records(
        self,
        *,
        spec: AgentRuntimeSpec,
        policy_index: int,
        prompt_batch,
        canonical_prompt_ids: list[str],
        partition_id: str,
        expected_rollout_n: int,
        validate: bool,
        manage_replica_lifecycle: bool = True,
    ) -> tuple[tuple[RolloutRecord, ...], KVBatchMeta, dict]:
        agent_id = spec.agent_id
        policy_batch, policy_uids = self._policy_prompt_batch(
            prompt_batch=prompt_batch,
            policy_index=policy_index,
            canonical_prompt_ids=canonical_prompt_ids,
            validate=validate,
        )
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in policy_uids]

        checkpoint_manager = self.checkpoint_managers[agent_id]
        lifecycle = (
            _active_rollout_policy(checkpoint_manager, self.global_steps) if manage_replica_lifecycle else nullcontext()
        )
        with lifecycle:
            tq.kv_batch_put(keys=policy_uids, partition_id=partition_id, tags=tags)
            self.agent_loop_managers[agent_id].generate_sequences(policy_batch)
            batch_meta, rollout_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id=partition_id,
                batch_size=len(policy_batch),
            )
            expected_trajectories = len(policy_batch) * expected_rollout_n
            if len(batch_meta.keys) != expected_trajectories:
                raise RuntimeError(
                    f"agent {agent_id!r} produced {len(batch_meta.keys)} trajectories, "
                    f"expected {expected_trajectories}; only one output per single-turn session is supported"
                )
            fields = [
                "prompts",
                "responses",
                "response_mask",
                "rm_scores",
                "raw_prompt",
                "hacpo_prompt_id",
                "extra_fields",
            ]
            data = tq.kv_batch_get(
                keys=batch_meta.keys,
                partition_id=partition_id,
                select_fields=fields,
            )
            records = materialize_rollout_records(
                data=data,
                keys=batch_meta.keys,
                tokenizer=self.tokenizers[agent_id],
                source_agent_id=agent_id,
                source_policy_version=self.global_steps,
                terminal_token_ids=self.generation_eos_token_ids[agent_id],
            )
            return records, batch_meta, rollout_metrics

    def _rollout_and_score_source(
        self,
        *,
        spec: AgentRuntimeSpec,
        policy_index: int,
        prompt_batch,
        canonical_prompt_ids: list[str],
    ) -> tuple[tuple[Trajectory, ...], KVBatchMeta, dict]:
        rollout_n = int(spec.config.actor_rollout_ref.rollout.n)
        records, batch_meta, rollout_metrics = self._generate_records(
            spec=spec,
            policy_index=policy_index,
            prompt_batch=prompt_batch,
            canonical_prompt_ids=canonical_prompt_ids,
            partition_id="train",
            expected_rollout_n=rollout_n,
            validate=False,
        )

        source_batch = build_model_tensordict(tuple(record.training_view() for record in records))
        tu.assign_non_tensor(
            source_batch,
            calculate_entropy=False,
            compute_loss=False,
            temperature=float(spec.config.actor_rollout_ref.rollout.temperature),
        )
        output = self.policy_wgs[spec.agent_id].compute_log_prob(source_batch)
        response_log_probs = response_from_nested(output["log_probs"], source_batch["response_mask"])
        trajectories = attach_source_log_probs(records, response_log_probs)
        return trajectories, batch_meta, rollout_metrics

    def _update_target(
        self,
        target: str,
        trajectories: tuple[Trajectory, ...],
        capabilities: dict[str, float],
    ) -> dict[str, float]:
        spec = self.specs_by_id[target]
        target_config = spec.config
        sources = self.agent_ids
        selected = _target_trajectories_self_first(
            trajectories,
            target=target,
            sources=sources,
        )
        source_ids = [trajectory.source_agent_id for trajectory in selected]
        prompt_ids = [trajectory.prompt_id for trajectory in selected]
        rewards = torch.tensor([trajectory.reward for trajectory in selected], dtype=torch.float32)

        advantage_config = self.config.hacpo.advantage
        source_reward_weights = {source: 1.0 / self.capability_tracker.ratio(source, target) for source in sources}
        advantages, baselines = compute_hacpo_advantages(
            rewards,
            prompt_ids,
            source_ids,
            learner_agent_id=target,
            capabilities=capabilities,
            normalize_by_std=bool(advantage_config.normalize_by_std),
            epsilon=float(advantage_config.epsilon),
            expected_source_agents=sources,
            source_reward_weights=source_reward_weights,
            require_equal_samples_per_source=True,
            disable_weighting_for_uniform_rewards=bool(advantage_config.disable_weighting_for_uniform_rewards),
            binary_baseline_guard=bool(advantage_config.binary_baseline_guard),
        )

        rollout_config = target_config.actor_rollout_ref.rollout
        trajectory_config = self.config.hacpo.trajectory
        target_max_prompt_length = trajectory_config.get("target_max_prompt_length", None)
        target_max_response_length = trajectory_config.get("target_max_response_length", None)
        views = tuple(
            build_training_view(
                trajectory,
                target_agent_id=target,
                tokenizer=self.tokenizers[target],
                max_prompt_length=(
                    int(rollout_config.prompt_length)
                    if target_max_prompt_length is None
                    else int(target_max_prompt_length)
                ),
                max_response_length=(
                    int(rollout_config.response_length)
                    if target_max_response_length is None
                    else int(target_max_response_length)
                ),
                apply_chat_template_kwargs=OmegaConf.to_container(
                    target_config.data.get("apply_chat_template_kwargs", {}),
                    resolve=True,
                ),
                overflow=str(self.config.hacpo.trajectory.overflow),
            )
            for trajectory in selected
        )
        is_self = torch.tensor([source == target for source in source_ids], dtype=torch.bool)
        capability_ratios = torch.tensor(
            [1.0 / source_reward_weights[source] for source in source_ids],
            dtype=torch.float32,
        )
        source_mean_log_probs = torch.tensor(
            [trajectory.source_mean_logp for trajectory in selected],
            dtype=torch.float32,
        )
        update_id = f"step-{self.global_steps}-target-{target}"
        batch = build_training_tensordict(
            views,
            source_mean_log_probs=source_mean_log_probs,
            advantages=advantages,
            is_self=is_self,
            source_to_target_capability_ratio=capability_ratios,
            update_id=update_id,
        )

        actor_config = target_config.actor_rollout_ref.actor
        if actor_config.use_kl_loss:
            ref_input = batch.clone()
            tu.assign_non_tensor(
                ref_input,
                calculate_entropy=False,
                compute_loss=False,
                temperature=float(rollout_config.temperature),
            )
            if self.ref_in_actor[target]:
                tu.assign_non_tensor_data(ref_input, "no_lora_adapter", True)
                ref_output = self.policy_wgs[target].compute_log_prob(ref_input)
            else:
                ref_output = self.policy_wgs[target].compute_ref_log_prob(ref_input)
            batch["ref_log_prob"] = response_from_nested(
                ref_output["log_probs"],
                batch["response_mask"],
            )

        rollout_n = int(rollout_config.n)
        mini_batch_size = int(actor_config.ppo_mini_batch_size) * rollout_n
        calculate_entropy = bool(actor_config.calculate_entropy or actor_config.entropy_coeff != 0.0)
        tu.assign_non_tensor(
            batch,
            calculate_entropy=calculate_entropy,
            global_batch_size=mini_batch_size,
            mini_batch_size=mini_batch_size,
            epochs=int(actor_config.ppo_epochs),
            seed=int(actor_config.data_loader_seed),
            dataloader_kwargs={"shuffle": bool(actor_config.shuffle)},
            temperature=float(rollout_config.temperature),
        )
        output = self.policy_wgs[target].update_actor(batch)
        raw_metrics = dict(tu.get(output, "metrics", {}))
        reduced = finalize_hacpo_metrics(reduce_metrics(raw_metrics))

        prefix = f"agent/{target}"
        metrics = {f"{prefix}/{name}": value for name, value in reduced.items()}
        metrics.update(
            {
                f"{prefix}/advantage/mean": advantages.mean().item(),
                f"{prefix}/advantage/std": advantages.float().std(correction=0).item(),
                f"{prefix}/baseline/mean": baselines.mean().item(),
                f"{prefix}/cross/capability_ratio_mean": capability_ratios[~is_self].mean().item()
                if bool((~is_self).any())
                else 1.0,
                f"{prefix}/trajectory/truncated_fraction": sum(view.truncated for view in views) / len(views),
            }
        )
        return metrics

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        del batch
        metrics.update(self._last_step_metrics)
        metrics["training/global_step"] = global_steps
        metrics["training/epoch"] = epoch
        for name, duration in timing_raw.items():
            metrics[f"timing_s/{name}"] = duration
        total_tokens = sum(
            len(trajectory.source_prompt_ids or ()) + trajectory.source_token_count
            for trajectory in self._last_trajectories
        )
        step_time = float(timing_raw.get("step", 0.0))
        metrics["perf/total_source_tokens"] = total_tokens
        if step_time > 0:
            metrics["perf/source_tokens_per_second"] = total_tokens / step_time

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        del batch, timing_raw
        trajectories = sorted(self._last_trajectories, key=lambda trajectory: trajectory.trajectory_id)
        self._dump_generations(
            inputs=[
                "\n".join(f"{message.role}: {message.content}" for message in trajectory.messages)
                for trajectory in trajectories
            ],
            outputs=[trajectory.response_text for trajectory in trajectories],
            gts=[None] * len(trajectories),
            scores=[trajectory.reward for trajectory in trajectories],
            reward_extra_infos_dict={
                "uid": [trajectory.trajectory_id for trajectory in trajectories],
                "source_agent": [trajectory.source_agent_id for trajectory in trajectories],
            },
            dump_path=rollout_data_dir,
        )

    def _validate(self) -> dict[str, float]:
        validation_specs = select_validation_agent_specs(self.config, self.agent_specs)
        validation_rows: dict[str, dict[str, list[Any]]] = {
            spec.agent_id: {"data_sources": [], "prompt_ids": [], "rewards": []} for _, spec in validation_specs
        }
        for policy_index, spec in validation_specs:
            checkpoint_manager = self.checkpoint_managers[spec.agent_id]
            logger.info("activating HACPO validation policy %s", spec.agent_id)
            try:
                with _active_rollout_policy(checkpoint_manager, self.global_steps):
                    for batch_dict in self.val_dataloader:
                        batch_size = len(batch_dict["raw_prompt"])
                        dispatch_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
                        data_sources = _validation_column(batch_dict, "data_source", batch_size, default="unknown")
                        problem_indices = (
                            _validation_column(batch_dict, "problem_idx", batch_size)
                            if "problem_idx" in batch_dict
                            else None
                        )
                        canonical_prompt_ids = validation_group_ids(data_sources, dispatch_ids, problem_indices)
                        source_by_prompt_id = dict(zip(canonical_prompt_ids, data_sources, strict=True))
                        batch_dict["uid"] = np.array(dispatch_ids, dtype=object)
                        prompt_batch = tu.get_tensordict(batch_dict)
                        val_n = int(spec.config.actor_rollout_ref.rollout.val_kwargs.n)
                        records, batch_meta, _ = self._generate_records(
                            spec=spec,
                            policy_index=policy_index,
                            prompt_batch=prompt_batch,
                            canonical_prompt_ids=canonical_prompt_ids,
                            partition_id="val",
                            expected_rollout_n=val_n,
                            validate=True,
                            manage_replica_lifecycle=False,
                        )
                        rows = validation_rows[spec.agent_id]
                        rows["data_sources"].extend(source_by_prompt_id[record.prompt_id] for record in records)
                        rows["prompt_ids"].extend(record.prompt_id for record in records)
                        rows["rewards"].extend(record.reward for record in records)
                        tq.kv_clear(keys=batch_meta.keys, partition_id="val")
            finally:
                logger.info("HACPO validation policy %s offloaded", spec.agent_id)

        metrics = {}
        for agent_id, rows in validation_rows.items():
            metrics.update(
                format_agent_validation_metrics(
                    agent_id=agent_id,
                    data_sources=rows["data_sources"],
                    prompt_ids=rows["prompt_ids"],
                    rewards=rows["rewards"],
                )
            )
        return metrics

    def _checkpoint_folder(self) -> str | None:
        resume_mode = str(self.config.trainer.resume_mode)
        if resume_mode == "disable":
            return None
        if resume_mode == "auto":
            root = str(self.config.trainer.default_local_dir)
            if not os.path.isabs(root):
                root = os.path.join(os.getcwd(), root)
            return find_latest_ckpt_path(root)
        if resume_mode == "resume_path":
            path = str(self.config.trainer.resume_from_path)
            if "global_step_" not in path:
                raise ValueError("trainer.resume_from_path must point to a global_step_* directory")
            return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        raise ValueError(f"unknown trainer.resume_mode {resume_mode!r}")

    def _load_checkpoint(self) -> None:
        self.global_steps = 0
        folder = self._checkpoint_folder()
        if folder is None:
            logger.info("HACPO training from scratch")
            return
        self.global_steps = int(os.path.basename(folder).split("global_step_")[-1])
        logger.info("resuming HACPO policies %s from %s", self.runtime_agent_ids, folder)

        topology_path = os.path.join(folder, "hacpo_topology.json")
        with open(topology_path) as file:
            saved_topology = json.load(file)
        # Ignore the removed graph field in older checkpoints.
        saved_summary = {
            "agents": [{"id": agent["id"], "model_path": agent["model_path"]} for agent in saved_topology["agents"]]
        }
        if saved_summary != public_agent_summary(self.agent_specs):
            raise ValueError("HACPO checkpoint agents/model paths do not match the current configuration")

        for spec in self.runtime_agent_specs:
            self.policy_wgs[spec.agent_id].load_checkpoint(
                local_path=os.path.join(folder, "agents", spec.agent_id, "actor"),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        data_path = os.path.join(folder, "data.pt")
        if os.path.exists(data_path):
            self.train_dataloader.load_state_dict(torch.load(data_path, weights_only=False))
        else:
            raise FileNotFoundError(f"HACPO checkpoint is missing dataloader state {data_path}")

        with open(os.path.join(folder, "hacpo_capability.json")) as file:
            self.capability_tracker.load_state_dict(json.load(file))

    def _save_checkpoint(self) -> None:
        from verl.utils.fs import local_mkdir_safe

        root = str(self.config.trainer.default_local_dir)
        folder = os.path.join(root, f"global_step_{self.global_steps}")
        local_mkdir_safe(folder)
        max_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        remote_root = self.config.trainer.default_hdfs_dir

        for spec in self.agent_specs:
            local_actor = os.path.join(folder, "agents", spec.agent_id, "actor")
            remote_actor = (
                None
                if remote_root is None
                else os.path.join(
                    remote_root,
                    f"global_step_{self.global_steps}",
                    "agents",
                    spec.agent_id,
                    "actor",
                )
            )
            self.policy_wgs[spec.agent_id].save_checkpoint(
                local_actor,
                remote_actor,
                self.global_steps,
                max_ckpt_to_keep=max_to_keep,
            )

        torch.save(self.train_dataloader.state_dict(), os.path.join(folder, "data.pt"))
        with open(os.path.join(folder, "hacpo_capability.json"), "w") as file:
            json.dump(self.capability_tracker.state_dict(), file, indent=2)
        with open(os.path.join(folder, "hacpo_topology.json"), "w") as file:
            json.dump(public_agent_summary(self.agent_specs), file, indent=2)

        local_mkdir_safe(root)
        with open(os.path.join(root, "latest_checkpointed_iteration.txt"), "w") as file:
            file.write(str(self.global_steps))


def validation_group_ids(
    data_sources: Sequence[object],
    fallback_ids: Sequence[object],
    problem_indices: Sequence[object] | None = None,
) -> list[str]:
    """Group repeated samples by dataset and problem index."""

    if len(data_sources) != len(fallback_ids):
        raise ValueError("data_sources and fallback_ids must have equal length")
    if problem_indices is None:
        return [str(value) for value in fallback_ids]
    if len(problem_indices) != len(fallback_ids):
        raise ValueError("problem_indices and fallback_ids must have equal length")

    group_ids = []
    for data_source, fallback_id, problem_index in zip(
        data_sources,
        fallback_ids,
        problem_indices,
        strict=True,
    ):
        if problem_index is None:
            group_ids.append(str(fallback_id))
        else:
            group_ids.append(f"{data_source}:{problem_index}")
    return group_ids


def format_agent_validation_metrics(
    *,
    agent_id: str,
    data_sources: Sequence[object],
    prompt_ids: Sequence[object],
    rewards: Sequence[float],
) -> dict[str, float]:
    """Return global and per-benchmark verl metrics for one policy."""

    if not rewards:
        raise ValueError(f"agent {agent_id!r} has no validation rewards")
    if len(data_sources) != len(prompt_ids) or len(prompt_ids) != len(rewards):
        raise ValueError("validation data_sources, prompt_ids, and rewards must have equal length")

    reward_array = np.asarray(rewards, dtype=np.float64)
    metrics = {
        f"val/{agent_id}/reward/mean": float(reward_array.mean()),
        f"val/{agent_id}/reward/positive_rate": float((reward_array > 0).mean()),
    }
    processed = process_validation_metrics(
        [str(value) for value in data_sources],
        [str(value) for value in prompt_ids],
        {"reward": reward_array.tolist()},
    )
    for data_source, var2metric2val in processed.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val)
            for metric_name, metric_value in metric2val.items():
                is_core = (
                    var_name == core_var
                    and metric_name.startswith(("mean", "maj", "best"))
                    and f"@{n_max}" in metric_name
                )
                section = "val-core" if is_core else "val-aux"
                metrics[f"{section}/{agent_id}/{data_source}/{var_name}/{metric_name}"] = float(metric_value)
    return metrics
