#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
import copy
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import ray
from omegaconf import OmegaConf
from ray.util.queue import Queue
from recipe.async_flow.agent_loop.agent_loop import AsyncFlowAgentLoopManager
from recipe.async_flow.config import get_resource_pool_spec, is_role_enabled
from recipe.async_flow.utils.metric.prometheus import marked_timer
from recipe.async_flow.utils.metrics_util import aggregate_metrics_before_reduce, reduce_timing_metrics
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.utils.transfer_queue.tq_mgr import TransferQueueManager
from recipe.async_flow.workers import (
    ActorForwardWorker,
    ActorTrainWorker,
    AdvantageWorker,
    ReferenceForwardWorker,
)
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.metric import reduce_metrics

log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


class AsyncFlowGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        self.actor_fwd_wg = None
        self.ref_fwd_wg = None
        self.advantage_wg = None
        self.actor_train_wg = None
        self.wg_names = ["actor_fwd_wg", "ref_fwd_wg", "advantage_wg", "actor_train_wg"]
        self.async_rollout_mode = True
        self.async_rollout_manager = None
        self.ckpt_backend = None

        # ── cluster trace auto-install (driver orchestration) ────────────
        if os.environ.get("VERL_CLUSTER_TRACE"):
            from recipe.async_flow.utils.cluster_trace.trace_logger import install

            install(role="driver_ctrl", rank=0)
        # ─────────────────────────────────────────────────────────────────

        self._init_transfer_queue()
        self._init_flow_control_queue()
        self._init_async_workers()
        self._init_ckpt_engine_manager()
        logger.info("AsyncFlowGRPOTrainer initialized")

    def _init_flow_control_queue(self):
        """初始化容量队列"""
        staleness = self.config.async_flow.staleness
        batch_size = self.config.data.train_batch_size
        max_capacity = (staleness + 1) * batch_size

        self.flow_control_queue = Queue(maxsize=max_capacity)
        logger.info(f"Flow controller: capacity_queue created with {staleness=}, {batch_size=}, {max_capacity=}")

    def _init_ckpt_engine_manager(self):
        logger.info("creating ckpt_engine_manager between train and rollout and fwd ....")
        from verl.utils.config import omega_conf_to_dataclass

        ckpt_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        if self.ckpt_backend == "flexfetch":
            from recipe.async_flow.utils.async_flow_flexfetch_checkpoint_engine import (
                AsyncFlowCheckpointEngineManager,
            )
        else:
            from recipe.async_flow.utils.async_flow_checkpoint_engine_hccl import AsyncFlowCheckpointEngineManager
        self.train_rollout_fwd_ckpt_manager = AsyncFlowCheckpointEngineManager(
            config=ckpt_engine_config,
            trainer_wg=self.actor_train_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
            fwd_wg=self.actor_fwd_wg,
        )

    def _init_transfer_queue(self):
        self.tq_manager = TransferQueueManager.remote()
        ray.get(self.tq_manager.init_ready.remote())
        self.tq_client = get_transferqueue_client()

        experience_columns = [
            "prompt",
            "prompt_length",
            "labels",
            "responses",
            "response_length",
            "model_version",
            "old_logprobs",
            "reference_logprobs",
            "rollout_log_probs",
            "reward",
            "advantage",
            "token_level_scores",
            "returns",
            "metrics",
            "input_ids",
            "attention_mask",
            "response_mask",
            "rm_scores",
            "position_ids",
            "prompt_uid",
            "raw_prompt",
            "data_source",
        ]
        experience_consumers = ["actor_forward", "ref_forward", "advantage", "actor_train"]

        n_samples = self.config.actor_rollout_ref.rollout.n
        topic = self.config.async_flow.get("experience_topic", "experience")
        self.tq_client.add_topic(
            prompts_num=1000,
            n_samples_per_prompt=n_samples,
            experience_columns=experience_columns,
            experience_consumers=experience_consumers,
            topic=topic,
        )

    def _init_async_workers(self):
        method_start_time = time.perf_counter()
        default_init_args = {"config": self.config}

        train_init_args = {"config": copy.deepcopy(self.config), "flow_control_queue": self.flow_control_queue}

        self.ckpt_backend = self.config.actor_rollout_ref.rollout.checkpoint_engine.backend
        if self.ckpt_backend == "flexfetch":
            OmegaConf.set_struct(train_init_args["config"].actor_rollout_ref, True)
            OmegaConf.update(
                train_init_args["config"],
                "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.flexfetch.is_trainer",
                True,
                force_add=True,  # 允许自动创建不存在的 key
            )

        worker_defs = [
            ("actor_fwd", "actor_fwd_wg", ActorForwardWorker, default_init_args),
            ("ref_fwd", "ref_fwd_wg", ReferenceForwardWorker, default_init_args),
            ("advantage", "advantage_wg", AdvantageWorker, default_init_args),
            ("actor_train", "actor_train_wg", ActorTrainWorker, train_init_args),
        ]

        tasks = []
        rc = self.config.async_resources
        self.worker_ngpus = 0
        for role, attr_name, worker_cls, init_args in worker_defs:
            if not is_role_enabled(rc, role):
                continue
            if role == "advantage":
                use_gpu = False
                process_on_nodes = [rc.advantage.num_cpus] * rc.advantage.nnodes
            else:
                use_gpu = True
                process_on_nodes = get_resource_pool_spec(rc, role)
                self.worker_ngpus += sum(process_on_nodes)

            tasks.append(
                {
                    "name": attr_name,
                    "worker_cls": worker_cls,
                    "resource_pool": RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=use_gpu),
                    "init_args": init_args,
                    "gpu_count": sum(process_on_nodes) if use_gpu else 0,
                }
            )

        sorted_tasks_for_pg = sorted(tasks, key=lambda t: t["gpu_count"], reverse=True)
        for t in sorted_tasks_for_pg:
            pg_start = time.perf_counter()
            t["resource_pool"].get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
            logger.info(
                f"PG ready for [{t['name']}] gpu_count={t['gpu_count']} cost time:{time.perf_counter() - pg_start}"
            )

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(
                    self._create_worker_group,
                    worker_cls=t["worker_cls"],
                    resource_pool=t["resource_pool"],
                    name_prefix=t["name"],
                    init_args=t["init_args"],
                ): t
                for t in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                setattr(self, task["name"], future.result())

        self.actor_rollout_wg = self.actor_train_wg
        logger.info(f"Finish trainer worker init with time:{time.perf_counter() - method_start_time}")

        start_time = time.perf_counter()
        self._init_reward_loop()
        self.async_rollout_manager = AsyncFlowAgentLoopManager.create(
            self.config,
            reward_loop_worker_handles=self.reward_loop_worker_handles,
        )
        logger.info(f"AsyncFlowAgentLoopManager initialized (standalone) cost time:{time.perf_counter()-start_time=}")

        logger.info(f"All Async workers initialized cost time:{time.perf_counter() - method_start_time}")

    def _init_reward_loop(self):
        from verl.experimental.reward_loop import RewardLoopManager

        rm_resource_pool = None
        rm_config = self.config.reward.reward_model
        if self.use_rm and rm_config.enable:
            if rm_config.enable_resource_pool:
                if rm_config.n_gpus_per_node <= 0:
                    raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
                if rm_config.nnodes <= 0:
                    raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

                reward_pool = [rm_config.n_gpus_per_node] * rm_config.nnodes
                rm_resource_pool = RayResourcePool(
                    process_on_nodes=reward_pool,
                    use_gpu=True,
                    max_colocate_count=3,
                    name_prefix="reward_pool",
                )
                rm_resource_pool.get_placement_groups(device_name=self.device_name)

        self.reward_loop_manager = RewardLoopManager(self.config, rm_resource_pool=rm_resource_pool)
        self.reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers
        logger.info(f"RewardLoopManager initialized with {len(self.reward_loop_worker_handles)} workers")

    def _create_worker_group(self, worker_cls, resource_pool, name_prefix, init_args) -> RayWorkerGroup:
        logger.info(f"Starting create worker_group with {worker_cls=},{name_prefix=} ")
        start_time = time.perf_counter()
        ray_cls = RayClassWithInitArgs(cls=worker_cls, **init_args)
        wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls,
            name_prefix=name_prefix,
            device_name=self.device_name,
        )
        logger.info(f"Finish create worker_group  [{name_prefix=}]cost time:{time.perf_counter() - start_time}")

        logger.info(f"Starting init_model [{name_prefix=}]")
        start_time = time.perf_counter()
        wg.init_model()
        logger.info(f"Finish init_model with  [{name_prefix=}]cost time:{time.perf_counter() - start_time}")

        start_time = time.perf_counter()
        if hasattr(wg, "init_ckpt_engine"):
            wg.init_ckpt_engine()
            logger.info(f"Finish init_ckpt_engine with  [{name_prefix=}]cost time:{time.perf_counter() - start_time}")
        return wg

    def _sync_weights_if_needed(self, current_model_version: int) -> int:
        """检查训练版本，若已更新则同步权重到 rollout 和 fwd workers。

        Returns:
            同步后的 model_version。
        """
        self._check_worker_health()
        try:
            train_versions = self.actor_train_wg.get_current_version()
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with train worker: {e}") from e
        current_train_version = train_versions[0]

        if current_train_version <= current_model_version:
            return current_model_version

        # 更新权重前，打印Metrics
        self._print_data_metrics(current_model_version)
        self._print_timing_metrics(self.actor_train_wg, current_model_version)
        self._print_timing_metrics(self.actor_fwd_wg, current_model_version)
        self._print_timing_metrics(self.ref_fwd_wg, current_model_version)

        logger.info(f"Syncing weights: v{current_model_version} -> v{current_train_version}")
        timing_raw = {}
        wait_for_inflight_requests = self.config.async_flow.get("wait_for_inflight_requests", True)

        is_last_step = self.global_steps >= self.total_training_steps
        self.progress_bar.update(1)
        self.global_steps += 1

        # Save checkpoint BEFORE syncing weights to rollout/fwd workers.
        # At this point the training thread is blocked waiting for weight_updated=True
        if self.config.trainer.save_freq > 0 and (
            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
        ):
            with marked_timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

        with marked_timer("sync_param", timing_raw):
            self.train_rollout_fwd_ckpt_manager.update_weights_rollout_fwd(
                version_id=current_train_version, wait_for_inflight=wait_for_inflight_requests
            )

        logger.info(f"[CKPT Engine Metrics] Sync param cost time:{timing_raw['sync_param']}")
        logger.info(f"Weight sync complete for version {current_train_version}")
        return current_train_version

    def _start_async_workers(self):
        for wg_name in self.wg_names:
            if hasattr(self, wg_name):
                getattr(self, wg_name).start_async_loop()
        logger.info("All async workers started")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        # resume model version from checkpoint
        self.train_rollout_fwd_ckpt_manager.update_weights_rollout_fwd(version_id=self.global_steps)

    def _print_data_metrics(self, current_model_version):
        """打印当前模型版本的精度指标。"""
        metrics = aggregate_metrics_before_reduce(self.actor_train_wg.get_metrics())[current_model_version]
        metrics_aggregated = reduce_metrics(metrics)  # 输入是dict{key: [values]}, 复用原生的reduce_metrics()
        metrics_aggregated.update({"training/global_step": self.global_steps})
        self.logger_tracking.log(data=metrics_aggregated, step=self.global_steps)

    def _print_timing_metrics(self, wg, current_model_version):
        """打印当前模型版本的性能指标。"""
        timing_wg_metrics = aggregate_metrics_before_reduce(wg.get_timing_metrics())
        if len(timing_wg_metrics) < 1:
            logger.info(f"No timing metrics get for version:{current_model_version}")
            return
        ((_version, _metrics),) = timing_wg_metrics.items()
        if _metrics["consumer_name"] == "actor_train":
            timing_wg_metrics = timing_wg_metrics[current_model_version]
        else:
            timing_wg_metrics = timing_wg_metrics[_version]
        timing_metrics_aggregated = reduce_timing_metrics(timing_wg_metrics)
        total_gpus = self.worker_ngpus + self.resource_pool_manager.get_n_gpus()
        timing_metrics_aggregated.update(
            {"perf/throughtput": timing_metrics_aggregated["perf/throughtput"] / total_gpus}
        )
        timing_metrics_aggregated.update(
            {"compute/throughtput": timing_metrics_aggregated["compute/throughtput"] / total_gpus}
        )
        self.logger_tracking.log(data=timing_metrics_aggregated, step=self.global_steps)

    def _check_worker_health(self) -> None:
        """Check health of all async workers and raise RuntimeError if any has failed."""
        for wg_name in self.wg_names:
            if not hasattr(self, wg_name):
                continue
            wg = getattr(self, wg_name)
            try:
                stats_list = wg.get_stats()
                for stats in stats_list:
                    if stats.get("last_error"):
                        raise RuntimeError(f"Worker {stats['consumer_name']} fatal error: {stats['last_error']}")
                    if not stats.get("running", True):
                        raise RuntimeError(
                            f"Worker {stats['consumer_name']} loop stopped unexpectedly "
                            f"(errors={stats.get('errors', 0)})"
                        )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to get stats from {wg_name} -- worker may have crashed: {e}") from e

        if self.async_rollout_manager is not None:
            error = self.async_rollout_manager.check_health()
            if error is not None:
                raise RuntimeError(f"AgentLoop error: {error}")

    def shutdown(self):
        for wg_name in self.wg_names:
            if hasattr(self, wg_name):
                try:
                    getattr(self, wg_name).stop_async_loop()
                except Exception as e:
                    logger.warning(f"Error stopping {wg_name}: {e}")
        if self.async_rollout_manager is not None:
            try:
                if hasattr(self.async_rollout_manager, "_loop") and self.async_rollout_manager._loop is not None:
                    self.async_rollout_manager._loop.call_soon_threadsafe(self.async_rollout_manager._loop.stop)
            except Exception as e:
                logger.warning(f"Error stopping agent loop: {e}")
        logger.info("All async workers stopped")

    def fit(self):
        """The main training loop."""
        from verl.utils.tracking import Tracking

        self.logger_tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        logger.info("====== Starting training ==========")

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        model_version = self.global_steps - 1

        # Initialize profiling variables
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        self._start_async_workers()

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self._check_worker_health()
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_prompts = len(batch.batch)
                data_waiting = 0
                while self.flow_control_queue.qsize() + num_prompts > self.flow_control_queue.maxsize:
                    if data_waiting % 60 == 0:
                        self._check_worker_health()
                    time.sleep(3)
                    logger.info(
                        f"Flow control queue backpressure: "
                        f"inflight({self.flow_control_queue.qsize()}) + new({num_prompts}) = "
                        f"{self.flow_control_queue.qsize() + num_prompts} > "
                        f"capacity({self.flow_control_queue.maxsize}), waiting for generation to complete..."
                    )

                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                model_version = self._sync_weights_if_needed(model_version)

                self.async_rollout_manager.generate_sequences(gen_batch_output)

                self.flow_control_queue.put_nowait_batch([1] * num_prompts)

                # Note: Validation is disabled in AsyncFlowGRPOTrainer

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                self.logger_tracking.log(data=metrics, step=self.global_steps)

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_train_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_train_wg, "async_calls_finalize_fn_exec"):
                        self.actor_train_wg.async_calls_finalize_fn_exec(blocking=True)
                    self.progress_bar.close()
                    return
