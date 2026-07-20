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
import asyncio
import logging
import threading
import time
from typing import Any, Optional
from uuid import uuid4

import ray
import recipe.async_flow.agent_loop.single_turn_agent_loop  # noqa: F401  trigger @register
import torch
from omegaconf import DictConfig, OmegaConf
from recipe.async_flow.utils.metric.prometheus import update_metric
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.vllm_rollout.vllm_async_server import AsyncFlowReplica

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import Tracking
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


def _extract_worker_idx(actor_name: str) -> int:
    parts = actor_name.split("_")
    for part in reversed(parts):
        try:
            return int(part)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot extract worker_idx from actor_name '{actor_name}': no valid integer found in name segments"
    )


class AsyncFlowLLMServerManager(AsyncLLMServerManager):
    """AsyncLLMServerManager with model_version support.

    Adds generate_with_version() that returns (TokenOutput, model_version, abort_stats)
    by querying the chosen server's model_version in parallel with generation.
    """

    async def generate_with_version(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[TokenOutput, int, dict]:
        """Generate tokens and return model_version from the chosen server.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters.
            image_data: Optional image data.
            video_data: Optional video data.

        Returns:
            Tuple[TokenOutput, int, dict]: (token output, model_version, abort_stats)
        """
        server_id, server = await self._acquire_server(request_id)
        try:
            partial_prompt_ids = prompt_ids.copy()
            output_list = []
            token_budget = self.config.data.get("max_response_length", 1)
            attempts = self.config.async_flow.get("rollout_max_resume_attempts", 1)
            attempt = 0
            while attempt < attempts:
                attempt += 1
                sampling_params["max_tokens"] = token_budget
                output, model_version = await asyncio.gather(
                    server.generate.remote(
                        request_id=uuid4().hex,
                        prompt_ids=partial_prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data,
                        video_data=video_data,
                    ),
                    server.get_model_version.remote(),
                )
                output_list.append(output)

                if output.stop_reason == "aborted":
                    logger.info(f"aborted request resume generation {output_list}")
                    partial_prompt_ids.extend(output.token_ids)
                    token_budget -= len(output.token_ids)
                    if token_budget <= 0:
                        break
                else:
                    break

            if len(output_list) != 1:
                response_ids = []
                response_logp = []
                for output in output_list:
                    response_ids.extend(output.token_ids)
                    if output.log_probs is not None:
                        response_logp.extend(output.log_probs)

                if len(response_logp) == 0:
                    response_logp = None
                output = TokenOutput(token_ids=response_ids, log_probs=response_logp, stop_reason=output.stop_reason)
                logger.info(f"aborted request completed {output_list} {output} {len(prompt_ids)} {len(response_ids)}")

            num_aborts = len(output_list) - 1
            abort_stats = {"is_partial": num_aborts > 0, "num_aborts": num_aborts}
            return output, model_version, abort_stats
        finally:
            self._release_server(server_id)


@ray.remote
class AsyncFlowAgentLoopWorker(AgentLoopWorker):
    """Custom worker that injects AsyncFlowLLMServerManager for model_version support."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        # ── cluster trace auto-install ───────────────────────────────────
        import os

        if os.environ.get("VERL_CLUSTER_TRACE"):
            from recipe.async_flow.utils.cluster_trace.trace_logger import _get_role, install

            actor_name = ray.get_runtime_context().get_actor_name()
            worker_idx = _extract_worker_idx(actor_name)
            role = _get_role(type(self).__name__)
            install(role=role, rank=worker_idx)
        # ────────────────────────────────────────────────────────────────

        self.server_manager = AsyncFlowLLMServerManager(config, servers, load_balancer_handle)
        super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)


class AsyncFlowAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        worker_group=None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.rollout_replica_class = AsyncFlowReplica
        self.agent_loop_workers_class = AsyncFlowAgentLoopWorker
        super().__init__(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)
        self.tq_client = get_transferqueue_client()

        self.prompt_pool = asyncio.Queue()
        self._semaphore = None
        self.inflight_limit = self.config.data.train_batch_size * (self.config.async_flow.staleness + 1)
        self._fatal_error = None
        self._error_lock = threading.Lock()

        self.total_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        self.task_monitor = {}
        self.global_steps = 0
        self.logger_tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def reset_monitor(self, step):
        num_workers = len(self.agent_loop_workers)
        self.task_monitor = {
            "step": step,
            "num_workers": num_workers,
            "task_done": 0,
            "total_size": self.total_size,
            "start_time": time.perf_counter(),
            "total_num_tokens": 0,
            "response_length": 0,  # Tensor sum
            "prompt_length": 0,  # Tensor sum
            "max_response": 0,  # Tensor sum
            "partial_count": 0,
            "total_aborts": 0,
            "max_aborts": 0,
        }

    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group=None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        instance = await super().create(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)
        instance._ensure_loop_running()
        return instance

    def _ensure_loop_running(self):
        """创建一个专用的线程运行 asyncio loop"""
        self._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._semaphore = asyncio.Semaphore(self.inflight_limit)
            # 初始化性能指标
            self.reset_monitor(self.global_steps)
            # 为每个 Worker 启动一个消费循环
            for worker in self.agent_loop_workers:
                self._loop.create_task(self._worker_pull_loop(worker))
            logger.info(f"Started {len(self.agent_loop_workers)} worker pull loops.")
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    def generate_sequences(self, prompts: DataProto) -> None:
        step = prompts.meta_info.get("global_steps", 0)
        batch_size = len(prompts)

        # 从配置里读n_sample
        num_chunks = batch_size // self.config.actor_rollout_ref.rollout.n

        # 调用原生 chunk 方法进行物理切分。这会返回一个 List[DataProto]，每个chunk的size为 n_sample
        chunks_list = prompts.chunk(num_chunks)

        def inject():
            for chunk in chunks_list:
                chunk.meta_info["step"] = step
                self.prompt_pool.put_nowait(chunk)

            logger.info(f"Injected {num_chunks} chunks (size {len(chunk)}) into pool.")

        self._loop.call_soon_threadsafe(inject)

    async def _worker_pull_loop(self, worker):
        worker_name = str(worker)
        while self._fatal_error is None:
            try:
                chunk = await self.prompt_pool.get()
                if self._fatal_error is not None:
                    break
                async with self._semaphore:
                    try:
                        future = worker.generate_sequences.remote(chunk)
                        result = await asyncio.wrap_future(future.future())
                        await self._process_and_writeback(result, chunk)
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        logger.exception(f"Inference/writeback error: {error_msg}")
                        with self._error_lock:
                            self._fatal_error = f"Worker {worker_name}: {error_msg}"
                        break
                self.prompt_pool.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"Critical: {type(e).__name__}: {str(e)}"
                logger.exception(error_msg)
                with self._error_lock:
                    self._fatal_error = error_msg
                break

    async def _process_and_writeback(self, result: DataProto, original_chunk: DataProto):
        """融合原生的 Metric 累加逻辑"""
        await self._put_batch_to_tq(result, original_chunk)

        response_len_vec = result.batch["response_mask"].sum(dim=-1).float()
        prompt_len_vec = result.batch["attention_mask"].sum(dim=-1).float() - response_len_vec
        tokens = int(result.batch["attention_mask"].sum().item())
        max_response_batch = response_len_vec.max().item()

        monitor = self.task_monitor
        if monitor:
            monitor["task_done"] += len(result.batch)
            monitor["total_num_tokens"] += tokens
            monitor["response_length"] += response_len_vec.sum().item()
            monitor["prompt_length"] += prompt_len_vec.sum().item()
            monitor["max_response"] = max(max_response_batch, monitor["max_response"])

            # Partial rollout statistics
            if result.non_tensor_batch and "is_partial" in result.non_tensor_batch:
                is_partial_list = result.non_tensor_batch["is_partial"]
                num_aborts_list = result.non_tensor_batch.get("num_aborts")
                for i, is_partial in enumerate(is_partial_list):
                    if is_partial:
                        monitor["partial_count"] += 1
                    if num_aborts_list is not None:
                        aborts = int(num_aborts_list[i])
                        monitor["total_aborts"] += aborts
                        monitor["max_aborts"] = max(monitor["max_aborts"], aborts)

            if monitor["task_done"] >= monitor["total_size"]:
                self._log_final_metrics(self.global_steps, monitor)
                self.global_steps = self.global_steps + 1
                self.reset_monitor(self.global_steps)

    async def _put_batch_to_tq(self, batch_with_response: DataProto, chunk: DataProto):
        batch_size = len(batch_with_response.batch)
        topic = self.config.async_flow.experience_topic

        # 计算长度
        prompt_tensors = batch_with_response.batch["prompts"]
        prompt_lengths = torch.full((batch_size, 1), prompt_tensors.shape[1], dtype=torch.long)
        response_tensors = batch_with_response.batch["responses"]
        response_lengths = batch_with_response.batch["response_mask"].sum(dim=-1).unsqueeze(1)

        labels = [None] * batch_size
        if chunk.non_tensor_batch and "reward_model" in chunk.non_tensor_batch:
            rm_data = chunk.non_tensor_batch["reward_model"]
            labels = [item.get("ground_truth") if isinstance(item, dict) else item for item in rm_data]

        prompt_uid = [None] * batch_size
        if chunk.non_tensor_batch and "uid" in chunk.non_tensor_batch:
            prompt_uid = list(chunk.non_tensor_batch["uid"])

        model_version = [torch.tensor(0, dtype=torch.long)] * batch_size
        if batch_with_response.non_tensor_batch and "model_version" in batch_with_response.non_tensor_batch:
            model_version_list = batch_with_response.non_tensor_batch["model_version"]
            model_version = [torch.tensor(item if item else 0, dtype=torch.long) for item in model_version_list]

        data_dict = {
            "prompt": prompt_tensors.cpu(),
            "prompt_length": prompt_lengths.cpu(),
            "responses": response_tensors.cpu(),
            "response_length": response_lengths.cpu(),
            "model_version": model_version,
            "labels": labels,
            "prompt_uid": prompt_uid,
            "input_ids": batch_with_response.batch["input_ids"].cpu(),
            "attention_mask": batch_with_response.batch["attention_mask"].cpu(),
            "response_mask": batch_with_response.batch["response_mask"].cpu(),
            "position_ids": batch_with_response.batch["position_ids"].cpu(),
            "raw_prompt": list(batch_with_response.non_tensor_batch.get("raw_prompt", [])),
            "data_source": list(chunk.non_tensor_batch.get("data_source", [])),
        }

        # 可选字段处理
        for key in ["rollout_log_probs", "rm_scores"]:
            if key in batch_with_response.batch:
                data_dict[key] = batch_with_response.batch[key].cpu()

        timing_raw = {}
        with simple_timer("put_data", timing_raw):
            await self.tq_client.put_experience_async(data_dict=data_dict, topic=topic, version=int(model_version[0]))

    def _log_final_metrics(self, current_version, monitor):
        timing_metrics: dict[str, Any] = {}
        tp_size = self.config.actor_rollout_ref.rollout.get("tensor_parallel_size", 1)

        duration_s = time.perf_counter() - monitor["start_time"]
        total_responses = monitor["response_length"]
        total_prompts = monitor["prompt_length"]
        max_response = monitor["max_response"]

        actual_global_tps = (total_responses / tp_size) / max(duration_s, 1e-6)

        timing_metrics["rollout/total_num_tokens"] = monitor["total_num_tokens"]
        timing_metrics["rollout/e2e_max_rollout"] = duration_s
        timing_metrics["rollout/throughput"] = actual_global_tps
        timing_metrics["rollout/total_prompts"] = total_prompts
        timing_metrics["rollout/total_responses"] = total_responses
        timing_metrics["rollout/max_response"] = max_response

        # Partial rollout metrics
        total_sequences = monitor["task_done"]
        partial_count = monitor["partial_count"]
        total_aborts = monitor["total_aborts"]
        max_aborts = monitor["max_aborts"]
        partial_ratio = partial_count / max(total_sequences, 1)

        timing_metrics["rollout/partial_ratio"] = partial_ratio
        timing_metrics["rollout/partial_count"] = partial_count
        timing_metrics["rollout/total_aborts"] = total_aborts
        timing_metrics["rollout/max_aborts_per_sequence"] = max_aborts

        self.logger_tracking.log(data=timing_metrics, step=current_version)

        update_metric("af_rollout_sequence_length", monitor["total_num_tokens"], labels={"step": current_version})
        update_metric("af_rollout_duration_seconds", duration_s, labels={"step": current_version})
        logger.info(f"Step {current_version} Metrics logged. TPS: {actual_global_tps:.2f}")

    def check_health(self) -> Optional[str]:
        """Check if the agent loop has encountered a fatal error.

        Returns:
            Error message string if a fatal error occurred, None otherwise.
        """
        with self._error_lock:
            return self._fatal_error
