# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import random
import time
from pprint import pformat

import numpy as np
import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncLLMServerClient,
    FullyAsyncLLMServerManager,
    FullyAsyncRollouter,
)
from verl.protocol import DataProto
from verl.utils.profiler import marked_timer
from verl.utils.ray_utils import auto_await

from .agent_loop.agent_loop import AgentLoopWorkerForRewardQueue
from .reward_queue import RewardQueueClient
from .utils import (
    SampleAggregator,
    SubRewardDataItem,
    _ScoredSubItem,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncAgentLoopManager(FullyAsyncAgentLoopManager):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        self.enable_reward_queue = config.async_training.get("enable_reward_queue", False) if config else False
        if self.enable_reward_queue:
            self.agent_loop_workers_class = ray.remote(AgentLoopWorkerForRewardQueue)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_single_for_reward_queue(self, batch: DataProto) -> tuple[DataProto, DataProto, float, float]:
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(batch)
        return await asyncio.wrap_future(output_future.future())


ray_metadata = FullyAsyncRollouter.__ray_metadata__
OriginalRollouter = ray_metadata.modified_class


@ray.remote(num_cpus=10, max_concurrency=100)
class Rollouter(OriginalRollouter):
    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)

        self.reward_queue_client = None
        self.reward_loop_worker_handles = None

        self.sample_aggregator = SampleAggregator()
        self.rollout_n: int | None = None
        self.scoring_paused = False

        self.enable_reward_queue = config.async_training.get("enable_reward_queue", False)

    def _init_async_objects(self):
        self.lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        # `_scoring_resume_event` signals that the scoring is currently running (scoring_paused == False).
        self._scoring_resume_event = asyncio.Event()
        self._scoring_resume_event.set()

    async def set_reward_queue_client(self, reward_queue_client: RewardQueueClient):
        async with self.lock:
            self.reward_queue_client = reward_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            if hasattr(self, "llm_server_manager") and self.llm_server_manager is not None:
                self.max_concurrent_samples = len(self.llm_server_manager.get_replicas()) * 16
                self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            else:
                self.max_concurrent_samples = self.max_required_samples
            self.max_queue_size = self.max_required_samples

            self.rollout_n = self.config.actor_rollout_ref.rollout.n
            if self.enable_reward_queue:
                self._init_reward_queue_size()

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def _init_reward_queue_size(self):
        reward_queue_size = self.config.async_training.get("reward_queue_size", None)
        if reward_queue_size is not None:
            self.max_reward_queue_size = reward_queue_size * self.rollout_n
        else:
            self.max_reward_queue_size = self.max_required_samples * self.rollout_n

    async def _init_async_rollout_manager(self):
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        self.llm_server_manager = await FullyAsyncLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
        )
        self.async_rollout_manager = await AsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    def get_max_reward_queue_size(self):
        return getattr(self, "max_reward_queue_size", None)

    def get_reward_loop_worker_handles(self):
        if hasattr(self, "reward_loop_manager") and self.reward_loop_manager is not None:
            if hasattr(self.reward_loop_manager, "reward_loop_workers"):
                return self.reward_loop_manager.reward_loop_workers
        return None

    def set_reward_loop_worker_handles(self, handles):
        self.reward_loop_worker_handles = handles

    async def reset_staleness(self):
        """
        Reset staleness samples after parameter update.
        Returns timing_raw dictionary for metrics.
        """
        async with self.lock:
            self.paused = False
            self.scoring_paused = False
            # Wake the drain loop in _processor_worker so it can exit early and resume submitting
            # new samples to idle replicas instead of waiting for long-tail in-flight tasks.
            self._resume_event.set()
            self._scoring_resume_event.set()
            # every time param change, reset staleness_samples
            self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
            timing_raw = {}
            rollout_version_time = max(time.time() - self.step_start_time, 1e-6)
            if self.idle_start_time > self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
            else:
                rollout_active_time = rollout_version_time
                idle_ratio = 0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f}"
            )
            self.step_start_time = time.time()

        return timing_raw

    def do_validate(self):
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    # Drain: wait for either (a) at least one active task to finish, or
                    # (b) a resume signal (reset_staleness / monitor flipping paused=False) to
                    # break the drain early so new samples can be submitted to free replicas.
                    # We do NOT hold the lock during the wait, so publishers can acquire it to
                    # update paused / staleness_samples concurrently.
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                        if resume_future in done:
                            print(
                                "[FullyAsyncRollouter][Processor] "
                                "Drain interrupted by resume signal, resuming generation early "
                                f"(active tasks remaining: {len(self.active_tasks)})"
                            )
                            break

                    # block until resuming
                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue
            # Get sample from appropriate queue and immediately mark task as done
            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            self.staleness_samples += 1

            if rollout_sample is None:
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                # Signal RewardQueue that no more items will be produced
                if self.enable_reward_queue:
                    await self.reward_queue_client.shutdown()
                break

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            # Submit single sample processing
            if self.paused:
                await self._resume_event.wait()
            async with self.lock:
                if self.enable_reward_queue:
                    task = safe_create_task(
                        self._process_sample_with_reward_queue(rollout_sample),
                        name=rollout_sample.sample_id,
                        task_set=self.active_tasks,
                    )
                else:
                    task = safe_create_task(
                        self._process_single_sample_streaming(rollout_sample),
                        name=rollout_sample.sample_id,
                        task_set=self.active_tasks,
                    )

    async def _process_sample_with_reward_queue(self, rollout_sample: RolloutSample):
        batch = rollout_sample.full_batch
        n = len(batch)
        sample_id = rollout_sample.sample_id
        generate_start = time.time()

        inference_tasks = {}
        for i in range(n):
            sub_batch = batch[i : i + 1]
            task = asyncio.create_task(self.async_rollout_manager.generate_single_for_reward_queue(sub_batch))
            inference_tasks[task] = i

        pending = set(inference_tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                i = inference_tasks[task]
                try:
                    padded_dp, reward_input_dp, inf_start, inf_end = task.result()
                except Exception as e:
                    print(f"[FullyAsyncRollouter][Processor] Inference FAILED for {sample_id}[{i}]: {e}")
                    self.dropped_stale_samples += 1
                    continue

                sub_item = SubRewardDataItem(
                    reward_input=reward_input_dp,
                    padded_output=padded_dp,
                    sample_id=sample_id,
                    epoch=rollout_sample.epoch,
                    sub_index=i,
                    total_count=n,
                    inference_start_timestamp=inf_start,
                    inference_end_timestamp=inf_end,
                )

                success = await self.reward_queue_client.put_sample(sub_item)
                if not success:
                    self.dropped_stale_samples += 1

        generate_end = time.time()

        self.processed_sample_count += 1
        if self.processed_sample_count % 10 == 0:
            rq_stats = await self.reward_queue_client.get_statistics()
            print(
                f"[FullyAsyncRollouter][Processor] Inference progress: "
                f"processed={self.processed_sample_count} "
                f"rq_size={rq_stats['queue_size']} sample_id={sample_id} "
                f"generate_time={generate_end - generate_start:.2f}s"
            )

    async def _reward_consumer_worker(self):
        active_reward_tasks = set()
        max_concurrent_rewards = getattr(self, "max_concurrent_samples", 16) * (self.rollout_n or 1)
        self.scoring_paused = False
        diagnostic_counter = 0

        while True:
            diagnostic_counter += 1
            if diagnostic_counter % 50 == 0:
                try:
                    rq_stats = await self.reward_queue_client.get_statistics()
                    mq_stats = await self.message_queue_client.get_statistics()
                    print(
                        f"[FullyAsyncRollouter][RewardConsumer] diagnostic: "
                        f"rq_size={rq_stats['queue_size']} rq_produced={rq_stats['total_produced']} "
                        f"rq_consumed={rq_stats['total_consumed']} "
                        f"mq_size={mq_stats['queue_size']} mq_produced={mq_stats['total_produced']} "
                        f"active_reward_tasks={len(active_reward_tasks)} "
                        f"aggregator_pending={self.sample_aggregator.pending_groups_count} "
                        f"scoring_paused={self.scoring_paused} "
                        f"staleness_samples={self.staleness_samples}"
                    )
                except Exception as e:
                    print(f"[FullyAsyncRollouter][RewardConsumer] diagnostic failed: {e}")

            if self.scoring_paused:
                await self._scoring_resume_event.wait()
            if self.scoring_paused or await self._should_pause_scoring():
                mq_stats = await self.message_queue_client.get_statistics()
                print(
                    f"[FullyAsyncRollouter][RewardConsumer] Pausing scoring: "
                    f"mq_size={mq_stats['queue_size']} >= max_queue_size={self.max_queue_size}"
                )
                async with self.lock:
                    self.scoring_paused = True
                    self._scoring_resume_event.clear()

                scoring_resume_future = asyncio.ensure_future(self._scoring_resume_event.wait())
                try:
                    while active_reward_tasks and not scoring_resume_future.done():
                        wait_set = set(active_reward_tasks) | {scoring_resume_future}
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {scoring_resume_future}
                        if actual_done:
                            for task in actual_done:
                                active_reward_tasks.discard(task)
                                await task
                        if scoring_resume_future in done:
                            print(
                                "[FullyAsyncRollouter][RewardConsumer] "
                                "Drain interrupted by resume signal, resuming scoring early "
                                f"(active_reward_tasks remaining: {len(active_reward_tasks)})"
                            )
                            break

                    if not scoring_resume_future.done():
                        await scoring_resume_future
                finally:
                    if not scoring_resume_future.done():
                        scoring_resume_future.cancel()
                        await asyncio.gather(scoring_resume_future, return_exceptions=True)
                continue

            while len(active_reward_tasks) >= max_concurrent_rewards:
                if not active_reward_tasks:
                    break
                done_tasks, active_reward_tasks = await asyncio.wait(
                    active_reward_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    await task

            result = await self.reward_queue_client.get_sample()
            if result is None:
                while active_reward_tasks:
                    done_tasks, active_reward_tasks = await asyncio.wait(
                        active_reward_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done_tasks:
                        await task
                break

            item, _ = result

            task = safe_create_task(
                self._process_single_sub_reward_item(item),
                name=f"reward_{item.sample_id}_{item.sub_index}",
                task_set=active_reward_tasks,
            )

    async def _process_single_sub_reward_item(self, item: SubRewardDataItem):
        sample_id = item.sample_id

        if not self.reward_loop_worker_handles:
            print(
                f"[FullyAsyncRollouter][RewardConsumer] No reward_loop_worker_handles, "
                f"discard: {sample_id}[{item.sub_index}]"
            )
            self.dropped_stale_samples += 1
            return

        reward_start = time.time()
        try:
            selected_worker = random.choice(self.reward_loop_worker_handles)
            score_result = await selected_worker.compute_score.remote(item.reward_input)
            score = score_result["reward_score"]
            reward_extra_info = score_result.get("reward_extra_info", {})
            reward_compute_time = score_result.get("reward_compute_time", 0.0)
        except Exception as e:
            reward_compute_time = 0.0
            score = -1.0
            reward_extra_info = {}
            print(f"[FullyAsyncRollouter][RewardConsumer] Scoring FAILED for {sample_id}[{item.sub_index}]: {e}")
        reward_end = time.time()

        scored_item = _ScoredSubItem(
            sub_index=item.sub_index,
            padded_output=item.padded_output,
            score=score,
            reward_extra_info=reward_extra_info,
            reward_start=reward_start,
            reward_end=reward_end,
            reward_compute_time=reward_compute_time,
            inference_start=item.inference_start_timestamp,
            inference_end=item.inference_end_timestamp,
        )
        is_complete = self.sample_aggregator.add_scored_item(
            sample_id=sample_id,
            total_count=item.total_count,
            epoch=item.epoch,
            scored_item=scored_item,
        )

        if is_complete:
            await self._finalize_sample(sample_id)

    async def _finalize_sample(self, sample_id: str):
        group = self.sample_aggregator.get_and_remove(sample_id)
        sorted_items = [group.items[i] for i in sorted(group.items.keys())]
        total_count = group.total_count

        scores = [si.score for si in sorted_items]
        all_reward_extra_info = [si.reward_extra_info for si in sorted_items]
        reward_start_ts_list = [si.reward_start for si in sorted_items]
        reward_end_ts_list = [si.reward_end for si in sorted_items]
        reward_compute_time_list = [si.reward_compute_time for si in sorted_items]
        inference_start_list = [si.inference_start for si in sorted_items]
        inference_end_list = [si.inference_end for si in sorted_items]

        failed_count = sum(1 for s in scores if s == -1.0)
        print(
            f"[FullyAsyncRollouter][RewardConsumer] Score summary for {sample_id}: "
            f"bsz={total_count} scores={scores} failed={failed_count} "
            f"abnormal_score_threshold={self.config.trainer.get('dapo_threshold', 0)}"
        )

        final_batch = DataProto.concat([si.padded_output for si in sorted_items])

        prompt_length = final_batch.batch["prompts"].shape[1]
        response_mask = final_batch.batch["response_mask"]
        attention_mask = final_batch.batch["attention_mask"]
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
        device = response_mask.device
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32, device=device)
        rm_scores[torch.arange(response_mask.size(0), device=device), valid_response_length] = torch.tensor(
            scores, dtype=torch.float32, device=device
        )
        final_batch.batch["rm_scores"] = rm_scores

        final_batch.non_tensor_batch["score"] = np.array(scores)
        reward_extra_keys = []
        if all_reward_extra_info and isinstance(all_reward_extra_info[0], dict):
            reward_extra_keys = list(all_reward_extra_info[0].keys())
            for key in reward_extra_keys:
                final_batch.non_tensor_batch[key] = np.array(
                    [info.get(key) for info in all_reward_extra_info], dtype=object
                )
        final_batch.non_tensor_batch["reward_compute_time"] = np.array(reward_compute_time_list)
        final_batch.non_tensor_batch["reward_start_timestamp"] = np.array(reward_start_ts_list)
        final_batch.non_tensor_batch["reward_end_timestamp"] = np.array(reward_end_ts_list)
        final_batch.non_tensor_batch["inference_start_timestamp"] = np.array(inference_start_list)
        final_batch.non_tensor_batch["inference_end_timestamp"] = np.array(inference_end_list)

        if reward_extra_keys:
            final_batch.meta_info["reward_extra_keys"] = reward_extra_keys

        final_batch.non_tensor_batch["uid"] = np.array([f"uid_{sample_id}"] * len(final_batch), dtype=object)
        rollout_status = await self.get_statistics()

        rollout_sample = RolloutSample(
            full_batch=final_batch,
            sample_id=sample_id,
            epoch=group.epoch,
            rollout_status=rollout_status,
        )

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )

        if success:
            self.total_generated_samples += 1
            if self.total_generated_samples % 10 == 0:
                mq_stats = await self.message_queue_client.get_statistics()
                print(
                    f"[FullyAsyncRollouter][RewardConsumer] Put to MQ success: "
                    f"total_generated={self.total_generated_samples} "
                    f"mq_size={mq_stats['queue_size']} sample_id={sample_id}"
                )
        else:
            self.dropped_stale_samples += 1
            print(
                f"[FullyAsyncRollouter][RewardConsumer] Put to MQ DROPPED: "
                f"sample_id={sample_id} total_dropped={self.dropped_stale_samples}"
            )

    async def _should_pause_scoring(self) -> bool:
        mq_stats = await self.message_queue_client.get_statistics()
        mq_size = mq_stats["queue_size"]
        if mq_size >= self.max_queue_size:
            return True
        return False

    def _maybe_create_reward_consumer_task(self):
        if self.enable_reward_queue:
            return safe_create_task(self._reward_consumer_worker(), name="reward_consumer_task")
        return None

    async def _maybe_wait_reward_consumer(self):
        if self.reward_consumer_task:
            await self.reward_consumer_task
            print("[FullyAsyncRollouter] Reward consumer completed")

    async def _maybe_cancel_reward_consumer(self):
        if self.reward_consumer_task and not self.reward_consumer_task.done():
            self.reward_consumer_task.cancel()
            await asyncio.gather(self.reward_consumer_task, return_exceptions=True)

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(
            f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}"
            + (" (with RewardQueue)" if self.enable_reward_queue else "")
        )

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")
        self.reward_consumer_task = self._maybe_create_reward_consumer_task()

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            tasks_to_wait = [self.feed_task, self.processor_task]
            if self.reward_consumer_task:
                tasks_to_wait.append(self.reward_consumer_task)

            done, pending = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            print("[FullyAsyncRollouter] pending_queue joined")

            await self._maybe_wait_reward_consumer()

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            await self._maybe_cancel_reward_consumer()

            self.feed_task = None
            self.processor_task = None
            self.reward_consumer_task = None

            # Send a finish signal
            await self.message_queue_client.put_sample(sample=None)

            async with self.lock:
                self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.scoring_paused = False
            self.running = True
            self._resume_event.set()
            self._scoring_resume_event.set()

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] resume rollouter.")
                    self._resume_event.set()

            # Trigger scoring recovery
            if self.enable_reward_queue:
                await self._maybe_resume_scoring()

    async def _maybe_resume_scoring(self):
        if self.scoring_paused and not await self._should_pause_scoring():
            async with self.lock:
                self.scoring_paused = False
                print("[FullyAsyncRollouter][Monitor] Resuming scoring, calling _scoring_resume_event.set()")
                self._scoring_resume_event.set()

    async def _should_pause_reward_queue(self) -> bool:
        reward_queue_stats = await self.reward_queue_client.get_statistics()
        reward_queue_size = reward_queue_stats["queue_size"]

        if reward_queue_size >= self.max_reward_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause] "
                    f"due to RewardQueue full: size={reward_queue_size}, max={self.max_reward_queue_size}"
                )
            return True
        return False

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        if self.enable_reward_queue:
            return await self._should_pause_reward_queue()

        return super()._should_pause_generation()

    async def get_statistics(self) -> dict:
        queue_stats = await self.message_queue_client.get_statistics()
        reward_queue_stats = await self.reward_queue_client.get_statistics() if self.reward_queue_client else None

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        if reward_queue_stats:
            stats["monitor/queue/reward_queue_size"] = reward_queue_stats["queue_size"]
            stats["reward_queue/total_produced"] = reward_queue_stats["total_produced"]
            stats["reward_queue/total_consumed"] = reward_queue_stats["total_consumed"]
            stats["reward_queue/dropped_samples"] = reward_queue_stats["dropped_samples"]

        if hasattr(self, "max_reward_queue_size"):
            stats["static/max_reward_queue_size"] = self.max_reward_queue_size
        if self.rollout_n is not None:
            stats["static/rollout_n"] = self.rollout_n
        if self.sample_aggregator is not None:
            stats["aggregator/pending_groups_count"] = self.sample_aggregator.pending_groups_count
            stats["aggregator/total_pending"] = self.sample_aggregator.total_pending

        return stats
