# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig
from recipe.partial_rollout.prompt_manager import RolloutPrompt

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker, get_trajectory_info
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


@ray.remote
class PartialRolloutAgentLoopWorker(AgentLoopWorker):
    """Continuous-loop worker with trajectory-grained pull pacing.

    Overrides upstream `AgentLoopWorker.generate_sequences` so the n
    trajectories of one prompt are awaited via `asyncio.wait(FIRST_COMPLETED)`
    instead of `asyncio.gather` — every per-trajectory completion decrements
    `self.inflight_traj` and signals `self._slot_event`. The worker's outer
    `run_continuous` loop watches that event and pulls the next prompt as soon
    as enough trajectories have completed across in-flight prompts to make
    room for another n, even while some prompts are still on their long-tail.

    Aborted `client.generate(...)` calls retry inside
    `FullyLLMServerClient.generate()`, so the worker is oblivious to the
    cancel/resume cycle the manager wires around `update_weights`. The loop
    runs for the actor's lifetime; Ray actor destruction at training end
    terminates it.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        prompt_manager_handle: ray.actor.ActorHandle,
        teacher_client: Optional[dict[str, LLMServerClient]] = None,
        reward_loop_worker_handles: Optional[list[ray.actor.ActorHandle]] = None,
    ):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.prompt_manager_handle = prompt_manager_handle
        self.inflight_traj: int = 0
        self._slot_event: asyncio.Event = asyncio.Event()

    async def generate_for_prompt(self, batch: DataProto) -> DataProto:
        """Copy of upstream `AgentLoopWorker.generate_sequences` with the final
        `asyncio.gather` replaced by an `asyncio.wait(FIRST_COMPLETED)` loop
        that updates `self.inflight_traj` after every per-trajectory
        completion and signals `self._slot_event` so `run_continuous` can
        consider pulling the next prompt.
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )

        outputs: list = [None] * len(tasks)
        idx_of = {t: i for i, t in enumerate(tasks)}
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                outputs[idx_of[t]] = t.result()
            self.inflight_traj -= len(done)
            self._slot_event.set()

        return self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=validate)

    async def run_continuous(self, max_inflight_prompts: int) -> None:
        n = self.rollout_config.n
        max_inflight_traj = max_inflight_prompts * n

        pull = self.prompt_manager_handle.pull_prompts.remote
        push = self.prompt_manager_handle.push_prompts.remote

        running: set[asyncio.Task] = set()
        pull_task: Optional[asyncio.Task] = None

        slot_wait = asyncio.ensure_future(self._slot_event.wait())
        running.add(slot_wait)

        while True:
            if pull_task is None and self.inflight_traj + n <= max_inflight_traj:
                slots = (max_inflight_traj - self.inflight_traj) // n
                pull_task = asyncio.ensure_future(pull(slots))
                running.add(pull_task)

            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            running -= done

            push_list: list[RolloutPrompt] = []
            for t in done:
                if t is slot_wait:
                    self._slot_event.clear()
                    slot_wait = asyncio.ensure_future(self._slot_event.wait())
                    running.add(slot_wait)
                elif t is pull_task:
                    pull_task = None
                    prompts = t.result()
                    self.inflight_traj += len(prompts) * n
                    running.update(asyncio.create_task(self._run_one(rp)) for rp in prompts)
                else:
                    push_list.append(t.result())

            if push_list:
                # Fire-and-forget; doesn't block the next pull.
                push(push_list)

    async def _run_one(self, rp: RolloutPrompt) -> RolloutPrompt:
        rp.gen_batch_output = await self.generate_for_prompt(rp.gen_batch_output)
        return rp


class PartialRolloutAgentLoopManager(AgentLoopManager):
    """PartialRollout manager. Differences vs upstream:

    - Builds PartialRolloutAgentLoopWorker with an extra `prompt_manager_handle` arg so
      cross-step partial-rollout state lives in a Ray actor instead of being
      threaded through generate_sequences kwargs.
    - Skips automatic worker creation in `create()` because workers need the
      prompt manager (and the trainer's `PartialRolloutLLMServerManager` for cancel/
      resume), which the trainer wires in post-init via
      `init_agent_loop_workers(...)`.
    - Exposes `cancel` / `resume` that target replicas owned by the decoupled
      `LLMServerManager`. The trainer brackets `update_weights` with these so
      PartialRollout's continuously-running workers don't generate trajectories that
      straddle a weight update. Necessary because the naive checkpoint_engine
      backend (PartialRollout's default) short-circuits before its own abort/resume.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: Optional[dict[str, LLMServerClient]] = None,
        reward_loop_worker_handles: Optional[list[ray.actor.ActorHandle]] = None,
    ):
        self.agent_loop_workers_class = PartialRolloutAgentLoopWorker
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        # Set by the trainer via init_agent_loop_workers; until then, calling
        # generate_sequences / cancel / resume is a programming error.
        self.rollout_prompt_manager: Optional[ray.actor.ActorHandle] = None
        self.llm_server_manager: Optional[LLMServerManager] = None

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the manager but defer worker spawn.

        Upstream's create eagerly spawns AgentLoopWorkers; PartialRollout can't because
        each worker needs `rollout_prompt_manager`, which the trainer only has
        after `RayPPOTrainer.init_workers` returns. The trainer calls
        `init_agent_loop_workers` immediately after, so the deferral window is
        a couple of statements wide.
        """
        return cls(*args, **kwargs)

    @auto_await
    async def init_agent_loop_workers(
        self,
        rollout_prompt_manager: ray.actor.ActorHandle,
        llm_server_manager: LLMServerManager,
    ):
        self.rollout_prompt_manager = rollout_prompt_manager
        self.llm_server_manager = llm_server_manager
        await self._init_agent_loop_workers()
        # Spawn each worker's persistent loop fire-and-forget. Workers immediately
        # start polling `prompt_manager.pull_prompts` (which blocks on an
        # asyncio.Event until the trainer's first push_batch). Mid-trajectory
        # aborts triggered by the trainer's cancel() are retried with
        # accumulated context inside `FullyLLMServerClient.generate()`, so
        # workers don't need their own pause gate.
        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        num_workers = len(self.agent_loop_workers)
        max_inflight_prompts = (train_batch_size + num_workers - 1) // num_workers
        # Fire-and-forget — workers run until Ray terminates them at training end.
        for worker in self.agent_loop_workers:
            worker.run_continuous.remote(max_inflight_prompts)

    async def _init_agent_loop_workers(self):
        # Mirrors upstream `AgentLoopManager._init_agent_loop_workers` (verl
        # /experimental/agent_loop/agent_loop.py) but injects
        # `rollout_prompt_manager` between `llm_client` and `teacher_client`
        # in the PartialRolloutAgentLoopWorker constructor.
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.rollout_prompt_manager,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Block until the prompt manager has a full training batch ready.

        Cross-step abort/resume cycle:

        - `resume()` at the start lifts the pause set by the previous step's
          `cancel()`. Workers' retried `client.generate(...)` calls — aborted
          last step and waiting inside `FullyLLMServerClient.generate()`'s
          retry loop — submit fresh against the just-updated weights.
        - `cancel()` after `pull_batch` aborts whatever the continuous
          workers are generating at the moment the trainer takes ownership
          of the batch. Without it, workers would keep producing trajectories
          across the upcoming forward/backward/update_weights with stale
          weights — wasted compute (`sleep_replicas` would block them anyway)
          and, if any token slipped through, off-policy contamination.
          Needed because the naive checkpoint_engine backend (PartialRollout's default)
          does not itself bracket abort/resume around `update_weights`.

        Per-sample weight-version tracking lives in `gen_batch.meta_info`
        (set by the trainer) and `FullyLLMServerClient.generate()`'s retry
        loop; no per-worker fan-out needed.
        """
        await self.resume()
        if prompts.meta_info.get("validate", False):
            # Validation uses the upstream `generate_sequences` path which
            # spawns its own per-call rollouts via the llm_client; the
            # continuous workers are unaffected and stay blocked in
            # `pull_prompts` until the next training-side `push_batch`.
            # No cancel at end — validation's per-call rollouts have already
            # completed when super() returns; canceling would only race with
            # the continuous workers' in-flight generations, which are still
            # legitimate work for the next training batch.
            return await super().generate_sequences(prompts)

        # pull_batch is async server-side and blocks on an internal
        # asyncio.Event until done_queue >= batch_size. One round-trip, no
        # manager-side polling, no per-step ~10k empty Ray RPCs.
        output = await self.rollout_prompt_manager.pull_batch.remote()
        await self.cancel()
        return output

    async def cancel(self):
        await self.llm_server_manager.cancel()

    async def resume(self):
        await self.llm_server_manager.resume()
