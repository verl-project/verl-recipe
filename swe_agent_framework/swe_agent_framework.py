# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
SWE-Agent Framework — Agent Gateway integration.

Replaces the old ModelProxy + TrajectoryReconstructor pattern with
GatewayActor-backed online trajectory collection via
OpenAICompatibleAgentFramework.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from recipe.swe_agent.config import (
    SWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
    build_sweagent_yaml,
)
from recipe.swe_agent.subprocess_runner import cleanup_instance_containers, execute_swe_agent

from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.framework.types import SessionHandle, SessionRewardContext, SessionRuntime
from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_rollout_and_model_config(config: DictConfig) -> tuple[DictConfig, DictConfig]:
    if config.get("actor_rollout_ref"):
        return config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
    else:
        return config.rollout, config.model


async def initialize_llm_servers(
    config: DictConfig,
    worker_group=None,
    rollout_resource_pool=None,
):
    rollout_config, model_config = _get_rollout_and_model_config(config)
    rollout_replica_class = get_rollout_replica_class(rollout_config.name)

    rollout_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    world_size = (
        worker_group.world_size
        if worker_group
        else rollout_config.n_gpus_per_node * rollout_config.nnodes
    )
    num_replicas = world_size // rollout_world_size

    rollout_replicas = [
        rollout_replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=rollout_config.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]

    if worker_group and rollout_config.name != "trtllm":
        await asyncio.gather(*[r.init_hybrid(worker_group) for r in rollout_replicas])
    elif worker_group and rollout_config.name == "trtllm":
        await asyncio.gather(
            *[r.init_hybrid_colocated(worker_group, rollout_resource_pool) for r in rollout_replicas]
        )
    else:
        await asyncio.gather(*[r.init_standalone() for r in rollout_replicas])

    server_handles = [r._server_handle for r in rollout_replicas]
    server_addresses = [r._server_address for r in rollout_replicas]
    logger.info(f"SWEAgentFramework: LLM servers ready at {server_addresses}")
    return rollout_replicas, server_handles, server_addresses


# ---------------------------------------------------------------------------
# SWEAgentFramework
# ---------------------------------------------------------------------------


class SWEAgentFramework(OpenAICompatibleAgentFramework):
    """SWE-Agent framework backed by the agent gateway.

    """

    def __init__(
        self,
        session_runtime: SessionRuntime,
        tokenizer,
        runtime_config: SWEAgentRuntimeConfig,
        rollout_replicas: list,
        max_model_len: int = 0,
    ):
        super().__init__(
            session_runtime=session_runtime,
            agent_runner=self._run_swe_agent,
            reward_fn=self._score_session,
            wait_for_completion_after_agent_run=False,
        )
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.rollout_replicas = rollout_replicas
        self._max_model_len = max_model_len


    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
    ) -> "SWEAgentFramework":
        """Initialise LLM servers then construct the framework."""
        rollout_config, model_config = _get_rollout_and_model_config(config)

        rollout_replicas, server_handles, server_addresses = await initialize_llm_servers(
            config, worker_group, rollout_resource_pool
        )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)

        load_balancer = GlobalRequestLoadBalancer.remote(
            server_actor_ids=server_addresses,
        )

        gateway_count = int(OmegaConf.select(config, "actor_rollout_ref.rollout.agent_framework.gateway_count", default=1))
        servers = list(zip(server_addresses, server_handles, strict=True))

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)

        from verl.agent.gateway.runtime import GatewayServingRuntime
        session_runtime = GatewayServingRuntime(
            servers,
            load_balancer,
            gateway_count=gateway_count,
            gateway_actor_kwargs={"tokenizer": tokenizer},
        )

        # Load SWE-agent runtime config from agent_framework.config_path
        swe_agent_config_path = OmegaConf.select(
            config, "actor_rollout_ref.rollout.agent_framework.config_path", default=None
        )
        if swe_agent_config_path:
            raw = OmegaConf.to_container(OmegaConf.load(swe_agent_config_path), resolve=True)
            # Support both plain dict and list-of-configs formats
            if isinstance(raw, list):
                raw = next((c for c in raw if isinstance(c, dict) and c.get("name") == "swe_agent"), raw[0])
            # Strip keys not part of SWEAgentRuntimeConfig (e.g., agent_framework meta-config)
            if isinstance(raw, dict):
                raw = {k: v for k, v in raw.items() if k in ("sandbox_config", "agent")}
            runtime_config = build_runtime_config(raw)
        else:
            runtime_config = build_runtime_config({})

        max_model_len = int(getattr(rollout_config, "max_model_len", 0) or 0)
        return cls(session_runtime, tokenizer, runtime_config, rollout_replicas, max_model_len)

    def start_profile(self, **kwargs):
        for replica in self.rollout_replicas:
            replica.start_profile(**kwargs)

    def stop_profile(self):
        for replica in self.rollout_replicas:
            replica.stop_profile()

    # ------------------------------------------------------------------
    # _run_session override — inject extra_info from sample_fields
    # ------------------------------------------------------------------

    async def _run_session(self, *, prompts, raw_prompt, sample_index: int):
        """Override to forward extra_info to _run_swe_agent and attach reward_info to trajectories."""
        import inspect
        from dataclasses import replace as dc_replace

        import torch
        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from verl.utils import tensordict_utils as tu

        session_id = self._build_session_id(prompts=prompts, sample_index=sample_index)
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data

        session = await self.session_runtime.create_session(session_id)
        try:
            extra_info = sample_fields.get("extra_info", {})
            if isinstance(extra_info, (bytes, str)):
                try:
                    extra_info = json.loads(extra_info)
                except (json.JSONDecodeError, TypeError):
                    extra_info = {}
            patch, problem_statement, repo_path = await self._run_swe_agent(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                extra_info=extra_info,
            )
            session_trajectories = await self.session_runtime.finalize_session(session_id)
        except Exception:
            await self.session_runtime.abort_session(session_id)
            raise

        reward_info = {"patch": patch, "problem_statement": problem_statement, "repo_path": repo_path}
        session_trajectories = [dc_replace(t, reward_info=reward_info) for t in session_trajectories]

        ctx = SessionRewardContext(trajectories=session_trajectories, sample_fields=sample_fields)
        scores = self._score_session(ctx)
        if inspect.isawaitable(scores):
            scores = await scores
        return (
            [dc_replace(traj, reward_score=float(s)) for traj, s in zip(session_trajectories, scores, strict=True)],
            sample_fields,
        )

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    async def _run_swe_agent(
        self,
        raw_prompt,
        session: SessionHandle,
        sample_index: int,
        extra_info: dict,
    ) -> tuple[Optional[str], str, Optional[str]]:
        """Run one SWE-Agent episode. Returns (patch, problem_statement, repo_path)."""
        problem_statement = extra_info.get("problem_statement", "") or ""
        repo_path = extra_info.get("repo_path", None)
        base_commit = extra_info.get("base_commit", "HEAD")
        problem_instance_id = str(extra_info.get("instance_id", "") or "")

        run_cfg = apply_data_overrides(self.runtime_config, extra_info)
        sb = run_cfg.sandbox_config

        sandbox_overrides = extra_info.get("sandbox_overrides", {}) or {}
        if isinstance(sandbox_overrides, str):
            try:
                sandbox_overrides = json.loads(sandbox_overrides)
            except json.JSONDecodeError:
                sandbox_overrides = {}

        use_preexisting_repo = bool(sandbox_overrides.get("use_preexisting_repo", False))
        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "testbed") or "testbed")
        preexisting_repo_reset = bool(sandbox_overrides.get("preexisting_repo_reset", False))

        if not use_preexisting_repo and not repo_path:
            if sb.docker_image.startswith("sweb.eval."):
                use_preexisting_repo = True
            else:
                repo_path = "/workspace/repo"

        run_slot = await self._acquire_run_slot(sb.max_parallel_tasks_per_worker, sb.output_dir)
        try:
            patch = await self._launch_agent(
                problem_statement=problem_statement,
                repo_path=repo_path,
                cfg=run_cfg,
                gateway_base_url=session.base_url,
                repo_base_commit=base_commit,
                use_preexisting_repo=use_preexisting_repo,
                preexisting_repo_name=preexisting_repo_name,
                preexisting_repo_reset=preexisting_repo_reset,
                problem_statement_id=problem_instance_id,
            )
        finally:
            self._release_run_slot(run_slot)

        return patch, problem_statement, repo_path

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def _score_session(self, ctx: SessionRewardContext) -> list[float]:
        from recipe.swe_agent.reward import compute_score

        sample_fields = ctx.sample_fields
        data_source = sample_fields.get("data_source", "swe_agent")
        ground_truth = sample_fields.get("reward_model", {})
        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("ground_truth", ground_truth)

        scores = []
        for trajectory in ctx.trajectories:
            solution_str = self.tokenizer.decode(trajectory.response_ids, skip_special_tokens=True)
            score = compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info={**trajectory.reward_info, "num_turns": trajectory.num_turns},
            )
            scores.append(float(score))
        return scores

    # ------------------------------------------------------------------
    # Agent launch
    # ------------------------------------------------------------------

    async def _launch_agent(
        self,
        problem_statement: str,
        repo_path: Optional[str],
        cfg: SWEAgentRuntimeConfig,
        *,
        gateway_base_url: str,
        repo_base_commit: str = "HEAD",
        use_preexisting_repo: bool = False,
        preexisting_repo_name: str = "testbed",
        preexisting_repo_reset: bool = False,
        problem_statement_id: str = "",
    ) -> Optional[str]:
        """Generate config, run SWE-Agent subprocess, cleanup containers."""
        instance_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        instance_output_dir = os.path.join(cfg.sandbox_config.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)
        exec_dir = tempfile.mkdtemp(prefix=f"swe_exec_{instance_id}_")

        if use_preexisting_repo:
            yaml_repo_path, yaml_repo_type = preexisting_repo_name, "preexisting"
        else:
            yaml_repo_path, yaml_repo_type = repo_path or "/workspace/repo", "local"

        yaml_str = build_sweagent_yaml(
            cfg,
            instance_id=instance_id,
            repo_path=yaml_repo_path,
            output_dir=instance_output_dir,
            gateway_base_url=gateway_base_url,
            max_input_tokens=self._max_model_len,
            repo_type=yaml_repo_type,
            repo_base_commit=repo_base_commit,
            preexisting_repo_reset=preexisting_repo_reset,
        )
        config_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_swe_config_{instance_id}.yaml",
            delete=False,
            encoding="utf-8",
        )
        config_file.write(yaml_str)
        config_file.close()
        config_path = config_file.name

        try:
            patch = await execute_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                instance_id=instance_id,
                output_dir=instance_output_dir,
                repo_path=repo_path or "",
                exec_dir=exec_dir,
                swe_agent_timeout=cfg.sandbox_config.swe_agent_timeout,
                problem_statement_id=problem_statement_id,
            )
            return patch
        except Exception as e:
            logger.exception(f"[{instance_id}] SWE-Agent execution failed: {e}")
            return None
        finally:
            await cleanup_instance_containers(instance_id)
            try:
                os.unlink(config_path)
            except OSError:
                pass
            shutil.rmtree(exec_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Run-slot throttle (limits concurrent Docker containers per worker)
    # ------------------------------------------------------------------

    @classmethod
    def _slot_lock_dir(cls, output_dir: str) -> str:
        digest = hashlib.sha1(os.path.abspath(output_dir).encode()).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), f"verl_swe_agent_slots_{digest}")

    @classmethod
    async def _acquire_run_slot(
        cls,
        max_parallel_tasks_per_worker: int,
        output_dir: str,
    ) -> Optional[tuple[int, int]]:
        if max_parallel_tasks_per_worker <= 0:
            return None
        lock_dir = cls._slot_lock_dir(output_dir)
        os.makedirs(lock_dir, exist_ok=True)
        while True:
            for slot_idx in range(max_parallel_tasks_per_worker):
                lock_path = os.path.join(lock_dir, f"slot_{slot_idx}.lock")
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0), 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(fd, 0)
                    os.write(fd, f"pid={os.getpid()}\n".encode())
                    return fd, slot_idx
                except BlockingIOError:
                    os.close(fd)
            await asyncio.sleep(0.2)

    @staticmethod
    def _release_run_slot(run_slot: Optional[tuple[int, int]]) -> None:
        if run_slot is None:
            return
        fd, _ = run_slot
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
