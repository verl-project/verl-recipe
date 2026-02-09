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
SWE Agent Loop — External Control Multi-turn Interaction Mode.

Intercepts SWE-Agent model calls through ModelProxy, allowing VERL to
control generation and collect training trajectories.

Delegated responsibilities:
  - Config merge / dataclass: ``config.runtime_config``
  - SWE-Agent CLI YAML generation: ``config.yaml_builder``
  - Subprocess lifecycle: ``runtime.subprocess_runner``
  - Docker cleanup: ``runtime.container_cleanup``
  - Temp repo creation: ``utils.repo_manager``
  - Message normalisation: ``utils.message_utils``
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

from recipe.swe_agent.config import (
    SWEAgentRuntimeConfig,
    SWEAgentYAMLBuilder,
    apply_data_overrides,
    build_runtime_config,
)
from recipe.swe_agent.runtime.container_cleanup import cleanup_instance_containers
from recipe.swe_agent.runtime.model_proxy import ModelProxy
from recipe.swe_agent.runtime.subprocess_runner import execute_swe_agent
from recipe.swe_agent.utils.message_utils import normalize_openai_messages
from recipe.swe_agent.utils.repo_manager import cleanup_temp_repo, create_temp_repo

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ──────────────────────────────────────────────────────────────────────
# SWEAgentLoop
# ──────────────────────────────────────────────────────────────────────


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    """SWE Agent Loop — External control multi-turn interaction mode."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
            **kwargs,
        )

        # ── Build structured runtime config (YAML baseline) ──
        self.runtime_config: SWEAgentRuntimeConfig = build_runtime_config(yaml_kwargs=kwargs)

        logger.info(
            f"SWE Agent Loop initialised "
            f"(max_turns={self.runtime_config.max_turns}, "
            f"max_steps={self.runtime_config.max_steps}, "
            f"max_parallel_tasks_per_worker={self.runtime_config.max_parallel_tasks_per_worker})"
        )

    @classmethod
    def _slot_lock_dir(cls, output_dir: str) -> str:
        """Return lock directory for cross-process run-slot coordination."""
        digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), f"verl_swe_agent_slots_{digest}")

    @classmethod
    async def _acquire_run_slot(
        cls,
        max_parallel_tasks_per_worker: int,
        output_dir: str,
    ) -> Optional[tuple[int, int]]:
        """Acquire one cross-process run slot."""
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
        """Release a previously acquired run slot."""
        if run_slot is None:
            return
        fd, _ = run_slot
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one SWE-Agent episode and return the trajectory."""
        run_start_time = time.time()
        agent_task: Optional[asyncio.Task] = None
        model_proxy: Optional[ModelProxy] = None
        run_slot: Optional[tuple[int, int]] = None

        # ── 1. Parse input ──
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}

        problem_statement = extra_info.get("problem_statement", "") or kwargs.get("problem_statement", "")
        repo_path = extra_info.get("repo_path", None) or kwargs.get("repo_path", None)
        repo_content = extra_info.get("repo_content", None)
        if repo_content is None:
            repo_content = kwargs.get("repo_content", None)
        repo_name = extra_info.get("repo", "")
        base_commit = extra_info.get("base_commit", "HEAD")
        problem_instance_id = str(extra_info.get("instance_id", "") or "")
        sandbox_overrides = extra_info.get("sandbox_overrides", {}) or {}
        if isinstance(sandbox_overrides, str):
            try:
                sandbox_overrides = json.loads(sandbox_overrides)
            except json.JSONDecodeError:
                sandbox_overrides = {}

        # ── 2. Per-instance config (data-affine overrides) ──
        run_cfg = apply_data_overrides(self.runtime_config, extra_info)

        # ── 3. Temp repo ──
        temp_repo_dir = None
        if repo_content is not None and not repo_path:
            temp_repo_dir = await create_temp_repo(repo_content)
            repo_path = temp_repo_dir
            logger.info(f"Created temporary repo at: {repo_path}")
        elif not repo_path:
            repo_path = "" if repo_name else "/workspace/repo"

        use_preexisting_repo_override = sandbox_overrides.get("use_preexisting_repo", None)
        if use_preexisting_repo_override is None:
            use_preexisting_repo = (
                bool(repo_name) and not bool(repo_path) and run_cfg.docker_image.startswith("sweb.eval.")
            )
        else:
            use_preexisting_repo = bool(use_preexisting_repo_override)

        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "testbed") or "testbed")
        preexisting_repo_reset = bool(sandbox_overrides.get("preexisting_repo_reset", False))

        if use_preexisting_repo:
            logger.info(
                "Using preexisting repo in container: "
                f"repo_name={preexisting_repo_name}, reset={preexisting_repo_reset}"
            )

        logger.info(f"Starting SWE Agent Loop for problem: {problem_statement[:100]}...")

        try:
            if run_cfg.max_parallel_tasks_per_worker > 0:
                logger.info(
                    "Waiting for SWE-agent run slot "
                    f"(max_parallel_tasks_per_worker={run_cfg.max_parallel_tasks_per_worker})"
                )
                run_slot = await self._acquire_run_slot(
                    run_cfg.max_parallel_tasks_per_worker,
                    run_cfg.output_dir,
                )
                logger.info(
                    "Acquired SWE-agent run slot "
                    f"(slot={run_slot[1]}, max_parallel_tasks_per_worker={run_cfg.max_parallel_tasks_per_worker})"
                )

            model_proxy = ModelProxy(port=run_cfg.proxy_port)

            # ── 4. Start ModelProxy ──
            await model_proxy.start_server(max_retries=run_cfg.max_port_retries)
            logger.info(f"ModelProxy started on port {model_proxy.port}")

            # ── 5. Launch SWE-Agent subprocess ──
            agent_task = asyncio.create_task(
                self._launch_agent(
                    problem_statement,
                    repo_path,
                    run_cfg,
                    model_proxy_port=model_proxy.port,
                    repo_base_commit=base_commit,
                    use_preexisting_repo=use_preexisting_repo,
                    preexisting_repo_name=preexisting_repo_name,
                    preexisting_repo_reset=preexisting_repo_reset,
                    problem_statement_id=problem_instance_id,
                    repo_name=repo_name,
                )
            )

            # ── 6. Interaction loop ──
            (
                patch,
                num_turns,
                initial_prompt_ids,
                all_response_ids,
                all_response_mask,
                all_response_logprobs,
            ) = await self._interaction_loop(
                agent_task=agent_task,
                sampling_params=sampling_params,
                max_turns=run_cfg.max_turns,
                request_timeout=run_cfg.proxy_timeout,
                model_proxy=model_proxy,
            )

            # ── 7. Drain agent task ──
            if not agent_task.done():
                patch = await self._drain_agent_task(agent_task, num_turns >= run_cfg.max_turns)

            # ── 8. Log latency ──
            total_elapsed = time.time() - run_start_time
            logger.info(
                f"SWE Agent Loop completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s"
            )

            # ── 9. Build output ──
            return self._build_output(
                initial_prompt_ids=initial_prompt_ids,
                all_response_ids=all_response_ids,
                all_response_mask=all_response_mask,
                all_response_logprobs=all_response_logprobs,
                num_turns=num_turns,
                patch=patch,
                problem_statement=problem_statement,
                repo_path=repo_path,
            )

        finally:
            if agent_task is not None and not agent_task.done():
                agent_task.cancel()
                try:
                    await asyncio.wait_for(agent_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if model_proxy is not None:
                await model_proxy.stop_server()
            cleanup_temp_repo(temp_repo_dir)
            if run_slot is not None:
                self._release_run_slot(run_slot)
                logger.info(f"Released SWE-agent run slot (slot={run_slot[1]})")

    # ------------------------------------------------------------------
    # Interaction loop (extracted for readability)
    # ------------------------------------------------------------------

    async def _interaction_loop(
        self,
        agent_task: asyncio.Task,
        sampling_params: dict[str, Any],
        max_turns: int,
        request_timeout: float,
        model_proxy: ModelProxy,
    ) -> tuple[
        Optional[str],  # patch
        int,  # num_turns
        Optional[list[int]],  # initial_prompt_ids
        list[int],  # all_response_ids
        list[int],  # all_response_mask
        list[float],  # all_response_logprobs
    ]:
        """Run the main turn-by-turn interaction with SWE-Agent via ModelProxy."""
        initial_prompt_ids: Optional[list[int]] = None
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []
        prev_messages: Optional[list[dict]] = None
        num_turns = 0
        patch: Optional[str] = None

        while True:
            # Pre-check: agent already done?
            if agent_task.done():
                try:
                    patch = await agent_task
                except Exception as e:
                    logger.exception(f"SWE-Agent task failed: {e}")
                break

            # Race: model request vs. agent completion
            request_task = asyncio.create_task(model_proxy.get_request())
            done, pending = await asyncio.wait(
                {request_task, agent_task},
                timeout=request_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                logger.error(f"Both request and agent tasks timed out after {request_timeout}s")
                request_task.cancel()
                try:
                    await request_task
                except (asyncio.CancelledError, Exception):
                    pass
                break

            if agent_task in done:
                if request_task in pending:
                    request_task.cancel()
                    try:
                        await request_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    patch = await agent_task
                except Exception as e:
                    logger.exception(f"SWE-Agent task failed: {e}")
                break

            # Process the model request
            try:
                model_request = request_task.result()
            except Exception as e:
                logger.exception(f"Error getting model request: {e}")
                continue

            messages = normalize_openai_messages(model_request.messages)

            # Append tool/observation tokens (mask=0)
            if prev_messages is not None and len(messages) > len(prev_messages):
                new_msgs = messages[len(prev_messages) :]
                tool_msgs = [m for m in new_msgs if m.get("role") != "assistant"]
                if tool_msgs:
                    tool_ids = self.tokenizer.apply_chat_template(
                        tool_msgs,
                        add_generation_prompt=False,
                        tokenize=True,
                    )
                    all_response_ids.extend(tool_ids)
                    all_response_mask.extend([0] * len(tool_ids))
                    all_response_logprobs.extend([0.0] * len(tool_ids))

            # Build prompt & truncate
            prompt_ids = await self.apply_chat_template(messages)
            prompt_ids = self._truncate_prompt(prompt_ids, num_turns)
            if initial_prompt_ids is None:
                initial_prompt_ids = prompt_ids

            # Generate
            output = await self.server_manager.generate(
                request_id=str(uuid.uuid4()),
                prompt_ids=prompt_ids,
                sampling_params={**sampling_params, "logprobs": sampling_params.get("logprobs", True)},
            )

            response_ids = output.token_ids
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Collect model tokens (mask=1)
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                all_response_logprobs.extend(output.log_probs)
            else:
                all_response_logprobs.extend([0.0] * len(response_ids))

            num_turns += 1
            prev_messages = messages

            # Send response back
            await model_proxy.send_response(response_text, request=model_request)

            logger.info(f"Turn {num_turns}: {len(response_ids)} model tokens, total_seq={len(all_response_ids)}")

            if num_turns >= max_turns:
                logger.warning(f"Max turns reached ({num_turns}/{max_turns})")
                break

        return patch, num_turns, initial_prompt_ids, all_response_ids, all_response_mask, all_response_logprobs

    # ------------------------------------------------------------------
    # Agent launch (config gen + subprocess)
    # ------------------------------------------------------------------

    async def _launch_agent(
        self,
        problem_statement: str,
        repo_path: str,
        cfg: SWEAgentRuntimeConfig,
        *,
        model_proxy_port: int,
        repo_base_commit: str = "HEAD",
        use_preexisting_repo: bool = False,
        preexisting_repo_name: str = "testbed",
        preexisting_repo_reset: bool = False,
        problem_statement_id: str = "",
        repo_name: str = "",
    ) -> Optional[str]:
        """Generate config, run SWE-Agent subprocess, cleanup."""
        instance_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        instance_output_dir = os.path.join(cfg.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)
        exec_dir = tempfile.mkdtemp(prefix=f"swe_exec_{instance_id}_")

        # Generate YAML config for SWE-Agent CLI
        config_path = self._write_agent_yaml(
            instance_id,
            repo_path,
            instance_output_dir,
            cfg,
            model_proxy_port=model_proxy_port,
            repo_base_commit=repo_base_commit,
            use_preexisting_repo=use_preexisting_repo,
            preexisting_repo_name=preexisting_repo_name,
            preexisting_repo_reset=preexisting_repo_reset,
            repo_name=repo_name,
        )

        try:
            patch = await execute_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                instance_id=instance_id,
                output_dir=instance_output_dir,
                repo_path=repo_path,
                exec_dir=exec_dir,
                swe_agent_timeout=cfg.swe_agent_timeout,
                proxy_port=model_proxy_port,
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

    def _write_agent_yaml(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        cfg: SWEAgentRuntimeConfig,
        *,
        model_proxy_port: int,
        repo_base_commit: str = "HEAD",
        use_preexisting_repo: bool = False,
        preexisting_repo_name: str = "testbed",
        preexisting_repo_reset: bool = False,
        repo_name: str = "",
    ) -> str:
        """Build and write SWE-Agent CLI YAML, return file path."""
        max_input_tokens = int(getattr(self.config.actor_rollout_ref.rollout, "max_model_len", 0) or 0)
        if use_preexisting_repo:
            yaml_repo_path = preexisting_repo_name
            yaml_repo_type = "preexisting"
        elif repo_name and not repo_path:
            yaml_repo_path = repo_name
            yaml_repo_type = "github"
        else:
            yaml_repo_path = repo_path
            yaml_repo_type = "local"

        builder = SWEAgentYAMLBuilder.from_config(
            cfg,
            instance_id=instance_id,
            repo_path=yaml_repo_path,
            output_dir=output_dir,
            model_proxy_port=model_proxy_port,
            max_input_tokens=max_input_tokens,
            repo_type=yaml_repo_type,
            repo_base_commit=repo_base_commit,
            preexisting_repo_reset=preexisting_repo_reset,
        )
        f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_swe_config_{instance_id}.yaml",
            delete=False,
            encoding="utf-8",
        )
        f.write(builder.to_yaml())
        f.close()
        return f.name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_prompt(self, prompt_ids: list[int], turn: int) -> list[int]:
        """Left-truncate prompt to fit within max_model_len."""
        max_model_len = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
        if max_model_len and len(prompt_ids) >= max_model_len:
            min_gen_tokens = 256
            max_prompt_tokens = max_model_len - min_gen_tokens
            original_len = len(prompt_ids)
            prompt_ids = prompt_ids[-max_prompt_tokens:]
            logger.warning(
                f"Turn {turn + 1}: prompt truncated {original_len} → {len(prompt_ids)} (max_model_len={max_model_len})"
            )
        return prompt_ids

    @staticmethod
    async def _drain_agent_task(agent_task: asyncio.Task, max_turns_reached: bool) -> Optional[str]:
        """Wait for / cancel the SWE-Agent background task."""
        if max_turns_reached:
            logger.warning("Cancelling SWE-Agent task due to max_turns limit")
            agent_task.cancel()
            try:
                return await asyncio.wait_for(agent_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return None
        else:
            try:
                return await asyncio.wait_for(agent_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for agent task completion")
                return None

    def _build_output(
        self,
        *,
        initial_prompt_ids: Optional[list[int]],
        all_response_ids: list[int],
        all_response_mask: list[int],
        all_response_logprobs: list[float],
        num_turns: int,
        patch: Optional[str],
        problem_statement: str,
        repo_path: str,
    ) -> AgentLoopOutput:
        """Assemble the final ``AgentLoopOutput``."""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        final_prompt_ids = initial_prompt_ids if initial_prompt_ids else [pad_token_id]

        if all_response_ids:
            final_response_ids = all_response_ids[:max_response_length]
            final_response_mask = all_response_mask[:max_response_length]
        else:
            final_response_ids = [pad_token_id]
            final_response_mask = [0]

        if all_response_logprobs:
            final_response_logprobs = all_response_logprobs[: len(final_response_ids)]
        else:
            final_response_logprobs = [0.0] * len(final_response_ids)

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=final_response_logprobs,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=0.0,
                tool_calls=0.0,
                num_preempted=-1,
            ),
            extra_fields={
                "patch": patch,
                "problem_statement": problem_statement,
                "repo_path": repo_path,
            },
        )
