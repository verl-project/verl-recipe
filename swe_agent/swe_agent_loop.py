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
  - Config merge / dataclass: ``config``
  - SWE-Agent CLI YAML generation: ``config.build_sweagent_yaml``
  - Subprocess lifecycle + Docker cleanup: ``subprocess_runner``
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
    apply_data_overrides,
    build_runtime_config,
    build_sweagent_yaml,
)
from recipe.swe_agent.model_proxy import ModelProxy
from recipe.swe_agent.subprocess_runner import cleanup_instance_containers, execute_swe_agent
from recipe.swe_agent.trajectory import AlignedTrajectory, TrajectoryReconstructor, TurnRecord

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ──────────────────────────────────────────────────────────────────────
# Inline utilities (message normalisation + temp repo management)
# ──────────────────────────────────────────────────────────────────────


def normalize_openai_messages(openai_messages: list[dict]) -> list[dict]:
    """Normalize OpenAI-format messages for ``tokenizer.apply_chat_template``.

    Handles ``content`` as a list of text blocks, None, or non-string.
    """
    messages: list[dict] = []
    for msg in openai_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(
                        part.get("text", str(part)) if part.get("type") == "text" else str(part.get("text", part))
                    )
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


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
        data_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )

        # ── Build structured runtime config (YAML baseline) ──
        # The @register decorator may overwrite the global registry entry with
        # just {"_target_": ...}, stripping YAML-loaded fields like sandbox_config.
        # When that happens, reload from the YAML file directly.
        effective_kwargs = kwargs
        if "sandbox_config" not in kwargs:
            effective_kwargs = self._reload_yaml_config(trainer_config, kwargs)
        self.runtime_config: SWEAgentRuntimeConfig = build_runtime_config(yaml_kwargs=effective_kwargs)

        sb = self.runtime_config.sandbox_config
        logger.info(
            f"SWE Agent Loop initialised "
            f"(max_model_calls_per_instance={sb.max_model_calls_per_instance}, "
            f"max_parallel_tasks_per_worker={sb.max_parallel_tasks_per_worker})"
        )

    @staticmethod
    def _reload_yaml_config(trainer_config, kwargs: dict) -> dict:
        """Reload agent loop config from YAML when registry entry was stripped by @register."""
        from omegaconf import OmegaConf

        from verl.experimental.agent_loop.utils import resolve_config_path

        try:
            cfg = trainer_config.config if hasattr(trainer_config, "config") else trainer_config
            rollout = cfg.actor_rollout_ref.rollout
            yaml_path = rollout.agent.agent_loop_config_path
            if yaml_path:
                resolved = resolve_config_path(yaml_path)
                configs = OmegaConf.load(resolved)
                for c in configs:
                    if getattr(c, "name", None) == "swe_agent":
                        merged = OmegaConf.to_container(c, resolve=True)
                        merged.update(kwargs)
                        logger.info(f"Reloaded YAML config for swe_agent from {resolved}")
                        return merged
        except Exception as e:
            logger.warning(f"Failed to reload YAML config: {e}")
        return kwargs

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
        sb = run_cfg.sandbox_config
        pc = run_cfg.proxy_config

        # ── 3. Resolve repo ──
        use_preexisting_repo = bool(sandbox_overrides.get("use_preexisting_repo", False))
        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "testbed") or "testbed")
        preexisting_repo_reset = bool(sandbox_overrides.get("preexisting_repo_reset", False))

        if not use_preexisting_repo and not repo_path:
            if sb.docker_image.startswith("sweb.eval."):
                use_preexisting_repo = True
            else:
                repo_path = "/workspace/repo"

        if use_preexisting_repo:
            logger.info(
                "Using preexisting repo in container: "
                f"repo_name={preexisting_repo_name}, reset={preexisting_repo_reset}"
            )

        logger.info(f"Starting SWE Agent Loop for problem: {problem_statement[:100]}...")

        try:
            if sb.max_parallel_tasks_per_worker > 0:
                logger.info(
                    f"Waiting for SWE-agent run slot (max_parallel_tasks_per_worker={sb.max_parallel_tasks_per_worker})"
                )
                run_slot = await self._acquire_run_slot(
                    sb.max_parallel_tasks_per_worker,
                    sb.output_dir,
                )
                logger.info(
                    "Acquired SWE-agent run slot "
                    f"(slot={run_slot[1]}, max_parallel_tasks_per_worker={sb.max_parallel_tasks_per_worker})"
                )

            model_proxy = ModelProxy(port=pc.port)

            # ── 4. Start ModelProxy ──
            await model_proxy.start_server(max_retries=pc.max_port_retries)
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
                )
            )

            # ── 6. Interaction loop ──
            # Use problem_instance_id (or a fresh UUID) as sticky session key
            # so that all turns route to the same vLLM replica for prefix
            # KV-cache reuse and implicit load-balancing via the server manager.
            session_id = problem_instance_id if problem_instance_id else str(uuid.uuid4())
            (
                patch,
                num_turns,
                turn_records,
            ) = await self._interaction_loop(
                agent_task=agent_task,
                sampling_params=sampling_params,
                max_model_calls_per_instance=sb.max_model_calls_per_instance,
                request_timeout=pc.timeout,
                model_proxy=model_proxy,
                session_id=session_id,
            )

            # ── 7. Drain agent task ──
            if not agent_task.done():
                patch = await self._drain_agent_task(agent_task, num_turns >= sb.max_model_calls_per_instance)

            # ── 8. Reconstruct trajectory ──
            total_elapsed = time.time() - run_start_time
            logger.info(
                f"SWE Agent Loop completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s"
            )

            aligned_trajectory = await TrajectoryReconstructor(self._render_chat_ids).reconstruct(turn_records)
            if aligned_trajectory.ok:
                effective_num_turns = aligned_trajectory.num_turns
            else:
                effective_num_turns = 0
                failure = aligned_trajectory.failure
                logger.warning(
                    "Alignment failed for SWE-Agent episode "
                    f"(turn={failure.turn_index if failure else 'n/a'}, "
                    f"reason={failure.reason if failure else 'unknown'}, "
                    f"details={failure.details if failure and failure.details else 'n/a'})"
                )

            # ── 9. Build output ──
            return self._build_output(
                aligned_trajectory=aligned_trajectory,
                num_turns=effective_num_turns,
                actual_num_turns=num_turns,
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
        max_model_calls_per_instance: int,
        request_timeout: float,
        model_proxy: ModelProxy,
        session_id: Optional[str] = None,
    ) -> tuple[
        Optional[str],  # patch
        int,  # num_turns
        list[TurnRecord],  # turn_records
    ]:
        """Run the main turn-by-turn interaction with SWE-Agent via ModelProxy.

        Args:
            session_id: Sticky session identifier.  When provided, every
                ``server_manager.generate()`` call within this episode uses
                the *same* ``request_id`` so that
                :class:`AsyncLLMServerManager` routes all turns to the same
                vLLM replica, maximising prefix KV-cache reuse.
        """
        turn_records: list[TurnRecord] = []
        num_turns = 0
        patch: Optional[str] = None

        # Use a fixed session_id for sticky routing & prefix cache reuse.
        if session_id is None:
            session_id = str(uuid.uuid4())

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

            # Build prompt
            prompt_ids = await self._render_chat_ids(messages, add_generation_prompt=True)

            # Early-stop if prompt leaves insufficient generation room.
            # Check both max_model_len and the vLLM budget (prompt_length + response_length).
            rollout_cfg = getattr(self.config, "actor_rollout_ref", self.config).rollout
            max_model_len = int(getattr(rollout_cfg, "max_model_len", 0) or 0)
            cfg_prompt_len = int(getattr(rollout_cfg, "prompt_length", 0) or 0)
            cfg_response_len = int(
                getattr(rollout_cfg, "response_length", 0) or getattr(self.data_config, "max_response_length", 4096)
            )
            vllm_budget = cfg_prompt_len + cfg_response_len if cfg_prompt_len else max_model_len
            effective_limit = (
                min(max_model_len, vllm_budget) if max_model_len and vllm_budget else (max_model_len or vllm_budget)
            )
            min_gen_tokens = 256
            remaining = max((effective_limit - len(prompt_ids)) if effective_limit else cfg_response_len, 0)
            if effective_limit and remaining < min_gen_tokens:
                logger.warning(
                    f"Turn {num_turns + 1}: remaining budget {remaining} < {min_gen_tokens} "
                    f"(prompt_len={len(prompt_ids)}, effective_limit={effective_limit}, "
                    f"max_model_len={max_model_len}, vllm_budget={vllm_budget}), "
                    f"stopping interaction (reward will be 0)"
                )
                break

            # Cap generation to the actual remaining budget.
            gen_max_tokens = min(cfg_response_len, remaining)

            # Generate — use fixed session_id for sticky routing & prefix cache
            output = await self.server_manager.generate(
                request_id=session_id,
                prompt_ids=prompt_ids,
                sampling_params={
                    **sampling_params,
                    "logprobs": sampling_params.get("logprobs", True),
                    "max_tokens": gen_max_tokens,
                },
            )

            response_ids = list(output.token_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_logprobs = list(output.log_probs) if output.log_probs is not None else [0.0] * len(response_ids)

            turn_records.append(
                TurnRecord(
                    turn_index=num_turns + 1,
                    request_id=model_request.request_id,
                    messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                    prompt_ids=list(prompt_ids),
                    response_ids=response_ids,
                    response_text=response_text,
                    response_logprobs=response_logprobs,
                    finish_reason="stop",
                    model=model_request.model,
                    temperature=model_request.temperature,
                    max_tokens=model_request.max_tokens,
                    extra_params=model_request.extra_params,
                )
            )

            num_turns += 1

            # Send response back
            await model_proxy.send_response(response_text, request=model_request)

            logger.info(f"Turn {num_turns}: {len(response_ids)} model tokens")

            if num_turns >= max_model_calls_per_instance:
                logger.warning(f"Max model calls reached ({num_turns}/{max_model_calls_per_instance})")
                break

        return patch, num_turns, turn_records

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
    ) -> Optional[str]:
        """Generate config, run SWE-Agent subprocess, cleanup."""
        instance_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        instance_output_dir = os.path.join(cfg.sandbox_config.output_dir, instance_id)
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
        )

        try:
            patch = await execute_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                instance_id=instance_id,
                output_dir=instance_output_dir,
                repo_path=repo_path,
                exec_dir=exec_dir,
                swe_agent_timeout=cfg.sandbox_config.swe_agent_timeout,
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
    ) -> str:
        """Build and write SWE-Agent CLI YAML, return file path."""
        rollout_cfg = getattr(self.config, "actor_rollout_ref", self.config).rollout
        max_input_tokens = int(getattr(rollout_cfg, "max_model_len", 0) or 0)

        if use_preexisting_repo:
            yaml_repo_path, yaml_repo_type = preexisting_repo_name, "preexisting"
        else:
            yaml_repo_path, yaml_repo_type = repo_path, "local"

        yaml_str = build_sweagent_yaml(
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
        f.write(yaml_str)
        f.close()
        return f.name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _drain_agent_task(agent_task: asyncio.Task, max_model_calls_reached: bool) -> Optional[str]:
        """Wait for / cancel the SWE-Agent background task.

        When max_model_calls is reached, wait briefly for the subprocess to
        finish naturally (e.g. complete a pending submit) before cancelling.
        """
        if max_model_calls_reached:
            # Give the agent a grace period to finish (submit + patch write)
            try:
                return await asyncio.wait_for(agent_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Cancelling SWE-Agent task due to max_model_calls_per_instance limit")
                agent_task.cancel()
                try:
                    return await asyncio.wait_for(agent_task, timeout=15.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    return None
        else:
            try:
                return await asyncio.wait_for(agent_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for agent task completion")
                return None

    async def _render_chat_ids(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        """Render chat messages into token ids using the exact runtime template path."""
        if add_generation_prompt:
            return await self.apply_chat_template(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            **getattr(self, "apply_chat_template_kwargs", {}),
        )

    def _build_output(
        self,
        *,
        aligned_trajectory: AlignedTrajectory,
        num_turns: int,
        actual_num_turns: int,
        patch: Optional[str],
        problem_statement: str,
        repo_path: str,
    ) -> AgentLoopOutput:
        """Assemble the final ``AgentLoopOutput``."""
        rollout_cfg = getattr(self.config, "actor_rollout_ref", self.config).rollout
        max_response_length = rollout_cfg.response_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        final_prompt_ids = (
            aligned_trajectory.initial_prompt_ids if aligned_trajectory.initial_prompt_ids else [pad_token_id]
        )

        if aligned_trajectory.response_ids:
            final_response_ids = aligned_trajectory.response_ids[:max_response_length]
            final_response_mask = aligned_trajectory.response_mask[:max_response_length]
        else:
            final_response_ids = [pad_token_id]
            final_response_mask = [1]

        # Guarantee at least one mask=1 token (required by rollout correction)
        if not any(m == 1 for m in final_response_mask):
            final_response_ids.append(pad_token_id)
            final_response_mask.append(1)

        if aligned_trajectory.response_logprobs:
            final_response_logprobs = aligned_trajectory.response_logprobs[: len(final_response_ids)]
        else:
            final_response_logprobs = [0.0] * len(final_response_ids)
        if len(final_response_logprobs) < len(final_response_ids):
            final_response_logprobs.extend([0.0] * (len(final_response_ids) - len(final_response_logprobs)))

        failure = aligned_trajectory.failure

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
                "num_turns": num_turns,
                "actual_num_turns": actual_num_turns,
                "problem_statement": problem_statement,
                "repo_path": repo_path,
                "alignment_failed": not aligned_trajectory.ok,
                "alignment_failure_reason": failure.reason if failure is not None else "",
                "alignment_failure_turn": failure.turn_index if failure is not None else -1,
                "alignment_failure_details": failure.details if failure is not None else "",
            },
        )
