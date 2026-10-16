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
"""ThunderAgent extension for the PR #110 Dynamo rollout backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

import ray
import yaml

from .dynamo_async_server import DynamoHttpServer, DynamoReplica

logger = logging.getLogger(__file__)

# (program_id, parent_program_id, is_final) of the request being dispatched.
_REQUEST_PROGRAM: ContextVar[tuple[str, Optional[str], bool] | None] = ContextVar(
    "dynamo_thunderagent_request_program",
    default=None,
)
_THUNDERAGENT_BACKEND_MODEL_SUFFIX = "--verl-thunderagent-backend"


class FinalRequestError(RuntimeError):
    """The frontend answered a ThunderAgent session-final request with a non-2xx status."""

    def __init__(self, status: int, body: str):
        super().__init__(f"ThunderAgent final request failed status={status} body={body[:2000]!r}")
        self.status = status


class DynamoThunderAgentHttpServer(DynamoHttpServer):
    """Add program-aware routing while preserving PR #110's server stack.

    The frontend selects the ThunderAgent endpoint. ThunderAgent then selects
    a complete engine (worker, DP rank) pair; never pre-pin a bare engine rank
    in the frontend request, where session affinity requires an explicit worker.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thunderagent_process: Optional[subprocess.Popen] = None
        self._thunderagent_log_fp = None
        if self._thunderagent_enabled() and not self._thunderagent_native():
            self._router_mode = "round-robin"
        self._validate_thunderagent_router_mode()

    def _thunderagent_config(self) -> dict[str, Any]:
        config = self._dynamo_cfg().get("thunderagent", {}) or {}
        if not isinstance(config, dict):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent must be a mapping")
        return config

    def _thunderagent_enabled(self) -> bool:
        value = self._thunderagent_config().get("enabled", False)
        if not isinstance(value, bool):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.enabled must be a boolean")
        return value

    def _thunderagent_native(self) -> bool:
        """Schedule programs with Dynamo's builtin ThunderAgent plugin (inside the KV router) instead of
        launching the Python prototype as a separate router process."""
        if not self._thunderagent_enabled():
            return False
        value = self._thunderagent_config().get("native", False)
        if not isinstance(value, bool):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.native must be a boolean")
        return value

    def _thunderagent_native_parameters(self) -> dict[str, Any]:
        value = self._thunderagent_config().get("native_parameters", {}) or {}
        if not isinstance(value, dict):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.native_parameters must be a mapping")
        return dict(value)

    def _validate_thunderagent_router_mode(self) -> None:
        if self._thunderagent_native() and self._router_mode != "kv":
            raise ValueError(
                "thunderagent.native=true runs inside the KV router; set rollout.engine_kwargs.dynamo.router_mode=kv "
                f"(got {self._router_mode!r})"
            )

    def _thunderagent_policy(self) -> dict[str, Any]:
        """The frontend policy file selecting the plugin's classifier and worker selector."""
        classifier: dict[str, Any] = {"type": "thunderagent"}
        parameters = self._thunderagent_native_parameters()
        if parameters:
            classifier["parameters"] = parameters
        return {
            "request_classifier": classifier,
            "worker_selection": {
                "aggregated": "thunderagent",
                "instances": [{"name": "thunderagent", "type": "thunderagent"}],
            },
        }

    def _write_thunderagent_policy(self) -> str:
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        os.makedirs(log_root, exist_ok=True)
        path = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_thunderagent_policy.yaml")
        with open(path, "w") as fp:
            yaml.safe_dump(self._thunderagent_policy(), fp, sort_keys=False)
        return path

    def _frontend_router_args(self) -> list[str]:
        args = super()._frontend_router_args()
        if not self._thunderagent_native():
            return args
        # soft affinity lets the plugin repack a session between turns
        policy = self._write_thunderagent_policy()
        return [*args, "--router-policy-config", policy, "--router-session-affinity-mode", "soft"]

    def _thunderagent_router_block_size(self) -> int:
        value = int(self._thunderagent_config().get("router_block_size", 16))
        if value <= 0:
            raise ValueError("rollout.engine_kwargs.dynamo.thunderagent.router_block_size must be positive")
        return value

    def _thunderagent_extra_args(self) -> list[str]:
        value = self._thunderagent_config().get("extra_args", []) or []
        if not isinstance(value, list):
            raise TypeError("rollout.engine_kwargs.dynamo.thunderagent.extra_args must be a list")
        return [str(arg) for arg in value]

    def _build_thunderagent_cmd(self) -> list[str]:
        model_name = self._served_model_name or self.model_config.local_path
        return [
            sys.executable,
            "-m",
            "dynamo.thunderagent_router",
            "--endpoint",
            f"{self._namespace}.backend.generate",
            "--model-name",
            str(model_name),
            "--model-path",
            str(self.model_config.local_path),
            "--router-block-size",
            str(self._thunderagent_router_block_size()),
            *self._thunderagent_extra_args(),
        ]

    def _build_vllm_cmd(
        self,
        served_model_name: str,
        tp: int,
        kv_events_config_json: Optional[str] = None,
    ) -> list[str]:
        thunderagent_enabled = self._thunderagent_enabled()
        worker_model_name = (
            f"{served_model_name}{_THUNDERAGENT_BACKEND_MODEL_SUFFIX}"
            if thunderagent_enabled and not self._thunderagent_native()
            else served_model_name
        )
        command = super()._build_vllm_cmd(worker_model_name, tp, kv_events_config_json=kv_events_config_json)
        if not thunderagent_enabled:
            return command

        block_size = str(self._thunderagent_router_block_size())
        configured_block_size = None
        for index, arg in enumerate(command):
            if arg == "--block-size" and index + 1 < len(command):
                configured_block_size = command[index + 1]
            elif arg.startswith("--block-size="):
                configured_block_size = arg.partition("=")[2]
        if configured_block_size is not None and configured_block_size != block_size:
            raise ValueError(
                f"Dynamo vLLM and ThunderAgent router block sizes must match: {configured_block_size} != {block_size}"
            )
        if configured_block_size is None:
            command.extend(["--block-size", block_size])
        return command

    def _build_sglang_cmd(self, served_model_name: str, tp: int, nccl_port=None, **kwargs) -> list[str]:
        """TA worker-side glue for the sglang engine, mirroring _build_vllm_cmd.

        Extra keyword arguments (kv_events_config_json, ...) belong to the base builder and
        pass through untouched.

        Rename the worker's served model so the frontend cannot route requests
        (or session-final probes) directly to it — with ThunderAgent on, the
        ONLY handler advertising the public model name must be the TA router,
        otherwise generation silently bypasses program affinity. Page-size
        alignment is handled by the base builder (it falls back to
        thunderagent.router_block_size); an explicit sglang.page_size that
        contradicts the router block size fails fast here, same as the vLLM
        --block-size check.
        """
        if not self._thunderagent_enabled() or self._thunderagent_native():
            return super()._build_sglang_cmd(served_model_name, tp, nccl_port=nccl_port, **kwargs)
        explicit_page_size = self._sglang_cfg().get("page_size")
        router_block_size = self._thunderagent_router_block_size()
        if explicit_page_size is not None and int(explicit_page_size) != int(router_block_size):
            raise ValueError(
                "Dynamo sglang and ThunderAgent router block sizes must match: "
                f"sglang.page_size={explicit_page_size} != thunderagent.router_block_size={router_block_size}"
            )
        worker_model_name = f"{served_model_name}{_THUNDERAGENT_BACKEND_MODEL_SUFFIX}"
        return super()._build_sglang_cmd(worker_model_name, tp, nccl_port=nccl_port, **kwargs)

    def _start_thunderagent(self) -> None:
        if not self._thunderagent_enabled() or self._thunderagent_native() or self._thunderagent_process is not None:
            return
        env = os.environ.copy()
        env.update(self._dynamo_env_vars())
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_thunderagent.log")
        self._thunderagent_log_fp = open(log_path, "w")
        command = self._build_thunderagent_cmd()
        logger.info("[DynamoHttpServer] starting ThunderAgent (log=%s): %s", log_path, " ".join(command))
        self._thunderagent_process = subprocess.Popen(
            command,
            env=env,
            stdout=self._thunderagent_log_fp,
            stderr=subprocess.STDOUT,
        )

    def _start_frontend(self) -> None:
        if self._thunderagent_enabled():
            self._start_thunderagent()
        super()._start_frontend()

    def _raise_if_subprocess_died(self) -> None:
        super()._raise_if_subprocess_died()
        process = self._thunderagent_process
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"dynamo ThunderAgent exited rc={process.returncode}")

    def _frontend_headers(self, request_id: str) -> dict[str, str]:
        # Build on the base headers: register.py always serves through this class, so
        # the base router_session_affinity_ttl_secs header (x-dynamo-session-id keyed on
        # the trajectory request id) would otherwise never reach the wire. An active
        # ThunderAgent program overrides it -- same header, which the frontend matches
        # case-insensitively, so drop the base spelling before adding TA's.
        headers = super()._frontend_headers(request_id)
        request_program = _REQUEST_PROGRAM.get()
        if request_program is not None:
            session_id, parent_session_id, is_final = request_program
            headers.pop("x-dynamo-session-id", None)
            headers["X-Dynamo-Session-ID"] = session_id
            if parent_session_id:
                # The frontend normalizes this into agent_context.parent_session_id,
                # which lets a subagent-aware scheduler rebuild the program tree.
                headers["X-Dynamo-Parent-Session-ID"] = parent_session_id
            if is_final:
                headers["X-Dynamo-Session-Final"] = "true"
        return headers

    async def generate(
        self,
        *args,
        thunderagent_session_id: Optional[str] = None,
        thunderagent_parent_session_id: Optional[str] = None,
        **kwargs,
    ):
        if not self._thunderagent_enabled():
            return await super().generate(*args, **kwargs)
        if not thunderagent_session_id:
            raise RuntimeError("ThunderAgent generation requires a non-empty session ID")
        if thunderagent_parent_session_id == thunderagent_session_id:
            raise ValueError("ThunderAgent parent session ID must differ from the session ID")
        if self._use_direct_generate():
            raise RuntimeError("direct_generate bypasses ThunderAgent and cannot be enabled")

        token = _REQUEST_PROGRAM.set((thunderagent_session_id, thunderagent_parent_session_id, False))
        try:
            return await super().generate(*args, **kwargs)
        finally:
            _REQUEST_PROGRAM.reset(token)

    def _finalize_max_attempts(self) -> int:
        attempts = int(self._thunderagent_config().get("finalize_max_attempts", 3))
        if attempts <= 0:
            raise ValueError("thunderagent.finalize_max_attempts must be positive")
        return attempts

    def _finalize_retry_delay(self) -> float:
        delay = float(self._thunderagent_config().get("finalize_retry_delay_s", 0.1))
        if delay < 0:
            raise ValueError("thunderagent.finalize_retry_delay_s must be non-negative")
        return delay

    def _validate_final_response(self, status: int, body: str) -> None:
        if not 200 <= status < 300:
            raise FinalRequestError(status, body)
        if not body.strip() or self._thunderagent_native():
            # the native plugin lets the one-token final request complete on a worker
            return
        data = json.loads(body)
        if data.get("choices"):
            raise RuntimeError("ThunderAgent final request was bypassed and reached a model worker")

    async def finalize_program(self, session_id: str) -> bool:
        """Release a ThunderAgent program with the session-final request.

        Returns False without posting while the engines are paused in native mode: the request
        would park on a worker until resume_generation (the Python prototype answers it itself).
        """
        if not self._thunderagent_enabled():
            return True
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if self._thunderagent_native() and not self._generation_resumed.is_set():
            return False

        request_id = f"thunderagent-final-{uuid4().hex}"
        payload = {
            "model": self._served_model_name or self.model_config.local_path,
            "prompt": [0],
            "max_tokens": 1,
            "stream": False,
        }
        token = _REQUEST_PROGRAM.set((session_id, None, True))
        try:
            last_error = None
            attempts = self._finalize_max_attempts()
            for attempt in range(1, attempts + 1):
                try:
                    status, body = await self._frontend_post(payload, request_id)
                except Exception as error:
                    last_error = error
                else:
                    if status not in {408, 429} and status < 500:
                        self._validate_final_response(status, body)
                        return True
                    last_error = FinalRequestError(status, body)
                if attempt < attempts:
                    await asyncio.sleep(self._finalize_retry_delay())
            raise RuntimeError(f"ThunderAgent final request failed after {attempts} attempts") from last_error
        finally:
            _REQUEST_PROGRAM.reset(token)

    def _health_probe_timeout(self) -> float:
        timeout = float(self._thunderagent_config().get("health_probe_timeout_s", 60.0))
        if timeout <= 0:
            raise ValueError("thunderagent.health_probe_timeout_s must be positive")
        return timeout

    async def _probe_final_request(self) -> None:
        """Release a throwaway program to prove the session-final path works; the completions
        route may answer 404 for a moment after /health passes, so a 404 is waited out."""
        deadline = time.monotonic() + self._health_probe_timeout()
        while True:
            try:
                await self.finalize_program(f"thunderagent-health-{time.time_ns()}")
                return
            except FinalRequestError as error:
                if error.status != 404 or time.monotonic() >= deadline:
                    raise
            await asyncio.sleep(1.0)

    async def _healthcheck_frontend(self, expected_workers: int) -> None:
        await super()._healthcheck_frontend(
            expected_workers + int(self._thunderagent_enabled() and not self._thunderagent_native())
        )
        if self._thunderagent_enabled():
            await self._probe_final_request()

    async def shutdown(self) -> None:
        frontend = self._frontend_process
        self._frontend_process = None
        if frontend is not None:
            self._stop_one(frontend, "frontend", 15)

        thunderagent = self._thunderagent_process
        self._thunderagent_process = None
        if thunderagent is not None:
            self._stop_one(thunderagent, "ThunderAgent", 15)

        try:
            await super().shutdown()
        finally:
            if self._thunderagent_log_fp is not None:
                self._thunderagent_log_fp.close()
                self._thunderagent_log_fp = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_thunderagent_process"] = None
        state["_thunderagent_log_fp"] = None
        return state


class DynamoThunderAgentReplica(DynamoReplica):
    """Use the ThunderAgent-aware HTTP server for the PR #110 replica."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(DynamoThunderAgentHttpServer)


__all__ = ["DynamoThunderAgentHttpServer", "DynamoThunderAgentReplica"]
