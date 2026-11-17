# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""E2B-backed ReTool code interpreter.

verl creates and releases a tool instance for every tool call, and E2B limits how fast
sandboxes can be created. So this tool keeps a small pool of sandboxes per agent loop
worker and runs each call in fresh Linux namespaces inside one of them: a new PID, mount,
network and IPC namespace, a read-only root, an empty /tmp, and an unprivileged user.
Nothing a call writes or starts survives into the next call.
"""

import asyncio
import logging
import os
import time
from typing import Any
from uuid import uuid4

from recipe.retool.retool import normalize_code

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TIMEOUT_TEXT = "TimeoutError: Code execution timed out."
UNAVAILABLE_TEXT = "Code execution is temporarily unavailable; the code was not run."

_REMOTE = "/opt/verl-retool"
# Runs as PID 1 of the new namespaces, so every process the code starts dies with it.
_RUNNER = """#!/bin/bash
set -eu
mount --make-rprivate /
mount --bind / /
mount -o remount,bind,ro /
mount -t tmpfs -o size=16m,mode=1777 tmpfs /tmp
mount -t tmpfs -o size=16m,mode=1777 tmpfs /dev/shm
cd /tmp
ulimit -u 64
ulimit -f 1024
exec setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs \\
  python3 -I -B -c 'import os, runpy, sys; os.write(3, b"ready"); os.close(3); \
runpy.run_path(sys.argv[1], run_name="__main__")' "$1"
"""
_UNSHARE = "unshare --mount --pid --fork --kill-child=KILL --net --ipc --mount-proc"


class _Unusable(Exception):
    """The sandbox could not run the code; unrelated to the code itself."""


def _load_e2b():
    """Import the E2B SDK lazily so the backend stays optional."""
    try:
        import e2b
        import httpx
    except ImportError as exc:
        raise ImportError(
            "CustomE2BSandboxTool requires the optional 'e2b' dependency. Install it with `pip install e2b`."
        ) from exc
    retryable = (
        _Unusable,
        e2b.TimeoutException,
        e2b.SandboxNotFoundException,
        e2b.RateLimitException,
        e2b.exceptions.GuestUnresponsiveException,
        e2b.exceptions.ServiceBusyException,
        httpx.TransportError,
    )
    return e2b.AsyncSandbox, retryable


class _Lease:
    """One tool call's hold on a pool slot."""

    def __init__(self):
        self.sandbox = None
        self.holds_slot = False
        self.reusable = False


class CustomE2BSandboxTool(BaseTool):
    """Execute Python code in a pooled E2B sandbox, isolated per call."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("CustomE2BSandboxTool requires the E2B_API_KEY environment variable")
        self._sandbox_cls, self._retryable = _load_e2b()

        self.pool_size = int(config.get("pool_size", 8))
        self.create_interval = float(config.get("create_interval", 2.0))
        self.default_timeout = int(config.get("default_timeout", 30))
        self.sandbox_timeout = int(config.get("sandbox_timeout", 600))
        self.max_output_bytes = int(config.get("max_output_bytes", 65536))
        self.template = config.get("template", "code-interpreter-v1")
        if min(self.pool_size, self.default_timeout, self.sandbox_timeout, self.max_output_bytes) < 1:
            raise ValueError("pool_size, default_timeout, sandbox_timeout and max_output_bytes must be positive")
        self._metadata = {
            **{str(k): str(v) for k, v in (config.get("metadata") or {}).items()},
            "framework": "verl",
            "tool": self.name,
            "pool": uuid4().hex[:12],
        }

        t, n = self.default_timeout, self.max_output_bytes
        self._command = (
            f"cd {_REMOTE} && rm -f status; "
            f"timeout --signal=KILL {t}s {_UNSHARE} bash {_REMOTE}/run.sh {_REMOTE}/solution.py "
            f"> out 2> err 3> status; rc=$?; "
            f'echo "$rc $(cat status 2>/dev/null)"; head -c {n} out; head -c {n} err'
        )

        self._leases: dict[str, _Lease] = {}
        self._idle: list = []
        self._slots = asyncio.Semaphore(self.pool_size)
        self._create_lock = asyncio.Lock()
        self._last_create = 0.0
        self._renewed: dict[str, float] = {}

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Register one tool call. The sandbox is taken in execute(), where failures can be retried."""
        instance_id = instance_id or str(uuid4())
        self._leases[instance_id] = _Lease()
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Run the code and return stdout followed by stderr."""
        code = normalize_code(parameters.get("code", ""))
        lease = self._leases[instance_id]
        await self._slots.acquire()
        lease.holds_slot = True

        for _ in range(2):
            try:
                lease.sandbox = await self._take_sandbox()
                text = await self._run(lease.sandbox, code)
                lease.reusable = True
                return ToolResponse(text=text), 0.0, {}
            except Exception as exc:
                if lease.sandbox is None:
                    logger.warning("E2B sandbox could not be created: %s: %s", type(exc).__name__, exc)
                else:
                    # The exception text may quote the code, so only its type is logged.
                    logger.warning("E2B sandbox %s failed with %s", lease.sandbox.sandbox_id, type(exc).__name__)
                await self._discard(lease.sandbox)
                lease.sandbox = None
                if not isinstance(exc, self._retryable):
                    break
        return ToolResponse(text=UNAVAILABLE_TEXT), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Return the sandbox to the pool, or kill it if the call did not finish cleanly."""
        lease = self._leases.pop(instance_id, None)
        if lease is None:
            return
        try:
            if lease.reusable:
                self._idle.append(lease.sandbox)
            else:
                # Cancelled or failed mid-call: remote code may still be running.
                await self._discard(lease.sandbox)
        finally:
            if lease.holds_slot:
                self._slots.release()

    async def close(self) -> None:
        """Kill the idle sandboxes. verl does not call this; idle sandboxes otherwise expire on their own."""
        idle, self._idle = self._idle, []
        await asyncio.gather(*(self._discard(sandbox) for sandbox in idle))

    async def _take_sandbox(self):
        while self._idle:
            sandbox = self._idle.pop()
            # The only keep-alive: a pool that gets no calls expires after sandbox_timeout.
            if time.monotonic() - self._renewed.get(sandbox.sandbox_id, 0.0) < self.sandbox_timeout / 3:
                return sandbox
            try:
                await sandbox.set_timeout(self.sandbox_timeout)
            except BaseException as exc:
                # Usually expired while the pool sat idle, e.g. during a long model update.
                logger.info("E2B sandbox %s is gone (%s), replacing it", sandbox.sandbox_id, type(exc).__name__)
                await self._discard(sandbox)
                if not isinstance(exc, Exception):
                    raise
                continue
            self._renewed[sandbox.sandbox_id] = time.monotonic()
            return sandbox
        return await self._new_sandbox()

    async def _new_sandbox(self):
        async with self._create_lock:
            delay = self.create_interval - (time.monotonic() - self._last_create)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_create = time.monotonic()
            sandbox = await self._sandbox_cls.create(
                template=self.template,
                timeout=self.sandbox_timeout,
                metadata=self._metadata,
                secure=True,
                allow_internet_access=False,
                lifecycle={"on_timeout": "kill"},
            )
        try:
            await sandbox.commands.run(f"install -d -m 755 {_REMOTE}", user="root", timeout=15)
            await sandbox.files.write(f"{_REMOTE}/run.sh", _RUNNER, user="root")
        except BaseException:
            await self._discard(sandbox)
            raise
        self._renewed[sandbox.sandbox_id] = time.monotonic()
        return sandbox

    async def _run(self, sandbox, code: str) -> str:
        await sandbox.files.write(f"{_REMOTE}/solution.py", code, user="root")
        result = await sandbox.commands.run(self._command, user="root", timeout=self.default_timeout + 15)
        header, _, output = result.stdout.partition("\n")
        exit_code, _, status = header.partition(" ")
        if status != "ready":
            raise _Unusable("runner did not start")
        if exit_code == "137":
            output = f"{output.rstrip()}\n{TIMEOUT_TEXT}".lstrip()
        return output

    async def _discard(self, sandbox) -> None:
        if sandbox is None:
            return
        self._renewed.pop(sandbox.sandbox_id, None)
        try:
            await sandbox.kill()
        except Exception as exc:
            # Never raise from cleanup; the sandbox timeout reclaims what a failed kill leaves behind.
            logger.warning("E2B sandbox %s kill failed with %s", sandbox.sandbox_id, type(exc).__name__)
