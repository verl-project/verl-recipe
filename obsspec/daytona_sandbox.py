from __future__ import annotations

import asyncio
import atexit
import io
import logging
import math
import posixpath
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import tomllib
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources, SessionExecuteRequest
from daytona.common.errors import (
    DaytonaFileNotFoundError,
    DaytonaNotFoundError,
    DaytonaProcessExecutionTimeoutError,
    DaytonaTimeoutError,
)

logger = logging.getLogger(__name__)

_ENVIRONMENT_PATH = PurePosixPath("environment")
_SHELL_STATE_FILE = "/tmp/.persistent_shell_state"
_REWARD_PATH = "/logs/verifier/reward.txt"
_VERIFIER_COMMAND = "mkdir -p /logs/verifier && bash /tests/test.sh"
_CLIENT: AsyncDaytona | None = None
_CLIENT_LOCK = asyncio.Lock()


@dataclass
class ExecResult:
    return_code: int
    stdout: str
    stderr: str


@dataclass
class _Task:
    cpus: float | None
    memory_mb: int | None
    verifier_files: dict[str, bytes]


def _wrap_persistent(command: str) -> str:
    state = _SHELL_STATE_FILE
    return (
        f"[ -f {state} ] && source {state} 2>/dev/null\n"
        f"{command}\n"
        "__command_rc=$?\n"
        f'{{ declare -p; declare -f; printf "cd %q\\n" "$PWD"; }} > {state} 2>/dev/null\n'
        "exit $__command_rc"
    )


def _extract_task(task_binary: bytes, root: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(bytes(task_binary)), mode="r:gz") as archive:
        members = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                continue
            if not (member.isfile() or member.isdir()):
                continue
            if path == PurePosixPath("task.toml") or (path.parts and path.parts[0] in {"environment", "tests"}):
                members.append(member)
        archive.extractall(root, members=members)


def _load_task(root: Path) -> _Task:
    task_toml_path = root / "task.toml"
    dockerfile_path = root / _ENVIRONMENT_PATH / "Dockerfile"
    tests_path = root / "tests"
    if not task_toml_path.is_file():
        raise ValueError("Task archive does not contain task.toml")
    if not dockerfile_path.is_file():
        raise ValueError("Task archive does not contain environment/Dockerfile")
    if not (tests_path / "test.sh").is_file():
        raise ValueError("Task archive does not contain tests/test.sh")

    environment = tomllib.loads(task_toml_path.read_text(encoding="utf-8")).get("environment", {})
    cpus = environment.get("cpus")
    memory_mb = environment.get("memory_mb")
    cpus = float(cpus) if cpus is not None else None
    memory_mb = int(memory_mb) if memory_mb is not None else None
    if cpus is not None and cpus <= 0:
        raise ValueError("environment.cpus must be positive")
    if memory_mb is not None and memory_mb <= 0:
        raise ValueError("environment.memory_mb must be positive")

    verifier_files = {
        f"/{path.relative_to(root).as_posix()}": path.read_bytes() for path in tests_path.rglob("*") if path.is_file()
    }
    return _Task(cpus=cpus, memory_mb=memory_mb, verifier_files=verifier_files)


async def _get_client() -> AsyncDaytona:
    global _CLIENT
    if _CLIENT is None:
        async with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = AsyncDaytona()
    return _CLIENT


def _close_client() -> None:
    global _CLIENT
    client = _CLIENT
    _CLIENT = None
    if client is not None:
        asyncio.run(client.close())


atexit.register(_close_client)


class DaytonaSandboxEnvironment:
    def __init__(self, task_binary: bytes, sandbox_timeout: float) -> None:
        if sandbox_timeout <= 0:
            raise ValueError("sandbox_timeout must be positive")
        self._task_binary = bytes(task_binary)
        self._sandbox_timeout = math.ceil(sandbox_timeout)
        self._verifier_files: dict[str, bytes] = {}
        self._sandbox = None

    async def setup(self) -> None:
        if self._sandbox is not None:
            return
        client = await _get_client()
        with tempfile.TemporaryDirectory(prefix="verl-daytona-") as temp_dir:
            root = Path(temp_dir)
            _extract_task(self._task_binary, root)
            task = _load_task(root)
            self._verifier_files = task.verifier_files
            resource_kwargs = {}
            if task.cpus is not None:
                resource_kwargs["cpu"] = max(1, math.ceil(task.cpus))
            if task.memory_mb is not None:
                resource_kwargs["memory"] = max(1, math.ceil(task.memory_mb / 1024))
            resources = Resources(**resource_kwargs) if resource_kwargs else None
            params = CreateSandboxFromImageParams(
                image=Image.from_dockerfile(root / _ENVIRONMENT_PATH / "Dockerfile"),
                resources=resources,
                auto_delete_interval=0,
            )
            create_task = asyncio.create_task(client.create(params=params, timeout=self._sandbox_timeout))
            try:
                self._sandbox = await asyncio.shield(create_task)
            except asyncio.CancelledError:
                try:
                    self._sandbox = await asyncio.wait_for(create_task, timeout=30)
                except (asyncio.CancelledError, TimeoutError):
                    create_task.cancel()
                if self._sandbox is not None:
                    await self._sandbox.delete()
                    self._sandbox = None
                raise

    async def _raw_exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("Environment is not set up")
        process_timeout = max(1, math.ceil(timeout)) if timeout is not None else None
        wrapped = f"bash -c {shlex.quote(command)}"
        if process_timeout is not None:
            wrapped = f"timeout {process_timeout} {wrapped}"
        session_id = str(uuid4())
        try:
            await self._sandbox.process.create_session(session_id)
            response = await self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=wrapped, run_async=True),
                timeout=process_timeout,
            )
            if response.cmd_id is None:
                raise RuntimeError("Daytona returned no command ID")
            command_id = response.cmd_id
            deadline = asyncio.get_running_loop().time() + process_timeout + 5 if process_timeout is not None else None
            command_status = await self._sandbox.process.get_session_command(session_id, command_id)
            while command_status.exit_code is None:
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"Command timed out after {timeout} seconds")
                await asyncio.sleep(0.1)
                command_status = await self._sandbox.process.get_session_command(session_id, command_id)
            logs = await self._sandbox.process.get_session_command_logs(session_id, command_id)
        except (DaytonaProcessExecutionTimeoutError, DaytonaTimeoutError) as exc:
            raise TimeoutError(str(exc)) from exc
        return_code = int(command_status.exit_code)
        if return_code == 124 and timeout is not None:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        return ExecResult(
            return_code=return_code,
            stdout=logs.stdout or "",
            stderr=logs.stderr or "",
        )

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        return await self._raw_exec(_wrap_persistent(command), timeout=timeout)

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        if self._sandbox is None:
            raise RuntimeError("Environment is not set up")
        directories = sorted({posixpath.dirname(path) for path in self._verifier_files})
        await self._raw_exec("mkdir -p " + " ".join(shlex.quote(path) for path in directories), timeout=timeout)
        for path, content in self._verifier_files.items():
            await self._sandbox.fs.upload_file(content, path)
        try:
            result = await self._raw_exec(_VERIFIER_COMMAND, timeout=timeout)
        except TimeoutError:
            logger.warning("Verifier timed out after %s seconds", timeout)
            return 0.0, "verifier_timeout"
        raw = await self._download_text(_REWARD_PATH)
        if raw is None:
            logger.warning("Verifier produced no reward file (exit=%s)", result.return_code)
            return 0.0, "verifier_error"
        try:
            return float(raw), None
        except ValueError:
            return 0.0, "verifier_error"

    async def _download_text(self, path: str) -> str | None:
        if self._sandbox is None:
            raise RuntimeError("Environment is not set up")
        try:
            content = await self._sandbox.fs.download_file(path)
        except (DaytonaFileNotFoundError, DaytonaNotFoundError):
            return None
        if content is None:
            return None
        return bytes(content).decode("utf-8", errors="replace").strip()

    async def cleanup(self) -> None:
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        self._sandbox = None
        try:
            await sandbox.delete()
        except DaytonaNotFoundError:
            return
