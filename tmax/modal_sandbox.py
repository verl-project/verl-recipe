from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import modal
import tomllib

logger = logging.getLogger(__name__)

_APP_NAME = "verl-sandbox"
_ENVIRONMENT_PATH = Path("environment")
_SHELL_STATE_FILE = "/tmp/.tmax_shell_state"
_REWARD_PATH = "/logs/verifier/reward.txt"
_VERIFIER_COMMAND = "mkdir -p /logs/verifier && bash /tests/test.sh"


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
        f"__tmax_rc=$?\n"
        f'{{ declare -p; declare -f; printf "cd %q\\n" "$PWD"; }} > {state} 2>/dev/null\n'
        f"exit $__tmax_rc"
    )


def _extract_task(task_binary: bytes, root: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(bytes(task_binary)), mode="r:gz") as archive:
        members = []
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                continue
            if not (member.isfile() or member.isdir()):
                continue
            if path == Path("task.toml") or (path.parts and path.parts[0] in {"environment", "tests"}):
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
    return _Task(
        cpus=cpus,
        memory_mb=memory_mb,
        verifier_files=verifier_files,
    )


class ModalSandboxEnvironment:
    def __init__(self, task_binary: bytes, sandbox_timeout: float) -> None:
        self._task_binary = bytes(task_binary)
        self._sandbox_timeout = int(sandbox_timeout)
        self._verifier_files: dict[str, bytes] = {}
        self._sandbox: modal.Sandbox | None = None

    async def setup(self) -> None:
        if self._sandbox is not None:
            return
        app = await modal.App.lookup.aio(_APP_NAME, create_if_missing=True)
        with tempfile.TemporaryDirectory(prefix="tmax-modal-") as temp_dir:
            root = Path(temp_dir)
            _extract_task(self._task_binary, root)
            task = _load_task(root)
            environment_dir = root / _ENVIRONMENT_PATH
            self._verifier_files = task.verifier_files
            image = modal.Image.from_dockerfile(
                environment_dir / "Dockerfile",
                context_dir=environment_dir,
            )
            self._sandbox = await modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                app=app,
                image=image,
                timeout=self._sandbox_timeout,
                cpu=(task.cpus, task.cpus) if task.cpus is not None else None,
                memory=(task.memory_mb, task.memory_mb) if task.memory_mb is not None else None,
            )

    async def _raw_exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("Environment is not set up")
        process_timeout = max(1, int(timeout)) if timeout is not None else None
        try:
            process = await self._sandbox.exec.aio(
                "bash",
                "-c",
                command,
                timeout=process_timeout,
            )
            stdout, stderr = await asyncio.gather(
                process.stdout.read.aio(),
                process.stderr.read.aio(),
            )
            return_code = await process.wait.aio()
        except modal.exception.TimeoutError as exc:
            raise TimeoutError(str(exc)) from exc
        if return_code == -1 and timeout is not None:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        return ExecResult(
            return_code=return_code,
            stdout=stdout or "",
            stderr=stderr or "",
        )

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        return await self._raw_exec(_wrap_persistent(command), timeout=timeout)

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        if self._sandbox is None:
            raise RuntimeError("Environment is not set up")
        for path, content in self._verifier_files.items():
            await self._sandbox.filesystem.write_bytes.aio(content, path)
        try:
            result = await self._raw_exec(_VERIFIER_COMMAND, timeout=timeout)
        except TimeoutError:
            logger.warning("Verifier timed out after %s seconds", timeout)
            return 0.0, "verifier_timeout"
        try:
            raw = (await self._sandbox.filesystem.read_text.aio(_REWARD_PATH)).strip()
        except (FileNotFoundError, modal.exception.SandboxFilesystemNotFoundError):
            raw = None
        if raw is None:
            logger.warning("Verifier produced no reward file (exit=%s)", result.return_code)
            return 0.0, "verifier_error"
        try:
            return float(raw), None
        except ValueError:
            return 0.0, "verifier_error"

    async def cleanup(self) -> None:
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        self._sandbox = None
        try:
            await sandbox.terminate.aio()
        finally:
            await sandbox.detach.aio()
