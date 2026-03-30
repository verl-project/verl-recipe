"""SDK client for submitting agent runs to the Harbor Agent Run server.

Usage::

    from .client import AgentRunClient

    client = AgentRunClient(server_url="http://harbor-server:8080")
    result = client.submit(
        task_path="/data/tasks/my-task",
        agent_name="claude_code",
        model_name="anthropic/claude-sonnet-4-20250514",
    )
    print(result.rewards)
    print(result.rollout_details)
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from pathlib import Path
from typing import Any

import httpx

from .models import AgentConfig, AgentRunResult, RolloutDetail, VerifierConfig

logger = logging.getLogger(__name__)


class AgentRunError(Exception):
    """Raised when the server returns an HTTP error."""


def _create_task_archive(task_path: str) -> tuple[bytes, str]:
    """Create a gzipped tar archive of the local task directory.

    Returns:
        A tuple of (archive_bytes, original_dir_name).
    """
    task_dir = Path(task_path).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Archive the directory under its own name so the server can
        # find it as <tmp>/<dir_name>/
        tar.add(str(task_dir), arcname=task_dir.name)
    buf.seek(0)
    return buf.getvalue(), task_dir.name


class AgentRunClient:
    """Synchronous and async client for submitting agent runs.

    The API is designed to feel similar to locally instantiating a Harbor
    Trial, but the actual execution happens on the remote server.
    The local task directory is automatically archived and uploaded to
    the server with each request.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 1200.0,
    ):
        """Initialize the client.

        Args:
            server_url: Base URL of the Harbor Agent Run server
                (e.g. ``http://localhost:8080``).
            timeout: HTTP request timeout in seconds. Since agent runs can
                take a long time, the default is 30 minutes.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def submit(
        self,
        task_path: str,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        disable_verifier: bool = False,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        llm_proxy_url: str | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Submit an agent run and block until the result is available.

        The local ``task_path`` directory is archived and uploaded to the
        server automatically.

        Args:
            task_path: Path to the local task directory. The directory and
                its contents will be uploaded to the server.
            agent_name: Name of a built-in Harbor agent (e.g. ``"claude_code"``).
            agent_import_path: Python import path of a custom agent
                (e.g. ``"my_agents:MyAgent"``). Mutually exclusive with
                ``agent_name``.
            model_name: LLM model name passed to the agent.
            agent_kwargs: Additional keyword arguments for the agent.
            timeout_multiplier: Multiplier for all timeout values.
            disable_verifier: If ``True``, skip verification after the agent run.
            environment_overrides: Optional dict to override environment params
                (e.g. ``{"override_cpus": 4}``).
            environment_kwargs: Optional dict of extra keyword arguments passed
                directly to the environment constructor (merged with server
                defaults, overriding any conflicting keys).
            llm_proxy_url: URL of the LLM proxy server. When set, the remote
                agent will use this as its LLM endpoint so that token_ids
                and logprobs can be captured for RL training.
            **kwargs: Extra fields forwarded to the agent config.

        Returns:
            An :class:`AgentRunResult` containing ``rewards`` and
            ``rollout_details``.

        Raises:
            AgentRunError: If the server returns an HTTP error.
            FileNotFoundError: If ``task_path`` does not exist locally.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g. Jupyter).
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.submit_async(
                        task_path=task_path,
                        agent_name=agent_name,
                        agent_import_path=agent_import_path,
                        model_name=model_name,
                        agent_kwargs=agent_kwargs,
                        timeout_multiplier=timeout_multiplier,
                        disable_verifier=disable_verifier,
                        environment_overrides=environment_overrides,
                        environment_kwargs=environment_kwargs,
                        llm_proxy_url=llm_proxy_url,
                        **kwargs,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.submit_async(
                    task_path=task_path,
                    agent_name=agent_name,
                    agent_import_path=agent_import_path,
                    model_name=model_name,
                    agent_kwargs=agent_kwargs,
                    timeout_multiplier=timeout_multiplier,
                    disable_verifier=disable_verifier,
                    environment_overrides=environment_overrides,
                    environment_kwargs=environment_kwargs,
                    llm_proxy_url=llm_proxy_url,
                    **kwargs,
                )
            )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def submit_async(
        self,
        task_path: str,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        disable_verifier: bool = False,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        llm_proxy_url: str | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Async version of :meth:`submit`.

        Archives the local task directory and uploads it to the server.
        """
        merged_kwargs = dict(agent_kwargs or {})
        merged_kwargs.update(kwargs)

        agent_config = AgentConfig(
            name=agent_name,
            import_path=agent_import_path,
            model_name=model_name,
            llm_proxy_url=llm_proxy_url,
            kwargs=merged_kwargs,
        )

        verifier_config = VerifierConfig(disable=disable_verifier)

        payload = {
            "task_path": task_path,
            "agent": agent_config.model_dump(exclude_none=True),
            "timeout_multiplier": timeout_multiplier,
            "verifier": verifier_config.model_dump(),
        }
        if environment_overrides:
            payload["environment_overrides"] = environment_overrides
        if environment_kwargs:
            payload["environment_kwargs"] = environment_kwargs

        # Archive the local task directory
        archive_bytes, _dir_name = await asyncio.to_thread(
            _create_task_archive, task_path
        )

        url = f"{self.server_url}/api/v1/runs"
        logger.info("Submitting agent run to %s task=%s", url, task_path)

        import json

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    data={"metadata": json.dumps(payload)},
                    files={"task_archive": ("task.tar.gz", archive_bytes, "application/gzip")},
                )
        except Exception as e:
            return AgentRunResult(
                run_id="",
                status="error",
                rewards={"reward": 0.0},
                rollout_details=None,
                error_message=str(e),
            )

        if response.status_code != 200:
            return AgentRunResult(
                run_id="",
                status="error",
                rewards={"reward": 0.0},
                rollout_details=None,
                error_message=f"Server returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        return AgentRunResult.model_validate(data)

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    async def submit_batch_async(
        self,
        requests: list[dict[str, Any]],
    ) -> list[AgentRunResult]:
        """Submit multiple agent runs concurrently.

        Args:
            requests: List of keyword argument dicts, each passed to
                :meth:`submit_async`.

        Returns:
            List of results in the same order as the input requests.
        """
        tasks = [self.submit_async(**req) for req in requests]
        return await asyncio.gather(*tasks)

    def submit_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[AgentRunResult]:
        """Synchronous version of :meth:`submit_batch_async`."""
        return asyncio.run(self.submit_batch_async(requests))
