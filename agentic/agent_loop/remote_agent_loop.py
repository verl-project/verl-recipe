"""RemoteAgentLoop: a verl-compatible agent loop that delegates execution
to a remote third-party agent while capturing LLM traffic via an HTTP proxy.

The proxy can run in two modes:

**Ray actor mode** (default)
    The proxy runs as a Ray named actor (``ProxyServerActor``) on the head
    node, managed by :mod:`recipe.agentic.proxyserver.proxy_server`.  Each
    ``RemoteAgentLoop`` instance communicates with the proxy entirely via
    HTTP — it does not hold a direct Python reference to the proxy object.

**Standalone proxy mode** (``PROXY_SERVER_URL`` set)
    The proxy runs as a standalone service.  The agent loop connects to
    the proxy URL directly (skipping Ray-based discovery) and starts an
    :class:`InferenceWorkerClient` that bridges the framework's vLLM
    servers to the proxy via WebSocket.

URL scheme
~~~~~~~~~~
The proxy base URL (cluster-internal) is, e.g.,
``http://10.0.1.5:9123``.  For a given ``trial_id`` the *external*
agent receives::

    agent_base_url = f"http://{LLM_PROXY_IP}:{port}/{trial_id}/v1"

where ``LLM_PROXY_IP`` is an environment variable specifying an IP that is
reachable from outside the Ray cluster (e.g. the Harbor server). When the
LLM proxy is co-located with the trainer, this is just the trainer node's
externally reachable IP.

Session management (register / get / complete / delete) is done via
the proxy's REST endpoints, allowing ``RemoteAgentLoop`` to run on
any node in the Ray cluster.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from transformers import AutoProcessor, AutoTokenizer

from recipe.agentic.agent_loop.config import RemoteAgentConfig
from recipe.agentic.proxyserver.models import SessionRecord
from recipe.agentic.serversdk.client import AgentRunClient

logger = logging.getLogger(__name__)
# Make sure diagnostic INFO/DEBUG lines surface in Ray worker stdout even
# when the root logger is configured at WARNING level.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)
logger.propagate = True
# Allow operator override; default to DEBUG while debugging local trial issues.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))
# Late imports to avoid hard dependency on verl at module level.
try:
    from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                         AgentLoopOutput,
                                                         DictConfigWrap,
                                                         register)
    from verl.workers.rollout.llm_server import LLMServerClient
except ImportError:  # pragma: no cover
    from dataclasses import dataclass
    from dataclasses import field as dc_field

    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    @dataclass
    class AgentLoopOutput:  # type: ignore[no-redef]
        prompt_ids: list[int] = dc_field(default_factory=list)
        response_ids: list[int] = dc_field(default_factory=list)
        response_mask: list[int] = dc_field(default_factory=list)
        response_logprobs: Optional[list[float]] = None
        routed_experts: Any = None
        multi_modal_data: Optional[dict] = None
        reward_score: Optional[float] = None
        num_turns: int = 0
        metrics: Any = None
        extra_fields: dict = dc_field(default_factory=dict)

    class LLMServerClient:  # type: ignore[no-redef]
        pass

    class DictConfigWrap:  # type: ignore[no-redef]
        pass

    def register(name):  # type: ignore[no-redef]
        def decorator(cls):
            return cls
        return decorator


# ---------------------------------------------------------------------------
# RemoteAgentLoop
# ---------------------------------------------------------------------------

@register("remote_agent")
class RemoteAgentLoop(AgentLoopBase):
    """Agent loop that delegates execution to a remote third-party agent.

    Instead of running the agent loop locally (like verl's
    ``ToolAgentLoop``), this implementation:

    1. Discovers the proxy server URL via a lightweight Ray RPC
       (``get_proxy_url()``).  The proxy itself is a Ray named actor
       managed by :mod:`recipe.agentic.proxyserver.proxy_server`.
    2. Registers a unique *trial_id* with the proxy via HTTP.
    3. Calls the remote harbor server via the SDK, telling the agent to
       use ``http://{LLM_PROXY_IP}:{port}/{trial_id}/v1`` as its OpenAI
       ``base_url`` (cluster-external URL).
    4. After the agent finishes, collects the recorded ``token_ids`` and
       ``logprobs`` from the proxy session via HTTP and reconstructs a
       verl-compatible ``AgentLoopOutput``.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config
        rollout_cfg = config.actor_rollout_ref.rollout

        # All remote-agent settings come from environment variables
        remote_cfg = RemoteAgentConfig.from_env()

        self.harbor_server_url: str = remote_cfg.harbor_server_url
        self.harbor_timeout: float = remote_cfg.harbor_timeout

        self.remote_agent_name: str | None = remote_cfg.agent_name
        self.remote_agent_import_path: str | None = remote_cfg.agent_import_path
        self.remote_model_name: str | None = remote_cfg.model_name
        self.remote_agent_kwargs: dict[str, Any] = dict(remote_cfg.agent_kwargs)

        self.max_retries: int = remote_cfg.max_retries
        self.retry_base_delay: float = remote_cfg.retry_base_delay

        self.task_path_template: str = remote_cfg.task_path_template

        # Local Harbor task roots — resolved from the trainer config so that
        # ``data.train_harbor_dir`` / ``data.val_harbor_dir`` (set by the
        # recipe's yaml) can be used to look up a task directory by
        # ``instance_id`` instead of relying on ``task_path_template``.
        # An env override ``HARBOR_DATASET_DIRS`` (``:`` or ``,`` separated)
        # is also honored, useful for standalone proxy mode where the worker
        # may not see the trainer config.
        self.harbor_search_dirs: list[Path] = self._collect_harbor_search_dirs(config)

        self.environment_overrides: dict[str, str] = dict(remote_cfg.environment_overrides)
        self.environment_kwargs: dict[str, Any] = dict(remote_cfg.environment_kwargs)
        self.environment_import_path: str = remote_cfg.environment_import_path
        self.use_local_trial: bool = remote_cfg.use_local_trial

        # Standalone proxy mode
        self.proxy_server_url: str | None = remote_cfg.proxy_server_url
        self._inference_worker = None

        # Sequence length limits
        self.prompt_length = rollout_cfg.prompt_length
        self.response_length = rollout_cfg.response_length

    # ------------------------------------------------------------------
    # Task-path resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_harbor_search_dirs(config: Any) -> list[Path]:
        """Collect local Harbor task roots from config + env.

        Reads ``data.train_harbor_dir`` and ``data.val_harbor_dir`` from the
        trainer config (both optional, may be ``None``) and additionally any
        path listed in ``$HARBOR_DATASET_DIRS`` (``:`` or ``,`` separated).
        Non-existent paths are silently skipped.
        """
        dirs: list[Path] = []
        try:
            data_cfg = config.get("data", {}) if hasattr(config, "get") else {}
        except Exception:  # noqa: BLE001 - defensive: any config shape
            data_cfg = {}
        for key in ("train_harbor_dir", "val_harbor_dir"):
            try:
                value = data_cfg.get(key, None) if hasattr(data_cfg, "get") else None
            except Exception:  # noqa: BLE001
                value = None
            if value:
                dirs.append(Path(os.path.expanduser(str(value))))

        env_dirs = os.getenv("HARBOR_DATASET_DIRS")
        if env_dirs:
            for raw in env_dirs.replace(":", ",").split(","):
                raw = raw.strip()
                if raw:
                    dirs.append(Path(os.path.expanduser(raw)))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[Path] = []
        for d in dirs:
            key = str(d)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def _resolve_task_path(
        self,
        instance_id: str,
        explicit_local_path: str | None = None,
    ) -> str:
        """Resolve ``task_path`` for a given ``instance_id``.

        Resolution order:

        1. ``explicit_local_path`` (the ``local_task_path`` column emitted by
           :mod:`recipe.agentic.dataset.local_harbor`) if it points to an
           existing directory — this is the fast path when the dataset row
           already carries the absolute path.
        2. ``<root>/<instance_id>`` for each configured harbor root
           (``data.train_harbor_dir``, ``data.val_harbor_dir``, then
           ``$HARBOR_DATASET_DIRS``).
        3. ``self.task_path_template.format(instance_id=...)`` — kept for
           backward compatibility with the old env-driven workflow.
        """
        if explicit_local_path:
            candidate = Path(os.path.expanduser(str(explicit_local_path)))
            if candidate.is_dir():
                return str(candidate)

        if instance_id:
            for root in self.harbor_search_dirs:
                candidate = root / instance_id
                if candidate.is_dir():
                    return str(candidate)

        return self.task_path_template.format(instance_id=instance_id)

    # ------------------------------------------------------------------
    # HTTP helpers — interact with the proxy via its REST endpoints
    # ------------------------------------------------------------------

    def _get_proxy_url(self) -> str:
        """Discover the proxy server URL.

        In standalone mode (``PROXY_SERVER_URL`` set), returns the configured
        URL directly.  Otherwise, performs a lightweight Ray RPC to the
        named proxy actor.

        Raises ``RuntimeError`` if the proxy cannot be found.
        """
        if self.proxy_server_url:
            return self.proxy_server_url

        from recipe.agentic.proxyserver.ray_actor import get_proxy_url

        url = get_proxy_url()
        if url is None:
            raise RuntimeError(
                "Proxy server actor not found.  Make sure the recipe calls "
                "start_proxy_server() before training starts, or set "
                "PROXY_SERVER_URL for standalone proxy mode."
            )
        return url

    async def _ensure_inference_worker(self) -> None:
        """Start the inference worker client if in standalone proxy mode.

        The worker connects to the proxy's WebSocket endpoint and bridges
        inference requests to the local vLLM rollout servers.  This is a
        no-op if the proxy is running in local (Ray actor) mode.

        The worker is started lazily on the first call and reused across
        all subsequent ``run()`` invocations.
        """
        if not self.proxy_server_url:
            return
        if self._inference_worker is not None:
            return

        from recipe.agentic.proxyserver.worker_client import \
            InferenceWorkerClient

        # Build the WebSocket URL from the proxy HTTP URL
        ws_url = self.proxy_server_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        if not ws_url.endswith("/"):
            ws_url += "/"
        ws_url += "ws/worker"

        # Get the verl GlobalRequestLoadBalancer handle from the in-cluster
        # proxy named actor.  In the new verl API ``self.server_manager`` is
        # an ``LLMServerClient`` that fronts a load balancer and no longer
        # exposes ``server_handles`` directly.  The agentic recipe registers
        # the LB handle on the driver via ``start_proxy_server()`` (Ray-actor
        # mode) or ``start_lb_registry()`` (standalone-proxy mode), so we
        # look it up by name here and let ``InferenceWorkerClient`` route
        # requests through it.
        import ray
        from recipe.agentic.proxyserver.ray_actor import PROXY_ACTOR_NAME

        try:
            proxy_actor = ray.get_actor(PROXY_ACTOR_NAME)
        except ValueError as exc:
            raise RuntimeError(
                "Cannot start inference worker: proxy named actor "
                f"'{PROXY_ACTOR_NAME}' not found. Ensure the recipe calls "
                "start_proxy_server() before training starts."
            ) from exc
        load_balancer = await proxy_actor.get_load_balancer.remote()
        model_path = self.config.actor_rollout_ref.model.path

        self._inference_worker = InferenceWorkerClient(
            proxy_ws_url=ws_url,
            load_balancer=load_balancer,
            model_path=model_path,
        )
        await self._inference_worker.start()
        logger.info(
            "Inference worker started, connecting to proxy at %s", ws_url
        )

    async def _register_session(self, proxy_url: str, trial_id: str) -> None:
        """POST /sessions/{trial_id} to register a new session."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{proxy_url}/sessions/{trial_id}") as resp:
                resp.raise_for_status()

    async def _get_session_data(self, proxy_url: str, trial_id: str) -> SessionRecord | None:
        """GET /sessions/{trial_id} to retrieve recorded session data."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{proxy_url}/sessions/{trial_id}") as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                data = await resp.json()
                return SessionRecord(**data)

    async def _complete_session(self, proxy_url: str, trial_id: str) -> None:
        """POST /sessions/{trial_id}/complete to mark session completed."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{proxy_url}/sessions/{trial_id}/complete") as resp:
                resp.raise_for_status()

    async def _delete_session(self, proxy_url: str, trial_id: str) -> None:
        """DELETE /sessions/{trial_id} to remove session data."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{proxy_url}/sessions/{trial_id}") as resp:
                resp.raise_for_status()

    # ------------------------------------------------------------------

    async def _submit_with_retry(
        self,
        proxy_url: str,
        trial_id: str,
        task_path: str,
        agent_base_url: str,
        agent_kwargs: dict[str, Any],
    ):
        """Submit an agent run with exponential backoff retry.

        On each retry the proxy session is reset (deleted + re-registered
        via HTTP) to avoid double-recording from partial runs.

        When ``use_local_trial`` is ``True``, the trial is executed in-process
        via :class:`harbor.trial.trial.Trial` instead of being sent to a remote
        Harbor HTTP server.

        Returns:
            ``AgentRunResult`` built from the trial result.
        """
        from recipe.agentic.serversdk.models import AgentRunResult

        # Build environment kwargs from config (populated via REMOTE_AGENT_ENVIRONMENT_KWARGS)
        env_kwargs = dict(self.environment_kwargs)
        env_overrides = dict(self.environment_overrides)
        env_overrides.setdefault("OPENAI_API_KEY", "verl-proxy")
        env_overrides.setdefault("OPENAI_BASE_URL", agent_base_url)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                # Reset session before retry via HTTP
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying trial %s (attempt %d/%d) after %.1fs delay",
                    trial_id, attempt + 1, self.max_retries, delay,
                )
                await asyncio.sleep(delay)
                await self._delete_session(proxy_url, trial_id)
                await self._register_session(proxy_url, trial_id)

            try:
                if self.use_local_trial:
                    result = await self._run_local_trial(
                        trial_id=trial_id,
                        task_path=task_path,
                        agent_kwargs=agent_kwargs,
                        env_overrides=env_overrides,
                        env_kwargs=env_kwargs,
                    )
                else:
                    result = await self._run_remote_trial(
                        trial_id=trial_id,
                        task_path=task_path,
                        agent_base_url=agent_base_url,
                        agent_kwargs=agent_kwargs,
                        env_overrides=env_overrides,
                        env_kwargs=env_kwargs,
                    )

                if result.status == "completed":
                    return result

                if result.status in ("failed", "error"):
                    last_error = RuntimeError(
                        f"Agent run {result.status}: {result.error or 'unknown'}"
                    )
                    logger.warning(
                        "Trial %s attempt %d/%d %s: %s",
                        trial_id, attempt + 1, self.max_retries,
                        result.status, result.error,
                    )
                    continue

                # Unexpected status — still return it
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Trial %s attempt %d/%d raised %s: %s",
                    trial_id, attempt + 1, self.max_retries,
                    type(e).__name__, e,
                )
                continue

        # All retries exhausted — return a synthetic error result
        logger.error(
            "Trial %s: all %d retries exhausted. Last error: %s",
            trial_id, self.max_retries, last_error,
        )
        return AgentRunResult(
            run_id="",
            status="error",
            rewards={"reward": 0.0},
            rollout_details=None,
            error=f"All {self.max_retries} retries exhausted: {last_error}",
        )

    async def _run_remote_trial(
        self,
        trial_id: str,
        task_path: str,
        agent_base_url: str,
        agent_kwargs: dict[str, Any],
        env_overrides: dict[str, str],
        env_kwargs: dict[str, Any],
    ):
        """Submit the trial to the remote Harbor HTTP server via AgentRunClient."""
        client = AgentRunClient(
            server_url=self.harbor_server_url,
            timeout=self.harbor_timeout,
        )
        return await client.submit_async(
            task_path=task_path,
            agent_name=self.remote_agent_name,
            agent_import_path=self.remote_agent_import_path,
            model_name=self.remote_model_name,
            environment_overrides=env_overrides,
            environment_kwargs=env_kwargs,
            agent_kwargs=agent_kwargs,
            llm_proxy_url=agent_base_url,
        )

    async def _run_local_trial(
        self,
        trial_id: str,
        task_path: str,
        agent_kwargs: dict[str, Any],
        env_overrides: dict[str, str],
        env_kwargs: dict[str, Any],
    ):
        """Run the trial in-process via :class:`harbor.trial.trial.Trial`."""
        import socket

        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import \
            EnvironmentConfig as TrialEnvironmentConfig
        from harbor.models.trial.config import TaskConfig, TrialConfig
        from harbor.trial.trial import Trial
        from recipe.agentic.serversdk.models import AgentRunResult

        # ------------------------------------------------------------------
        # Diagnostic logging — surface every input that may cause [Errno 2]
        # or K8s ApiException(0) before the trial actually starts.
        # ------------------------------------------------------------------
        try:
            host = socket.gethostname()
        except Exception:
            host = "<unknown>"

        logger.info(
            "[local_trial:%s] host=%s pid=%d HOME=%s cwd=%s",
            trial_id, host, os.getpid(),
            os.environ.get("HOME"), os.getcwd(),
        )
        logger.info(
            "[local_trial:%s] task_path=%s exists=%s import_path=%s",
            trial_id, task_path, os.path.isdir(task_path),
            self.environment_import_path,
        )
        logger.info(
            "[local_trial:%s] env_kwargs=%s", trial_id, env_kwargs,
        )
        kubeconfig = env_kwargs.get("kubeconfig") if isinstance(env_kwargs, dict) else None
        if kubeconfig:
            kc_path = os.path.expanduser(str(kubeconfig))
            logger.info(
                "[local_trial:%s] kubeconfig=%s expanded=%s exists=%s readable=%s",
                trial_id, kubeconfig, kc_path,
                os.path.exists(kc_path),
                os.access(kc_path, os.R_OK) if os.path.exists(kc_path) else False,
            )

        trial_config = TrialConfig(
            trial_name=trial_id,
            task=TaskConfig(path=task_path),
            agent=TrialAgentConfig(
                name=self.remote_agent_name,
                import_path=self.remote_agent_import_path,
                model_name=self.remote_model_name,
                kwargs=agent_kwargs,
                env=env_overrides,
            ),
            environment=TrialEnvironmentConfig(
                import_path=self.environment_import_path,
                env=env_overrides,
                kwargs=env_kwargs,
            ),
        )

        try:
            trial = await Trial.create(trial_config)
        except Exception as e:
            logger.exception(
                "[local_trial:%s] Trial.create failed: %s: %s",
                trial_id, type(e).__name__, e,
            )
            raise

        try:
            trial_result = await trial.run()
        except Exception as e:
            logger.exception(
                "[local_trial:%s] trial.run raised %s: %s",
                trial_id, type(e).__name__, e,
            )
            raise

        if trial_result.exception_info is None:
            rewards = (
                trial_result.verifier_result.rewards
                if trial_result.verifier_result is not None
                else {}
            )
            return AgentRunResult(
                run_id=trial_id,
                status="completed",
                rewards=rewards or {},
                rollout_details=None,
                error=None,
            )

        exc = trial_result.exception_info
        # Dump full stack trace if available, so [Errno 2] becomes locatable.
        tb_text = getattr(exc, "traceback", None) or getattr(exc, "stack_trace", None)
        logger.error(
            "[local_trial:%s] trial returned exception %s: %s\n%s",
            trial_id, exc.exception_type, exc.exception_message,
            tb_text or "<no traceback in exception_info>",
        )
        # Also write the full exception_info repr to make sure nothing is hidden.
        try:
            logger.debug(
                "[local_trial:%s] full exception_info=%r", trial_id, exc,
            )
        except Exception:
            pass
        return AgentRunResult(
            run_id=trial_id,
            status="error",
            rewards={"reward": 0.0},
            rollout_details=None,
            error=f"{exc.exception_type}: {exc.exception_message}",
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run a single rollout via a remote agent.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Dataset fields, must include ``raw_prompt``.

        Returns:
            ``AgentLoopOutput`` with reconstructed token_ids, masks, and
            logprobs captured by the proxy.
        """
        import shortuuid
        messages = list(kwargs["raw_prompt"])
        instance_id = kwargs.get("instance_id", "")
        trial_id = instance_id + "-" + shortuuid.uuid()

        # 0. If using a standalone proxy, ensure the inference worker is up
        await self._ensure_inference_worker()

        # 1. Discover proxy URL (cluster-internal) via Ray named actor or env
        proxy_url = self._get_proxy_url()
        parsed = urlparse(proxy_url)
        proxy_port = parsed.port

        # 2. Register session with the proxy via HTTP
        await self._register_session(proxy_url, trial_id)

        try:
            # 3. Build the agent's base_url using LLM_PROXY_IP (cluster-external)
            llm_proxy_ip = os.getenv("LLM_PROXY_IP", "0.0.0.0")
            if llm_proxy_ip == "0.0.0.0":
                logger.warning(
                    "LLM_PROXY_IP is not set, falling back to 0.0.0.0. "
                    "The remote agent may not be able to reach the proxy."
                )
            agent_base_url = f"http://{llm_proxy_ip}:{proxy_port}/{trial_id}/v1"

            agent_kwargs = dict(self.remote_agent_kwargs)
            agent_kwargs["model_base_url"] = agent_base_url
            agent_kwargs["session_id"] = trial_id
            agent_kwargs["temperature"] = sampling_params.get("temperature", 1.0)
            agent_kwargs["top_p"] = sampling_params.get("top_p", 1.0)

            # 4. Call the remote harbor server with retry
            task_path = self._resolve_task_path(
                instance_id,
                explicit_local_path=kwargs.get("local_task_path"),
            )
            result = await self._submit_with_retry(
                proxy_url=proxy_url,
                trial_id=trial_id,
                task_path=task_path,
                agent_base_url=agent_base_url,
                agent_kwargs=agent_kwargs,
            )

            # 5. Collect session data from the proxy via HTTP
            await self._complete_session(proxy_url, trial_id)
            session = await self._get_session_data(proxy_url, trial_id)

            # Build metrics with harbor result status
            metrics: dict[str, Any] = {
                "harbor_status": result.status,
            }
            if result.error:
                metrics["harbor_error"] = result.error

            if session is None or not session.turns:
                logger.warning(
                    "Session %s has no recorded turns — the agent may not "
                    "have called the proxy.",
                    trial_id,
                )
                message = kwargs.get("problem_statement", "")
                prompt_ids = self._tokenize_messages([{"role": "user", "content": message}])
                # Return at least an EOS token so downstream tokenizer.pad
                # produces a proper tensor instead of an empty list (which
                # would crash with AttributeError on .dim()).
                # Use the harbor reward if available (the agent may have
                # succeeded even though the proxy didn't record turns, e.g.
                # due to a streaming disconnect); fall back to 0.0 so that
                # _compute_score is skipped — the remote agent dataset may
                # lack fields like 'data_source' that the reward manager
                # requires.
                eos_token_id = self.tokenizer.eos_token_id
                reward_score = 0.0
                if result.rewards:
                    reward_score = sum(result.rewards.values())
                return AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=[eos_token_id],
                    response_mask=[0],
                    response_logprobs=[0.0],
                    reward_score=reward_score,
                    num_turns=0,
                    metrics=metrics,
                )

            # 6. Reconstruct verl output - use messages from session
            initial_messages = session.turns[0].request_messages
            output = self._reconstruct_output(session, initial_messages)
            output.metrics = metrics

            if result.rewards:
                output.reward_score = sum(result.rewards.values())

            return output

        finally:
            await self._delete_session(proxy_url, trial_id)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize message content fields.

        Some agents send ``content`` as a list of content-parts (OpenAI
        multi-part format).  The tokenizer's ``apply_chat_template``
        expects plain strings, so we flatten them here.
        """
        normalized = []
        for msg in messages:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif "text" in part:
                            text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                msg["content"] = "\n".join(text_parts) if text_parts else ""
            normalized.append(msg)
        return normalized

    def _tokenize_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize messages using the chat template."""
        messages = self._normalize_messages(messages)
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            return model_inputs["input_ids"].squeeze(0).tolist()
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        )

    def _tokenize_tool_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize tool / user response messages for the inter-turn gap."""
        if not messages:
            return []
        messages = self._normalize_messages(messages)
        sys_ids = self._get_system_prompt_ids()
        if self.processor is not None:
            raw = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            inputs = self.processor(text=[raw], return_tensors="pt")
            ids = inputs["input_ids"].squeeze(0).tolist()
        else:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
        if sys_ids and ids[: len(sys_ids)] == sys_ids:
            ids = ids[len(sys_ids):]
        return ids

    def _get_system_prompt_ids(self) -> list[int]:
        if hasattr(self, "_cached_sys_ids"):
            return self._cached_sys_ids
        try:
            if self.processor is not None:
                raw = self.processor.apply_chat_template(
                    [], add_generation_prompt=False, tokenize=False,
                )
                inputs = self.processor(text=[raw], return_tensors="pt")
                self._cached_sys_ids = inputs["input_ids"].squeeze(0).tolist()
            else:
                self._cached_sys_ids = self.tokenizer.apply_chat_template(
                    [], add_generation_prompt=False, tokenize=True,
                )
        except Exception:
            self._cached_sys_ids = []
        return self._cached_sys_ids

    # ------------------------------------------------------------------
    # Output reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_output(
        self,
        session: SessionRecord,
        initial_messages: list[dict[str, Any]],
    ) -> AgentLoopOutput:
        """Build ``AgentLoopOutput`` from the proxy's session recording.

        For each LLM turn the completion tokens get ``mask=1``; for every
        inter-turn gap (tool / user responses) the tokens get ``mask=0``
        with ``logprobs=0.0``.
        """
        prompt_ids = self._tokenize_messages(initial_messages)

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        num_turns = 0

        for i, turn in enumerate(session.turns):
            # LLM completion tokens → mask=1
            response_ids.extend(turn.completion_token_ids)
            response_mask.extend([1] * len(turn.completion_token_ids))
            response_logprobs.extend(turn.completion_logprobs)
            num_turns += 1

            # Inter-turn content (tool / user messages) → mask=0
            if i + 1 < len(session.turns):
                next_turn = session.turns[i + 1]
                prev_count = len(turn.request_messages)
                next_count = len(next_turn.request_messages)

                if next_count > prev_count:
                    new_messages = next_turn.request_messages[prev_count:]
                    tool_messages = [
                        m for m in new_messages
                        if m.get("role") in ("tool", "user", "system")
                    ]
                    if tool_messages:
                        tool_ids = self._tokenize_tool_messages(tool_messages)
                        response_ids.extend(tool_ids)
                        response_mask.extend([0] * len(tool_ids))
                        response_logprobs.extend([0.0] * len(tool_ids))
                        num_turns += len(tool_messages)

        # Truncate to response_length
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=num_turns,
            metrics={},
        )
