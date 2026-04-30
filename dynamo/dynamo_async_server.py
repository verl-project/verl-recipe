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
"""DynamoHttpServer + DynamoReplica (recipe-side).

m2: same Ray actor name prefix delta as vLLM (``dynamo_*``); m2 e2e
    pipeline is functionally identical to vLLM colocate.

m3 (this file, gated by VERL_DYNAMO_ENABLE_RUNTIME=1): after vLLM
    AsyncLLM is up, additionally:
      * create_runtime(discovery_backend="file") same-process
      * register_model(ModelInput.Tokens, ...) so dynamo-side
        components (router, KVBM, frontend in m4) can discover this
        worker via /tmp/dynamo file discovery
      * KvEventPublisher tap on vLLM's KV cache events (when prefix
        caching + kv events are enabled) so the KV router can compute
        overlap scores

m3 does NOT start a dynamo Frontend; verl's generate path still goes
through the existing vLLM AsyncLLM (server_handle.<m>.remote()). m4
flips generate to the Frontend HTTP API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Optional

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer as _VllmHttpServer,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMReplica as _VllmReplica,
)

logger = logging.getLogger(__name__)

# Opt-in. Default off so existing m2 deployments on vllm017_latest.sqsh
# (which lacks the dynamo SDK) keep working. Set to "1" only after
# switching to dynamo_vllm_*.sqsh, which ships dynamo + vllm together.
_DYNAMO_RUNTIME_ENV = "VERL_DYNAMO_ENABLE_RUNTIME"
_DYNAMO_FRONTEND_ENV = "VERL_DYNAMO_ENABLE_FRONTEND"

# Where DistributedRuntime registers worker endpoints when discovery is
# "file" (matches dynamo's default; frontend in m4 reads from the same path
# unless DYN_FILE_KV is overridden in its env).
_DEFAULT_FILE_KV = "/tmp/dynamo/file_kv"

# Frontend health-check budget: dynamo Rust runtime startup + axum bind +
# file_kv discovery sync.
_FRONTEND_READY_TIMEOUT_S = 120
_FRONTEND_READY_POLL_S = 1.0


def _dynamo_runtime_enabled() -> bool:
    return os.environ.get(_DYNAMO_RUNTIME_ENV, "0") == "1"


def _dynamo_frontend_enabled() -> bool:
    # m4 implies m3 (frontend has nothing to discover without registered workers).
    return (
        _dynamo_runtime_enabled()
        and os.environ.get(_DYNAMO_FRONTEND_ENV, "0") == "1"
    )


class DynamoHttpServer(_VllmHttpServer):
    """Per-node Ray actor.

    m2: identical to vLLMHttpServer.
    m3: after super().run_server() finishes, optionally bring up a
        same-process dynamo DistributedRuntime and register this worker.
    """

    # m3 state, populated only when the runtime is enabled.
    _dynamo_runtime: Optional[Any] = None
    _dynamo_kv_publishers: Optional[list] = None
    _dynamo_generate_endpoint: Optional[Any] = None
    # m4 state, populated only when the frontend is enabled (master replica only).
    _dynamo_frontend_proc: Optional[subprocess.Popen] = None
    _vllm_server_port: Optional[int] = None  # original vLLM uvicorn port, kept for diagnostics

    async def run_server(self, args):  # noqa: D401 — match parent signature
        await super().run_server(args)
        if not _dynamo_runtime_enabled():
            return
        try:
            await self._setup_dynamo_runtime(args)
        except Exception:
            # Use warning so the message survives Ray's default actor log
            # filter (info gets dropped). m3/m4 must never break m2 generate;
            # on failure the worker keeps serving via the vLLM OpenAI server
            # that super() already started.
            logger.warning(
                "[m3/m4] dynamo runtime setup failed for replica=%s node=%s; "
                "continuing on vLLM-native path. Traceback follows.",
                self.replica_rank,
                self.node_rank,
                exc_info=True,
            )
            self._dynamo_runtime = None

    async def _setup_dynamo_runtime(self, args):
        """Bring up an in-process DistributedRuntime + register_model.

        Pre: ``self.engine`` is a live ``vllm.AsyncLLM`` (set by parent
        ``run_server``). ``self.replica_rank`` and ``self.node_rank`` are
        set. Every replica registers its own ``worker_{r}_{n}/generate``
        endpoint so the Frontend's KV router has the full set of workers
        to route between. The master replica then launches the Frontend
        (m4); non-master replicas redirect their advertised server
        address to the master Frontend (see ``_connect_to_master_frontend``).
        """
        # Imports are local so the module loads in environments that
        # don't ship the dynamo SDK (e.g. m2 baseline on vllm017).
        from dynamo.common.utils.runtime import create_runtime
        from dynamo.llm import (
            KvEventPublisher,
            ModelInput,
            ModelRuntimeConfig,
            ModelType,
            register_model,
        )

        os.makedirs(_DEFAULT_FILE_KV, exist_ok=True)
        os.environ.setdefault("DYN_DISCOVERY_BACKEND", "file")
        os.environ.setdefault("DYN_FILE_KV", _DEFAULT_FILE_KV)

        runtime, _loop = create_runtime(
            discovery_backend=os.environ["DYN_DISCOVERY_BACKEND"],
            request_plane="tcp",
            event_plane="zmq",
            use_kv_events=True,
        )
        self._dynamo_runtime = runtime

        # One endpoint per worker. Frontend (m4) finds workers by listing
        # all endpoints under verl_dynamo.worker.* in the file_kv tree.
        endpoint_path = (
            f"verl_dynamo.worker_{self.replica_rank}_{self.node_rank}.generate"
        )
        gen_endpoint = runtime.endpoint(endpoint_path)
        self._dynamo_generate_endpoint = gen_endpoint

        # Pull the runtime numbers vLLM computed at engine init.
        cache_cfg = self.engine.vllm_config.cache_config
        sched_cfg = self.engine.vllm_config.scheduler_config
        rt_cfg = ModelRuntimeConfig()
        rt_cfg.total_kv_blocks = cache_cfg.num_gpu_blocks or 0
        rt_cfg.max_num_seqs = sched_cfg.max_num_seqs
        rt_cfg.max_num_batched_tokens = sched_cfg.max_num_batched_tokens

        served_model_name = getattr(args, "served_model_name", None) or args.model

        # Plan B: every replica registers its own endpoint so the Frontend
        # discovers all 8 workers (one per replica) and KV router has real
        # routing space. Same logical model name, distinct endpoint paths
        # (worker_{replica_rank}_{node_rank}/generate); dynamo's
        # discovery joins them under one model card.
        await register_model(
            ModelInput.Tokens,  # KV router requirement
            ModelType.Chat | ModelType.Completions,
            gen_endpoint,
            args.model,
            served_model_name,
            context_length=self.engine.vllm_config.model_config.max_model_len,
            kv_cache_block_size=cache_cfg.block_size,
            runtime_config=rt_cfg,
        )
        logger.warning(
            "[m3] replica=%s registered model=%s endpoint=%s blocks=%s",
            self.replica_rank,
            served_model_name,
            endpoint_path,
            rt_cfg.total_kv_blocks,
        )

        # KV-event tap. Optional: only meaningful when prefix caching is on
        # and vLLM was started with kv_events_config.endpoint set. We
        # don't synthesize a kv_events_config here — that's a vLLM startup
        # concern (engine_kwargs.kv_events_config). If absent, KV router
        # falls back to approximate routing.
        kv_events_cfg = getattr(self.engine.vllm_config, "kv_events_config", None)
        if (
            cache_cfg.enable_prefix_caching
            and kv_events_cfg is not None
            and getattr(kv_events_cfg, "enable_kv_cache_events", False)
        ):
            try:
                publisher = KvEventPublisher(
                    endpoint=gen_endpoint,
                    kv_block_size=cache_cfg.block_size,
                    zmq_endpoint=kv_events_cfg.endpoint.replace("*", "127.0.0.1"),
                    zmq_topic="",
                    enable_local_indexer=False,
                    dp_rank=0,
                )
                self._dynamo_kv_publishers = [publisher]
                logger.info(
                    "[m3] KvEventPublisher attached at %s",
                    kv_events_cfg.endpoint,
                )
            except Exception:
                logger.exception(
                    "[m3] KvEventPublisher init failed; router will fall back "
                    "to approximate routing."
                )
                self._dynamo_kv_publishers = None
        else:
            logger.info(
                "[m3] kv_events not configured on vLLM (prefix_caching=%s, "
                "kv_events_config=%s); KV router will approximate.",
                cache_cfg.enable_prefix_caching,
                kv_events_cfg,
            )

        # m4: master replica launches Frontend and rebinds its own
        # server_port to it. Non-master replicas poll the master Ray actor
        # and rebind their own (server_address, server_port) to the master
        # Frontend, so verl's per-replica server_handle.get_server_address
        # all return the same Frontend endpoint -> all generate traffic
        # goes through the KV router.
        if not _dynamo_frontend_enabled():
            return
        if self.replica_rank == 0 and self.node_rank == 0:
            await self._launch_dynamo_frontend()
        else:
            await self._connect_to_master_frontend()

    async def _launch_dynamo_frontend(self):
        """Spawn ``python -m dynamo.frontend --router-mode kv`` and rebind
        ``self._server_port`` to it so verl's agent loop reaches the Frontend
        instead of vLLM's own OpenAI server.

        Pre: ``self._server_port`` is the vLLM uvicorn port (set by
        ``run_uvicorn`` in parent ``run_server``); ``self._dynamo_runtime``
        is alive and at least one ``register_model`` has fired (this one)
        so the Frontend has a worker to route to.
        """
        from verl.utils.net_utils import get_free_port

        # Pick a port and immediately release the holding socket so the
        # Frontend subprocess can bind it. There's a tiny TOCTOU window;
        # if it bites us in practice we'll switch to UDS or fixed port.
        fe_port, holder = get_free_port(self._server_address)
        if holder is not None:
            holder.close()

        env = {
            **os.environ,
            "DYN_DISCOVERY_BACKEND": os.environ.get("DYN_DISCOVERY_BACKEND", "file"),
            "DYN_FILE_KV": os.environ.get("DYN_FILE_KV", _DEFAULT_FILE_KV),
        }
        cmd = [
            "python",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            "kv",
            "--http-port",
            str(fe_port),
        ]
        # Redirect Frontend stdout/stderr to a file so router decision logs
        # (KV overlap scores, worker selection) survive after the actor exits
        # and so the buffer doesn't fill up with no reader.
        fe_log_path = os.path.join("/tmp/dynamo", "frontend.log")
        os.makedirs("/tmp/dynamo", exist_ok=True)
        fe_log_fp = open(fe_log_path, "w")  # closed in shutdown via Popen.terminate

        logger.warning(
            "[m4] spawning dynamo Frontend: %s (env DYN_FILE_KV=%s, log=%s)",
            cmd,
            env["DYN_FILE_KV"],
            fe_log_path,
        )
        self._dynamo_frontend_proc = subprocess.Popen(  # noqa: S603 — controlled cmd
            cmd,
            env=env,
            stdout=fe_log_fp,
            stderr=subprocess.STDOUT,
        )

        await self._wait_frontend_ready(fe_port, _FRONTEND_READY_TIMEOUT_S)

        self._vllm_server_port = self._server_port
        self._server_port = fe_port
        logger.warning(
            "[m4] dynamo Frontend ready at %s:%s (was vLLM uvicorn at port %s); "
            "verl agent loop now routes through KV router.",
            self._server_address,
            fe_port,
            self._vllm_server_port,
        )

    async def is_dynamo_frontend_ready(self) -> bool:
        """Master-side query; non-master replicas poll this before
        rebinding their server address to the master Frontend."""
        return (
            self._dynamo_frontend_proc is not None
            and self._dynamo_frontend_proc.poll() is None
            and self._vllm_server_port is not None  # set after _server_port swap
        )

    async def _connect_to_master_frontend(self):
        """Non-master: wait for master to publish its Frontend, then
        redirect this replica's server address to the master Frontend.

        After the rebind, ``self.get_server_address()`` (inherited from
        vLLMHttpServer) returns ``(master_host, master_frontend_port)``,
        which is what ``vLLMReplica.launch_servers`` advertises to verl.
        """
        # NOTE: this prefix is the one DynamoReplica._get_server_name_prefix
        # returns; DynamoHttpServer instances don't carry that method, so
        # we hardcode here to avoid a cross-class lookup.
        master_name = "dynamo_server_0_0"
        deadline = time.monotonic() + _FRONTEND_READY_TIMEOUT_S
        master = None
        last_err: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                if master is None:
                    master = ray.get_actor(master_name)
                ready = await master.is_dynamo_frontend_ready.remote()
                if ready:
                    addr, port = await master.get_server_address.remote()
                    self._vllm_server_port = self._server_port
                    self._server_port = port
                    self._server_address = addr
                    logger.warning(
                        "[m4] replica=%s redirected to master Frontend at %s:%s "
                        "(was vLLM port %s)",
                        self.replica_rank,
                        addr,
                        port,
                        self._vllm_server_port,
                    )
                    return
            except ValueError as e:
                # ray.get_actor before master is registered.
                last_err = f"ValueError: {e}"
            except Exception as e:  # noqa: BLE001 — diagnostic only
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_FRONTEND_READY_POLL_S)
        raise TimeoutError(
            f"[m4] master Frontend not advertised within {_FRONTEND_READY_TIMEOUT_S}s "
            f"(last error: {last_err})"
        )

    async def _wait_frontend_ready(self, port: int, timeout: float):
        """Poll ``GET /v1/models`` until 200 or timeout. Raises on timeout
        (the calling try/except in ``run_server`` will fall back to m3-only
        behavior, leaving generate on the vLLM port)."""
        import aiohttp

        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/v1/models"
        last_err: Optional[str] = None
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if self._dynamo_frontend_proc.poll() is not None:
                    raise RuntimeError(
                        f"[m4] Frontend exited early rc={self._dynamo_frontend_proc.returncode}"
                    )
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return
                        last_err = f"HTTP {resp.status}"
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(_FRONTEND_READY_POLL_S)
        raise TimeoutError(
            f"[m4] dynamo Frontend at port {port} not ready within {timeout}s "
            f"(last error: {last_err})"
        )

    async def shutdown(self, *args, **kwargs):  # type: ignore[override]
        # m4: stop the Frontend before tearing down the runtime; otherwise
        # in-flight HTTP requests would race against runtime shutdown.
        if self._dynamo_frontend_proc is not None:
            try:
                self._dynamo_frontend_proc.terminate()
                try:
                    self._dynamo_frontend_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._dynamo_frontend_proc.kill()
                    self._dynamo_frontend_proc.wait(timeout=5)
            except Exception:
                logger.exception("[m4] dynamo Frontend shutdown raised; ignoring.")
            self._dynamo_frontend_proc = None
        # Tear down dynamo runtime so its background tasks unwind
        # cleanly while the vLLM engine is still alive.
        if self._dynamo_runtime is not None:
            try:
                self._dynamo_runtime.shutdown()
            except Exception:
                logger.exception("[m3] dynamo runtime shutdown raised; ignoring.")
            self._dynamo_runtime = None
            self._dynamo_kv_publishers = None
            self._dynamo_generate_endpoint = None
        # vLLMHttpServer may or may not define shutdown — call only if it does.
        parent_shutdown = getattr(_VllmHttpServer, "shutdown", None)
        if callable(parent_shutdown):
            result = parent_shutdown(self, *args, **kwargs)
            if asyncio.iscoroutine(result):
                await result


class DynamoReplica(_VllmReplica):
    """Replica for the dynamo backend.

    Reuses the entire vLLMReplica launch pipeline but binds the Ray actor
    name to ``dynamo_*`` so ServerAdapter (which reads ``config.name``)
    lands on these actors and not on someone else's vllm replicas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebind to the recipe-side server class so subclass hooks (m3)
        # take effect. vLLMReplica.__init__ already set this to
        # ``ray.remote(vLLMHttpServer)``; we replace it.
        self.server_class = ray.remote(DynamoHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"


__all__ = ["DynamoHttpServer", "DynamoReplica"]
