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
"""DynamoHttpServer + DynamoReplica.

Reference impl: nemo_rl/models/generation/dynamo/dynamo_generation.py.
  1. Reserves no GPUs itself; trainer workers in colocated mode already claim
     them. We only forward CUDA_VISIBLE_DEVICES into the engine subprocesses.
  2. Spawns + watchdogs etcd / nats-server / dynamo.{vllm,sglang} × N /
     dynamo.frontend. Which engine is picked is documented at ENGINE_VLLM below.
  3. Never holds an in-process engine (no AsyncLLM, no sglang Engine). The
     actor's generate() method is only an HTTP client shim to dynamo.frontend;
     it does not generate locally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import ray
import requests
from ray.actor import ActorHandle

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_torch_device
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# (attr_name, display_name, stop_timeout_seconds) — order = teardown order.
# Stop consumers first (frontend), then producers (workers), then infra
# (NATS, etcd). Keep parallel to nemo-rl's _SUBPROCESS_REGISTRY but with the
# engine worker pool added as a list-typed entry. The display name also lands in
# watchdog errors ("dynamo engine_workers[1] exited rc=..."), so keep it
# engine-neutral.
_SUBPROCESS_REGISTRY: list[tuple[str, str, int]] = [
    ("_frontend_process", "frontend", 15),
    ("_engine_processes", "engine_workers", 30),
    ("_nats_process", "NATS", 10),
    ("_etcd_process", "etcd", 10),
]

# Default verl-side rank-offset env var read by the vLLM WorkerExtension
# (see recipe/dynamo/dynamo_worker_extension.py). Must be passed per-shard
# when spawning dynamo.vllm so the vLLM TP rank inside the subprocess maps
# to the same node-local rank that the trainer side computes. Set for sglang
# shards too, but unread there: DynamoSGLangRollout derives the same mapping
# trainer-side from shard = local_rank // tp.
_REPLICA_RANK_ENV = "VERL_REPLICA_RANK"

# Where an engine worker exposes its system HTTP (/metrics plus the /engine/*
# routes). Each subprocess gets its own port (allocated by the actor). Optional
# for vllm, which is controlled over the ZMQ sidecar below; mandatory for
# sglang, whose only control plane is /engine/control/* on this port.
_DYN_SYSTEM_PORT_ENV = "DYN_SYSTEM_PORT"

# Verl-private control sidecar (see _dynamo_vllm_with_control.py) listens on
# this ZMQ endpoint per subprocess; the actor uses it to bridge collective_rpc.
# vllm only — sglang shards are spawned without a sidecar.
_CONTROL_ZMQ_ENV = "VERL_DYNAMO_CONTROL_ZMQ"

_FRONTEND_READY_TIMEOUT_S = float(os.getenv("VERL_DYNAMO_FE_READY_TIMEOUT", "600"))
_FRONTEND_READY_POLL_S = 2.0
_ETCD_READY_TIMEOUT_S = 30.0
_NATS_READY_TIMEOUT_S = 30.0
_WATCHDOG_INTERVAL_S = 5.0
_VLLM_TCPSTORE_PORT_BASE = int(os.getenv("VERL_DYNAMO_VLLM_PORT_BASE", "20000"))
_KV_EVENT_PORT_BASE = int(os.getenv("VERL_DYNAMO_KV_EVENT_PORT_BASE", "42000"))
# Opt-in per-worker system-status/metrics port base. Kept well below 32768 so
# Dynamo's Rust runtime (which parses DYN_SYSTEM_PORT as i16) accepts it, and
# below the 20000 vLLM-TCPStore window to avoid overlap. Only used when
# rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true.
_SYSTEM_METRICS_PORT_BASE = int(os.getenv("VERL_DYNAMO_SYSTEM_METRICS_PORT_BASE", "11000"))

# Inference engine behind Dynamo. "vllm" keeps the historical path (thin
# _dynamo_vllm_with_control wrapper + ZMQ control sidecar + worker_extension_cls);
# "sglang" launches bare `python -m dynamo.sglang` and drives it entirely through
# the /engine/control/* HTTP routes dynamo.sglang registers natively, so it needs
# no sidecar and no worker extension. Selected via
# rollout.engine_kwargs.dynamo.engine.
ENGINE_VLLM = "vllm"
ENGINE_SGLANG = "sglang"
_SUPPORTED_ENGINES = (ENGINE_VLLM, ENGINE_SGLANG)


@dataclass(frozen=True)
class _DynamoWorkerSpec:
    """One engine subprocess (dynamo.vllm or dynamo.sglang) on this Ray actor."""

    replica_rank: int
    cuda_visible_devices: str
    rank_offset: int
    label: str


# --------------------------------------------------------------------------- #
# DynamoHttpServer
# --------------------------------------------------------------------------- #


class DynamoHttpServer:
    """Ray actor: GPU placeholder + dynamo subprocess watchdog.

    Lifecycle (driven by ``DynamoReplica.launch_servers``):
      __init__ → store config + cuda_visible_devices, no subprocesses yet
      launch_server(master_address, master_port, dp_rpc_port):
        node 0 (master): _start_etcd → _start_nats → _start_engine_workers
                         → _start_frontend → _healthcheck_frontend
        node N (slave) : just _start_engine_workers, pointing to master etcd/nats
      generate / wake_up / sleep / collective_rpc / ... :
        generate goes through master dynamo.frontend HTTP
        control ops reach each subprocess over the engine's own control plane:
        the ZMQ sidecar (collective_rpc) for vllm, /engine/control/* HTTP for
        sglang
      shutdown : SIGTERM each entry of _SUBPROCESS_REGISTRY in order.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        worker_specs: Optional[list[dict[str, Any]]] = None,
        expected_workers: Optional[int] = None,
    ):
        # Match vLLMHttpServer's __init__ contract so vLLMReplica.launch_servers
        # can spin us up unchanged. We do NOT instantiate vLLM AsyncLLM.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ[_REPLICA_RANK_ENV] = str(replica_rank)

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.rollout_mode = rollout_mode
        # workers handle is captured for parity with vLLMHttpServer; we don't
        # use it (no in-process engine, no collective_rpc destination here).
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self._cuda_visible_devices = cuda_visible_devices
        self._worker_specs: Optional[list[_DynamoWorkerSpec]] = (
            [_DynamoWorkerSpec(**spec) for spec in worker_specs] if worker_specs is not None else None
        )
        self._expected_workers = expected_workers

        # Set by ServerAdapter.update_weights to tag generations.
        self.global_steps: Optional[int] = None

        # Partial-rollout gate: abort_all_requests() clears it and pauses the
        # engines; resume_generation() resumes the engines and sets it.
        # generate() waits on it so new / client-retried requests queue here
        # instead of reaching a paused engine (abort_all_requests pauses both engines).
        self._generation_resumed = asyncio.Event()
        self._generation_resumed.set()

        # Sleep/wake around training follows verl's rollout.free_cache_engine.
        # An explicit engine_kwargs.dynamo.free_engine_on_train may only
        # restate it — a contradiction between the two switches used to no-op
        # silently (OOM in colocate training); now it fails at startup.
        verl_free_cache_engine = bool(getattr(self.config, "free_cache_engine", False))
        explicit_free_engine = self._dynamo_cfg().get("free_engine_on_train")
        if (
            explicit_free_engine is not None
            and self._dynamo_cfg_bool("free_engine_on_train", verl_free_cache_engine) != verl_free_cache_engine
        ):
            raise ValueError(
                f"engine_kwargs.dynamo.free_engine_on_train={explicit_free_engine!r} contradicts "
                f"rollout.free_cache_engine={verl_free_cache_engine}. This switch now mirrors "
                "rollout.free_cache_engine and is no longer independently tunable. To keep engines "
                "resident during training (the old free_engine_on_train=false behavior), set "
                "rollout.free_cache_engine=false as well."
            )
        self._check_sleep_mode_pairing()
        self._free_engine_on_train_flag = verl_free_cache_engine

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port: Optional[int] = None  # = frontend_port once ready

        # Master-side ports — these are filled in on node_rank==0 in
        # launch_server and exposed to slaves via get_master_address.
        self._etcd_port: Optional[int] = None
        self._etcd_peer_port: Optional[int] = None
        self._nats_port: Optional[int] = None
        self._frontend_port: Optional[int] = None

        # Slave-side: cache the master address that launch_server received.
        # Slaves use this to compute ETCD_ENDPOINTS / NATS_SERVER for their
        # subprocesses without re-querying master.
        self._master_address: Optional[str] = None
        self._master_etcd_port: Optional[int] = None
        self._master_nats_port: Optional[int] = None

        # dynamo namespace — share across all replicas in this job. Distinct
        # ETCD_ENDPOINTS / data dirs per job already isolate state, so a
        # single namespace is fine and matches the frontend's filter.
        dynamo_cfg = self._dynamo_cfg()
        self._namespace: str = dynamo_cfg.get("namespace", "verl_dynamo")
        self._router_mode: str = dynamo_cfg.get("router_mode", "kv")

        # Subprocess handles.
        self._etcd_process: Optional[subprocess.Popen] = None
        self._nats_process: Optional[subprocess.Popen] = None
        self._frontend_process: Optional[subprocess.Popen] = None
        self._engine_processes: list[subprocess.Popen] = []
        self._etcd_data_dir: Optional[str] = None
        self._frontend_log_fp = None
        self._engine_log_fps: list = []
        self._engine_log_paths: list[str] = []
        self._allocated_tcp_ports: set[int] = set()
        self._direct_generate_idx: int = 0
        self._direct_generate_lock = asyncio.Lock()
        self._logged_engine_data_token_ids = False
        self._logged_missing_engine_data = False

        # Async HTTP client to the Dynamo frontend. Created lazily on the actor
        # event loop. Replaces the blocking ``asyncio.to_thread(requests.post)``
        # data plane, whose default thread pool (~32 workers) caps concurrency
        # far below the hundreds of in-flight turns an agentic-RL step issues.
        self._http_session: Optional[Any] = None
        self._http_session_lock = asyncio.Lock()

        # vllm only: per-subprocess control sidecar endpoints (filled in
        # _start_engine_workers); used by the collective_rpc bridge. The sglang
        # equivalent is _engine_control_endpoints below.
        self._control_endpoints: list[str] = []
        # Per-worker /metrics endpoints (host:port), populated only when
        # enable_worker_system_metrics is on. These expose engine-level
        # vllm:prefix_cache_* that the frontend endpoint does not.
        self._worker_metrics_endpoints: list[str] = []
        # sglang only: per-shard http://host:DYN_SYSTEM_PORT base URLs. This is the
        # whole control plane for that engine (weight refit, memory occupation,
        # cache flush), so unlike the vLLM path it is not optional.
        self._engine_control_endpoints: list[str] = []
        self._sglang_clients: Optional[list] = None
        # Which memory tags this node's sglang shards currently have released.
        # SINGLE OWNER of that state: both DynamoHttpServer.sleep/wake_up and the
        # per-rank SGLangServerAdapter.release/resume route through
        # sglang_release/sglang_resume below. Tracking it per-adapter instead let the
        # actor release (which UNREGISTERS the worker from discovery) while the
        # adapter believed nothing was released, skipped the resume, and left the
        # frontend answering 503 "Model is not ready to serve requests yet".
        self._sglang_released_tags: set[str] = set()
        # Per-shard truth behind the node-level view above (index = local shard
        # index, same order as _engine_control_endpoints). A fan-out can succeed on
        # some shards and fail on others; recording only the all-or-nothing outcome
        # let the next release re-release the shards that had already succeeded
        # (the torch_memory_saver double-unbind described in sglang_release).
        self._sglang_released_by_shard: list[set[str]] | None = None
        # Guards the check-then-act in sglang_release/sglang_resume. Ray async
        # actors run methods concurrently on one event loop, and every shard's
        # adapter on this node calls in parallel (4 shards at TP=2 on 8 GPUs), so
        # without this all of them read the tag set, all decide they must act, and
        # all reach the engine. Created lazily: __init__ may run outside a loop.
        self._sglang_tag_lock: asyncio.Lock | None = None

        # Filled in by _start_engine_workers; consumed by generate() to build
        # the OpenAI completions payload.
        self._served_model_name: Optional[str] = None

        # Watchdog state.
        self._watchdog_task: Optional[asyncio.Task] = None
        self._shutdown_requested: bool = False

        logger.info(
            "[DynamoHttpServer] init replica=%s node=%s nnodes=%s gpus=%s cvd=%s",
            self.replica_rank,
            self.node_rank,
            self.nnodes,
            self.gpus_per_node,
            cuda_visible_devices,
        )

    # ------------------------------------------------------------------ #
    # config helpers
    # ------------------------------------------------------------------ #

    def _dynamo_cfg(self) -> dict:
        """Return ``rollout.engine_kwargs.dynamo`` dict (or empty)."""
        return (self.config.engine_kwargs or {}).get("dynamo", {}) or {}

    def _engine_kind(self) -> str:
        """Which inference engine Dynamo fronts: ``vllm`` (default) or ``sglang``."""
        engine = str(self._dynamo_cfg().get("engine", ENGINE_VLLM)).lower()
        if engine not in _SUPPORTED_ENGINES:
            raise ValueError(f"rollout.engine_kwargs.dynamo.engine must be one of {_SUPPORTED_ENGINES}, got {engine!r}")
        return engine

    def _is_sglang(self) -> bool:
        return self._engine_kind() == ENGINE_SGLANG

    def _sglang_cfg(self) -> dict:
        """Return ``rollout.engine_kwargs.dynamo.sglang`` dict (or empty)."""
        return self._dynamo_cfg().get("sglang", {}) or {}

    def get_engine_control_endpoints(self) -> list[str]:
        """Base URLs of this node's per-shard ``/engine/*`` control planes.

        Ordered by shard index, so a trainer rank can index with
        ``local_rank // tensor_model_parallel_size`` and reach the engine that owns
        its GPUs. sglang only — the vLLM path uses ``_control_endpoints`` (ZMQ).
        """
        return list(self._engine_control_endpoints)

    def _sglang_control_clients(self) -> list:
        """Lazily build one control client per local sglang shard."""
        if self._sglang_clients is None:
            from recipe.dynamo.dynamo_sglang_engine import DynamoSGLangControlClient

            timeout_s = float(self._dynamo_cfg().get("request_timeout_s", 600))
            self._sglang_clients = [
                DynamoSGLangControlClient(url, timeout_s=timeout_s) for url in self._engine_control_endpoints
            ]
        return self._sglang_clients

    def _tag_lock(self) -> asyncio.Lock:
        if self._sglang_tag_lock is None:
            self._sglang_tag_lock = asyncio.Lock()
        return self._sglang_tag_lock

    def _sglang_shard_state(self) -> list[set[str]]:
        """Per-shard released-tag sets, sized lazily to the number of control clients."""
        n = len(self._sglang_control_clients())
        if self._sglang_released_by_shard is None or len(self._sglang_released_by_shard) != n:
            self._sglang_released_by_shard = [set() for _ in range(n)]
        return self._sglang_released_by_shard

    def _refresh_node_released_view(self) -> None:
        shards = self._sglang_released_by_shard or []
        self._sglang_released_tags = set().union(*shards) if shards else set()

    async def sglang_release(self, tags: list[str]):
        """Release only tags not already released; double-release corrupts the pool.

        Symmetric to the filter in ``sglang_resume`` — and load-bearing for the same
        reason, just with a nastier failure mode. verl releases before every weight
        sync but resumes ``weights`` and ``kv_cache`` at different points, so a second
        release arrives while ``kv_cache`` is still released. Calling
        ``release_memory_occupation(["kv_cache"])`` twice makes torch_memory_saver
        unbind a region it has already unbound
        (``tms_torch_free: only support interesting region``); the pool survives as a
        Python object but its storage is no longer GPU-backed, so the next prefill
        dies inside a Triton kernel with the very indirect

            ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)

        which surfaces to the trainer only as HTTP 500 "Failed to fold completions
        stream" from the frontend (job 16283764: 257 of them). Nothing in that chain
        names memory release, so the guard has to be here.
        """
        async with self._tag_lock():
            per_shard = self._sglang_shard_state()
            targets = []
            for idx, released in enumerate(per_shard):
                wanted = [t for t in tags if t not in released]
                if wanted:
                    targets.append((idx, wanted))
            if not targets:
                logger.info(
                    "[DynamoHttpServer] sglang release%s skipped, already released (released=%s)",
                    list(tags),
                    sorted(self._sglang_released_tags),
                )
                return
            outcomes = await self._sglang_control_fanout(
                "release_memory_occupation", [(idx, {"tags": wanted}) for idx, wanted in targets]
            )
            errors = []
            for (idx, wanted), result in zip(targets, outcomes, strict=True):
                if isinstance(result, Exception):
                    errors.append((idx, result))
                else:
                    per_shard[idx].update(wanted)
            self._refresh_node_released_view()
            logger.info(
                "[DynamoHttpServer] sglang released %s on %d/%d shard(s) (now released=%s)",
                sorted({t for _, w in targets for t in w}),
                len(targets) - len(errors),
                len(targets),
                sorted(self._sglang_released_tags),
            )
            if errors:
                # Successful shards are already recorded above, so a retry only
                # touches the ones that failed instead of double-releasing the rest.
                raise RuntimeError(
                    f"dynamo.sglang release_memory_occupation failed on shard(s) {errors}; "
                    f"state recorded per shard, retry will skip the {len(targets) - len(errors)} that succeeded"
                )

    async def sglang_resume(self, tags: list[str]):
        """Resume only tags actually released; resuming others kills the scheduler.

        SGLang raises ``KeyError`` inside ``weight_updater.resume_memory_occupation``
        for a tag it never released, and that takes the whole scheduler process down
        (seen from the trainer as ``aiohttp.ServerDisconnectedError``). verl resumes
        weights unconditionally before every weight sync — including the first, when
        nothing has been released — so this filter is load-bearing, not defensive.
        """
        async with self._tag_lock():
            per_shard = self._sglang_shard_state()
            targets = [(idx, sorted(released)) for idx, released in enumerate(per_shard) if released]
            if not targets:
                logger.info(
                    "[DynamoHttpServer] sglang resume%s skipped (released=%s)",
                    list(tags),
                    sorted(self._sglang_released_tags),
                )
                return
            # Resume EVERY released tag, not just the requested one. Dynamo's sglang
            # handler re-registers the worker into discovery on the first resume, so a
            # partial resume makes the shard routable while part of its memory is still
            # released. verl asks for "weights" first (it wants live tensors to write
            # the weight sync into) and "kv_cache" only after the sync, which left a
            # ~9 s window where the frontend happily routed prefills at a shard whose
            # req_to_token_pool had no GPU backing:
            #   16:42:53.311 resumed ['weights'] -> worker added back to routing pool
            #   16:43:01.841 Scheduler hit an exception:
            #     ValueError: Pointer argument (at 0) cannot be accessed from Triton
            #                 (cpu tensor?)   [write_req_to_token_pool_triton]
            # which reached the trainer only as 257x HTTP 500 "Failed to fold
            # completions stream" (job 16441143). Resuming kv_cache early is safe: its
            # contents are stale, and the weight sync's flush_cache drops them.
            widened = sorted({t for _, w in targets for t in w})
            if set(widened) != set(tags):
                logger.info(
                    "[DynamoHttpServer] sglang resume%s widened to %s so the shard "
                    "does not rejoin discovery half-restored",
                    list(tags),
                    widened,
                )
            outcomes = await self._sglang_control_fanout(
                "resume_memory_occupation", [(idx, {"tags": wanted}) for idx, wanted in targets]
            )
            errors = []
            for (idx, wanted), result in zip(targets, outcomes, strict=True):
                if isinstance(result, Exception):
                    errors.append((idx, result))
                else:
                    per_shard[idx].difference_update(wanted)
            self._refresh_node_released_view()
            logger.info(
                "[DynamoHttpServer] sglang resumed %s on %d/%d shard(s) (now released=%s)",
                widened,
                len(targets) - len(errors),
                len(targets),
                sorted(self._sglang_released_tags),
            )
            if errors:
                raise RuntimeError(
                    f"dynamo.sglang resume_memory_occupation failed on shard(s) {errors}; "
                    f"state recorded per shard, retry will skip the {len(targets) - len(errors)} that succeeded"
                )

    async def _sglang_control_fanout(self, method: str, targets: list[tuple[int, dict]]) -> list:
        """Call ``method(**kwargs)`` on the selected shards concurrently.

        Returns one entry per target, in order; a failed shard yields its exception
        instead of raising, so callers can record which shards succeeded.
        """
        clients = self._sglang_control_clients()
        if not targets:
            return []
        return await asyncio.gather(
            *[getattr(clients[idx], method)(**kw) for idx, kw in targets],
            return_exceptions=True,
        )

    async def _sglang_control_all(self, method: str, *args, **kwargs):
        """Fan a control call out to every local sglang shard, concurrently.

        Same rationale as ``_engine_method_all``'s parallel dispatch: several of
        these (weight refit, memory release) are synchronization points inside the
        engine, and serialising them across shards deadlocks when the trainer side
        is waiting on all shards to arrive together. Raises if any shard failed;
        release/resume use ``_sglang_control_fanout`` instead because they must
        record partial success.
        """
        clients = self._sglang_control_clients()
        if not clients:
            return []
        results = await asyncio.gather(
            *[getattr(c, method)(*args, **kwargs) for c in clients],
            return_exceptions=True,
        )
        errors = [(c.base_url, r) for c, r in zip(clients, results, strict=True) if isinstance(r, Exception)]
        if errors:
            raise RuntimeError(f"dynamo.sglang control call {method!r} failed on {errors}")
        return results

    def _dynamo_cfg_bool(self, key: str, default: bool) -> bool:
        value = self._dynamo_cfg().get(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _check_sleep_mode_pairing(self) -> None:
        """rollout.free_cache_engine=true requires rollout.enable_sleep_mode=true, on both engines.

        Without sleep mode neither engine actually frees anything: vLLM's sleep() is a
        silent no-op, and sglang's release_memory_occupation deregisters the worker but
        keeps the memory (its memory saver is armed by --enable-memory-saver, which
        follows enable_sleep_mode). The trainer would believe memory was freed while
        nothing happened, then OOM later with no log line naming the cause.
        """
        if bool(getattr(self.config, "free_cache_engine", False)) and not bool(
            getattr(self.config, "enable_sleep_mode", True)
        ):
            raise ValueError(
                "rollout.free_cache_engine=true requires rollout.enable_sleep_mode=true for the "
                "dynamo backend: without sleep mode vLLM sleep() is a silent no-op and sglang "
                "release_memory_occupation frees nothing (--enable-memory-saver follows "
                "enable_sleep_mode), so training would OOM with no error at the sleep site. "
                "Set both true, or both false."
            )

    def _free_engine_on_train(self) -> bool:
        """Sleep/wake of the colocated engine workers around training.

        Mirrors verl's rollout.free_cache_engine (validated in __init__), so
        the trainer-side release/resume gating and the dynamo-side engine
        sleep/wake can never silently diverge.
        """
        return self._free_engine_on_train_flag

    def _enable_rl_mode(self) -> bool:
        """Enable Dynamo's RL/TITO-friendly mode (DYN_ENABLE_RL, both engines).

        Distinct from ``dynamo.sglang.enable_rl``, which is the ``--enable-rl``
        CLI flag on the sglang worker itself (see _build_sglang_cmd).
        """
        return self._dynamo_cfg_bool("enable_rl", True)

    def _request_engine_data(self) -> bool:
        """Ask Dynamo to return vLLM engine token data via nvext.

        vllm only: dynamo populates nvext.engine_data on that path alone, which
        is why _start_engine_workers refuses an engine=sglang run that leaves
        request_completion_token_ids unset.
        """
        return self._dynamo_cfg_bool("request_engine_data", self._enable_rl_mode())

    def _request_completion_token_ids(self) -> bool:
        """Ask Dynamo to return top-level nvext.completion_token_ids."""
        return self._dynamo_cfg_bool("request_completion_token_ids", False)

    def _dynamo_env_vars(self) -> dict[str, str]:
        """Common env vars for all dynamo subprocesses on this node.

        On master we point at our own etcd/nats; on slaves we point at the
        master's (set in launch_server). Mirrors nemo-rl
        ``DynamoVllmGeneration._dynamo_env_vars`` (dynamo_generation.py:212).
        """
        if self.node_rank == 0:
            host = self._server_address
            etcd_port = self._etcd_port
            nats_port = self._nats_port
        else:
            host = self._master_address
            etcd_port = self._master_etcd_port
            nats_port = self._master_nats_port
        assert host and etcd_port and nats_port, f"dynamo env vars missing host/ports: {host}/{etcd_port}/{nats_port}"
        env = {
            "ETCD_ENDPOINTS": f"http://{host}:{etcd_port}",
            "NATS_SERVER": f"nats://{host}:{nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
            "DYN_LOG": os.environ.get(
                "DYN_LOG",
                "dynamo_llm::http::service::metrics=warn,"
                "dynamo_runtime::pipeline::network::ingress::push_handler=warn,"
                "dynamo_llm::http::service::service_v2=warn,info",
            ),
        }
        env["DYN_ENABLE_RL"] = "true" if self._enable_rl_mode() else "false"
        return env

    # ------------------------------------------------------------------ #
    # verl interface — addresses
    # ------------------------------------------------------------------ #

    def get_master_address(self):
        """Return ``(host, etcd_port, nats_port)`` for slave actors.

        Position-compatible with vLLMHttpServer.get_master_address (which
        returns ``(master_address, master_port, dp_rpc_port)``); slaves read
        the second/third values as etcd_port/nats_port.
        """
        assert self.node_rank == 0, "non-master node has no master address"
        assert self._etcd_port and self._nats_port, "etcd/nats not started yet"
        return self._server_address, self._etcd_port, self._nats_port

    def get_server_address(self):
        """Return ``(frontend_host, frontend_port)`` for the trainer.

        On master: returns this node's frontend. On slaves: returns the
        master's frontend (via cache populated in launch_server). All trainer
        ranks reach the same frontend, regardless of which node they're on.
        """
        assert self._server_port is not None, "server not launched yet"
        return self._server_address, self._server_port

    # ------------------------------------------------------------------ #
    # verl interface — launch
    # ------------------------------------------------------------------ #

    async def launch_server(
        self,
        master_address: Optional[str] = None,
        master_port: Optional[int] = None,
        dp_rpc_port: Optional[int] = None,
        start_healthcheck: bool = True,
    ):
        """Start subprocesses on this node.

        master_address / master_port / dp_rpc_port semantics differ from
        vLLM's: we re-purpose master_port for etcd_port and dp_rpc_port for
        nats_port (see get_master_address).
        """
        if self.node_rank == 0:
            await self._launch_master(start_healthcheck=start_healthcheck)
        else:
            assert master_address and master_port and dp_rpc_port, (
                f"slave node_rank={self.node_rank} requires master_address/"
                f"etcd_port/nats_port; got "
                f"({master_address}, {master_port}, {dp_rpc_port})"
            )
            self._master_address = master_address
            self._master_etcd_port = int(master_port)
            self._master_nats_port = int(dp_rpc_port)
            await self._launch_slave()

        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _launch_master(self, start_healthcheck: bool = True):
        """Master: etcd + nats + engine workers + frontend + healthcheck."""
        # Reserve ports up-front so we know all of them before starting.
        self._etcd_port = self._configured_or_allocated_port("etcd_port", bind_wildcard=True)
        self._etcd_peer_port = self._configured_or_allocated_port("etcd_peer_port", bind_wildcard=True)
        self._nats_port = self._configured_or_allocated_port("nats_port", bind_wildcard=True)
        # Frontend port: 0 = auto, else honor config.
        self._frontend_port = self._configured_or_allocated_port("frontend_http_port", bind_wildcard=True)

        self._start_etcd()
        self._start_nats()
        self._start_engine_workers()
        self._start_frontend()

        # Expose frontend to trainer.
        self._server_port = self._frontend_port
        if start_healthcheck:
            await self.wait_frontend_ready()
            # Verify control-plane reachability so refit failures surface at
            # startup instead of silently dropping weight updates mid-training.
            # Soft-fail by default; set VERL_DYNAMO_REFIT_STRICT=1 for fail-fast.
            # IMPORTANT: must run AFTER wait_frontend_ready so the engine
            # subprocesses are fully booted — neither control plane answers
            # earlier (the vllm sidecar captures its AsyncLLM only once the
            # engine is alive; sglang registers /engine/* just as late). In the
            # shared-pool path, launch_servers calls wait_frontend_ready
            # externally (start_healthcheck=False here) and runs
            # _self_test_refit_path explicitly there.
            await self._self_test_refit_path()
        logger.info(
            "[DynamoHttpServer] master ready: frontend=http://%s:%s",
            self._server_address,
            self._frontend_port,
        )

    async def _launch_slave(self):
        """Slave: engine workers only, pointing at master etcd/nats."""
        self._start_engine_workers()
        # Slave doesn't run frontend/healthcheck; trainer reaches master FE.
        # We still set _server_port so get_server_address works — it returns
        # the master frontend port (advertised by DynamoReplica via __init__).
        # DynamoReplica sets it via set_master_frontend_port below.
        # Until then, get_server_address asserts.

    # Called by DynamoReplica.launch_servers after master.get_server_address
    # returns, so all slaves answer with the same (master_host, fe_port).
    def set_master_frontend(self, host: str, port: int):
        self._server_address = host
        self._server_port = port

    def _compute_expected_workers(self) -> int:
        if self._expected_workers is not None:
            return self._expected_workers
        tp = self.config.tensor_model_parallel_size
        per_node = max(1, self.gpus_per_node // tp)
        return per_node * self.nnodes

    async def wait_frontend_ready(self, expected_workers: Optional[int] = None):
        """Wait for the frontend to see all Dynamo workers for this replica."""
        if self.node_rank != 0:
            return
        await self._healthcheck_frontend(expected_workers=expected_workers or self._compute_expected_workers())

    # ------------------------------------------------------------------ #
    # subprocess starters
    # ------------------------------------------------------------------ #

    def _start_etcd(self):
        if self._etcd_process is not None:
            return
        self._etcd_data_dir = tempfile.mkdtemp(prefix="verl_dynamo_etcd_")
        peer_url = f"http://{self._server_address}:{self._etcd_peer_port}"
        env = os.environ.copy()
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        cmd = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{self._etcd_port}",
            "--advertise-client-urls",
            f"http://{self._server_address}:{self._etcd_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{self._etcd_peer_port}",
            "--initial-advertise-peer-urls",
            peer_url,
            "--initial-cluster",
            f"default={peer_url}",
            "--data-dir",
            self._etcd_data_dir,
            "--heartbeat-interval",
            "500",
            "--election-timeout",
            "5000",
        ]
        logger.info("[DynamoHttpServer] starting etcd: %s", " ".join(cmd))
        self._etcd_process = subprocess.Popen(cmd, env=env)
        self._wait_for_etcd(_ETCD_READY_TIMEOUT_S)

    def _wait_for_etcd(self, timeout: float):
        url = f"http://localhost:{self._etcd_port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._etcd_process and self._etcd_process.poll() is not None:
                raise RuntimeError(f"etcd exited with rc={self._etcd_process.returncode} before becoming healthy")
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    logger.info("[DynamoHttpServer] etcd healthy on :%s", self._etcd_port)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"etcd did not become healthy within {timeout}s")

    def _start_nats(self):
        if self._nats_process is not None:
            return
        configured_port = int(self._dynamo_cfg().get("nats_port", 0) or 0)
        max_attempts = 1 if configured_port else 8
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            cmd = ["nats-server", "-p", str(self._nats_port)]
            logger.info(
                "[DynamoHttpServer] starting NATS (attempt %s/%s): %s",
                attempt,
                max_attempts,
                " ".join(cmd),
            )
            self._nats_process = subprocess.Popen(cmd)
            try:
                self._wait_for_process_port(
                    self._nats_process,
                    self._nats_port,
                    _NATS_READY_TIMEOUT_S,
                    "NATS",
                )
                return
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "[DynamoHttpServer] NATS failed on port %s: %s",
                    self._nats_port,
                    exc,
                )
                self._terminate_process(self._nats_process, "NATS", timeout=5)
                self._nats_process = None
                if configured_port:
                    break
                self._allocated_tcp_ports.discard(self._nats_port)
                self._nats_port = self._configured_or_allocated_port("nats_port", bind_wildcard=True)
        raise RuntimeError(f"NATS failed to start after {max_attempts} attempts: {last_error}")

    @staticmethod
    def _wait_for_port(port: int, timeout: float, label: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    logger.info("[DynamoHttpServer] %s port :%s open", label, port)
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{label} did not open port {port} within {timeout}s")

    @staticmethod
    def _wait_for_process_port(
        process: subprocess.Popen,
        port: int,
        timeout: float,
        label: str,
    ):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{label} exited with rc={process.returncode} before opening port {port}")
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    logger.info("[DynamoHttpServer] %s port :%s open", label, port)
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{label} did not open port {port} within {timeout}s")

    def _start_engine_workers(self):
        """Spawn N dynamo.vllm (or dynamo.sglang) subprocesses on this node.

        Each subprocess is one DP shard; gets a contiguous TP-slice of GPUs
        from cuda_visible_devices. CUDA_VISIBLE_DEVICES + VERL_ZMQ_BASE_TRAINER_RANK
        are passed in env so verl's vLLM WorkerExtension's _get_zmq_handle picks the
        correct global rank for ZMQ-IPC weight bucket transfer (see §11.3).

        Everything downstream of the GPU slicing is engine-dependent and gated on
        ``is_sglang`` below: the sglang path skips the ZMQ sidecar, the KV-event
        and VLLM_PORT allocations and the vLLM-only env, and instead requires
        DYN_SYSTEM_PORT so /engine/control/* is reachable.
        """
        tp = self.config.tensor_model_parallel_size
        cvd_list = [s for s in self._cuda_visible_devices.split(",") if s]
        assert len(cvd_list) % tp == 0, (
            f"GPUs ({len(cvd_list)}) on this node not divisible by TP ({tp}); cvd={self._cuda_visible_devices}"
        )
        n_local_shards = len(cvd_list) // tp
        # Persist subprocess logs under VERL_DYNAMO_LOG_DIR (e.g. a /workspace
        # path) so they survive the container, falling back to /tmp.
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        log_dir = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_node{self.node_rank}")
        os.makedirs(log_dir, exist_ok=True)

        served_model_name = (
            self._dynamo_cfg().get("served_model_name")
            or getattr(self.model_config, "served_model_name", None)
            or self.model_config.local_path
        )
        self._served_model_name = served_model_name

        worker_specs = self._worker_specs
        if worker_specs is None:
            worker_specs = [
                _DynamoWorkerSpec(
                    replica_rank=self.replica_rank,
                    cuda_visible_devices=",".join(cvd_list[shard_idx_local * tp : (shard_idx_local + 1) * tp]),
                    rank_offset=shard_idx_local * tp,
                    label=f"replica{self.replica_rank}_shard{shard_idx_local}",
                )
                for shard_idx_local in range(n_local_shards)
            ]

        is_sglang = self._is_sglang()

        for spec_idx, spec in enumerate(worker_specs):
            worker_cvd = spec.cuda_visible_devices
            # sglang needs no verl-private ZMQ sidecar: dynamo.sglang registers
            # control/* engine routes itself, reached over DYN_SYSTEM_PORT below.
            control_endpoint = None
            if not is_sglang:
                control_port = self._allocate_tcp_port(bind_wildcard=False)
                control_endpoint = f"tcp://{self._server_address}:{control_port}"
                self._control_endpoints.append(control_endpoint)

            vllm_port = None if is_sglang else self._allocate_vllm_tcpstore_port(spec_idx)
            # Allocate a registered (<32768, i16-safe) system-status port so this
            # worker exposes /metrics (on vllm incl. pass-through
            # vllm:prefix_cache_hits_total/queries_total) and the /engine/* routes.
            # Default ON, unconditionally sets DYN_SYSTEM_PORT=server_port (its fixed port avoids the i16 issue;
            # we use a fixed low port via _allocate_stable_node_port for the same
            # reason). Set enable_worker_system_metrics=false to restore the legacy
            # no-DYN_SYSTEM_PORT behaviour.
            # For sglang this port is NOT optional: /engine/control/* is the only
            # control plane there (weight refit, memory occupation, cache flush),
            # so refuse to honour enable_worker_system_metrics=false.
            enable_worker_metrics = self._dynamo_cfg_bool("enable_worker_system_metrics", True)
            if is_sglang and not enable_worker_metrics:
                raise ValueError(
                    "engine=sglang requires enable_worker_system_metrics=true: the "
                    "dynamo.sglang control plane (/engine/control/*) is served on "
                    "DYN_SYSTEM_PORT, so disabling it leaves no way to sync weights."
                )
            # Fail fast on the default token-id config for sglang (2026-08-31).
            # request_completion_token_ids defaults to False, and the other nvext
            # channel that is on by default -- engine_data -- is populated only by
            # dynamo.vllm. So an engine=sglang run that does not set this flag gets no
            # token ids and falls back to re-encoding the response text, i.e. the
            # trainer scores tokens the engine never sampled. Nothing downstream can
            # reveal that: response_length, grad_norm and the reward all keep normal
            # values. A launch-time refusal is the only place it can be caught, which
            # is why an explicit `false` is still honoured (opt out knowingly) while
            # the *unset* default is rejected.
            if is_sglang and self._dynamo_cfg().get("request_completion_token_ids") is None:
                raise ValueError(
                    "engine=sglang requires request_completion_token_ids to be set "
                    "explicitly. Add\n"
                    "  ++actor_rollout_ref.rollout.engine_kwargs.dynamo."
                    "request_completion_token_ids=true\n"
                    "(the default False yields no token ids on sglang: engine_data is "
                    "a vLLM-only channel, so generation silently trains on re-encoded "
                    "text). Set it to false explicitly if you really intend that."
                )
            # Re-checked here (not only in __init__) so a server constructed without
            # __init__ -- the CPU unit tests -- still refuses the split.
            self._check_sleep_mode_pairing()
            # KV events are what feed the KV router's index, on BOTH engines, and both
            # take the same vLLM-style --kv-events-config JSON. On sglang the engine's
            # own ZmqEventPublisher binds `endpoint` and dynamo's DynamoSglangPublisher
            # (components/src/dynamo/sglang/publisher.py:298) subscribes to it and
            # re-publishes on the event plane -- but ONLY when kv_events_config is set:
            # dynamo.sglang/args.py derives use_kv_events=False otherwise and the router
            # runs on its approximate (predict-on-route only, 120s TTL) index. Every
            # sglang kv-router run before 2026-09-09 ran that way (worker logs:
            # "Derived use_kv_events=False from kv_events_config=None"; over 100 steps
            # the router estimated 0.865 prefix hit while the engines measured 0.627).
            # Placed after the config validations so a bad sglang config fails
            # before any port is allocated. Opt out with engine_kwargs.dynamo.enable_kv_events=false (e.g. for
            # round-robin, where the events are pure overhead).
            publish_kv_events = self._kv_events_enabled()
            kv_event_port = self._allocate_kv_event_port(spec_idx) if publish_kv_events else None
            kv_events_config_json = self._build_kv_events_config_json(kv_event_port) if publish_kv_events else None
            system_metrics_port = (
                self._allocate_stable_node_port(_SYSTEM_METRICS_PORT_BASE, spec_idx, window=8)
                if enable_worker_metrics
                else None
            )

            env = os.environ.copy()
            env.update(self._dynamo_env_vars())
            env["CUDA_VISIBLE_DEVICES"] = worker_cvd
            # verl's base vLLMColocateWorkerExtension._get_zmq_handle consumes
            # this natively (int(base) + dp-resolved local rank; dynamo shards
            # run dp=1 so the resolver is the identity) — the recipe no longer
            # overrides _get_zmq_handle.
            env["VERL_ZMQ_BASE_TRAINER_RANK"] = str(spec.rank_offset)
            env[_REPLICA_RANK_ENV] = str(spec.replica_rank)
            # Match verl's native vLLM colocated path: both trainer-side
            # BucketedWeightSender and vLLM-side BucketedWeightReceiver include
            # the Ray job id in their shared /tmp IPC socket name.
            env["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()
            if not is_sglang:
                # vLLM's multiproc executor uses VLLM_PORT for its local TCPStore.
                # A node can host many TP=1 Dynamo shards, so leaving this random can
                # collide under concurrent startup.
                env["VLLM_PORT"] = str(vllm_port)
                env["VLLM_HOST_IP"] = self._server_address
                env["MASTER_ADDR"] = self._server_address
                env["MASTER_PORT"] = str(vllm_port)

            # Ensure subprocess can ``import recipe.dynamo._dynamo_vllm_with_control``
            # even when ray runtime_env doesn't propagate the driver's PYTHONPATH.
            # Compute the verl root from the location of this module; works on
            # any node since /workspace is the shared mount.
            recipe_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            existing_pp = env.get("PYTHONPATH", "")
            if recipe_root not in existing_pp.split(":"):
                env["PYTHONPATH"] = f"{recipe_root}:{existing_pp}" if existing_pp else recipe_root
            # vllm's control plane is this verl-private ZMQ sidecar rather than
            # DYN_SYSTEM_PORT, which is set further down (only when worker metrics
            # are on) from an i16-safe low port because dynamo's Rust runtime
            # parses that variable as i16 and rejects ephemeral ports >= 32768.
            if control_endpoint is not None:
                env[_CONTROL_ZMQ_ENV] = control_endpoint
            # Defensively unset any DYN_SYSTEM_* leaking from caller env.
            for k in list(env.keys()):
                if k.startswith("DYN_SYSTEM_"):
                    del env[k]
            # Opt-in worker metrics: set a low (i16-safe) DYN_SYSTEM_PORT AFTER the
            # defensive unset above, so the worker exposes /metrics. Recorded for
            # the monitoring sidecar / Prometheus to scrape engine-level KV hits.
            if system_metrics_port is not None:
                env[_DYN_SYSTEM_PORT_ENV] = str(system_metrics_port)
                worker_metrics_endpoint = f"{self._server_address}:{system_metrics_port}"
                self._worker_metrics_endpoints.append(worker_metrics_endpoint)
                self._record_worker_metrics_endpoint(worker_metrics_endpoint)
                if is_sglang:
                    host = (
                        f"[{self._server_address}]"
                        if is_valid_ipv6_address(self._server_address)
                        else self._server_address
                    )
                    self._engine_control_endpoints.append(f"http://{host}:{system_metrics_port}")
            if not is_sglang:
                # Mirrors nemo_rl/dynamo_worker.py:308-310.
                env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                env["VLLM_SKIP_P2P_CHECK"] = "1"
                env["VLLM_NO_USAGE_STATS"] = "1"
            # Deterministic block hashes across workers: without a fixed seed,
            # Python's randomized hashing can leak into block-hash derivation so
            # the same prefix hashes differently per worker → the router logs
            # "block_hash mismatch" and prefix-cache hits collapse.
            env.setdefault("PYTHONHASHSEED", "0")

            if is_sglang:
                cmd = self._build_sglang_cmd(
                    served_model_name,
                    tp,
                    nccl_port=self._allocate_tcp_port(),
                    kv_events_config_json=kv_events_config_json,
                )
            else:
                cmd = self._build_vllm_cmd(
                    served_model_name,
                    tp,
                    kv_events_config_json=kv_events_config_json,
                )

            stdout_path = os.path.join(log_dir, f"{spec.label}.log")
            stdout_fp = open(stdout_path, "w")
            self._engine_log_fps.append(stdout_fp)
            self._engine_log_paths.append(stdout_path)

            logger.info(
                "[DynamoHttpServer] starting dynamo.%s shard %s/%s "
                "(replica=%s, rank_offset=%s, GPUs=%s, vllm_port=%s, kv_event_port=%s, control=%s, "
                "DYN_ENABLE_RL=%s, request_engine_data=%s, log=%s): %s",
                self._engine_kind(),
                spec_idx,
                len(worker_specs),
                spec.replica_rank,
                spec.rank_offset,
                worker_cvd,
                vllm_port,
                kv_event_port,
                control_endpoint,
                env.get("DYN_ENABLE_RL"),
                self._request_engine_data(),
                stdout_path,
                " ".join(cmd),
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_fp,
                stderr=subprocess.STDOUT,
            )
            self._engine_processes.append(proc)

    def _record_worker_metrics_endpoint(self, endpoint: str) -> None:
        """Append a worker /metrics endpoint to a per-(replica,node) file so an
        external monitoring sidecar can discover and scrape it. The engine-level
        vllm:prefix_cache_* metrics live on the worker, not the frontend, so this
        is how the Dynamo arm becomes comparable to the vLLM arm. No-op unless
        VERL_DYNAMO_WORKER_METRICS_DIR is set."""
        out_dir = os.environ.get("VERL_DYNAMO_WORKER_METRICS_DIR")
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"r{self.replica_rank}_n{self.node_rank}.endpoints")
            with open(path, "a") as f:
                f.write(endpoint + "\n")
                f.flush()
        except Exception as exc:  # best-effort; never block worker startup
            logger.warning("[DynamoHttpServer] failed to record worker metrics endpoint %s: %s", endpoint, exc)

    def get_worker_metrics_endpoints(self) -> list[str]:
        """host:port of each engine worker's /metrics on this node (empty unless
        enable_worker_system_metrics was on)."""
        return list(self._worker_metrics_endpoints)

    def _allocate_tcp_port(self, bind_wildcard: bool = False) -> int:
        """Allocate a port for a subprocess and avoid duplicates in this actor.

        vLLM KV event publishers bind ``tcp://*:<port>``. Checking only the
        node IP can miss conflicts with wildcard listeners, so those ports are
        probed on 0.0.0.0. We also keep a local reservation set so a burst of
        shard launches does not accidentally reuse a just-released port — that
        set is also what keeps the sglang shards' ``--nccl-port`` distinct.
        """
        address = (
            "0.0.0.0" if bind_wildcard and not is_valid_ipv6_address(self._server_address) else self._server_address
        )
        for _ in range(128):
            family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
            with socket.socket(family=family, type=socket.SOCK_STREAM) as sock:
                sock.bind((address, 0))
                port = sock.getsockname()[1]
            if port in self._allocated_tcp_ports:
                continue
            self._allocated_tcp_ports.add(port)
            return port
        raise RuntimeError(f"failed to allocate unique TCP port for address={address}")

    def _can_bind_tcp_port(self, port: int, bind_wildcard: bool = False) -> bool:
        address = (
            "0.0.0.0" if bind_wildcard and not is_valid_ipv6_address(self._server_address) else self._server_address
        )
        family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
        sock = socket.socket(family=family, type=socket.SOCK_STREAM)
        try:
            sock.bind((address, port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _allocate_stable_node_port(self, base: int, shard_idx: int, window: int = 8) -> int:
        """Pick a stable node-local port for concurrently launched shards.

        Engine ports get bound in child processes, well after the parent picked
        them: vLLM's TCPStore and KV event publisher, and the DYN_SYSTEM_PORT
        system server on both engines. If the parent only probes a random free
        port and releases it, another shard can claim it before the child binds.
        Use deterministic, non-ephemeral per-node/per-shard windows to avoid
        those startup races.
        """
        replica_slot = self.replica_rank % 4
        node_slot = self.node_rank % 16
        start = base + replica_slot * 2048 + node_slot * 128 + shard_idx * window
        for port in range(start, start + window):
            if port in self._allocated_tcp_ports:
                continue
            if self._can_bind_tcp_port(port, bind_wildcard=True):
                self._allocated_tcp_ports.add(port)
                return port
        return self._allocate_tcp_port(bind_wildcard=True)

    def _allocate_vllm_tcpstore_port(self, shard_idx: int) -> int:
        """Pick a stable low port for vLLM's local TCPStore."""
        return self._allocate_stable_node_port(_VLLM_TCPSTORE_PORT_BASE, shard_idx)

    def _kv_events_enabled(self) -> bool:
        """``engine_kwargs.dynamo.enable_kv_events`` (default true): publish KV cache
        events from every engine shard so the KV router indexes real block
        residency instead of only its predict-on-route guesses."""
        return self._dynamo_cfg_bool("enable_kv_events", True)

    def _allocate_kv_event_port(self, shard_idx: int) -> int:
        """Pick a port for the engine's ZMQ KV event publisher (vLLM and sglang).

        vLLM binds this port inside a child process after model loading. Fixed
        node-local ports are easy to collide with stale subprocesses from a
        previous failed job, so use a random currently-free port by default.
        """
        if self._dynamo_cfg_bool("stable_kv_event_ports", False):
            return self._allocate_stable_node_port(_KV_EVENT_PORT_BASE, shard_idx, window=16)
        return self._allocate_tcp_port(bind_wildcard=True)

    def _configured_or_allocated_port(self, key: str, bind_wildcard: bool = False) -> int:
        configured = int(self._dynamo_cfg().get(key, 0) or 0)
        if configured:
            if configured in self._allocated_tcp_ports:
                raise RuntimeError(f"duplicate configured Dynamo port {configured} for {key}")
            self._allocated_tcp_ports.add(configured)
            return configured
        return self._allocate_tcp_port(bind_wildcard=bind_wildcard)

    @staticmethod
    def _build_kv_events_config_json(kv_event_port: int) -> str:
        return json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_event_port}",
                "enable_kv_cache_events": True,
            }
        )

    def _build_vllm_cmd(
        self,
        served_model_name: str,
        tp: int,
        kv_events_config_json: Optional[str] = None,
    ) -> list[str]:
        """Construct the dynamo.vllm CLI for one DP shard.

        We launch our own thin entrypoint instead of plain ``-m dynamo.vllm``
        so we can:
          1. inject ``--worker-extension-cls
             recipe.dynamo.dynamo_worker_extension.vLLMDynamoColocateWorkerExtension``
             so the WorkerExtension reads VERL_ZMQ_BASE_TRAINER_RANK in
             _get_zmq_handle (§11.3 of design doc).
          2. start a control ZMQ listener for engine.collective_rpc bridge.
        """
        cmd = [
            sys.executable,
            "-m",
            "recipe.dynamo._dynamo_vllm_with_control",
            "--model",
            self.model_config.local_path,
            "--served-model-name",
            served_model_name,
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]
        if self.config.max_model_len:
            cmd += ["--max-model-len", str(self.config.max_model_len)]
        if self.config.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(self.config.max_num_batched_tokens)]
        if self.config.max_num_seqs:
            cmd += ["--max-num-seqs", str(self.config.max_num_seqs)]
        if self.config.dtype:
            cmd += ["--dtype", self.config.dtype]
        if self.model_config.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if self.config.enforce_eager:
            cmd += ["--enforce-eager"]
        if self.config.enable_chunked_prefill:
            cmd += ["--enable-chunked-prefill"]
        if self.config.enable_prefix_caching:
            cmd += ["--enable-prefix-caching"]
        if self.config.enable_sleep_mode:
            cmd += ["--enable-sleep-mode"]
        # Worker-extension class is the plumbing for verl's update_weights_from_ipc.
        cmd += [
            "--worker-extension-cls",
            "recipe.dynamo.dynamo_worker_extension.vLLMDynamoColocateWorkerExtension",
        ]
        # For TP=1, avoid vLLM's multiproc executor. Dynamo already launches
        # one process per shard, and an extra local WorkerProc/TCPStore has
        # caused EADDRINUSE races during concurrent startup on multi-shard
        # nodes. TP>1 still needs a local distributed executor.
        executor_backend = self._dynamo_cfg().get("distributed_executor_backend")
        if not executor_backend:
            executor_backend = "uni" if tp == 1 else "mp"
        cmd += ["--distributed-executor-backend", str(executor_backend)]
        # Same contract as the sglang builder: no config -> no flag (enable_kv_events=false).
        if kv_events_config_json:
            cmd += ["--kv-events-config", kv_events_config_json]
        # Pass through extra args from rollout.engine_kwargs.dynamo.extra_args.
        extra = self._dynamo_cfg().get("extra_args") or []
        if isinstance(extra, list):
            cmd += [str(x) for x in extra]
        return cmd

    def _build_sglang_cmd(
        self,
        served_model_name: str,
        tp: int,
        nccl_port: Optional[int] = None,
        kv_events_config_json: Optional[str] = None,
    ) -> list[str]:
        """Construct the ``dynamo.sglang`` CLI for one DP shard.

        Unlike the vLLM path this launches the stock entrypoint — no verl wrapper —
        because the control plane we need is already registered by
        ``handler_base.register_engine_routes`` and reachable over DYN_SYSTEM_PORT.

        ``dynamo.sglang.args`` forwards the full ``ServerArgs.add_cli_args`` surface,
        so anything not mapped below can still be passed through
        ``engine_kwargs.dynamo.extra_args``.
        """
        cmd = [
            sys.executable,
            "-m",
            "dynamo.sglang",
            "--model-path",
            self.model_config.local_path,
            "--served-model-name",
            served_model_name,
            "--tp-size",
            str(tp),
            "--mem-fraction-static",
            str(self.config.gpu_memory_utilization),
        ]
        # Pin the torch.distributed rendezvous port (2026-08-31). Left unset,
        # ServerArgs.nccl_port is None and SGLang picks its own; the four shards on a
        # node choose concurrently and collide, which surfaces as
        #   DistNetworkError: server socket failed to listen. port: 31981, EADDRINUSE
        # inside the scheduler, then SIGQUIT, and the watchdog reports the useless
        # "exited rc=-9" (job 3671). _allocate_tcp_port keeps a per-actor reservation
        # set, and all shards on a node share one actor, so ports are distinct by
        # construction rather than by luck.
        if nccl_port is not None:
            cmd += ["--nccl-port", str(nccl_port)]
        if self.config.max_model_len:
            cmd += ["--context-length", str(self.config.max_model_len)]
        if self.config.max_num_batched_tokens:
            cmd += ["--chunked-prefill-size", str(self.config.max_num_batched_tokens)]
        if self.config.max_num_seqs:
            cmd += ["--max-running-requests", str(self.config.max_num_seqs)]
        if self.config.dtype:
            cmd += ["--dtype", self.config.dtype]
        if self.model_config.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if self.config.enforce_eager:
            # SGLang's inverse of vLLM's --enforce-eager.
            cmd += ["--disable-cuda-graph"]
        if not self.config.enable_prefix_caching:
            # SGLang's radix cache is on by default; vLLM's prefix cache is not.
            cmd += ["--disable-radix-cache"]
        if self.config.enable_sleep_mode:
            # Without this SGLang's torch_memory_saver never arms, and
            # release_memory_occupation silently frees nothing.
            cmd += ["--enable-memory-saver"]

        # Preserve the worker's RL flag. After Dynamo PR #13951 it no longer
        # registers call_tokenizer_manager; cache flush is configured below.
        if self._sglang_cfg().get("enable_rl", True):
            cmd += ["--enable-rl"]

        # KV-router page size. ThunderAgent's router_block_size must match, same
        # as it must match vLLM's --block-size on the other path.
        page_size = self._sglang_cfg().get("page_size")
        if page_size is None:
            ta_cfg = self._dynamo_cfg().get("thunderagent", {}) or {}
            page_size = ta_cfg.get("router_block_size")
        if page_size is not None:
            cmd += ["--page-size", str(page_size)]

        # KV events for the KV router: same JSON as the vLLM path. sglang's
        # ZmqEventPublisher binds the endpoint (offset per DP rank by dynamo's
        # publisher) and dynamo.sglang flips use_kv_events=True in the worker's
        # runtime config from this flag alone -- there is no other switch.
        if kv_events_config_json:
            cmd += ["--kv-events-config", kv_events_config_json]

        # Token-in/token-out: skip detokenization so the trainer scores exactly the
        # ids the engine produced. NB dynamo's llm_engine.py force-disables this
        # when its metrics hook needs a tokenizer, so treat it as a request.
        if self._sglang_cfg().get("skip_tokenizer_init", False):
            cmd += ["--skip-tokenizer-init"]

        extra = self._sglang_cfg().get("extra_args") or self._dynamo_cfg().get("extra_args") or []
        extra = [str(x) for x in extra] if isinstance(extra, list) else []

        # Readiness and weight refit both need a native cache-flush endpoint.
        # Honor an explicit public-path override without registering it twice.
        routes = [extra[i + 1] for i, arg in enumerate(extra[:-1]) if arg == "--engine-route"]
        routes += [arg.split("=", 1)[1] for arg in extra if arg.startswith("--engine-route=")]
        if not any(route.split(":", 1)[0].split("=", 1)[0] == "flush_cache" for route in routes):
            cmd += ["--engine-route", "flush_cache:tm"]

        # Attention backend: default to flashinfer, mirroring verl's native sglang
        # server (async_sglang_server.py). SGLang's own default on Hopper is fa3,
        # and fa3 with the KV-router page size (16) decodes ~8% slower per step
        # (Qwen2.5-0.5B, H100, 32x512 fixed-length: fa3/page16 8.09 ms/step vs
        # flashinfer 7.44; native verl 7.47). That gap was the whole
        # "dynamo generation is 9% slower" delta -- the frontend hop itself is <1%.
        # verl also avoids fa3 because fa3 + cuda graph is broken on sglang>=0.5.12.
        # Explicit engine_kwargs.dynamo.sglang.attention_backend or an
        # --attention-backend in extra_args wins.
        attention_backend = self._sglang_cfg().get("attention_backend")
        if attention_backend is None and not any(x.startswith("--attention-backend") for x in extra):
            attention_backend = "flashinfer"
        if attention_backend:
            cmd += ["--attention-backend", str(attention_backend)]

        cmd += extra
        return cmd

    def _start_frontend(self):
        if self._frontend_process is not None:
            return
        env = os.environ.copy()
        env.update(self._dynamo_env_vars())
        # In RL mode the frontend binds an extra "RL worker discovery" listener
        # on DYN_RL_PORT (default 8001, dynamo service_v2.rs). With two pools
        # on one node (separate_async: hybrid on trainer GPUs + standalone on
        # rollout GPUs) the fixed default collides and the second frontend
        # exits rc=1 — allocate a free port per pool.
        if "DYN_RL_PORT" not in env:
            env["DYN_RL_PORT"] = str(self._allocate_tcp_port(bind_wildcard=True))

        cmd = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(self._frontend_port),
            "--http-host",
            "0.0.0.0",
            "--router-mode",
            self._router_mode,
            "--discovery-backend",
            "etcd",
            "--namespace-prefix",
            self._namespace,
        ]
        cmd += self._frontend_router_args()
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        log_path = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_frontend.log")
        self._frontend_log_fp = open(log_path, "w")
        logger.info(
            "[DynamoHttpServer] starting dynamo.frontend on :%s (DYN_ENABLE_RL=%s, request_engine_data=%s, log=%s): %s",
            self._frontend_port,
            env.get("DYN_ENABLE_RL"),
            self._request_engine_data(),
            log_path,
            " ".join(cmd),
        )
        self._frontend_process = subprocess.Popen(cmd, env=env, stdout=self._frontend_log_fp, stderr=subprocess.STDOUT)

    def _frontend_router_args(self) -> list[str]:
        """Return Dynamo frontend router tuning args.

        Targets Dynamo v1.2.0's ``dynamo.frontend`` router CLI:
          * ``--active-decode-blocks-threshold`` is now a *fraction* of KV block
            utilization and must be in ``[0.0, 1.0]`` (the frontend rejects
            out-of-range values); pass the literal ``"None"`` to disable the
            check. The prefill thresholds likewise accept ``"None"`` to disable.
            We default all three to disabled so the KV router routes purely by
            cache affinity instead of shedding load — the original intent behind
            the previous (now out-of-range) ``1000.0`` sentinels.
          * ``--router-predict-on-route`` (boolean) was removed in v1.2.0 and
            replaced in v1.3.0 (the version installed here) by
            ``--router-predicted-ttl-secs <ttl>``. We enable route-time
            speculative insert BY DEFAULT — essential for RL, where n=16
            same-burst siblings + per-step KV-cache clears otherwise cause
            ParentBlockNotFound storms. Knobs: ``router_predict_on_route``
            (bool, default true) and ``router_predicted_ttl_secs`` (default
            120.0; set to None/0 to disable).

        Legacy configs in this repo documented the pre-v1.2.0 "disable" sentinels
        (e.g. ``active_decode_blocks_threshold: 1000.0``). Those are out of v1.2.0's
        valid range and would now make the frontend exit before healthcheck, so we
        normalize out-of-range / sentinel values back to ``"None"`` (disabled).
        """
        if self._router_mode != "kv":
            # Session affinity is a plain router option (dynamo router_args.py), not
            # a KV-router one: the frontend wraps its simple router in
            # SessionAffinityPushRouter too, so emit it in every mode.
            return self._session_affinity_args() + self._frontend_extra_args()

        cfg = self._dynamo_cfg()
        args: list[str] = []

        # Dynamo v1.3.0 exposes --router-predicted-ttl-secs <ttl>: speculatively insert the
        # routed prefix (short TTL) so siblings / post-clear requests see it
        # immediately; the real event later promotes it. Kept independent of enable_nemo_router_tuning so
        # the fix applies even when threshold tuning is off. Disable via
        # router_predict_on_route=false or router_predicted_ttl_secs=None/0.
        if self._dynamo_cfg_bool("router_predict_on_route", True):
            predicted_ttl = cfg.get("router_predicted_ttl_secs", 120.0)
            if not self._is_disabled_threshold(predicted_ttl):
                try:
                    ttl_val = float(predicted_ttl)
                except (TypeError, ValueError):
                    ttl_val = 0.0
                if ttl_val > 0:
                    args += ["--router-predicted-ttl-secs", str(ttl_val)]

        # Router load-shedding thresholds, only when nemo router tuning is on.
        # The installed dynamo.frontend parses these with type=float and rejects
        # the literal "None"; OMIT a flag when it normalizes to disabled (absent
        # = frontend default = route by cache affinity, no load shedding).
        if self._dynamo_cfg_bool("enable_nemo_router_tuning", True):
            decode = self._normalize_decode_blocks_threshold(cfg.get("active_decode_blocks_threshold", "None"))
            if decode != "None":
                args += ["--active-decode-blocks-threshold", decode]
            prefill = self._normalize_prefill_threshold(
                "active_prefill_tokens_threshold", cfg.get("active_prefill_tokens_threshold", "None")
            )
            if prefill != "None":
                args += ["--active-prefill-tokens-threshold", prefill]
            prefill_frac = self._normalize_prefill_threshold(
                "active_prefill_tokens_threshold_frac",
                cfg.get("active_prefill_tokens_threshold_frac", "None"),
            )
            if prefill_frac != "None":
                args += ["--active-prefill-tokens-threshold-frac", prefill_frac]
        return args + self._session_affinity_args() + self._frontend_extra_args()

    def _session_affinity_ttl_secs(self) -> Optional[int]:
        """``engine_kwargs.dynamo.router_session_affinity_ttl_secs`` -> seconds, or None.

        None / 0 / "None" leave affinity off (the frontend default). With a TTL the
        frontend pins every request carrying the same ``x-dynamo-session-id`` to
        the worker that served the first one, for TTL seconds after the last hit.
        For verl that key is the trajectory's request_id (see _frontend_headers),
        so the later turns of a multi-turn rollout land on the worker that already
        holds their prefix regardless of what the KV index knows -- the index only
        learns about blocks from KV events (see enable_kv_events) and from
        predict-on-route entries that expire after router_predicted_ttl_secs.
        The frontend accepts 1..=31_536_000 (dynamo router_args.py); reject other
        values here so a typo fails at config time, not as a frontend that never
        turns healthy.
        """
        value = self._dynamo_cfg().get("router_session_affinity_ttl_secs")
        if self._is_disabled_threshold(value):
            return None
        try:
            ttl = int(float(value))
        except (TypeError, ValueError) as e:
            raise ValueError(
                "rollout.engine_kwargs.dynamo.router_session_affinity_ttl_secs must be a number "
                f"of seconds in 1..31536000, or None, got {value!r}"
            ) from e
        if ttl <= 0:
            return None
        if ttl > 31_536_000:
            raise ValueError(
                f"rollout.engine_kwargs.dynamo.router_session_affinity_ttl_secs={value!r} exceeds the "
                "frontend's maximum of 31536000 seconds"
            )
        return ttl

    def _session_affinity_args(self) -> list[str]:
        ttl = self._session_affinity_ttl_secs()
        return ["--router-session-affinity-ttl-secs", str(ttl)] if ttl is not None else []

    def _frontend_headers(self, request_id: str) -> dict[str, str]:
        """Per-request headers for the frontend POST.

        ``x-dynamo-session-id`` is the frontend's session-affinity key
        (lib/llm/src/protocols/agents.rs). verl's tool agent loop mints ONE
        request_id per trajectory and reuses it across turns
        (tool_agent_loop.py), so the id we already send as X-Request-Id doubles as
        the session key. Only sent when affinity is configured: the same header
        also populates the frontend's agent_context, so keep the wire identical
        to before for everyone who has not opted in.
        """
        headers = {"X-Request-Id": str(request_id)}
        if self._session_affinity_ttl_secs() is not None:
            headers["x-dynamo-session-id"] = str(request_id)
        return headers

    @staticmethod
    def _is_disabled_threshold(value: Any) -> bool:
        """True if the configured value means "disable this check"."""
        if value is None:
            return True
        return str(value).strip().lower() in {"none", "null", ""}

    def _normalize_decode_blocks_threshold(self, value: Any) -> str:
        """Clamp ``--active-decode-blocks-threshold`` to v1.2.0's [0.0, 1.0].

        The frontend rejects out-of-range fractions, and legacy repo configs
        still carry the pre-v1.2.0 ``1000.0`` "disable" sentinel. Treat anything
        outside the valid range (including that sentinel) as disabled (``None``).
        """
        if self._is_disabled_threshold(value):
            return "None"
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "active_decode_blocks_threshold=%r is not a number; disabling the check.",
                value,
            )
            return "None"
        if 0.0 <= parsed <= 1.0:
            return str(parsed)
        logger.warning(
            "active_decode_blocks_threshold=%r is outside Dynamo v1.2.0's valid "
            "[0.0, 1.0] range (likely a legacy disable sentinel); disabling the check.",
            value,
        )
        return "None"

    def _normalize_prefill_threshold(self, key: str, value: Any) -> str:
        """Normalize prefill busy thresholds for v1.2.0.

        v1.2.0 puts no upper bound on the prefill thresholds, so unlike the
        decode-blocks fraction they never crash the frontend — the legacy repo
        sentinels (``1000000000000`` / ``1000.0``) still parse and effectively
        disable the check. So we only coerce explicit ``None``/empty values to
        ``"None"`` and pass any valid number through unchanged.
        """
        if self._is_disabled_threshold(value):
            return "None"
        try:
            float(value)
        except (TypeError, ValueError):
            logger.warning("%s=%r is not a number; disabling the check.", key, value)
            return "None"
        return str(value)

    def _frontend_extra_args(self) -> list[str]:
        extra = self._dynamo_cfg().get("frontend_extra_args") or []
        if isinstance(extra, str):
            return [extra]
        if isinstance(extra, list):
            return [str(x) for x in extra]
        raise TypeError(
            f"rollout.engine_kwargs.dynamo.frontend_extra_args must be a list or string, got {type(extra).__name__}"
        )

    async def _healthcheck_frontend(self, expected_workers: int):
        url = f"http://localhost:{self._frontend_port}/health"
        deadline = time.monotonic() + _FRONTEND_READY_TIMEOUT_S
        last_err: Optional[str] = None
        while time.monotonic() < deadline:
            self._raise_if_subprocess_died()
            try:
                # use blocking requests in a thread to avoid pulling aiohttp here
                resp = await asyncio.to_thread(requests.get, url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    instances = data.get("instances") or []
                    n_gen = sum(1 for i in instances if i.get("endpoint") == "generate")
                    if n_gen >= expected_workers:
                        logger.info(
                            "[DynamoHttpServer] frontend healthy: %s/%s workers registered",
                            n_gen,
                            expected_workers,
                        )
                        return
                    last_err = f"only {n_gen}/{expected_workers} workers registered"
                else:
                    last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_FRONTEND_READY_POLL_S)
        raise RuntimeError(
            f"dynamo frontend not healthy within {_FRONTEND_READY_TIMEOUT_S}s "
            f"(expected {expected_workers} workers; last={last_err})"
        )

    # ------------------------------------------------------------------ #
    # watchdog
    # ------------------------------------------------------------------ #

    def _raise_if_subprocess_died(self):
        for attr, name, _ in _SUBPROCESS_REGISTRY:
            proc = getattr(self, attr, None)
            if proc is None:
                continue
            if isinstance(proc, list):
                for i, p in enumerate(proc):
                    if p.poll() is not None:
                        log_hint = ""
                        if name == "engine_workers" and i < len(self._engine_log_paths):
                            log_hint = f" (log={self._engine_log_paths[i]})"
                        raise RuntimeError(f"dynamo {name}[{i}] exited rc={p.returncode}{log_hint}")
            else:
                if proc.poll() is not None:
                    raise RuntimeError(f"dynamo {name} exited rc={proc.returncode}")

    async def _watchdog_loop(self):
        try:
            while not self._shutdown_requested:
                self._raise_if_subprocess_died()
                await asyncio.sleep(_WATCHDOG_INTERVAL_S)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("[DynamoHttpServer] watchdog detected death")
            raise

    # ------------------------------------------------------------------ #
    # verl interface — generate / RPC
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
        priority: int = 0,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ):
        """Dispatch generation through the Dynamo frontend HTTP router.

        The actor only manages the subprocess stack; token generation goes
        through the OpenAI-style frontend so Dynamo can route across registered
        workers. Engine-agnostic: the frontend hides which engine is behind it.

        Remaining kwargs from verl clients (e.g. priority scheduling hints)
        are accepted and ignored; Dynamo's KV router owns request scheduling.
        Multimodal inputs are rejected loudly: silently dropping them would
        train on text-only prompts while reporting success.
        """
        if image_data is not None or video_data is not None or audio_data is not None or mm_processor_kwargs:
            return self._build_token_output(
                stop_reason="error: Dynamo frontend generate does not support multimodal inputs",
            )

        # Partial rollout: while the engines are paused (abort_all_requests),
        # answer immediately with an empty aborted output instead of parking
        # the coroutine. FullyAsyncLLMServerClient polls with a 1s retry loop,
        # so parked coroutines would only pin this actor's max_concurrency
        # slots — with every in-flight trajectory parked, the control calls
        # that eventually open the gate (resume_generation) could never be
        # scheduled, deadlocking the first on_step_end.
        include_log_probs = bool(sampling_params.get("logprobs", False))
        if not self._generation_resumed.is_set():
            return self._aborted_output(include_log_probs)

        if self._use_direct_generate():
            return await self._generate_direct(prompt_ids, sampling_params, request_id)

        try:
            request_id = request_id or f"dynamo-{time.time_ns()}"
            # verl reuses one request_id for every turn of a trajectory; it is the
            # session-affinity key (_frontend_headers), not a per-request id. A
            # trajectory running Claude Code subagents has several turns in flight
            # at once, so the id the frontend tracks per request must be unique.
            # The caller's id stays as a readable prefix.
            http_request_id = f"{request_id}-{uuid.uuid4().hex}"
            payload = self._build_frontend_completion_payload(prompt_ids, sampling_params, http_request_id)
            status, body_text = await self._frontend_post(payload, request_id, http_request_id)
            if status == 400 and "Unsupported parameter" in body_text:
                fallback_payload = self._drop_frontend_extension_fields(payload)
                if fallback_payload is not payload:
                    logger.warning(
                        "Dynamo frontend rejected optional request extension fields; "
                        "retrying without them (request_id=%s, removed=%s)",
                        request_id,
                        sorted(set(payload) - set(fallback_payload)),
                    )
                    payload = fallback_payload
                    status, body_text = await self._frontend_post(payload, request_id, http_request_id)
            if status != 200:
                raise RuntimeError(
                    "Dynamo frontend /v1/completions failed "
                    f"status={status} body={body_text[:2000]!r} "
                    f"payload_summary={self._payload_debug_summary(payload)}"
                )
            return self._completion_response_to_token_output(json.loads(body_text), include_log_probs=include_log_probs)
        except Exception:
            if not self._generation_resumed.is_set():
                # The engines are paused (abort_all_requests): an in-flight
                # request whose frontend response errored out was almost
                # certainly killed by the pause. Report it as aborted-empty so
                # FullyAsyncLLMServerClient retries after resume instead of
                # failing the trajectory. (A clean abort response still
                # returns partial tokens via finish_reason="abort".)
                logger.warning(
                    "[generate] frontend dispatch failed while engines are paused; treating as aborted (request_id=%s)",
                    request_id,
                )
                return self._aborted_output(include_log_probs)
            logger.exception("[generate] frontend dispatch failed (request_id=%s)", request_id)
            raise

    def _frontend_completions_url(self) -> str:
        assert self._server_port is not None, "frontend server not ready"
        host = f"[{self._server_address}]" if is_valid_ipv6_address(self._server_address) else self._server_address
        return f"http://{host}:{self._server_port}/v1/completions"

    async def _get_http_session(self):
        """Lazily create one keep-alive aiohttp session on the actor loop.

        A single pooled, fully async client lets the async actor keep hundreds
        of turn-level requests in flight concurrently, instead of serializing
        them through the default thread pool used by
        ``asyncio.to_thread(requests.post)``. The connection limit defaults to
        unlimited (0) so the Dynamo frontend/KV router — not this client — owns
        load shedding; it can be capped via
        ``rollout.engine_kwargs.dynamo.frontend_connection_limit``.
        """
        if self._http_session is not None and not self._http_session.closed:
            return self._http_session
        async with self._http_session_lock:
            if self._http_session is not None and not self._http_session.closed:
                return self._http_session
            import aiohttp

            limit = int(self._dynamo_cfg().get("frontend_connection_limit", 0))
            connector = aiohttp.TCPConnector(
                limit=limit,
                limit_per_host=limit,
                ttl_dns_cache=300,
            )
            self._http_session = aiohttp.ClientSession(connector=connector)
            return self._http_session

    async def _frontend_post(
        self, payload: dict[str, Any], request_id: str, http_request_id: Optional[str] = None
    ) -> tuple[int, str]:
        """POST one completion to the frontend; return ``(status, body_text)``.

        ``request_id`` keys the session headers; ``http_request_id``, when given,
        is the unique id of this one request.
        """
        import aiohttp

        headers = self._frontend_headers(request_id)
        if http_request_id is not None:
            headers["X-Request-Id"] = http_request_id
        session = await self._get_http_session()
        timeout = aiohttp.ClientTimeout(total=self._frontend_request_timeout_s())
        async with session.post(
            self._frontend_completions_url(),
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            return resp.status, await resp.text()

    def _frontend_request_timeout_s(self) -> float:
        value = self._dynamo_cfg().get("request_timeout_s", 600)
        try:
            timeout = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"rollout.engine_kwargs.dynamo.request_timeout_s must be a positive number, got {value!r}"
            ) from e
        if timeout <= 0:
            raise ValueError(f"rollout.engine_kwargs.dynamo.request_timeout_s must be positive, got {value!r}")
        return timeout

    @staticmethod
    def _drop_frontend_extension_fields(payload: dict[str, Any]) -> dict[str, Any]:
        extension_keys = {"request_id", "return_tokens_as_token_ids"}
        if not any(key in payload for key in extension_keys):
            return payload
        return {key: value for key, value in payload.items() if key not in extension_keys}

    def _build_frontend_completion_payload(self, prompt_ids, sampling_params, request_id: str) -> dict[str, Any]:
        from verl.utils.tokenizer import normalize_token_ids

        tokenizer = getattr(self.model_config, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("model_config.tokenizer is required for Dynamo frontend generation")
        model = self._served_model_name or self.model_config.local_path
        prompt_token_ids = normalize_token_ids(prompt_ids)
        sp = dict(sampling_params)
        max_tokens = sp.pop("max_tokens", None) or sp.pop("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = max(
                1,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_token_ids),
                ),
            )
        # Tool observations grow the prompt between turns. Match the direct
        # engine path by bounding explicit sampling limits as well as defaults.
        max_possible_tokens = max(1, self.config.prompt_length + self.config.response_length - len(prompt_token_ids))
        max_tokens = min(int(max_tokens), max_possible_tokens)
        sp.pop("logprobs", None)
        nvext = sp.pop("nvext", None)
        payload: dict[str, Any] = {
            "model": model,
            # Dynamo's OpenAI completions frontend accepts token-id prompts for
            # either engine, so we keep the same token-in path the native vLLM
            # backend uses rather than sending text.
            "prompt": prompt_token_ids,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        # OpenAI maps engine cancellation to "stop". Request the lossless
        # reason so async clients resume partial rollouts after weight updates.
        nvext_fields = ["detailed_finish_reason"]
        if self._request_engine_data():
            nvext_fields.append("engine_data")
        if self._request_completion_token_ids():
            nvext_fields.append("completion_token_ids")
        if nvext_fields:
            payload["nvext"] = self._merge_nvext_extra_fields(nvext, nvext_fields)
        elif nvext is not None:
            payload["nvext"] = nvext
        if self._dynamo_cfg_bool("include_payload_request_id", False):
            payload["request_id"] = str(request_id)
        return_tokens_as_token_ids = self._dynamo_cfg_bool("return_tokens_as_token_ids", False)
        if return_tokens_as_token_ids:
            payload["return_tokens_as_token_ids"] = True
        # Dynamo v1.2.0 only emits the ``token_id:<id>`` strings (our token-out
        # path) inside ``choice.logprobs.tokens``, and that array is dropped
        # entirely unless ``logprobs`` is requested. So requesting token ids
        # implies requesting logprobs. ``force_logprobs_for_token_ids`` is kept
        # as an independent override for callers that want logprobs without the
        # token-id formatting.
        if (
            sampling_params.get("logprobs", False)
            or return_tokens_as_token_ids
            or self._dynamo_cfg_bool("force_logprobs_for_token_ids", False)
        ):
            # Native vLLM uses SamplingParams(logprobs=0) to return generated
            # token logprobs only. The Dynamo OpenAI frontend can spend extra
            # time formatting logprob payloads; that is the cost of strict
            # token-in/token-out parity.
            payload["logprobs"] = 0
        sp.pop("prompt_logprobs", None)
        for key, value in sp.items():
            if value is not None:
                payload[key] = value
        return payload

    def _use_direct_generate(self) -> bool:
        value = self._dynamo_cfg().get("direct_generate", os.getenv("VERL_DYNAMO_DIRECT_GENERATE", "0"))
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    async def _generate_direct(self, prompt_ids, sampling_params, request_id):
        """Generate through the per-shard control sidecar instead of the frontend.

        This is primarily for smoke tests and debugging ai-dynamo/vLLM
        integration. It still exercises the spawned Dynamo vLLM shards but
        bypasses the OpenAI frontend path that has been observed to hang.

        vllm only — there are no control sidecars on the sglang path, so the
        guard below always trips there.
        """
        if not self._control_endpoints:
            raise RuntimeError("direct_generate=True requires Dynamo control sidecars")

        import pickle

        import zmq
        import zmq.asyncio

        from verl.utils.tokenizer import normalize_token_ids

        tokenizer = getattr(self.model_config, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("model_config.tokenizer is required for direct Dynamo generation")

        prompt_token_ids = normalize_token_ids(prompt_ids)
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        include_log_probs = bool(sampling_params.get("logprobs", False))
        direct_sampling_params = self._build_direct_sampling_params(prompt_token_ids, sampling_params)

        async with self._direct_generate_lock:
            endpoint = self._control_endpoints[self._direct_generate_idx % len(self._control_endpoints)]
            self._direct_generate_idx += 1

        req = {
            "kind": "generate_direct",
            "kwargs": {
                "token_ids": prompt_token_ids,
                "prompt_text": prompt_text,
                "sampling_params": direct_sampling_params,
                "request_id": request_id or f"direct-{time.time_ns()}",
                "include_log_probs": include_log_probs,
            },
        }

        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(endpoint)
            await sock.send(pickle.dumps(req))
            timeout = self._direct_request_timeout_s()
            reply_bytes = await asyncio.wait_for(sock.recv(), timeout=timeout)
            reply = pickle.loads(reply_bytes)
            if not reply.get("ok"):
                raise RuntimeError(f"direct_generate @ {endpoint} failed: {reply.get('error')}")
            result = reply.get("result") or {}
            token_ids = normalize_token_ids(result.get("token_ids") or [])
            if not token_ids:
                raise RuntimeError(f"direct_generate @ {endpoint} returned no tokens: {result}")
            log_probs = result.get("log_probs") if include_log_probs else None
            return self._build_token_output(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason=self._map_finish_reason(result.get("finish_reason")),
            )
        except Exception:
            logger.exception("[generate] direct sidecar request failed (request_id=%s)", request_id)
            raise
        finally:
            sock.close()

    def _build_direct_sampling_params(
        self, prompt_token_ids: list[int], sampling_params: dict[str, Any]
    ) -> dict[str, Any]:
        sp = dict(sampling_params)
        max_tokens = sp.pop("max_tokens", None) or sp.pop("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = max(
                1,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_token_ids),
                ),
            )
        max_possible_tokens = max(1, self.config.prompt_length + self.config.response_length - len(prompt_token_ids))
        sp["max_tokens"] = int(max(1, min(max_tokens, max_possible_tokens)))
        sp["logprobs"] = 0 if sp.pop("logprobs", False) else None
        sp.pop("prompt_logprobs", None)
        sp.setdefault("repetition_penalty", getattr(self.config, "repetition_penalty", 1.0))
        return {key: value for key, value in sp.items() if value is not None}

    def _direct_request_timeout_s(self) -> float:
        value = self._dynamo_cfg().get("direct_request_timeout_s", self._frontend_request_timeout_s())
        try:
            timeout = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"rollout.engine_kwargs.dynamo.direct_request_timeout_s must be a positive number, got {value!r}"
            ) from e
        if timeout <= 0:
            raise ValueError(f"rollout.engine_kwargs.dynamo.direct_request_timeout_s must be positive, got {value!r}")
        return timeout

    @staticmethod
    def _payload_debug_summary(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            prompt_summary = {
                "type": "list",
                "len": len(prompt),
                "head": prompt[:8],
            }
        else:
            prompt_summary = {
                "type": type(prompt).__name__,
                "len": len(prompt) if hasattr(prompt, "__len__") else None,
            }
        return {
            "keys": sorted(payload.keys()),
            "model": payload.get("model"),
            "request_id": payload.get("request_id"),
            "prompt": prompt_summary,
            "max_tokens": payload.get("max_tokens"),
            "logprobs": payload.get("logprobs"),
            "return_tokens_as_token_ids": payload.get("return_tokens_as_token_ids"),
            "nvext_extra_fields": (
                payload.get("nvext", {}).get("extra_fields") if isinstance(payload.get("nvext"), dict) else None
            ),
        }

    def _completion_response_to_token_output(self, data: dict[str, Any], include_log_probs: bool = False):
        from verl.utils.tokenizer import normalize_token_ids

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"frontend response has no choices: {data}")
        choice = choices[0]
        if "text" in choice:
            text = choice.get("text") or ""
        else:
            text = ((choice.get("message") or {}).get("content")) or ""
        tokenizer = getattr(self.model_config, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("model_config.tokenizer is required for Dynamo frontend generation")
        self._log_engine_data_token_ids_status(choice, data)
        finish_reason = self._completion_finish_reason(choice, data)
        is_aborted = self._map_finish_reason(finish_reason) == "aborted"

        token_ids = self._extract_completion_token_ids(
            choice, data, tokenizer, allow_text_mapping=not include_log_probs
        )
        used_text_fallback = token_ids is None
        if token_ids is None:
            if is_aborted:
                # Cancellation before the first token can omit the token-ID
                # extension entirely. There is no sampled text to reconstruct.
                return self._aborted_output(include_log_probs)
            if self._request_completion_token_ids():
                # Token ids were explicitly requested and the frontend still returned
                # none: re-encoding the text would make the trainer score tokens the
                # engine never sampled, with every metric staying plausible. Fail.
                raise RuntimeError(
                    "Dynamo frontend response carried no parseable token ids although "
                    "request_completion_token_ids=true; refusing to re-encode the text. "
                    f"Check the dynamo build/frontend flags. Response: {str(data)[:500]}"
                )
            self._text_reencode_hits = getattr(self, "_text_reencode_hits", 0) + 1
            hits = self._text_reencode_hits
            if hits <= 3 or hits % 100 == 0:
                logger.error(
                    "Dynamo frontend response did not include parseable token ids; falling back to "
                    "re-encoding the response text (hit #%d). The trainer is scoring re-tokenized text, "
                    "not the engine's sampled tokens. Set "
                    "engine_kwargs.dynamo.request_completion_token_ids=true for token-in/token-out.",
                    hits,
                )
            token_ids = normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))
        if not token_ids:
            if is_aborted:
                return self._aborted_output(include_log_probs)
            raise RuntimeError(f"Dynamo frontend returned an empty completion: {data}")
        # Guard against ai-dynamo/dynamo#14302: from 1.4.1 the sglang worker strips
        # the matched stop token from completion_token_ids on finish_reason=stop
        # while usage.completion_tokens still counts it. The trainer would then
        # never see EOS -- the policy cannot learn to stop and response_length
        # drifts to max_tokens -- with every other metric looking healthy. This
        # checkout (1.3.0 @ 94accc7389) predates the regression (probe 2026-09-08:
        # last id 151645, lengths equal on sglang direct and both dynamo modes);
        # an upgrade that reintroduces it now fails on the first response, not
        # fifty steps later. Aborted responses are exempt: their arrays are partial.
        usage_tokens = (data.get("usage") or {}).get("completion_tokens")
        if (
            not used_text_fallback
            and finish_reason in ("stop", "length")
            and isinstance(usage_tokens, int)
            and usage_tokens != len(token_ids)
        ):
            raise RuntimeError(
                f"Dynamo frontend returned {len(token_ids)} completion token ids but "
                f"usage.completion_tokens={usage_tokens} (finish_reason={finish_reason!r}): the token "
                "array is not the sampled sequence. Known cause: ai-dynamo/dynamo#14302 (stop token "
                "stripped from completion_token_ids in dynamo>=1.4.1; fixed by #14317). Refusing to "
                f"train on it. Response head: {str(data)[:300]}"
            )

        log_probs = None
        if include_log_probs:
            if used_text_fallback:
                if is_aborted:
                    return self._aborted_output(include_log_probs)
                # Re-encoded text token ids have no correspondence with the
                # sampled tokens the frontend logprobs describe — aligning them
                # would be silent data corruption. Point at the config instead.
                raise RuntimeError(
                    "calculate_log_probs=True requires a real token-id channel, but this response "
                    "had no parseable token ids (text re-encode fallback). Enable "
                    "engine_kwargs.dynamo.request_engine_data (or return_tokens_as_token_ids), "
                    "or disable calculate_log_probs."
                )
            try:
                # vLLM engine_data is exact -> strict; sglang is known to short
                # one logprob (dynamo#11640 area) -> loud mean-pad until the
                # patched channel is verified exact.
                log_probs = self._extract_completion_log_probs(
                    choice, len(token_ids), data, strict=not self._is_sglang()
                )
                if log_probs is None:
                    # No trustworthy logprob source at all (no engine_data
                    # channel, no frontend token_logprobs, or — sglang
                    # non-strict path — zero usable values after padding).
                    # Deferring this produces log_probs=None TokenOutputs that
                    # starve or crash the training consumer far from the root
                    # cause.
                    raise RuntimeError(
                        "calculate_log_probs is enabled but the response carries no usable logprob "
                        "source (neither nvext.engine_data.completion_logprobs nor "
                        "choice.logprobs.token_logprobs yielded values). Enable "
                        "engine_kwargs.dynamo.request_engine_data or fix the frontend config."
                    )
            except RuntimeError:
                if is_aborted:
                    # Partial data from an aborted request with inconsistent
                    # logprobs is untrusted — drop it and let the client retry.
                    return self._aborted_output(include_log_probs)
                raise
        return self._build_token_output(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=self._map_finish_reason(finish_reason),
        )

    def _log_engine_data_token_ids_status(self, choice: dict[str, Any], response: dict[str, Any]):
        """Log once whether Dynamo returned RL engine token data."""
        has_engine_token_ids = False
        for nvext in (response.get("nvext"), choice.get("nvext")):
            if not isinstance(nvext, dict):
                continue
            engine_data = nvext.get("engine_data")
            if not isinstance(engine_data, dict):
                continue
            token_ids = engine_data.get("completion_token_ids")
            if isinstance(token_ids, list) and token_ids:
                has_engine_token_ids = True
                break

        if has_engine_token_ids and not self._logged_engine_data_token_ids:
            logger.info("Dynamo response includes nvext.engine_data.completion_token_ids")
            self._logged_engine_data_token_ids = True
            return

        if self._request_engine_data() and not has_engine_token_ids and not self._logged_missing_engine_data:
            logger.warning(
                "Dynamo response did not include nvext.engine_data.completion_token_ids; "
                "falling back to legacy token-id extraction"
            )
            self._logged_missing_engine_data = True

    def _aborted_output(self, include_log_probs: bool = False):
        """Empty TokenOutput with stop_reason "aborted".

        Returned when abort_all_requests cut a request short (partial rollout) and no
        trustworthy partial data exists, so FullyAsyncLLMServerClient retries the same
        prompt after resume. Never pad fallback tokens here. ``log_probs`` is an empty
        list when the caller asked for logprobs so the (token_ids, log_probs) length
        invariant holds.
        """
        return self._build_token_output(
            token_ids=[],
            log_probs=[] if include_log_probs else None,
            stop_reason="aborted",
            allow_empty=True,
        )

    @staticmethod
    def _completion_finish_reason(choice: dict[str, Any], response: dict[str, Any]) -> Optional[str]:
        """Prefer Dynamo's engine reason over OpenAI's lossy stop mapping."""
        for extension in (choice.get("nvext"), response.get("nvext")):
            if isinstance(extension, dict):
                reason = extension.get("detailed_finish_reason")
                if isinstance(reason, str) and reason:
                    return reason
        return choice.get("finish_reason")

    @staticmethod
    def _map_finish_reason(finish_reason: Optional[str]) -> Optional[str]:
        """OpenAI finish_reason -> verl stop_reason; one rule for both engines and both
        dispatch paths (frontend and direct).

        ai-dynamo's handlers normalize vLLM's "abort" to "cancelled"
        (dynamo.common.utils.engine_response.normalize_finish_reason) before the Rust
        frontend exposes it in nvext.detailed_finish_reason; both mean the request was cut short, so
        partial-rollout resume triggers instead of a truncated trajectory silently
        entering training as completed. Unknown reasons pass through unchanged.
        """
        if finish_reason in ("stop", "length"):
            return "completed"
        if finish_reason in ("abort", "cancelled"):
            return "aborted"
        return finish_reason

    def _build_token_output(
        self,
        token_ids: Optional[list[int]] = None,
        log_probs: Optional[list[float]] = None,
        stop_reason: Optional[str] = None,
        allow_empty: bool = False,
    ):
        """Build a verl TokenOutput while preserving AgentLoop shape invariants."""
        from verl.workers.rollout.replica import TokenOutput

        if not allow_empty:
            token_ids = token_ids or self._fallback_token_ids()
        else:
            token_ids = token_ids or []
        if log_probs is not None and len(log_probs) != len(token_ids):
            raise RuntimeError(
                f"log_probs/token_ids length mismatch ({len(log_probs)} vs {len(token_ids)}); "
                "refusing to pad/truncate (fake logprobs would corrupt training data)."
            )
        # Empty (aborted-before-first-token) outputs carry NO version tag:
        # the client aggregates min/max_global_steps per attempt regardless of
        # token count, so tagging a token-less attempt inflates trajectory
        # version-span metrics. FullyAsyncLLMServerClient only registers
        # non-None versions, so omitting the key skips the attempt cleanly.
        extra_fields = {"global_steps": self.global_steps or 0} if token_ids else {}
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields=extra_fields,
        )

    def _fallback_token_ids(self) -> list[int]:
        """Return one harmless token so Dynamo never emits an empty response.

        WARNING: this is a correctness cliff, not a nicety. The engine's text is
        fine; only the token ids are missing, so the trainer scores a single EOS
        against a full response. Every metric stays plausible —
        response_length==1 for every sample, grad_norm==0, reward at its floor —
        and the run reports success. A 2-node 30B retool run was lost to exactly
        this (job 16270139) before the log line below existed.

        It fires when the frontend returned no token ids, i.e. when
        ``request_engine_data`` / ``request_completion_token_ids`` are off, or when
        the engine's handler does not implement the ``nvext`` fields.
        """
        self._fallback_token_id_hits = getattr(self, "_fallback_token_id_hits", 0) + 1
        if self._fallback_token_id_hits <= 3 or self._fallback_token_id_hits % 100 == 0:
            logger.error(
                "[DynamoHttpServer] NO TOKEN IDS in the frontend response (hit #%s) — "
                "substituting a single EOS token. Training will see 1-token responses and "
                "learn nothing while every metric still looks valid. Set "
                "engine_kwargs.dynamo.request_engine_data=true and "
                "request_completion_token_ids=true, and verify the engine's handler "
                "supports them.",
                self._fallback_token_id_hits,
            )
        tokenizer = getattr(self.model_config, "tokenizer", None)
        for attr in ("eos_token_id", "pad_token_id"):
            token_id = getattr(tokenizer, attr, None) if tokenizer is not None else None
            if token_id is not None:
                return [int(token_id)]
        return [0]

    @staticmethod
    def _merge_nvext_extra_fields(nvext: Any, fields: list[str]) -> dict[str, Any]:
        """Return nvext with requested extra_fields added."""
        merged = dict(nvext) if isinstance(nvext, dict) else {}
        extra_fields = merged.get("extra_fields")
        if isinstance(extra_fields, list):
            requested = list(extra_fields)
        else:
            requested = []
        for field in fields:
            if field not in requested:
                requested.append(field)
        merged["extra_fields"] = requested
        return merged

    @staticmethod
    def _extract_completion_token_ids(
        choice: dict[str, Any],
        response: Optional[dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        allow_text_mapping: bool = True,
    ) -> Optional[list[int]]:
        """Extract nvext / OpenAI-extension token ids when available.

        Covers both channels: ``engine_data`` (vllm) and the top-level
        ``completion_token_ids`` (the only one sglang populates).

        ``allow_text_mapping=False`` disables the per-token text->id mapping
        channel: it is CONTENT-dependent (tokens like "" don't round-trip),
        so ids recovered through it cannot be trusted to align with logprobs.
        The exact ``token_id:N`` string parse stays available either way.
        """
        from verl.utils.tokenizer import normalize_token_ids

        candidates: list[Any] = [
            choice.get("token_ids"),
            choice.get("output_token_ids"),
            choice.get("completion_token_ids"),
        ]

        nvext = (response or {}).get("nvext")
        if isinstance(nvext, dict):
            candidates.extend(
                [
                    nvext.get("completion_token_ids"),
                    nvext.get("output_token_ids"),
                    nvext.get("generated_token_ids"),
                ]
            )
            engine_data = nvext.get("engine_data")
            if isinstance(engine_data, dict):
                candidates.extend(
                    [
                        engine_data.get("completion_token_ids"),
                        engine_data.get("output_token_ids"),
                        engine_data.get("generated_token_ids"),
                    ]
                )
        choice_nvext = choice.get("nvext")
        if isinstance(choice_nvext, dict):
            candidates.extend(
                [
                    choice_nvext.get("completion_token_ids"),
                    choice_nvext.get("output_token_ids"),
                    choice_nvext.get("generated_token_ids"),
                    choice_nvext.get("token_ids"),
                ]
            )
            engine_data = choice_nvext.get("engine_data")
            if isinstance(engine_data, dict):
                candidates.extend(
                    [
                        engine_data.get("completion_token_ids"),
                        engine_data.get("output_token_ids"),
                        engine_data.get("generated_token_ids"),
                    ]
                )

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            candidates.extend(
                [
                    logprobs.get("token_ids"),
                    logprobs.get("output_token_ids"),
                    logprobs.get("completion_token_ids"),
                ]
            )

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return normalize_token_ids(candidate)
            except TypeError:
                logger.warning("Ignoring non-token-id completion field: %r", candidate)

        # Only inspect OpenAI logprob token strings after all authoritative
        # numeric token-id fields are absent. Some frontends include decoded
        # logprob tokens even when nvext already carries token ids; probing
        # those strings eagerly emits noisy warnings for tokens such as "" that
        # do not round-trip to exactly one tokenizer id.
        if isinstance(logprobs, dict):
            token_strings = logprobs.get("tokens")
            token_ids_from_strings = DynamoHttpServer._parse_token_id_strings(token_strings)
            if token_ids_from_strings is not None:
                return token_ids_from_strings
            if tokenizer is not None and allow_text_mapping:
                return DynamoHttpServer._encode_logprob_token_strings(token_strings, tokenizer)
        return None

    @staticmethod
    def _parse_token_id_strings(tokens: Any) -> Optional[list[int]]:
        """Parse Dynamo logprob token strings formatted as ``token_id:<id>``."""
        if not isinstance(tokens, list) or not tokens:
            return None
        token_ids: list[int] = []
        for token in tokens:
            if not isinstance(token, str):
                return None
            prefix, sep, suffix = token.partition(":")
            if prefix.strip() != "token_id" or not sep:
                return None
            try:
                token_ids.append(int(suffix.strip()))
            except ValueError:
                return None
        return token_ids

    @staticmethod
    def _encode_logprob_token_strings(tokens: Any, tokenizer: Any) -> Optional[list[int]]:
        """Encode OpenAI logprob token strings when explicit token ids are absent.

        This is a best-effort bridge for frontends that return
        ``choice.logprobs.tokens`` as decoded token strings. To avoid silently
        changing sequence length, only accept the result when each token string
        maps to exactly one tokenizer id. Otherwise the caller falls back to
        encoding the full completion text.
        """
        if not isinstance(tokens, list) or not tokens:
            return None
        token_ids: list[int] = []
        for token in tokens:
            if not isinstance(token, str):
                return None
            ids = tokenizer.encode(token, add_special_tokens=False)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if len(ids) != 1:
                logger.warning("Cannot map logprob token %r to one token id: %r", token, ids)
                return None
            token_ids.append(int(ids[0]))
        return token_ids

    @staticmethod
    def _extract_completion_log_probs(
        choice: dict[str, Any],
        token_count: int,
        response: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Optional[list[float]]:
        """Extract selected-token logprobs from OpenAI completions response."""
        for nvext in ((response or {}).get("nvext"), choice.get("nvext")):
            if not isinstance(nvext, dict):
                continue
            engine_data = nvext.get("engine_data")
            if not isinstance(engine_data, dict):
                continue
            values = engine_data.get("completion_logprobs")
            if isinstance(values, list):
                return DynamoHttpServer._normalize_log_probs(values, token_count, strict=strict)
            if isinstance(engine_data.get("completion_token_ids"), list):
                # dynamo's handler omits completion_logprobs when its own
                # token/logprob accounting misaligned (documented degradation).
                # The frontend-aggregated token_logprobs come from a different
                # accumulator — even a length match would be untrustworthy.
                raise RuntimeError(
                    "dynamo omitted engine_data.completion_logprobs (server-side token/logprob "
                    "misalignment, see worker log); refusing cross-provenance frontend logprobs."
                )

        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            return None
        values = logprobs.get("token_logprobs")
        if values is None:
            values = logprobs.get("logprobs")
        if values is None:
            return None
        return DynamoHttpServer._normalize_log_probs(values, token_count, strict=strict)

    # Count of responses whose logprobs had to be padded. Class-level because
    # _normalize_log_probs is a staticmethod shared by every replica in the process.
    _logprob_padding_events: int = 0

    @staticmethod
    def _logprob_fill_value(observed: list[float]) -> Optional[float]:
        """Least-harmful stand-in for a logprob the engine did not return.

        NOT 0.0 (2026-09-01). verl turns a logprob into a probability with exp(), so
        0.0 means "the engine was 100% certain" -- the maximum possible claim. Against
        an actor probability of ~1e-6 that is a log-ratio of ~14, and the K3 KL
        estimator in rollout_corr_helper (exp(r) - r - 1, *unclamped*) turns one such
        token into ~1.2e6, which a mean over ~900 tokens still leaves at ~1400.
        Measured: rollout_corr/k3_kl = 1635 on the dynamo path vs 0.0024 on native --
        a 680000x gap produced by 0.1% of tokens. The linear estimator (k1) stayed at
        0.0037 and hid it completely.

        The sequence mean is a fabrication too, but a *typical* one: it keeps the
        padded position in the same range as its neighbours, so exp() cannot explode.
        Returns None when nothing was observed -- then the caller must report "no
        logprobs" rather than invent a whole sequence.
        """
        if not observed:
            return None
        return sum(observed) / len(observed)

    @staticmethod
    def _normalize_log_probs(values: list[Any], token_count: int, strict: bool = True) -> Optional[list[float]]:
        """Validate (strict) or loudly pad (non-strict) selected-token logprobs.

        Per-engine policy:
        - strict=True (vLLM): fail fast on length mismatch or None entries. A
          padded value is a fake logprob that silently poisons
          rollout_log_probs -- under rollout_correction bypass_mode it feeds
          the policy-loss ratio directly. The vLLM engine_data channel is
          exact, so any mismatch is a real defect upstream.
        - strict=False (sglang): the engine is known to return one fewer
          logprob than token ids (dynamo#11640 area); pad LOUDLY with the
          sequence mean (see _logprob_fill_value for why not 0.0) and report.
          Returns None when nothing usable was observed. Tighten to strict
          once the sglang channel is verified exact under the patched image.
        """
        if strict:
            if len(values) != token_count:
                raise RuntimeError(
                    f"Dynamo returned {len(values)} logprobs for {token_count} tokens; "
                    "refusing to pad/truncate (fake logprobs would corrupt training data)."
                )
            checked: list[float] = []
            for index, value in enumerate(values):
                if value is None:
                    raise RuntimeError(
                        f"Dynamo returned a None logprob at position {index}/{token_count}; "
                        "refusing to substitute 0.0 (probability 1.0) for a real value."
                    )
                checked.append(float(value))
            return checked

        raw = list(values[:token_count])
        observed = [float(v) for v in raw if v is not None]
        fill = DynamoHttpServer._logprob_fill_value(observed)

        n_none = sum(1 for v in raw if v is None)
        n_pad = max(0, token_count - len(raw))

        if fill is None:
            # Zero usable values: say so instead of fabricating token_count entries.
            if n_none or n_pad:
                DynamoHttpServer._report_logprob_padding(0, token_count, n_none, n_pad, None)
            return None

        result = [fill if v is None else float(v) for v in raw]
        result.extend([fill] * n_pad)

        if n_none or n_pad:
            DynamoHttpServer._report_logprob_padding(token_count - n_none - n_pad, token_count, n_none, n_pad, fill)
        return result

    @staticmethod
    def _report_logprob_padding(usable: int, total: int, n_none: int, n_pad: int, fill: Optional[float]) -> None:
        cls = DynamoHttpServer
        cls._logprob_padding_events += 1
        n = cls._logprob_padding_events
        if n <= 3 or n % 100 == 0:
            logger.error(
                "[logprobs] engine returned %d/%d usable logprobs (None=%d, missing=%d); "
                "filling with %s. rollout_probs_diff_* / rollout_actor_probs_pearson_corr / "
                "rollout_corr/k3_kl are NOT trustworthy for these samples. occurrence=%d",
                usable,
                total,
                n_none,
                n_pad,
                f"sequence mean {fill:.4f}" if fill is not None else "nothing (returning None)",
                n,
            )

    async def collective_rpc(
        self,
        method,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        """Bridge collective_rpc to every dynamo.vllm subprocess on this node.

        Sends the request via ZMQ to each subprocess's control sidecar and
        awaits all replies in parallel. Each sidecar invokes
        ``engine_client.collective_rpc(method, args, kwargs)`` on its local
        AsyncLLM, so the verl WorkerExtension methods (update_weights_from_ipc,
        wake_up, sleep, ...) execute inside vLLM workers.

        vllm only: there is no worker-extension equivalent on the sglang path,
        where the same operations go through _sglang_control_all instead.
        """
        if not self._control_endpoints:
            raise NotImplementedError(
                "DynamoHttpServer.collective_rpc requires the per-shard ZMQ "
                "control sidecars, which only exist for engine=vllm and only "
                "after _start_engine_workers has run."
            )

        # Control sidecar protocol: REQ side sends pickled dict, RECVs reply.
        # Sequential for simplicity (one sidecar per shard, response time
        # similar across shards). If this becomes a bottleneck switch to
        # asyncio.gather over per-endpoint REQ sockets.
        import pickle

        import zmq
        import zmq.asyncio

        # Iterating the endpoints sequentially deadlocks
        # `update_weights_from_ipc`. That RPC blocks until the receiver's IPC
        # loop returns, but the loop returns only after the sender finishes;
        # the sender depends on a cupy NCCL broadcast which requires ALL
        # replicas' rollout actors to join the group. Sequentially, only
        # ep[0]'s workers are ever woken — the other replicas' receivers never
        # set up, the cupy broadcast hangs, everything deadlocks.
        #
        # So dispatch all sidecars CONCURRENTLY via asyncio.gather: every
        # replica's workers fire update_weights_from_ipc together, all REP
        # sockets bind, the broadcast progresses, the sender unblocks.
        method_name = method if isinstance(method, str) else method.__name__
        req = {
            "method": method_name,
            "args": args,
            "kwargs": kwargs or {},
            "timeout": timeout,
        }
        recv_timeout = timeout if timeout else 600

        ctx = zmq.asyncio.Context.instance()

        async def _call_one(idx: int, ep: str) -> Any:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(ep)
                await sock.send(pickle.dumps(req))
                reply_bytes = await asyncio.wait_for(sock.recv(), timeout=recv_timeout)
                reply = pickle.loads(reply_bytes)
                if not reply.get("ok"):
                    raise RuntimeError(f"control sidecar @ {ep} returned error: {reply.get('error')}")
                return reply.get("result")
            finally:
                sock.close()

        results = await asyncio.gather(*[_call_one(i, ep) for i, ep in enumerate(self._control_endpoints)])
        return results

    # ------------------------------------------------------------------ #
    # verl interface — lifecycle hooks, forwarded to the engine's control
    # plane (ZMQ sidecar for vllm, /engine/control/* for sglang). The ones
    # an engine has no route for stay no-ops; each says so.
    # ------------------------------------------------------------------ #

    async def wake_up(self, **kwargs):
        # NB: no node_rank guard — each per-node server wakes its OWN local
        # workers (control endpoints are node-local on both engines), so all
        # nodes must run.
        if not self._free_engine_on_train():
            logger.info("[DynamoHttpServer] wake_up: free_engine_on_train disabled, leaving Dynamo workers loaded")
            return
        if self._is_sglang():
            await self.sglang_resume(kwargs.get("tags") or ["kv_cache", "weights"])
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] wake_up: no control sidecar, skipping")
            return
        # bridge to engine.wake_up via control sidecar (engine method,
        # not collective_rpc — handled in sidecar). Weight/KV re-onlining can
        # exceed the default 120s on large models.
        await self._engine_method_all("wake_up", kwargs=kwargs, timeout=600)

    async def sleep(self, **kwargs):
        # NB: no node_rank guard — each per-node server sleeps its OWN local
        # workers (control endpoints are node-local on both engines), so all
        # nodes must run.
        if not self._free_engine_on_train():
            logger.info("[DynamoHttpServer] sleep: free_engine_on_train disabled, leaving Dynamo workers loaded")
            return
        if self._is_sglang():
            # SGLang releases by tag; vLLM by numeric level. Map them by what they
            # do to GPU memory, NOT by "lower level == fewer tags":
            #   vLLM sleep(level=1)  = offload weights to CPU + drop KV -> GPU freed
            #   sglang tags=["kv_cache"]          = drop KV, weights STAY on GPU
            #   sglang tags=["kv_cache","weights"] = both freed, restorable
            # So vLLM level 1 corresponds to BOTH sglang tags. An earlier revision
            # mapped level 1 -> kv_cache only and left ~31 GB of TP=2 Qwen3-30B
            # weights resident through the training step; the trainer's FSDP forward
            # then OOM'd with 92 MiB free (48.0 GiB trainer + 30.9 GiB engine on an
            # 80 GiB H100, job 16280143). Releasing weights is safe because a weight
            # sync always follows wake_up, so the restored region is overwritten
            # before it is read.
            level = int(kwargs.get("level", 1))
            del level  # both levels free weights; kept for interface parity
            await self.sglang_release(["kv_cache", "weights"])
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] sleep: no control sidecar, skipping")
            return
        # Level 1 offloads weights to CPU and drops KV, so wake_up restores the
        # weights from CPU and a refit is not required to make the engine usable
        # again. Level 2 would discard them outright.
        kwargs.setdefault("level", 1)
        await self._engine_method_all("sleep", kwargs=kwargs, timeout=600)

    async def clear_kv_cache(self):
        if self._is_sglang():
            await self._sglang_control_all("flush_cache")
            return
        if not self._control_endpoints:
            return
        await self._engine_method_all("reset_prefix_cache")

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    async def release_kv_cache(self):
        """Pre-weight-sync hook, per engine.

        vLLM: parity no-op — the KV cache contents were already invalidated by
        abort_all_requests (pause_generation with clear_cache=True) one step
        earlier in the CheckpointEngineManager choreography, and the KV pool
        GPU memory stays resident, same as upstream vLLM's stub (TODO
        upstream: true KV release).

        SGLang releases KV memory *for real* via
        release_memory_occupation(tags=["kv_cache"]), so resume_kv_cache()
        below has actual work to do on that path.
        """
        if self._is_sglang():
            await self.sglang_release(["kv_cache"])
        return None

    async def resume_kv_cache(self):
        """Post-weight-sync counterpart to release_kv_cache.

        vLLM: parity stub. SGLang really released the memory and also
        unregistered the worker from discovery, so the resume is what puts
        the shard back into the routing pool.
        """
        if self._is_sglang():
            await self.sglang_resume(["kv_cache"])
        return None

    async def wait_for_requests_to_drain(self):
        # vllm only. dynamo.sglang's /engine/control/* surface has no drain
        # route, so this is a no-op there (no control endpoints); the pre-refit
        # equivalent on that path is abort_all_requests below.
        if not self._control_endpoints:
            return
        await self._engine_method_all("wait_for_requests_to_drain")

    async def _self_test_sglang_control_plane(self):
        """Prove every local sglang shard answers on /engine/* before training starts.

        Also verifies that the explicit ``flush_cache:tm`` engine route is
        registered, so missing cache-flush support fails before weight refit.
        """
        clients = self._sglang_control_clients()
        if not clients:
            raise RuntimeError(
                "engine=sglang but no /engine control endpoints were recorded; "
                "check that DYN_SYSTEM_PORT was set for each dynamo.sglang shard."
            )
        await asyncio.gather(*[c.wait_ready(timeout_s=_FRONTEND_READY_TIMEOUT_S) for c in clients])

        if self._sglang_cfg().get("enable_rl", True):
            probe = await asyncio.gather(
                *[c.flush_cache() for c in clients],
                return_exceptions=True,
            )
            bad = [(c.base_url, r) for c, r in zip(clients, probe, strict=True) if isinstance(r, Exception)]
            if bad:
                raise RuntimeError(
                    "dynamo.sglang explicit flush_cache route is not reachable on "
                    f"{bad}. Register --engine-route flush_cache:tm; without it the "
                    "prefix cache cannot be flushed after a weight update and rollouts "
                    "will silently use stale weights."
                )
        logger.info("[DynamoHttpServer] sglang control plane OK on %s shard(s)", len(clients))

    async def probe_logprob_channel(self):
        """Startup probe: verify a trustworthy logprob source end-to-end.

        With calculate_log_probs on, a missing token-id/logprob channel only
        surfaces as per-request errors — the trainer then waits forever for
        trajectories that never arrive (silent hang). One probe request at
        launch turns that into an immediate, actionable startup failure.
        """
        probe_id = f"logprob-probe-{time.time_ns()}"
        # Right after "workers registered" the frontend's model registration
        # can still be propagating (observed: 404 with empty body ~16ms after
        # the health check passes). Retry within a window; deterministic
        # config errors just re-raise after the deadline — still fail-fast
        # relative to a training-time hang. Each attempt is HARD-capped with
        # asyncio.wait_for: without it, a connected-but-unresponsive frontend
        # holds the probe for the full request_timeout_s (600s default,
        # 1800s in the recommended config) and the deadline never fires.
        window_s = float(self._dynamo_cfg().get("logprob_probe_timeout_s", 60))
        deadline = time.monotonic() + window_s
        last_error: Optional[BaseException] = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if isinstance(last_error, asyncio.TimeoutError):
                    raise RuntimeError(
                        f"logprob channel probe timed out after {window_s}s: the frontend accepted "
                        "connections but never answered the probe request"
                    ) from last_error
                raise last_error if last_error is not None else RuntimeError("logprob probe never ran")
            try:
                output = await asyncio.wait_for(
                    self.generate(
                        prompt_ids=self._fallback_token_ids(),
                        sampling_params={"max_tokens": 1, "logprobs": True, "temperature": 0.0},
                        request_id=probe_id,
                        thunderagent_session_id=probe_id,
                    ),
                    timeout=max(1.0, min(15.0, remaining)),
                )
                break
            except (RuntimeError, asyncio.TimeoutError) as error:
                last_error = error
                await asyncio.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
            raise RuntimeError(
                f"logprob channel probe failed: got log_probs={output.log_probs!r} for {len(output.token_ids)} tokens"
            )
        finalize = getattr(self, "finalize_program", None)
        if finalize is not None:
            await finalize(probe_id)
        logger.info("[DynamoHttpServer] logprob channel probe OK")

    async def abort_all_requests(self, reset_prefix_cache: bool = True, reject_request: bool = False):
        """Abort every in-flight request on this node's engine shards and leave them paused.

        sglang: native ``tokenizer_manager.pause_generation(mode="abort")`` over the
        /engine/control plane. vLLM: ``AsyncLLM.pause_generation`` bridged through each
        shard's control sidecar. On both engines the resume gate closes first, so new and
        client-retried generate() calls answer aborted-empty until resume_generation().

        ``reject_request`` is accepted for current verl's replica API. Dynamo
        always rejects paused admissions with an aborted TokenOutput, for both
        flag values, so waiting requests cannot exhaust the actor's concurrency
        slots and prevent the resume RPC from running.
        """
        # sglang: native pause for V1 partial rollout. The engine-agnostic
        # resume gate closes FIRST (new and client-retried generate() calls
        # answer aborted-empty at the actor, never park), then sglang's
        # async tokenizer_manager.pause_generation(mode="abort") clears the
        # in-flight requests and pauses intake — the same call verl's native
        # V1 sglang server uses. NOT abort_request: that one is a sync method
        # returning None and dynamo's passthrough awaits results
        # unconditionally (HTTP 500 "can't await NoneType", first
        # colocate_async run). flush_cache stays after the pause: sglang's
        # flush no-ops while requests are running. resume_generation()
        # reopens intake + the gate (choreography step 8 in V0 and V1 alike).
        if self._is_sglang():
            self._generation_resumed.clear()
            await self._sglang_control_all("pause_generation", mode="abort")
            if reset_prefix_cache:
                await self._sglang_control_all("flush_cache")
            return {"aborted_count": -1, "request_ids": [], "paused": True}
        # vLLM: bridge AsyncLLM.pause_generation (vLLM >= 0.12) through each shard's
        # control sidecar -- aborts in-flight requests (the frontend returns their
        # partial tokens with finish_reason "abort"), drains, optionally clears
        # caches, and leaves the engines paused.
        if self._use_direct_generate():
            # The debug direct-generate path holds the sidecar's single
            # in-flight REP slot for the whole generation, so the pause
            # request would queue behind every running generation and time
            # out. Incompatible with abort semantics — fail fast.
            raise RuntimeError(
                "engine_kwargs.dynamo.direct_generate=true is incompatible with abort_all_requests "
                "(V1 async trainers pause engines every step); disable direct_generate."
            )
        self._generation_resumed.clear()
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] abort_all_requests: no control sidecar, skipping")
            return {"aborted_count": 0, "request_ids": [], "paused": False}
        # Pause failures must raise: sleeping (weight-offloading) an engine
        # that still has active requests is undefined behavior.
        await self._engine_method_all(
            "pause_generation",
            kwargs={"wait_for_inflight_requests": False, "clear_cache": reset_prefix_cache},
            timeout=600,
        )
        logger.info("[DynamoHttpServer] abort_all_requests: engines paused (node=%s)", self.node_rank)
        return {"aborted_count": None, "request_ids": [], "paused": True}

    async def resume_generation(self):
        """Resume request intake after abort_all_requests."""
        if self._is_sglang():
            # Counterpart to pause_generation(mode="abort"): without it the
            # engine stays paused and every post-sync generation queues
            # forever behind a closed intake.
            await self._sglang_control_all("continue_generation")
        if self._control_endpoints:
            await self._engine_method_all("resume_generation", timeout=120)
        self._generation_resumed.set()
        logger.info("[DynamoHttpServer] resume_generation: gate open (node=%s)", self.node_rank)
        return None

    async def start_profile(self, **kwargs):
        if self._is_sglang():
            await self._sglang_control_all("start_profile", **kwargs)
        return None

    async def stop_profile(self):
        if self._is_sglang():
            await self._sglang_control_all("stop_profile")
        return None

    async def _engine_method_all(self, method: str, kwargs: Optional[dict] = None, timeout: float = 600):
        """Like collective_rpc but invokes a top-level AsyncLLM method
        (wake_up / sleep / pause_generation / resume_generation /
        reset_prefix_cache / wait_for_requests_to_drain), not a
        worker-extension RPC. Distinguished by message kind.

        Raises RuntimeError when any shard reports failure — a silently
        skipped sleep/pause leaves the engine in a state the trainer no
        longer agrees with (e.g. sleeping an engine with active requests).

        vllm only, and a silent no-op elsewhere: with no control endpoints (the
        sglang path, or a slave before launch) it returns None, so every caller
        must route sglang through _sglang_control_all first.

        Dispatches in parallel for the same reason as collective_rpc: sequential
        iteration deadlocked update_weights_from_ipc, and deadlocks
        reset_prefix_cache post-refit as well."""
        if not self._control_endpoints:
            return None

        import pickle

        import zmq
        import zmq.asyncio

        ctx = zmq.asyncio.Context.instance()
        req = {
            "kind": "engine_method",
            "method": method,
            "kwargs": kwargs or {},
            "timeout": timeout,
        }

        async def _call_one(idx: int, ep: str) -> None:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(ep)
                await sock.send(pickle.dumps(req))
                reply_bytes = await asyncio.wait_for(sock.recv(), timeout=timeout)
                reply = pickle.loads(reply_bytes)
                if not reply.get("ok"):
                    raise RuntimeError(f"engine_method {method} failed @ {ep}: {reply.get('error')}")
            finally:
                sock.close()

        await asyncio.gather(*[_call_one(i, ep) for i, ep in enumerate(self._control_endpoints)])
        return None

    def get_num_engine_workers(self) -> int:
        """Total number of TP worker processes across all engine shards on this
        node. Used by DynamoRollout.update_weights to compute the NCCL group
        world_size = 1 (broadcaster) + N (engine workers)."""
        tp = int(self.config.tensor_model_parallel_size)
        n_shards = len(self._engine_control_endpoints) if self._is_sglang() else len(self._control_endpoints)
        return n_shards * tp

    async def _self_test_refit_path(self):
        """Verify the control-sidecar ⇄ AsyncLLM round-trip is alive.

        Refit (DynamoRollout.update_weights) routes weight bytes through
        ``collective_rpc("update_weights_from_ipc", ...)`` which depends on
        a working REQ-REP loop to each ``_dynamo_vllm_with_control``
        subprocess. A silent failure here (sidecar didn't start, control
        endpoint port collision, etc.) would let ``update_weights`` appear
        to succeed while actually losing every update — which is the failure
        this self-test exists to catch.

        This self-test sends one ``collective_rpc`` request with a
        deliberately invalid method name. A reachable sidecar will reply
        with a structured error response; an unreachable one will time out.
        Either response proves the IPC path is alive.

        Skipped when no control endpoints are registered (slave node / pre-launch).
        Soft-fail by default; set env ``VERL_DYNAMO_REFIT_STRICT=1`` to
        raise on failure (recommended).

        For engine=sglang the equivalent probe is an HTTP round-trip to each
        shard's /engine/* plane, and it is **always strict**: unlike the vLLM
        sidecar (which is verl-private and can degrade), a missing sglang control
        plane means weight sync cannot work at all, so failing at launch is
        strictly better than failing at the first refit.
        """
        if self._is_sglang():
            await self._self_test_sglang_control_plane()
            return
        if not self._control_endpoints:
            return

        import pickle

        import zmq
        import zmq.asyncio

        strict = os.environ.get("VERL_DYNAMO_REFIT_STRICT", "0") not in (
            "",
            "0",
            "false",
            "False",
        )

        # Ping the first endpoint only — one round-trip is sufficient to
        # prove the sidecar machinery is alive. We do not iterate all
        # endpoints here to keep startup overhead minimal.
        ep = self._control_endpoints[0]
        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(ep)
            # Use an UNKNOWN `kind` (not an unknown method under
            # kind=collective_rpc) so the sidecar bails at the kind-dispatch
            # `else` branch in _handle_request and returns a structured
            # error reply WITHOUT ever calling engine.collective_rpc.
            #
            # Previously we used kind="collective_rpc" + an invalid method
            # name, which dispatched into vLLM's worker RPC queue. The
            # AttributeError on workers got cached/queued and corrupted the
            # NEXT real engine.collective_rpc call (sleep) — sleep silently
            # failed, vLLM held its full 128 GiB, and the next trainer
            # all-gather OOM'd (job 2463154).
            req = {
                "kind": "__refit_self_test_probe__",
                "method": None,
                "args": (),
                "kwargs": {},
                "timeout": 5,
            }
            await sock.send(pickle.dumps(req))
            reply_bytes = await asyncio.wait_for(sock.recv(), timeout=10)
            reply = pickle.loads(reply_bytes)
            logger.info(
                "[DynamoHttpServer] refit self-test PASSED @ %s (sidecar responded ok=%s)",
                ep,
                reply.get("ok"),
            )
        except (asyncio.TimeoutError, Exception) as e:
            msg = (
                f"[DynamoHttpServer] refit self-test FAILED @ {ep}: "
                f"{type(e).__name__}: {e} — DynamoRollout.update_weights "
                f"will likely lose updates silently. Check that "
                f"dynamo.vllm control sidecars started."
            )
            if strict:
                raise RuntimeError(msg) from e
            logger.warning(msg)
        finally:
            sock.close()

    # ------------------------------------------------------------------ #
    # shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self):
        self._shutdown_requested = True
        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog_task = None

        # SIGTERM each entry in registry order, escalate to SIGKILL on timeout.
        for attr, name, timeout in _SUBPROCESS_REGISTRY:
            proc = getattr(self, attr, None)
            if proc is None:
                continue
            if isinstance(proc, list):
                for i, p in enumerate(proc):
                    self._stop_one(p, f"{name}[{i}]", timeout)
                setattr(self, attr, [])
            else:
                self._stop_one(proc, name, timeout)
                setattr(self, attr, None)

        # Close log fps; cleanup tmp dirs.
        for fp in self._engine_log_fps:
            try:
                fp.close()
            except Exception:
                pass
        self._engine_log_fps = []
        if self._frontend_log_fp is not None:
            try:
                self._frontend_log_fp.close()
            except Exception:
                pass
            self._frontend_log_fp = None

        if self._etcd_data_dir and os.path.isdir(self._etcd_data_dir):
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None

    @staticmethod
    def _stop_one(proc: subprocess.Popen, name: str, timeout: int):
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        logger.info("[DynamoHttpServer] stopped %s (rc=%s)", name, proc.returncode)

    # ------------------------------------------------------------------ #
    # pickle support — actor handles can be passed across actors; the
    # subprocesses themselves cannot be pickled.
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr, _, _ in _SUBPROCESS_REGISTRY:
            state[attr] = None if not attr.endswith("_processes") else []
        state["_watchdog_task"] = None
        state["_frontend_log_fp"] = None
        state["_engine_log_fps"] = []
        state["_engine_log_paths"] = []
        # aiohttp sessions are bound to a live event loop; rebuilt lazily.
        state["_sglang_clients"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# --------------------------------------------------------------------------- #
# DynamoReplica
# --------------------------------------------------------------------------- #


class _DynamoCheckpointEngineWorker(CheckpointEngineWorker):
    """CheckpointEngineWorker variant spawned with ``num_gpus=0``.

    The base ``Worker._setup_env_cuda_visible_devices`` reads
    ``ray.get_runtime_context().get_accelerator_ids()[device_name][0]`` when
    ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` — that list is empty
    for a ``num_gpus=0`` actor, so it IndexErrors. Our spawn helper injects
    ``CUDA_VISIBLE_DEVICES`` via ``runtime_env`` to pin the actor to the
    GPU shared with its paired engine subprocess (CUDA IPC needs same-GPU
    pairing), so torch sees that single GPU as device 0.
    """

    def _setup_env_cuda_visible_devices(self):  # type: ignore[override]
        get_torch_device().set_device(0)


class DynamoReplica(RolloutReplica):
    """Manages one logical Dynamo serving replica across N nodes.

    Mirrors vLLMReplica.launch_servers (one DynamoHttpServer actor per node)
    but with a master/slave split:
      - first actor (node_rank=0) starts etcd + nats + frontend in addition
        to its engine worker subprocesses,
      - other actors only start their workers, pointing at the master via
        get_master_address.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )
        # TP must fit within one node — see design doc §11.1.
        assert self.config.tensor_model_parallel_size <= self.gpus_per_replica_node, (
            f"TP={self.config.tensor_model_parallel_size} must be <= "
            f"gpus_per_node={self.gpus_per_replica_node} (CUDA IPC does not "
            f"cross hosts; raise dp/pp instead)."
        )
        self.server_class = ray.remote(DynamoHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"

    async def sleep(self):
        """Drain in-flight requests before the weight-offloading sleep.

        Mirrors vLLMReplica.sleep: the base class would sleep immediately,
        which is undefined behavior when requests are still active (e.g. the
        colocated-reward-model validation path sleeps without an abort).
        Unlike vLLM, dynamo servers are per-node — drain every one of them.
        """
        await asyncio.gather(*[server.wait_for_requests_to_drain.remote() for server in self.servers])
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def init_hybrid_worker_pool(self, worker_group):
        """Initialize Dynamo as worker pool for all rollout GPUs.

        The verl CheckpointEngineManager fans weight-sync hooks
        (update_weights / execute_checkpoint_engine / release_kv_cache /
        resume_kv_cache) out to per-rollout-rank Ray actors. Dynamo's
        rollout-side ``engine'' is a subprocess (not a Ray actor) plus a
        node-level DynamoHttpServer, so we cannot route those hooks through
        the trainer's borrowed WorkerDict handles — they don't expose the
        rollout-side methods. We therefore:

        1. Borrow the trainer worker_group only to look up per-GPU placement.
        2. Spawn the engine subprocesses on those GPUs (existing logic).
        3. Spawn dedicated CheckpointEngineWorker Ray actors (one per
           rollout rank) colocated on the same GPUs as the subprocess
           workers via env-injected CUDA_VISIBLE_DEVICES.
        4. Reassign ``self.workers`` to those CheckpointEngineWorker
           handles so framework hooks land where the methods actually
           live.
        """
        self.rollout_mode = RolloutMode.HYBRID
        # Hold trainer handles only as a placement lookup; replaced at the
        # end with dedicated rollout-side actors.
        self.workers = list(worker_group.workers)

        assert len(self.workers) % self.world_size == 0, (
            f"worker_group size {len(self.workers)} must be divisible by "
            f"dynamo logical replica world_size {self.world_size}"
        )
        num_logical_replicas = len(self.workers) // self.world_size
        await self._launch_shared_worker_pool(num_logical_replicas=num_logical_replicas)

        # Now that the engine subprocesses are alive on the GPUs identified
        # by self._trainer_worker_infos, spawn matching CheckpointEngineWorker
        # actors and adopt them as our framework-facing workers. Naive mode
        # must keep the trainer WorkerDict handles: its refit runs inside
        # WorkerDict.update_weights, and a CE actor's ServerAdapter would
        # collide with the WorkerDict adapter on the per-rank IPC socket
        # (observed as a stuck sender -> 30min NCCL watchdog abort).
        if self.config.checkpoint_engine.backend != "naive":
            self.workers = self._spawn_rollout_checkpoint_engine_workers()

    def _spawn_rollout_checkpoint_engine_workers(self) -> list[ActorHandle]:
        """Spawn one CheckpointEngineWorker Ray actor per rollout rank.

        Each actor:
        - ``num_gpus=0`` — Ray does not see it as competing for the trainer's
          resource pool slots.
        - ``CUDA_VISIBLE_DEVICES`` env-injected to the same GPU as the
          corresponding dynamo subprocess worker, so cupy/torch tensor
          allocations land where the subprocess can receive them via CUDA IPC.
        - ``RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE`` env match the
          trainer rank layout so the actor's ``server_adapter``
          (recipe.dynamo.dynamo_rollout.ServerAdapter) computes a
          ``zmq_handle`` that pairs 1:1 with the subprocess worker.
        - ``MASTER_ADDR / MASTER_PORT`` set so the CheckpointEngineWorker
          actors form their own torch.distributed cpu:gloo group on
          ``initialize_global_process_group_ray``. Port chosen to not
          collide with the trainer's existing distributed init.
        """
        worker_infos = self._trainer_worker_infos
        total_ranks = len(worker_infos)
        local_world_size = self.gpus_per_node
        master_port = str(int(os.environ.get("VERL_DYNAMO_CE_MASTER_PORT", "29600")))
        # CE worker rank 0 lands on the head node; on multi-node setups every
        # other rank needs head's reachable IP, not 127.0.0.1. Resolve via
        # ``ray.nodes()`` using the first worker's node_id.
        master_addr = "127.0.0.1"
        if worker_infos:
            head_node_id = worker_infos[0][0]
            for node in ray.nodes():
                if node.get("NodeID") == head_node_id:
                    master_addr = node.get("NodeManagerAddress") or master_addr
                    break

        # Resolve placement-group ids → PlacementGroup objects (deduped) so
        # each CE actor can colocate into the same bundle as its paired
        # trainer worker. ``get_placement_group`` raises if the id is unknown
        # (e.g. PG cleaned up); falling back to NodeAffinity keeps the actor
        # on the same node even if PG capture fails.
        trainer_pg_ids = self._trainer_worker_pg_ids
        trainer_pg_lookup: dict[str, Any] = {}
        for pg_id in trainer_pg_ids:
            if pg_id and pg_id not in trainer_pg_lookup:
                try:
                    trainer_pg_lookup[pg_id] = ray.util.get_placement_group(pg_id)
                except Exception:
                    trainer_pg_lookup[pg_id] = None

        actors: list[ActorHandle] = []
        for rank, (node_id, gpu_id) in enumerate(worker_infos):
            env_vars = {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "RANK": str(rank),
                "WORLD_SIZE": str(total_ranks),
                "LOCAL_RANK": str(rank % local_world_size),
                "LOCAL_WORLD_SIZE": str(local_world_size),
                "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }

            trainer_pg = trainer_pg_lookup.get(trainer_pg_ids[rank])
            if trainer_pg is not None:
                scheduling_strategy = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=trainer_pg,
                    placement_group_bundle_index=rank % local_world_size,
                    placement_group_capture_child_tasks=False,
                )
            else:
                scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )

            actor = (
                ray.remote(num_gpus=0, num_cpus=1)(_DynamoCheckpointEngineWorker)
                .options(
                    scheduling_strategy=scheduling_strategy,
                    runtime_env={"env_vars": env_vars},
                    name=f"dynamo_ce_worker_{self.replica_rank}_{rank}{self.name_suffix}",
                )
                .remote(
                    rollout_config=self.config,
                    model_config=self.model_config,
                    replica_rank=rank // self.world_size,
                )
            )
            actors.append(actor)
        return actors

    async def init_standalone_pool(self):
        """Standalone shared worker pool over rollout.nnodes × n_gpus_per_node.

        Mirrors verl's RolloutReplica.init_standalone but at POOL granularity:
        one resource pool + one CheckpointEngineWorker per rollout GPU (they
        receive weights over the checkpoint-engine first hop and forward them
        node-locally via CUDA-IPC), then the same shared master/slave dynamo
        stack as hybrid — one etcd/nats/frontend for the whole pool.
        """
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
        from verl.utils.device import get_device_name

        self.rollout_mode = RolloutMode.STANDALONE
        pool_nnodes = self.config.nnodes
        pool_gpus_per_node = self.config.n_gpus_per_node
        assert pool_nnodes > 0 and pool_gpus_per_node > 0, (
            "standalone dynamo pool requires rollout.nnodes > 0 and rollout.n_gpus_per_node > 0"
        )

        resource_pool_name = f"dynamo_rollout_pool_{self.replica_rank}{self.name_suffix}"
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={resource_pool_name: [pool_gpus_per_node] * pool_nnodes},
            mapping=None,
            max_colocate_count=2,
        )
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        # Same actor class and init args as RolloutReplica.get_ray_class_with_init_args,
        # plus a marker the adapters use to recognise the standalone pool. They cannot
        # infer it from replica_rank: the hybrid path (_spawn_rollout_checkpoint_engine_
        # workers) passes a per-logical-replica replica_rank to its workers as well.
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(CheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
            dynamo_standalone_pool=True,
        )
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            bin_pack=False,
            name_prefix=f"dynamo_rollout_standalone_{self.replica_rank}{self.name_suffix}",
            use_gpu=True,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers

        assert len(self.workers) % self.world_size == 0, (
            f"standalone pool size {len(self.workers)} must be divisible by "
            f"dynamo logical replica world_size {self.world_size}"
        )
        num_logical_replicas = len(self.workers) // self.world_size
        await self._launch_shared_worker_pool(num_logical_replicas=num_logical_replicas)

    async def _launch_shared_worker_pool(self, num_logical_replicas: int):
        """Launch a single frontend backed by all logical replica workers."""
        from verl.utils.device import get_resource_name

        tp = self.config.tensor_model_parallel_size
        worker_infos_raw = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                        ray.get_runtime_context().get_placement_group_id(),
                    )
                )
                for worker in self.workers
            ]
        )
        # Strip pg_id from the tuple stored as worker_infos (so existing
        # downstream consumers see the original 2-tuple shape), and resolve
        # the per-rank placement_group id to a PlacementGroup object so the
        # CE worker spawn can colocate into the trainer's bundle.
        worker_infos = [(node_id, gpu_id) for node_id, gpu_id, _ in worker_infos_raw]
        pg_ids = [pg_id for _, _, pg_id in worker_infos_raw]
        self._trainer_worker_infos = worker_infos
        self._trainer_worker_pg_ids = pg_ids

        node_order: list[str] = []
        node_to_workers: dict[str, list[ActorHandle]] = {}
        node_to_specs: dict[str, list[dict[str, Any]]] = {}

        def ensure_node(node_id: str):
            if node_id not in node_to_specs:
                node_order.append(node_id)
                node_to_workers[node_id] = []
                node_to_specs[node_id] = []

        for logical_replica_rank in range(num_logical_replicas):
            start = logical_replica_rank * self.world_size
            end = start + self.world_size
            replica_infos = worker_infos[start:end]
            replica_workers = self.workers[start:end]
            per_node_gpu_ids: dict[str, list[str]] = {}
            per_node_workers: dict[str, list[ActorHandle]] = {}
            for (node_id, gpu_id), worker in zip(replica_infos, replica_workers, strict=True):
                ensure_node(node_id)
                per_node_gpu_ids.setdefault(node_id, []).append(str(gpu_id))
                per_node_workers.setdefault(node_id, []).append(worker)

            for node_id, gpu_ids in per_node_gpu_ids.items():
                node_to_workers[node_id].extend(per_node_workers[node_id])
                assert len(gpu_ids) % tp == 0, (
                    f"logical_replica={logical_replica_rank} node={node_id} has "
                    f"{len(gpu_ids)} GPUs, not divisible by TP={tp}: {gpu_ids}"
                )
                for shard_idx in range(len(gpu_ids) // tp):
                    shard_gpus = gpu_ids[shard_idx * tp : (shard_idx + 1) * tp]
                    # POOL-global replica id (self.replica_rank carries the
                    # LLMServerManager start_rank offset): the engine-side ZMQ
                    # socket is named replica-{VERL_REPLICA_RANK}-rank-{...}
                    # and must (a) match the CE-sender side, which derives the
                    # same pool-global id, and (b) never collide with the other
                    # pool's engines when hybrid + standalone share a node in
                    # separate_async.
                    global_replica_rank = self.replica_rank + logical_replica_rank
                    node_to_specs[node_id].append(
                        {
                            "replica_rank": global_replica_rank,
                            "cuda_visible_devices": ",".join(shard_gpus),
                            "rank_offset": shard_idx * tp,
                            "label": f"replica{global_replica_rank}_shard{shard_idx}",
                        }
                    )

        expected_workers = sum(len(specs) for specs in node_to_specs.values())
        prefix = self._get_server_name_prefix()
        suffix = self.name_suffix
        self.servers = []

        for node_rank, node_id in enumerate(node_order):
            worker_specs = node_to_specs[node_id]
            node_cvd = ",".join(gpu for spec in worker_specs for gpu in spec["cuda_visible_devices"].split(",") if gpu)
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{suffix}"

            actor_env_vars = {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
            for env_key in ("VERL_DYNAMO_LOG_DIR", "PATH", "PYTHONPATH"):
                if os.environ.get(env_key):
                    actor_env_vars[env_key] = os.environ[env_key]

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": actor_env_vars},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=node_to_workers[node_id],
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=self.gpus_per_node,
                nnodes=len(node_order),
                cuda_visible_devices=node_cvd,
                worker_specs=worker_specs,
                expected_workers=expected_workers,
            )
            self.servers.append(server)

        master = self.servers[0]
        await master.launch_server.remote(start_healthcheck=False)
        master_host, master_etcd_port, master_nats_port = await master.get_master_address.remote()
        fe_host, fe_port = await master.get_server_address.remote()

        slave_launches = [
            server.launch_server.remote(
                master_address=master_host,
                master_port=master_etcd_port,
                dp_rpc_port=master_nats_port,
            )
            for server in self.servers[1:]
        ]
        if slave_launches:
            await asyncio.gather(*slave_launches)
            await asyncio.gather(*[server.set_master_frontend.remote(fe_host, fe_port) for server in self.servers[1:]])

        await master.wait_frontend_ready.remote(expected_workers=expected_workers)
        await master._self_test_refit_path.remote()
        if bool(getattr(self.config, "calculate_log_probs", False)):
            await master.probe_logprob_channel.remote()
        self._server_handle = master
        self._server_address = f"[{fe_host}]:{fe_port}" if is_valid_ipv6_address(fe_host) else f"{fe_host}:{fe_port}"
        logger.info(
            "[DynamoReplica pool] ready: server_address=%s logical_replicas=%s workers=%s nodes=%s",
            self._server_address,
            num_logical_replicas,
            expected_workers,
            len(node_order),
        )

    async def launch_servers(self):
        """Dynamo uses NeMo-style worker-pool entrypoints instead."""
        raise RuntimeError(
            "DynamoReplica.launch_servers() is disabled because the dynamo "
            "backend uses a single shared worker pool. Call "
            "DynamoReplica.init_hybrid_worker_pool(worker_group) or "
            "init_standalone_pool() via DynamoLLMServerManager instead."
        )


__all__ = ["DynamoHttpServer", "DynamoReplica"]
