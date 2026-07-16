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
     them. We only forward CUDA_VISIBLE_DEVICES into dynamo.vllm subprocesses.
  2. Spawns + watchdogs etcd / nats-server / dynamo.vllm × N / dynamo.frontend.
  3. Never holds an AsyncLLM. The actor's generate() method is only an
     HTTP client shim to dynamo.frontend; it does not generate locally.
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
from dataclasses import dataclass
from typing import Any, Optional

import ray
import requests
from ray.actor import ActorHandle

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# (attr_name, display_name, stop_timeout_seconds) — order = teardown order.
# Stop consumers first (frontend), then producers (workers), then infra
# (NATS, etcd). Keep parallel to nemo-rl's _SUBPROCESS_REGISTRY but with the
# vllm worker pool added as a list-typed entry.
_SUBPROCESS_REGISTRY: list[tuple[str, str, int]] = [
    ("_frontend_process", "frontend", 15),
    ("_vllm_processes", "vllm_workers", 30),
    ("_nats_process", "NATS", 10),
    ("_etcd_process", "etcd", 10),
]

# Default verl-side rank-offset env var read by the WorkerExtension
# (see recipe/dynamo/dynamo_worker_extension.py). Must be passed per-shard
# when spawning dynamo.vllm so the vLLM TP rank inside the subprocess maps
# to the same node-local rank that the trainer side computes.
_RANK_OFFSET_ENV = "VERL_DYNAMO_RANK_OFFSET"
_REPLICA_RANK_ENV = "VERL_REPLICA_RANK"

# Where dynamo.vllm exposes its system control HTTP. Each subprocess gets
# its own port (allocated by the actor).
_DYN_SYSTEM_PORT_ENV = "DYN_SYSTEM_PORT"

# Verl-private control sidecar (see _dynamo_vllm_with_control.py) listens on
# this ZMQ endpoint per subprocess; the actor uses it to bridge collective_rpc.
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


@dataclass(frozen=True)
class _DynamoWorkerSpec:
    """One dynamo.vllm subprocess to launch on this Ray actor."""

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
        node 0 (master): _start_etcd → _start_nats → _start_vllm_workers
                         → _start_frontend → _healthcheck_frontend
        node N (slave) : just _start_vllm_workers, pointing to master etcd/nats
      generate / wake_up / sleep / collective_rpc / ... :
        generate goes through master dynamo.frontend HTTP
        collective_rpc bridges to per-subprocess control sidecar
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
        self._vllm_processes: list[subprocess.Popen] = []
        self._etcd_data_dir: Optional[str] = None
        self._frontend_log_fp = None
        self._vllm_log_fps: list = []
        self._vllm_log_paths: list[str] = []
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

        # Per-subprocess control sidecar endpoints (filled in
        # _start_vllm_workers); used by collective_rpc bridge in v2.
        self._control_endpoints: list[str] = []
        # Per-worker /metrics endpoints (host:port), populated only when
        # enable_worker_system_metrics is on. These expose engine-level
        # vllm:prefix_cache_* that the frontend endpoint does not.
        self._worker_metrics_endpoints: list[str] = []

        # Filled in by _start_vllm_workers; consumed by generate() to build
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

    def _dynamo_cfg_bool(self, key: str, default: bool) -> bool:
        value = self._dynamo_cfg().get(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _free_engine_on_train(self) -> bool:
        """Opt-in sleep/wake of colocated vLLM workers around training."""
        return self._dynamo_cfg_bool("free_engine_on_train", False)

    def _enable_rl_mode(self) -> bool:
        """Enable Dynamo's RL/TITO-friendly vLLM mode."""
        return self._dynamo_cfg_bool("enable_rl", True)

    def _request_engine_data(self) -> bool:
        """Ask Dynamo to return vLLM engine token data via nvext."""
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
        """Master: etcd + nats + vllm workers + frontend + healthcheck."""
        # Reserve ports up-front so we know all of them before starting.
        self._etcd_port = self._configured_or_allocated_port("etcd_port", bind_wildcard=True)
        self._etcd_peer_port = self._configured_or_allocated_port("etcd_peer_port", bind_wildcard=True)
        self._nats_port = self._configured_or_allocated_port("nats_port", bind_wildcard=True)
        # Frontend port: 0 = auto, else honor config.
        self._frontend_port = self._configured_or_allocated_port("frontend_http_port", bind_wildcard=True)

        self._start_etcd()
        self._start_nats()
        self._start_vllm_workers()
        self._start_frontend()

        # Expose frontend to trainer.
        self._server_port = self._frontend_port
        if start_healthcheck:
            await self.wait_frontend_ready()
            # v2: verify control-sidecar reachability so refit failures surface
            # at startup instead of silently dropping weight updates mid-training.
            # Soft-fail by default; set VERL_DYNAMO_REFIT_STRICT=1 for fail-fast.
            # IMPORTANT: must run AFTER wait_frontend_ready so dynamo.vllm
            # subprocesses are fully booted (their control sidecars cannot
            # serve requests until the captured AsyncLLM is alive). In the
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
        """Slave: vllm workers only, pointing at master etcd/nats."""
        self._start_vllm_workers()
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

    def _start_vllm_workers(self):
        """Spawn N dynamo.vllm subprocesses on this node.

        Each subprocess is one DP shard; gets a contiguous TP-slice of GPUs
        from cuda_visible_devices. CUDA_VISIBLE_DEVICES + VERL_DYNAMO_RANK_OFFSET
        are passed in env so the WorkerExtension's _get_zmq_handle picks the
        correct global rank for ZMQ-IPC weight bucket transfer (see §11.3).
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

        for spec_idx, spec in enumerate(worker_specs):
            worker_cvd = spec.cuda_visible_devices
            control_port = self._allocate_tcp_port(bind_wildcard=False)
            control_endpoint = f"tcp://{self._server_address}:{control_port}"
            self._control_endpoints.append(control_endpoint)

            kv_event_port = self._allocate_kv_event_port(spec_idx)
            kv_events_config_json = self._build_kv_events_config_json(kv_event_port)
            vllm_port = self._allocate_vllm_tcpstore_port(spec_idx)
            # Allocate a registered (<32768, i16-safe) system-status port so this
            # dynamo.vllm worker exposes /metrics (incl. pass-through
            # vllm:prefix_cache_hits_total/queries_total) and the /engine/* routes.
            # Default ON, unconditionally sets DYN_SYSTEM_PORT=server_port (its fixed port avoids the i16 issue;
            # we use a fixed low port via _allocate_stable_node_port for the same
            # reason). Set enable_worker_system_metrics=false to restore the legacy
            # no-DYN_SYSTEM_PORT behaviour.
            enable_worker_metrics = self._dynamo_cfg_bool("enable_worker_system_metrics", True)
            system_metrics_port = (
                self._allocate_stable_node_port(_SYSTEM_METRICS_PORT_BASE, spec_idx, window=8)
                if enable_worker_metrics
                else None
            )

            env = os.environ.copy()
            env.update(self._dynamo_env_vars())
            env["CUDA_VISIBLE_DEVICES"] = worker_cvd
            env[_RANK_OFFSET_ENV] = str(spec.rank_offset)
            env[_REPLICA_RANK_ENV] = str(spec.replica_rank)
            # Match verl's native vLLM colocated path: both trainer-side
            # BucketedWeightSender and vLLM-side BucketedWeightReceiver include
            # the Ray job id in their shared /tmp IPC socket name.
            env["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()
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
            # NB: don't set DYN_SYSTEM_PORT — dynamo's Rust runtime parses it
            # as i16 and rejects ephemeral ports >= 32768. We use our own
            # control sidecar (VERL_DYNAMO_CONTROL_ZMQ) instead.
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
            # Mirrors nemo_rl/dynamo_worker.py:308-310.
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["VLLM_NO_USAGE_STATS"] = "1"
            # Deterministic block hashes across workers: without a fixed seed,
            # Python's randomized hashing can leak into block-hash derivation so
            # the same prefix hashes differently per worker → the router logs
            # "block_hash mismatch" and prefix-cache hits collapse.
            env.setdefault("PYTHONHASHSEED", "0")

            cmd = self._build_vllm_cmd(
                served_model_name,
                tp,
                kv_events_config_json=kv_events_config_json,
            )

            stdout_path = os.path.join(log_dir, f"{spec.label}.log")
            stdout_fp = open(stdout_path, "w")
            self._vllm_log_fps.append(stdout_fp)
            self._vllm_log_paths.append(stdout_path)

            logger.info(
                "[DynamoHttpServer] starting dynamo.vllm shard %s/%s "
                "(replica=%s, rank_offset=%s, GPUs=%s, vllm_port=%s, kv_event_port=%s, control=%s, "
                "DYN_ENABLE_RL=%s, request_engine_data=%s, log=%s): %s",
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
            self._vllm_processes.append(proc)

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
        """host:port of each vLLM worker's /metrics on this node (empty unless
        enable_worker_system_metrics was on)."""
        return list(self._worker_metrics_endpoints)

    def _allocate_tcp_port(self, bind_wildcard: bool = False) -> int:
        """Allocate a port for a subprocess and avoid duplicates in this actor.

        vLLM KV event publishers bind ``tcp://*:<port>``. Checking only the
        node IP can miss conflicts with wildcard listeners, so those ports are
        probed on 0.0.0.0. We also keep a local reservation set so a burst of
        shard launches does not accidentally reuse a just-released port.
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

        vLLM's TCPStore and KV event publisher both bind in child processes.
        If the parent only probes a random free port and releases it, another
        shard can claim it before the child binds. Use deterministic,
        non-ephemeral per-node/per-shard windows to avoid those startup races.
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

    def _allocate_kv_event_port(self, shard_idx: int) -> int:
        """Pick a port for vLLM's ZMQ KV event publisher.

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
        kv_events_config_json: str,
    ) -> list[str]:
        """Construct the dynamo.vllm CLI for one DP shard.

        We launch our own thin entrypoint instead of plain ``-m dynamo.vllm``
        so we can:
          1. inject ``--worker-extension-cls
             recipe.dynamo.dynamo_worker_extension.vLLMDynamoColocateWorkerExtension``
             so the WorkerExtension reads VERL_DYNAMO_RANK_OFFSET in
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
        cmd += ["--kv-events-config", kv_events_config_json]
        # Pass through extra args from rollout.engine_kwargs.dynamo.extra_args.
        extra = self._dynamo_cfg().get("extra_args") or []
        if isinstance(extra, list):
            cmd += [str(x) for x in extra]
        return cmd

    def _start_frontend(self):
        if self._frontend_process is not None:
            return
        env = os.environ.copy()
        env.update(self._dynamo_env_vars())

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
            return self._frontend_extra_args()

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
        return args + self._frontend_extra_args()

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
                        if name == "vllm_workers" and i < len(self._vllm_log_paths):
                            log_hint = f" (log={self._vllm_log_paths[i]})"
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
    ):
        """Dispatch generation through the Dynamo frontend HTTP router.

        the actor manages the
        subprocess stack, while token generation goes through the OpenAI-style
        frontend so Dynamo can route across registered workers.
        """
        if image_data is not None or video_data is not None:
            return self._build_token_output(
                stop_reason="error: Dynamo frontend generate does not support multimodal inputs",
            )

        if self._use_direct_generate():
            return await self._generate_direct(prompt_ids, sampling_params, request_id)

        try:
            include_log_probs = bool(sampling_params.get("logprobs", False))
            request_id = request_id or f"dynamo-{time.time_ns()}"
            payload = self._build_frontend_completion_payload(prompt_ids, sampling_params, request_id)
            status, body_text = await self._frontend_post(payload, request_id)
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
                    status, body_text = await self._frontend_post(payload, request_id)
            if status != 200:
                raise RuntimeError(
                    "Dynamo frontend /v1/completions failed "
                    f"status={status} body={body_text[:2000]!r} "
                    f"payload_summary={self._payload_debug_summary(payload)}"
                )
            return self._completion_response_to_token_output(json.loads(body_text), include_log_probs=include_log_probs)
        except Exception:
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

    async def _frontend_post(self, payload: dict[str, Any], request_id: str) -> tuple[int, str]:
        """POST one completion to the frontend; return ``(status, body_text)``."""
        import aiohttp

        session = await self._get_http_session()
        timeout = aiohttp.ClientTimeout(total=self._frontend_request_timeout_s())
        async with session.post(
            self._frontend_completions_url(),
            json=payload,
            headers={"X-Request-Id": str(request_id)},
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
                0,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_token_ids),
                ),
            )
        sp.pop("logprobs", None)
        nvext = sp.pop("nvext", None)
        payload: dict[str, Any] = {
            "model": model,
            # vLLM's OpenAI completions endpoint accepts token-id prompts. Keep
            # Dynamo on the same token-in path as the native vLLM backend.
            "prompt": prompt_token_ids,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        nvext_fields = []
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
                stop_reason="completed" if result.get("finish_reason") else None,
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
                0,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_token_ids),
                ),
            )
        max_possible_tokens = max(0, self.config.prompt_length + self.config.response_length - len(prompt_token_ids))
        sp["max_tokens"] = int(max(0, min(max_tokens, max_possible_tokens)))
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
        token_ids = self._extract_completion_token_ids(choice, data, tokenizer)
        if token_ids is None:
            logger.warning(
                "Dynamo frontend response did not include parseable token ids; falling back to text encode. "
                "For strict token-in/token-out parity, expose generated token ids in the frontend response."
            )
            token_ids = normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))
        if not token_ids:
            raise RuntimeError(f"Dynamo frontend returned an empty completion: {data}")
        log_probs = self._extract_completion_log_probs(choice, len(token_ids), data) if include_log_probs else None
        finish_reason = choice.get("finish_reason")
        if finish_reason == "stop" or finish_reason == "length":
            stop_reason = "completed"
        elif finish_reason == "abort":
            stop_reason = "aborted"
        else:
            stop_reason = finish_reason

        return self._build_token_output(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
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

    def _build_token_output(
        self,
        token_ids: Optional[list[int]] = None,
        log_probs: Optional[list[float]] = None,
        stop_reason: Optional[str] = None,
    ):
        """Build a verl TokenOutput while preserving AgentLoop shape invariants."""
        from verl.workers.rollout.replica import TokenOutput

        token_ids = token_ids or self._fallback_token_ids()
        if log_probs is not None:
            if len(log_probs) < len(token_ids):
                log_probs = log_probs + [0.0] * (len(token_ids) - len(log_probs))
            elif len(log_probs) > len(token_ids):
                log_probs = log_probs[: len(token_ids)]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields={"global_steps": self.global_steps or 0},
        )

    def _fallback_token_ids(self) -> list[int]:
        """Return one harmless token so Dynamo never emits an empty response."""
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
    ) -> Optional[list[int]]:
        """Extract vLLM OpenAI extension token ids when available."""
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
            if tokenizer is not None:
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
                return DynamoHttpServer._normalize_log_probs(values, token_count)

        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            return None
        values = logprobs.get("token_logprobs")
        if values is None:
            values = logprobs.get("logprobs")
        if values is None:
            return None
        return DynamoHttpServer._normalize_log_probs(values, token_count)

    @staticmethod
    def _normalize_log_probs(values: list[Any], token_count: int) -> list[float]:
        """Pad/truncate selected-token logprobs to match token ids."""
        result: list[float] = []
        for value in values[:token_count]:
            result.append(0.0 if value is None else float(value))
        if len(result) < token_count:
            result.extend([0.0] * (token_count - len(result)))
        return result

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

        v1: control sidecar isn't started yet, so we fail fast.
        """
        if not self._control_endpoints:
            raise NotImplementedError(
                "DynamoHttpServer.collective_rpc requires control sidecars "
                "(set rollout.engine_kwargs.dynamo.enable_control_sidecar=True "
                "to enable). v1 generation-only smoke does not need this."
            )

        # Control sidecar protocol: REQ side sends pickled dict, RECVs reply.
        # Sequential for simplicity (one sidecar per shard, response time
        # similar across shards). If this becomes a bottleneck switch to
        # asyncio.gather over per-endpoint REQ sockets.
        import pickle

        import zmq
        import zmq.asyncio

        # v4a-6 (Iter 7.5): Iter 7.4 revealed sequential endpoint iteration
        # deadlocks `update_weights_from_ipc`. That RPC blocks until the
        # receiver's IPC loop returns, but the loop returns only after
        # sender finishes; sender depends on cupy NCCL broadcast which
        # requires ALL replicas' rollout actors to join the group. With
        # sequential iter, only ep[0]'s workers are ever woken — the
        # other 3 replicas' receivers never set up, cupy broadcast hangs,
        # everything deadlocks.
        #
        # Fix: dispatch all sidecars CONCURRENTLY via asyncio.gather so
        # all 4 replicas' workers fire update_weights_from_ipc together,
        # all REP sockets bind, cupy broadcast progresses, sender unblocks.
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
    # verl interface — lifecycle no-ops (v1) / passthroughs (v2)
    # ------------------------------------------------------------------ #

    async def wake_up(self, **kwargs):
        # NB: no node_rank guard — each per-node server wakes its OWN local
        # workers (self._control_endpoints are node-local), so all nodes must run.
        if not self._free_engine_on_train():
            logger.info("[DynamoHttpServer] wake_up: free_engine_on_train disabled, leaving Dynamo workers loaded")
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] wake_up: no control sidecar, skipping")
            return
        # bridge to engine.wake_up via control sidecar (engine method,
        # not collective_rpc — handled in sidecar).
        await self._engine_method_all("wake_up", kwargs=kwargs)

    async def sleep(self, **kwargs):
        # NB: no node_rank guard — each per-node server sleeps its OWN local
        # workers (self._control_endpoints are node-local), so all nodes must run.
        if not self._free_engine_on_train():
            logger.info("[DynamoHttpServer] sleep: free_engine_on_train disabled, leaving Dynamo workers loaded")
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] sleep: no control sidecar, skipping")
            return
        # v1 can't refit weights, so use sleep level 1 (offload weights to CPU +
        # drop KV); wake_up restores weights from CPU — no refit needed.
        kwargs.setdefault("level", 1)
        await self._engine_method_all("sleep", kwargs=kwargs)

    async def clear_kv_cache(self):
        if not self._control_endpoints:
            return
        await self._engine_method_all("reset_prefix_cache")

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        if not self._control_endpoints:
            return
        await self._engine_method_all("wait_for_requests_to_drain")

    async def abort_all_requests(self, reset_prefix_cache: bool = True):
        # dynamo doesn't expose a global abort; v1 returns no-op result so
        # RolloutReplica.abort_all_requests's gather doesn't blow up.
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        return None

    async def start_profile(self, **kwargs):
        return None

    async def stop_profile(self):
        return None

    async def _engine_method_all(self, method: str, kwargs: Optional[dict] = None):
        """Like collective_rpc but invokes a top-level AsyncLLM method
        (wake_up / sleep / reset_prefix_cache / wait_for_requests_to_drain),
        not a worker-extension RPC. Distinguished by message kind.

        v4a-6 (Iter 7.5): same parallel-dispatch fix as collective_rpc.
        Sequential iter deadlocked update_weights_from_ipc and now also
        deadlocks reset_prefix_cache post-refit."""
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
        }

        async def _call_one(idx: int, ep: str) -> None:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(ep)
                await sock.send(pickle.dumps(req))
                reply_bytes = await asyncio.wait_for(sock.recv(), timeout=120)
                reply = pickle.loads(reply_bytes)
                if not reply.get("ok"):
                    logger.warning(
                        "[DynamoHttpServer] engine_method %s failed @ %s: %s",
                        method,
                        ep,
                        reply.get("error"),
                    )
            finally:
                sock.close()

        await asyncio.gather(*[_call_one(i, ep) for i, ep in enumerate(self._control_endpoints)])
        return None

    # ------------------------------------------------------------------ #
    # v3 refit: world-size helper for NCCL group setup
    # ------------------------------------------------------------------ #

    def get_num_engine_workers(self) -> int:
        """Total number of TP worker processes across all dynamo.vllm shards
        on this node. Used by DynamoRollout.update_weights to compute the
        NCCL group world_size = 1 (broadcaster) + N (engine workers)."""
        tp = int(self.config.tensor_model_parallel_size)
        return len(self._control_endpoints) * tp

    # ------------------------------------------------------------------ #
    # refit path self-test (v2 — verifies control sidecar reachability)
    # ------------------------------------------------------------------ #

    async def _self_test_refit_path(self):
        """Verify the control-sidecar ⇄ AsyncLLM round-trip is alive.

        Refit (DynamoRollout.update_weights, v2) routes weight bytes through
        ``collective_rpc("update_weights_from_ipc", ...)`` which depends on
        a working REQ-REP loop to each ``_dynamo_vllm_with_control``
        subprocess. A silent failure here (sidecar didn't start, control
        endpoint port collision, etc.) would let ``update_weights`` appear
        to succeed while actually losing all updates — that's the exact
        bug B v4 had pre-fix.

        This self-test sends one ``collective_rpc`` request with a
        deliberately invalid method name. A reachable sidecar will reply
        with a structured error response; an unreachable one will time out.
        Either response proves the IPC path is alive.

        Skipped when no control endpoints are registered (slave node / pre-launch).
        Soft-fail by default; set env ``VERL_DYNAMO_REFIT_STRICT=1`` to
        raise on failure (recommended once v2 is the default).
        """
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
            # all-gather OOM'd. (Observed in B v5 smoke iter 2, job 2463154.)
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
        for fp in self._vllm_log_fps:
            try:
                fp.close()
            except Exception:
                pass
        self._vllm_log_fps = []
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
        state["_vllm_log_fps"] = []
        state["_vllm_log_paths"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# --------------------------------------------------------------------------- #
# DynamoReplica
# --------------------------------------------------------------------------- #


class DynamoReplica(RolloutReplica):
    """Manages one logical Dynamo serving replica across N nodes.

    Mirrors vLLMReplica.launch_servers (one DynamoHttpServer actor per node)
    but with a master/slave split:
      - first actor (node_rank=0) starts etcd + nats + frontend in addition
        to its dynamo.vllm worker subprocesses,
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

    async def init_hybrid_worker_pool(self, worker_group):
        """Initialize Dynamo as worker pool for all rollout GPUs."""
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = list(worker_group.workers)

        assert len(self.workers) % self.world_size == 0, (
            f"worker_group size {len(self.workers)} must be divisible by "
            f"dynamo logical replica world_size {self.world_size}"
        )
        num_logical_replicas = len(self.workers) // self.world_size
        await self._launch_shared_worker_pool(num_logical_replicas=num_logical_replicas)

    async def _launch_shared_worker_pool(self, num_logical_replicas: int):
        """Launch a single frontend backed by all logical replica workers."""
        from verl.utils.device import get_resource_name

        tp = self.config.tensor_model_parallel_size
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )

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
                    node_to_specs[node_id].append(
                        {
                            "replica_rank": logical_replica_rank,
                            "cuda_visible_devices": ",".join(shard_gpus),
                            "rank_offset": shard_idx * tp,
                            "label": f"replica{logical_replica_rank}_shard{shard_idx}",
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
        """Dynamo uses a NeMo-style worker-pool entrypoint instead."""
        raise RuntimeError(
            "DynamoReplica.launch_servers() is disabled because the dynamo "
            "backend uses a single shared worker pool. Call "
            "DynamoReplica.init_hybrid_worker_pool(worker_group) via "
            "AgentLoopManager instead."
        )


__all__ = ["DynamoHttpServer", "DynamoReplica"]
