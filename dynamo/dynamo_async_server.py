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

Per dynamo_design_0507.md, the actor in this design:
  1. Reserves no GPUs itself; trainer workers in colocated mode already claim
     them. We only forward CUDA_VISIBLE_DEVICES into dynamo.vllm subprocesses.
  2. Spawns + watchdogs etcd / nats-server / dynamo.vllm × N / dynamo.frontend.
  3. Never holds an AsyncLLM. generate() is unsupported on the actor; the
     trainer agent loop talks to dynamo.frontend over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

import ray
import requests
from ray.actor import ActorHandle

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
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
        v1 — most are no-op or NotImplementedError
        v2 — collective_rpc bridges to per-subprocess control sidecar
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
    ):
        # Match vLLMHttpServer's __init__ contract so vLLMReplica.launch_servers
        # can spin us up unchanged. We do NOT instantiate vLLM AsyncLLM.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ[_REPLICA_RANK_ENV] = str(replica_rank)

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(
            model_config, dataclass_type=HFModelConfig
        )
        self.rollout_mode = rollout_mode
        # workers handle is captured for parity with vLLMHttpServer; we don't
        # use it (no in-process engine, no collective_rpc destination here).
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self._cuda_visible_devices = cuda_visible_devices

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

        # Per-subprocess control sidecar endpoints (filled in
        # _start_vllm_workers); used by collective_rpc bridge in v2.
        self._control_endpoints: list[str] = []

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
        assert host and etcd_port and nats_port, (
            f"dynamo env vars missing host/ports: {host}/{etcd_port}/{nats_port}"
        )
        return {
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
    ):
        """Start subprocesses on this node.

        master_address / master_port / dp_rpc_port semantics differ from
        vLLM's: we re-purpose master_port for etcd_port and dp_rpc_port for
        nats_port (see get_master_address).
        """
        if self.node_rank == 0:
            await self._launch_master()
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

    async def _launch_master(self):
        """Master: etcd + nats + vllm workers + frontend + healthcheck."""
        # Reserve ports up-front so we know all of them before starting.
        self._etcd_port = self._dynamo_cfg().get("etcd_port") or get_free_port(self._server_address)[0]
        self._etcd_peer_port = self._dynamo_cfg().get("etcd_peer_port") or get_free_port(self._server_address)[0]
        self._nats_port = self._dynamo_cfg().get("nats_port") or get_free_port(self._server_address)[0]
        # Frontend port: 0 = auto, else honor config.
        cfg_fe = self._dynamo_cfg().get("frontend_http_port", 0) or 0
        self._frontend_port = cfg_fe if cfg_fe else get_free_port(self._server_address)[0]

        self._start_etcd()
        self._start_nats()
        self._start_vllm_workers()
        self._start_frontend()

        expected_workers = self._compute_expected_workers()
        await self._healthcheck_frontend(expected_workers=expected_workers)

        # Expose frontend to trainer.
        self._server_port = self._frontend_port
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
        tp = self.config.tensor_model_parallel_size
        per_node = max(1, self.gpus_per_node // tp)
        return per_node * self.nnodes

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
            "--listen-client-urls", f"http://0.0.0.0:{self._etcd_port}",
            "--advertise-client-urls", f"http://{self._server_address}:{self._etcd_port}",
            "--listen-peer-urls", f"http://0.0.0.0:{self._etcd_peer_port}",
            "--initial-advertise-peer-urls", peer_url,
            "--initial-cluster", f"default={peer_url}",
            "--data-dir", self._etcd_data_dir,
            "--heartbeat-interval", "500",
            "--election-timeout", "5000",
        ]
        logger.info("[DynamoHttpServer] starting etcd: %s", " ".join(cmd))
        self._etcd_process = subprocess.Popen(cmd, env=env)
        self._wait_for_etcd(_ETCD_READY_TIMEOUT_S)

    def _wait_for_etcd(self, timeout: float):
        url = f"http://localhost:{self._etcd_port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._etcd_process and self._etcd_process.poll() is not None:
                raise RuntimeError(
                    f"etcd exited with rc={self._etcd_process.returncode} "
                    f"before becoming healthy"
                )
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
        cmd = ["nats-server", "-p", str(self._nats_port)]
        logger.info("[DynamoHttpServer] starting NATS: %s", " ".join(cmd))
        self._nats_process = subprocess.Popen(cmd)
        self._wait_for_port(self._nats_port, _NATS_READY_TIMEOUT_S, "NATS")

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
            f"GPUs ({len(cvd_list)}) on this node not divisible by TP ({tp}); "
            f"cvd={self._cuda_visible_devices}"
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

        for shard_idx_local in range(n_local_shards):
            worker_cvd = ",".join(cvd_list[shard_idx_local * tp : (shard_idx_local + 1) * tp])
            control_port = get_free_port(self._server_address)[0]
            control_endpoint = f"tcp://{self._server_address}:{control_port}"
            self._control_endpoints.append(control_endpoint)

            # dynamo defaults DYN_VLLM_KV_EVENT_PORT to 20080 — collides
            # across DP shards (and across replicas sharing a node).
            # Must fit in i16 (dynamo runtime parses some ports as i16)
            # so we draw deterministically from a per-job slice
            # [20080..32767), stepped by replica_rank * 64 + shard.
            kv_event_port = 20080 + (self.replica_rank * 64 + shard_idx_local) % (32767 - 20080)
            assert kv_event_port < 32768

            env = os.environ.copy()
            env.update(self._dynamo_env_vars())
            env["CUDA_VISIBLE_DEVICES"] = worker_cvd
            env[_RANK_OFFSET_ENV] = str(shard_idx_local * tp)
            env[_REPLICA_RANK_ENV] = str(self.replica_rank)
            env["DYN_VLLM_KV_EVENT_PORT"] = str(kv_event_port)

            # Ensure subprocess can ``import recipe.dynamo._dynamo_vllm_with_control``
            # even when ray runtime_env doesn't propagate the driver's PYTHONPATH.
            # Compute the verl root from the location of this module; works on
            # any node since /workspace is the shared mount.
            recipe_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            existing_pp = env.get("PYTHONPATH", "")
            if recipe_root not in existing_pp.split(":"):
                env["PYTHONPATH"] = (
                    f"{recipe_root}:{existing_pp}" if existing_pp else recipe_root
                )
            # NB: don't set DYN_SYSTEM_PORT — dynamo's Rust runtime parses it
            # as i16 and rejects ephemeral ports >= 32768. We use our own
            # control sidecar (VERL_DYNAMO_CONTROL_ZMQ) instead.
            env[_CONTROL_ZMQ_ENV] = control_endpoint
            # Defensively unset any DYN_SYSTEM_* leaking from caller env.
            for k in list(env.keys()):
                if k.startswith("DYN_SYSTEM_"):
                    del env[k]
            # Mirrors nemo_rl/dynamo_worker.py:308-310.
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["VLLM_NO_USAGE_STATS"] = "1"

            cmd = self._build_vllm_cmd(served_model_name, tp)

            stdout_path = os.path.join(log_dir, f"vllm_shard{shard_idx_local}.log")
            stdout_fp = open(stdout_path, "w")
            self._vllm_log_fps.append(stdout_fp)

            logger.info(
                "[DynamoHttpServer] starting dynamo.vllm shard %s/%s "
                "(GPUs=%s, control=%s, log=%s): %s",
                shard_idx_local,
                n_local_shards,
                worker_cvd,
                control_endpoint,
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

    def _build_vllm_cmd(self, served_model_name: str, tp: int) -> list[str]:
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
            "--model", self.model_config.local_path,
            "--served-model-name", served_model_name,
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
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
        # Distributed executor: MP for single-node TP (no Ray inside subprocess).
        cmd += ["--distributed-executor-backend", "mp"]
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
            "--http-port", str(self._frontend_port),
            "--http-host", "0.0.0.0",
            "--router-mode", self._router_mode,
            "--discovery-backend", "etcd",
            "--namespace-prefix", self._namespace,
        ]
        log_root = os.environ.get("VERL_DYNAMO_LOG_DIR", "/tmp")
        log_path = os.path.join(log_root, f"verl_dynamo_replica{self.replica_rank}_frontend.log")
        self._frontend_log_fp = open(log_path, "w")
        logger.info(
            "[DynamoHttpServer] starting dynamo.frontend on :%s (log=%s): %s",
            self._frontend_port,
            log_path,
            " ".join(cmd),
        )
        self._frontend_process = subprocess.Popen(
            cmd, env=env, stdout=self._frontend_log_fp, stderr=subprocess.STDOUT
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
                            n_gen, expected_workers,
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
                        raise RuntimeError(
                            f"dynamo {name}[{i}] exited rc={p.returncode}"
                        )
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
        """Dispatch generation to the local dynamo.vllm subprocess via the
        control sidecar. Bypasses dynamo's HTTP frontend (which was observed
        to hang in ai-dynamo 1.0.2 + vllm 0.17 when called from the verl
        agent loop). Trades KV-router routing for a working path; OK for
        v2 training smoke."""
        from verl.workers.rollout.replica import TokenOutput

        sp = dict(sampling_params)
        max_tokens = sp.pop("max_tokens", None) or sp.pop("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = max(
                0,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_ids),
                ),
            )
        sp["max_tokens"] = int(max_tokens)

        # Drop keys we don't want forwarded to vLLM SamplingParams.
        sp.pop("logprobs", None)  # vLLM uses int, verl can pass bool — easier to drop

        if not self._control_endpoints:
            logger.error("[generate] no control sidecar; cannot run generate")
            return TokenOutput(
                token_ids=[],
                stop_reason="error: no control sidecar",
                extra_fields={"global_steps": self.global_steps or 0},
            )

        # Pick a sidecar — for a single replica with single shard there is
        # only one. With multiple shards we'd want load balancing here, but
        # v2 smoke uses 1 replica × 1 shard.
        control_endpoint = self._control_endpoints[0]

        # Detokenize prompt_ids → text. vLLM 0.17 + dynamo intercepted
        # generate hangs on TokensPrompt; TextPrompt goes through a separate
        # path that works. Lossy at multi-turn chat boundaries; acceptable
        # for v2 smoke.
        tokenizer = getattr(self.model_config, "tokenizer", None)
        prompt_text = None
        if tokenizer is not None:
            prompt_text = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)

        request = {
            "kind": "generate_direct",
            "kwargs": {
                "token_ids": list(prompt_ids),
                "prompt_text": prompt_text,
                "sampling_params": sp,
                "request_id": str(request_id),
            },
            "timeout": 600,
        }

        import pickle

        import zmq
        import zmq.asyncio

        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(control_endpoint)
            await sock.send(pickle.dumps(request))
            reply_bytes = await asyncio.wait_for(sock.recv(), timeout=600)
        except Exception as e:
            logger.exception("[generate] sidecar dispatch failed (request_id=%s)", request_id)
            return TokenOutput(
                token_ids=[],
                stop_reason=f"error: {type(e).__name__}: {e}",
                extra_fields={"global_steps": self.global_steps or 0},
            )
        finally:
            sock.close()

        reply = pickle.loads(reply_bytes)
        if not reply.get("ok"):
            logger.warning("[generate] sidecar returned error: %s", reply.get("error"))
            return TokenOutput(
                token_ids=[],
                stop_reason=f"error: {reply.get('error', 'unknown')}",
                extra_fields={"global_steps": self.global_steps or 0},
            )

        result = reply.get("result") or {}
        token_ids = list(result.get("token_ids") or [])
        finish_reason = result.get("finish_reason")
        if finish_reason == "stop" or finish_reason == "length":
            stop_reason = "completed"
        elif finish_reason == "abort":
            stop_reason = "aborted"
        else:
            stop_reason = finish_reason

        return TokenOutput(
            token_ids=token_ids,
            stop_reason=stop_reason,
            extra_fields={"global_steps": self.global_steps or 0},
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

        v1: control sidecar isn't started yet, so we fail fast — call sites
        guard with self.config.free_cache_engine etc., so most paths skip.
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

        ctx = zmq.asyncio.Context.instance()
        results: list[Any] = []
        for ep in self._control_endpoints:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(ep)
                req = {
                    "method": method if isinstance(method, str) else method.__name__,
                    "args": args,
                    "kwargs": kwargs or {},
                    "timeout": timeout,
                }
                await sock.send(pickle.dumps(req))
                reply_bytes = await asyncio.wait_for(
                    sock.recv(), timeout=timeout if timeout else 600
                )
                reply = pickle.loads(reply_bytes)
                if not reply.get("ok"):
                    raise RuntimeError(
                        f"control sidecar @ {ep} returned error: {reply.get('error')}"
                    )
                results.append(reply.get("result"))
            finally:
                sock.close()
        return results

    # ------------------------------------------------------------------ #
    # verl interface — lifecycle no-ops (v1) / passthroughs (v2)
    # ------------------------------------------------------------------ #

    async def wake_up(self, **kwargs):
        if self.node_rank != 0:
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] wake_up: no control sidecar, skipping")
            return
        # v2: bridge to engine.wake_up via control sidecar (engine method,
        # not collective_rpc — handled in sidecar).
        await self._engine_method_all("wake_up", kwargs=kwargs)

    async def sleep(self, **kwargs):
        if self.node_rank != 0:
            return
        if not self._control_endpoints:
            logger.info("[DynamoHttpServer] sleep: no control sidecar, skipping")
            return
        await self._engine_method_all("sleep", kwargs=kwargs)

    async def clear_kv_cache(self):
        if self.node_rank != 0:
            return
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
        not a worker-extension RPC. Distinguished by message kind."""
        if not self._control_endpoints:
            return None

        import pickle

        import zmq
        import zmq.asyncio

        ctx = zmq.asyncio.Context.instance()
        for ep in self._control_endpoints:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                sock.connect(ep)
                req = {
                    "kind": "engine_method",
                    "method": method,
                    "kwargs": kwargs or {},
                }
                await sock.send(pickle.dumps(req))
                reply_bytes = await asyncio.wait_for(sock.recv(), timeout=120)
                reply = pickle.loads(reply_bytes)
                if not reply.get("ok"):
                    logger.warning(
                        "[DynamoHttpServer] engine_method %s failed @ %s: %s",
                        method, ep, reply.get("error"),
                    )
            finally:
                sock.close()
        return None

    # ------------------------------------------------------------------ #
    # shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self):
        self._shutdown_requested = True
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
            replica_rank, config, model_config, gpus_per_node,
            is_reward_model, is_teacher_model, name_suffix,
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

    async def launch_servers(self):
        """Same shape as vLLMReplica.launch_servers, with master/slave split."""
        from verl.utils.device import get_resource_name

        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} != world size {self.world_size}"
        )

        # gather (node_id, GPU id) per worker
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
        worker_node_ids = [info[0] for info in worker_infos]
        worker_cvds = [info[1] for info in worker_infos]

        nnodes = self.nnodes
        gppn = self.gpus_per_replica_node
        prefix = self._get_server_name_prefix()
        suffix = self.name_suffix

        for node_rank in range(nnodes):
            node_workers = self.workers[node_rank * gppn : (node_rank + 1) * gppn]
            node_cvd = ",".join(worker_cvds[node_rank * gppn : (node_rank + 1) * gppn])
            node_id = worker_node_ids[node_rank * gppn]
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
            # Forward optional verl-side knobs + PYTHONPATH so the dynamo.vllm
            # subprocess can ``import recipe.dynamo`` on every node, including
            # slave nodes where ray actors don't inherit the driver's env.
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
                workers=node_workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gppn,
                nnodes=nnodes,
                cuda_visible_devices=node_cvd,
            )
            self.servers.append(server)

        # 1) bring up master so it can publish etcd/nats/frontend.
        master = self.servers[0]
        await master.launch_server.remote()
        master_host, master_etcd_port, master_nats_port = await master.get_master_address.remote()
        fe_host, fe_port = await master.get_server_address.remote()

        # 2) bring up slaves (in parallel) using master's etcd/nats.
        slave_launches = []
        for server in self.servers[1:]:
            slave_launches.append(
                server.launch_server.remote(
                    master_address=master_host,
                    master_port=master_etcd_port,
                    dp_rpc_port=master_nats_port,
                )
            )
            # And tell them the master frontend so get_server_address answers.
            slave_launches.append(server.set_master_frontend.remote(fe_host, fe_port))
        if slave_launches:
            await asyncio.gather(*slave_launches)

        # 3) advertise the master frontend to the trainer. All ranks talk to
        # the same URL regardless of which node they're on.
        self._server_handle = master
        self._server_address = (
            f"[{fe_host}]:{fe_port}"
            if is_valid_ipv6_address(fe_host)
            else f"{fe_host}:{fe_port}"
        )
        logger.info(
            "[DynamoReplica %s] ready: server_address=%s",
            self.replica_rank, self._server_address,
        )


__all__ = ["DynamoHttpServer", "DynamoReplica"]
