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
"""CPU-only tests for the Dynamo × SGLang rollout backend.

Deliberately avoids importing ``dynamo_sglang_rollout`` at module scope: that
module pulls in verl's sglang ServerAdapter, which imports ``sglang`` and patches
its engine entrypoint at import time. These tests run in the vLLM container too.
"""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from recipe.dynamo.dynamo_async_server import DynamoHttpServer
from recipe.dynamo.dynamo_sglang_engine import (
    API_DYNAMO,
    API_NATIVE,
    LAUNCH_MODE_SIDECAR,
    LAUNCH_MODE_WORKER,
    ROUTE_UPDATE_WEIGHTS_FROM_TENSOR,
    DynamoSGLangControlClient,
    DynamoSGLangControlError,
    sglang_launch_mode,
)


def _make_server(dynamo_cfg: dict | None = None, **rollout_cfg) -> DynamoHttpServer:
    server = object.__new__(DynamoHttpServer)
    cfg = {
        "tensor_model_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 1024,
        "max_num_batched_tokens": None,
        "max_num_seqs": None,
        "dtype": None,
        "enforce_eager": False,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "enable_sleep_mode": False,
        "engine_kwargs": {"dynamo": dynamo_cfg if dynamo_cfg is not None else {}},
    }
    cfg.update(rollout_cfg)
    server.config = SimpleNamespace(**cfg)
    server.model_config = SimpleNamespace(local_path="/models/test-model", trust_remote_code=False)
    server._engine_control_endpoints = []
    server._control_endpoints = []
    server._sglang_clients = None
    return server


# --------------------------------------------------------------------------- #
# engine selection
# --------------------------------------------------------------------------- #


def test_engine_defaults_to_vllm():
    assert _make_server()._engine_kind() == "vllm"
    assert _make_server()._is_sglang() is False


def test_engine_sglang_selected():
    server = _make_server({"engine": "sglang"})
    assert server._engine_kind() == "sglang"
    assert server._is_sglang() is True


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        _make_server({"engine": "trtllm"})._engine_kind()


# --------------------------------------------------------------------------- #
# CLI mapping: vLLM flags -> SGLang ServerArgs flags
# --------------------------------------------------------------------------- #


def _cmd(dynamo_cfg=None, **rollout_cfg) -> list[str]:
    cfg = {"engine": "sglang"}
    cfg.update(dynamo_cfg or {})
    server = _make_server(cfg, **rollout_cfg)
    tp = server.config.tensor_model_parallel_size
    return server._build_sglang_cmd("test-model", tp)


def test_sglang_cmd_core_mapping():
    cmd = _cmd(tensor_model_parallel_size=4, gpu_memory_utilization=0.7, max_model_len=2048)
    assert cmd[1:3] == ["-m", "dynamo.sglang"]
    # The whole point of the mapping table: none of the vLLM spellings survive.
    for vllm_flag in (
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--enable-prefix-caching",
        "--enable-sleep-mode",
        "--worker-extension-cls",
    ):
        assert vllm_flag not in cmd, f"{vllm_flag} leaked into the sglang command line"
    assert cmd[cmd.index("--tp-size") + 1] == "4"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.7"
    assert cmd[cmd.index("--context-length") + 1] == "2048"
    assert cmd[cmd.index("--model-path") + 1] == "/models/test-model"


def test_sglang_cmd_inverted_flags():
    """SGLang's radix cache and CUDA graph default ON, so the flags invert."""
    on = _cmd(enable_prefix_caching=True, enforce_eager=False)
    assert "--disable-radix-cache" not in on
    assert "--disable-cuda-graph" not in on

    off = _cmd(enable_prefix_caching=False, enforce_eager=True)
    assert "--disable-radix-cache" in off
    assert "--disable-cuda-graph" in off


def test_sleep_mode_maps_to_memory_saver():
    assert "--enable-memory-saver" in _cmd(enable_sleep_mode=True)
    assert "--enable-memory-saver" not in _cmd(enable_sleep_mode=False)


def test_enable_rl_on_by_default():
    """Keep forwarding the worker's RL flag independently of native routes."""
    assert "--enable-rl" in _cmd()
    assert "--enable-rl" not in _cmd({"sglang": {"enable_rl": False}})


def test_cache_flush_route_is_registered_without_extra_args():
    for enable_rl in [True, False]:
        cmd = _cmd({"sglang": {"enable_rl": enable_rl}})
        assert cmd[cmd.index("--engine-route") + 1] == "flush_cache:tm"


@pytest.mark.parametrize(
    "extra",
    [
        ["--engine-route", "flush_cache:tm"],
        ["--engine-route=flush_cache:tm"],
        ["--engine-route", "flush_cache=custom_flush:tm"],
    ],
)
def test_cache_flush_route_preserves_explicit_registration(extra):
    cmd = _cmd({"sglang": {"extra_args": extra}})
    assert sum(arg.startswith("--engine-route") for arg in cmd) == 1
    assert cmd[-len(extra) :] == extra


def test_other_engine_routes_do_not_disable_cache_flush():
    cmd = _cmd({"sglang": {"extra_args": ["--engine-route", "get_load:tm"]}})
    assert "flush_cache:tm" in cmd and "get_load:tm" in cmd


def test_page_size_falls_back_to_thunderagent_block_size():
    cmd = _cmd({"thunderagent": {"router_block_size": 32}})
    assert cmd[cmd.index("--page-size") + 1] == "32"
    # explicit sglang.page_size wins
    cmd = _cmd({"sglang": {"page_size": 16}, "thunderagent": {"router_block_size": 32}})
    assert cmd[cmd.index("--page-size") + 1] == "16"


def test_attention_backend_defaults_to_flashinfer():
    """Mirror verl's native sglang server: SGLang's Hopper default (fa3) decodes ~8%
    slower per step with the router page size, and fa3 + cuda graph is broken on
    sglang>=0.5.12. Explicit config or an extra_args flag must still win."""
    cmd = _cmd()
    assert cmd[cmd.index("--attention-backend") + 1] == "flashinfer"
    cmd = _cmd({"sglang": {"attention_backend": "fa3"}})
    assert cmd[cmd.index("--attention-backend") + 1] == "fa3"
    cmd = _cmd({"sglang": {"extra_args": ["--attention-backend", "triton"]}})
    assert cmd.count("--attention-backend") == 1
    assert cmd[cmd.index("--attention-backend") + 1] == "triton"


def test_sglang_cmd_publishes_kv_events_like_vllm():
    """KV events feed the KV router's index. dynamo.sglang derives use_kv_events from
    --kv-events-config alone (dynamo/sglang/args.py), so without the flag the router
    only ever sees predict-on-route entries (100-step 30B run before 2026-09-09:
    router-estimated prefix hit 0.865 vs engine-measured 0.627)."""
    server = _make_server({"engine": "sglang"})
    assert server._kv_events_enabled() is True
    assert _make_server({"engine": "sglang", "enable_kv_events": False})._kv_events_enabled() is False
    js = DynamoHttpServer._build_kv_events_config_json(5557)
    # the JSON dynamo's own sglang docs use (docs/integrations/flexkv-integration.md)
    assert json.loads(js) == {
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": "tcp://*:5557",
        "enable_kv_cache_events": True,
    }
    cmd = server._build_sglang_cmd("test-model", 1, kv_events_config_json=js)
    assert cmd[cmd.index("--kv-events-config") + 1] == js
    # no config -> no flag (the worker then registers use_kv_events=False)
    assert "--kv-events-config" not in server._build_sglang_cmd("test-model", 1)


def test_vllm_cmd_kv_events_flag_follows_the_same_contract():
    """enable_kv_events=false must reach the vLLM shard as *no* --kv-events-config, exactly
    like the sglang builder; the launcher passes None in that case."""
    server = _make_server({"engine": "vllm"})
    js = DynamoHttpServer._build_kv_events_config_json(5557)
    cmd = server._build_vllm_cmd("test-model", 1, kv_events_config_json=js)
    assert cmd[cmd.index("--kv-events-config") + 1] == js
    assert "--kv-events-config" not in server._build_vllm_cmd("test-model", 1)
    assert "--kv-events-config" not in server._build_vllm_cmd("test-model", 1, kv_events_config_json=None)


def _router_args(dynamo_cfg: dict, mode: str) -> list[str]:
    server = _make_server(dynamo_cfg)
    server._router_mode = mode
    return server._frontend_router_args()


def test_session_affinity_flag_off_by_default_and_in_every_router_mode():
    for mode in ("kv", "round-robin"):
        assert "--router-session-affinity-ttl-secs" not in _router_args({"engine": "sglang"}, mode)
        args = _router_args({"engine": "sglang", "router_session_affinity_ttl_secs": 600}, mode)
        assert args[args.index("--router-session-affinity-ttl-secs") + 1] == "600"
    for off in (None, 0, "None", "null", ""):
        cfg = {"engine": "sglang", "router_session_affinity_ttl_secs": off}
        assert "--router-session-affinity-ttl-secs" not in _router_args(cfg, "kv")
    with pytest.raises(ValueError):
        _make_server({"engine": "sglang", "router_session_affinity_ttl_secs": "soon"})._session_affinity_ttl_secs()
    with pytest.raises(ValueError):
        _make_server({"engine": "sglang", "router_session_affinity_ttl_secs": 31_536_001})._session_affinity_ttl_secs()


def test_session_header_only_when_affinity_configured():
    assert _make_server({"engine": "sglang"})._frontend_headers("traj-1") == {"X-Request-Id": "traj-1"}
    on = _make_server({"engine": "sglang", "router_session_affinity_ttl_secs": 600})._frontend_headers("traj-1")
    assert on == {"X-Request-Id": "traj-1", "x-dynamo-session-id": "traj-1"}


def test_extra_args_forwarded():
    cmd = _cmd({"sglang": {"extra_args": ["--schedule-policy", "fcfs"]}})
    assert cmd[-2:] == ["--schedule-policy", "fcfs"]


# --------------------------------------------------------------------------- #
# control-plane invariants
# --------------------------------------------------------------------------- #


def test_sglang_requires_system_port():
    """DYN_SYSTEM_PORT is the sglang control plane, not an optional metrics extra."""
    server = _make_server({"engine": "sglang", "enable_worker_system_metrics": False})
    server.replica_rank = 0
    server.node_rank = 0
    server._cuda_visible_devices = "0"
    server._worker_specs = None
    with pytest.raises(ValueError, match="requires enable_worker_system_metrics"):
        server._start_engine_workers()


def test_num_engine_workers_counts_sglang_shards():
    server = _make_server({"engine": "sglang"}, tensor_model_parallel_size=2)
    server._engine_control_endpoints = ["http://h:11000", "http://h:11001"]
    assert server.get_num_engine_workers() == 4


# --------------------------------------------------------------------------- #
# shard mapping — the silent-corruption guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("local_world_size,tp", [(8, 1), (8, 2), (8, 4), (8, 8), (4, 2)])
def test_trainer_shard_index_matches_launcher_gpu_slicing(local_world_size, tp):
    """Every trainer rank must resolve to the shard that owns its GPU.

    The launcher (``_start_engine_workers``) gives shard *i* the GPU slice
    ``cvd[i*tp:(i+1)*tp]``. The adapter computes ``shard = local_rank // tp``.
    If those two ever drift, weight sync posts CUDA-IPC handles to an engine on a
    different GPU — which does not raise, it just trains against wrong weights.
    """
    cvd = [str(i) for i in range(local_world_size)]
    n_shards = local_world_size // tp
    launcher_owner = {}
    for shard_idx in range(n_shards):
        for gpu in cvd[shard_idx * tp : (shard_idx + 1) * tp]:
            launcher_owner[gpu] = shard_idx

    for local_rank in range(local_world_size):
        adapter_shard = local_rank // tp  # SGLangServerAdapter._shard_idx_local
        assert adapter_shard == launcher_owner[str(local_rank)]
        assert 0 <= adapter_shard < n_shards


def test_tp_group_ranks_are_contiguous_and_tp_aligned():
    """The shard TP process group is [k*tp, (k+1)*tp) over global ranks."""
    world_size, tp = 16, 4
    groups = [list(range(base, base + tp)) for base in range(0, world_size, tp)]
    assert len(groups) == world_size // tp
    assert [r for g in groups for r in g] == list(range(world_size))
    for rank in range(world_size):
        owning = [g for g in groups if rank in g]
        assert len(owning) == 1
        assert owning[0][0] == (rank // tp) * tp


# --------------------------------------------------------------------------- #
# control client
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def text(self):
        import json

        return json.dumps(self._payload)

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else {"status": "ok"}
        self.calls = []
        self.closed = False

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return _FakeResponse(self.status, self.payload)


def _client(session, **kwargs):
    client = DynamoSGLangControlClient("http://worker:11000", **kwargs)
    client._session = session
    return client


def test_route_url_shape():
    client = DynamoSGLangControlClient("http://worker:11000/")
    assert client.route_url("release_memory_occupation") == ("http://worker:11000/engine/release_memory_occupation")


@pytest.mark.asyncio
async def test_update_weights_from_tensor_base64_encodes():
    session = _FakeSession(payload={"success": True, "message": "ok"})
    client = _client(session)
    blobs = [b"\x00\x01\xff-not-utf8", b"second"]
    req = SimpleNamespace(serialized_named_tensors=blobs, load_format=None, flush_cache=False)

    await client.update_weights_from_tensor(req)

    url, body = session.calls[0]
    assert url.endswith(f"/engine/{ROUTE_UPDATE_WEIGHTS_FROM_TENSOR}")
    # JSON has no bytes; the wire form must be b64 and must round-trip exactly.
    assert [base64.b64decode(v) for v in body["serialized_named_tensors"]] == blobs
    assert body["flush_cache"] is False


@pytest.mark.asyncio
async def test_update_weights_survives_full_byte_range():
    """Every byte value must round-trip: a payload SGLang cannot deserialize does not
    return an error, it kills the worker process (M0c, job 16215105)."""
    session = _FakeSession(payload={"success": True})
    client = _client(session)
    blob = bytes(range(256))
    req = SimpleNamespace(serialized_named_tensors=[blob], load_format=None, flush_cache=True)

    await client.update_weights_from_tensor(req)

    _, body = session.calls[0]
    assert base64.b64decode(body["serialized_named_tensors"][0]) == blob


@pytest.mark.asyncio
async def test_error_status_body_raises():
    """A 200 with {"status": "error"} must not be mistaken for success."""
    client = _client(_FakeSession(payload={"status": "error", "message": "memory control not supported"}))
    with pytest.raises(DynamoSGLangControlError, match="memory control not supported"):
        await client.release_memory_occupation(tags=["kv_cache"])


@pytest.mark.asyncio
async def test_success_false_body_raises():
    client = _client(_FakeSession(payload={"success": False, "message": "deserialize failed"}))
    req = SimpleNamespace(serialized_named_tensors=[b"x"], load_format=None, flush_cache=False)
    with pytest.raises(DynamoSGLangControlError, match="deserialize failed"):
        await client.update_weights_from_tensor(req)


@pytest.mark.asyncio
async def test_http_error_raises():
    client = _client(_FakeSession(status=404, payload={"error": "not found"}))
    with pytest.raises(DynamoSGLangControlError, match="HTTP 404"):
        await client.flush_cache()


@pytest.mark.asyncio
async def test_flush_cache_uses_explicit_native_route():
    """PR13951 requires an explicitly configured native flush route."""
    session = _FakeSession()
    client = _client(session)
    await client.flush_cache()
    url, body = session.calls[0]
    assert url.endswith("/engine/flush_cache")
    assert body == {}


@pytest.mark.asyncio
async def test_memory_occupation_tags_passed_through():
    session = _FakeSession()
    client = _client(session)
    await client.release_memory_occupation(tags=["kv_cache", "weights"])
    assert session.calls == [
        ("http://worker:11000/engine/pause_generation", {"mode": "abort"}),
        ("http://worker:11000/engine/release_memory_occupation", {"tags": ["kv_cache", "weights"]}),
    ]


@pytest.mark.asyncio
async def test_memory_restore_precedes_generation_and_discovery_resume():
    session = _FakeSession()
    client = _client(session)
    await client.resume_memory_occupation(tags=["kv_cache", "weights"])
    assert session.calls == [
        ("http://worker:11000/engine/resume_memory_occupation", {"tags": ["kv_cache", "weights"]}),
        ("http://worker:11000/engine/continue_generation", {}),
    ]


@pytest.mark.asyncio
async def test_failed_memory_restore_keeps_generation_paused():
    session = _FakeSession(payload={"success": False, "message": "restore failed"})
    client = _client(session)
    with pytest.raises(DynamoSGLangControlError, match="restore failed"):
        await client.resume_memory_occupation(tags=["kv_cache", "weights"])
    assert [url for url, _ in session.calls] == ["http://worker:11000/engine/resume_memory_occupation"]


@pytest.mark.asyncio
async def test_failed_pause_prevents_memory_release():
    session = _FakeSession(status=500, payload={"error": "pause failed"})
    client = _client(session)
    with pytest.raises(DynamoSGLangControlError, match="HTTP 500"):
        await client.release_memory_occupation(tags=["kv_cache"])
    assert [url for url, _ in session.calls] == ["http://worker:11000/engine/pause_generation"]


@pytest.mark.asyncio
async def test_ready_probes_the_native_flush_route():
    session = _FakeSession()
    client = _client(session)
    assert await client.wait_ready(timeout_s=1, poll_s=0)
    assert session.calls == [("http://worker:11000/engine/flush_cache", {})]


# --------------------------------------------------------------------------- #
# sidecar mode: stock SGLang + dynamo.sglang.sidecar, native SGLang control API
# --------------------------------------------------------------------------- #


class _RawResponse(_FakeResponse):
    """A 200 whose body is not a JSON object, as SGLang's native routes answer."""

    def __init__(self, status, body_text):
        super().__init__(status, None)
        self._body_text = body_text

    async def text(self):
        return self._body_text

    async def json(self, content_type=None):
        return json.loads(self._body_text)  # raises for plain text, None for "null"


class _RawSession(_FakeSession):
    def __init__(self, body_text, status=200):
        super().__init__(status=status)
        self.body_text = body_text

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return _RawResponse(self.status, self.body_text)


def _native_client(session):
    client = DynamoSGLangControlClient("http://engine:2048", api=API_NATIVE)
    client._session = session
    return client


def _sidecar_cfg(**sglang):
    return {"engine": "sglang", "sglang": {"launch_mode": "sidecar", **sglang}}


def test_launch_mode_defaults_to_worker_and_is_validated():
    assert sglang_launch_mode(None) == LAUNCH_MODE_WORKER
    assert sglang_launch_mode({}) == LAUNCH_MODE_WORKER
    assert sglang_launch_mode({"launch_mode": " Sidecar "}) == LAUNCH_MODE_SIDECAR
    with pytest.raises(ValueError, match="launch_mode"):
        sglang_launch_mode({"launch_mode": "grpc"})
    assert _make_server({"engine": "sglang"})._is_sglang_sidecar() is False
    assert _make_server(_sidecar_cfg())._is_sglang_sidecar() is True
    # launch_mode only means something for the sglang engine
    assert _make_server({"engine": "vllm", "sglang": {"launch_mode": "sidecar"}})._is_sglang_sidecar() is False


def _sidecar_engine_cmd(dynamo_cfg=None, **rollout_cfg):
    server = _make_server(dynamo_cfg if dynamo_cfg is not None else _sidecar_cfg(), **rollout_cfg)
    return server._build_sglang_cmd(
        "test-model",
        server.config.tensor_model_parallel_size,
        nccl_port=5000,
        kv_events_config_json=DynamoHttpServer._build_kv_events_config_json(5557),
        sidecar_engine=("10.0.0.1", 2048, 2049),
    )


def test_sidecar_engine_cmd_is_stock_sglang_with_http_and_grpc():
    cmd = _sidecar_engine_cmd(tensor_model_parallel_size=4, gpu_memory_utilization=0.7, enable_sleep_mode=True)
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert cmd[cmd.index("--host") + 1] == "10.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "2048"
    assert cmd[cmd.index("--grpc-port") + 1] == "2049"
    # without it the sidecar silently drops the native /generate passthrough
    assert "--incremental-streaming-output" in cmd
    # shared engine mapping is unchanged
    assert cmd[cmd.index("--tp-size") + 1] == "4"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.7"
    assert cmd[cmd.index("--nccl-port") + 1] == "5000"
    assert "--enable-memory-saver" in cmd
    assert json.loads(cmd[cmd.index("--kv-events-config") + 1])["endpoint"] == "tcp://*:5557"
    # dynamo.sglang-only flags would make sglang.launch_server exit at argparse
    assert "--enable-rl" not in cmd
    assert not any(arg.startswith("--engine-route") for arg in cmd)


def test_sidecar_engine_rejects_engine_route_extra_args():
    with pytest.raises(ValueError, match="engine-route"):
        _sidecar_engine_cmd(_sidecar_cfg(extra_args=["--engine-route", "flush_cache:tm"]))
    with pytest.raises(ValueError, match="engine-route"):
        _sidecar_engine_cmd(_sidecar_cfg(extra_args=["--engine-route=flush_cache:tm"]))


def test_sidecar_cmd_raises_the_startup_deadline():
    server = _make_server(_sidecar_cfg())
    cmd = server._build_sglang_sidecar_cmd("http://10.0.0.1:2049")
    assert cmd[1:3] == ["-m", "dynamo.sglang.sidecar"]
    assert cmd[cmd.index("--grpc-endpoint") + 1] == "http://10.0.0.1:2049"
    # the sidecar's own default (300 s) is shorter than a large-model load
    assert int(cmd[cmd.index("--grpc-startup-deadline-secs") + 1]) >= 300

    server = _make_server(_sidecar_cfg(sidecar_startup_deadline_s=1800, sidecar_extra_args=["--grpc-connections", "8"]))
    cmd = server._build_sglang_sidecar_cmd("http://10.0.0.1:2049")
    assert cmd[cmd.index("--grpc-startup-deadline-secs") + 1] == "1800"
    assert cmd[-2:] == ["--grpc-connections", "8"]


def test_stable_kv_event_ports_avoid_ephemeral_range_and_are_unique():
    # The engine binds its KV-event port minutes after it is picked; a port in
    # Linux's ephemeral range (32768-60999) can be taken by an outgoing
    # connection in between (seen: ZMQError Address already in use, port 42160).
    seen = set()
    for replica_rank in range(4):
        for node_rank in range(16):
            server = _make_server({"engine": "sglang", "stable_kv_event_ports": True})
            server.replica_rank, server.node_rank = replica_rank, node_rank
            server._server_address = "127.0.0.1"
            server._allocated_tcp_ports = set()
            server._can_bind_tcp_port = lambda port, bind_wildcard=False: True
            for shard_idx in range(8):
                port = server._allocate_kv_event_port(shard_idx)
                assert 60999 < port <= 65535, (replica_rank, node_rank, shard_idx, port)
                assert port not in seen
                seen.add(port)


def test_spawn_sidecar_shard_starts_sidecar_then_engine(tmp_path, monkeypatch):
    server = _make_server(_sidecar_cfg(), tensor_model_parallel_size=2)
    server.replica_rank = 0
    server.node_rank = 0
    server._server_address = "127.0.0.1"
    server._allocated_tcp_ports = set()
    server._engine_processes = []
    server._engine_log_fps = []
    server._engine_log_paths = []

    launched = []

    class _Proc:
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            launched.append((cmd, env))

    monkeypatch.setattr("recipe.dynamo.dynamo_async_server.subprocess.Popen", _Proc)
    spec = SimpleNamespace(replica_rank=0, rank_offset=2, cuda_visible_devices="2,3", label="replica0_shard1")
    env = {"CUDA_VISIBLE_DEVICES": "2,3", "DYN_SYSTEM_PORT": "11008", "DYN_NAMESPACE": "ns"}
    try:
        server._spawn_sglang_sidecar_shard(
            spec=spec,
            spec_idx=1,
            n_specs=2,
            served_model_name="test-model",
            tp=2,
            env=env,
            kv_events_config_json=None,
            log_dir=str(tmp_path),
        )
    finally:
        for fp in server._engine_log_fps:
            fp.close()

    (sidecar_cmd, sidecar_env), (engine_cmd, engine_env) = launched
    assert sidecar_cmd[1:3] == ["-m", "dynamo.sglang.sidecar"]
    assert engine_cmd[1:3] == ["-m", "sglang.launch_server"]
    # sidecar dials exactly the engine's gRPC port; control goes to its HTTP port
    grpc_port = engine_cmd[engine_cmd.index("--grpc-port") + 1]
    http_port = engine_cmd[engine_cmd.index("--port") + 1]
    assert grpc_port != http_port
    assert sidecar_cmd[sidecar_cmd.index("--grpc-endpoint") + 1] == f"http://127.0.0.1:{grpc_port}"
    assert server._engine_control_endpoints == [f"http://127.0.0.1:{http_port}"]
    # GPUs belong to the engine; the Dynamo system port to the sidecar
    assert engine_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert sidecar_env["CUDA_VISIBLE_DEVICES"] == ""
    assert sidecar_env["DYN_SYSTEM_PORT"] == "11008"
    assert "DYN_SYSTEM_PORT" not in engine_env
    assert sidecar_env["DYN_NAMESPACE"] == "ns"
    # the caller's env is not mutated
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    # both processes are watched, logs line up with processes, one shard counted
    assert len(server._engine_processes) == 2
    assert [p.rsplit("/", 1)[-1] for p in server._engine_log_paths] == [
        "replica0_shard1_sidecar.log",
        "replica0_shard1_engine.log",
    ]
    assert server.get_num_engine_workers() == 2


def test_sidecar_mode_builds_native_control_clients():
    server = _make_server(_sidecar_cfg())
    server._engine_control_endpoints = ["http://h:2048", "http://h:2056"]
    assert [c.api for c in server._sglang_control_clients()] == [API_NATIVE, API_NATIVE]
    worker = _make_server({"engine": "sglang"})
    worker._engine_control_endpoints = ["http://h:11000"]
    assert [c.api for c in worker._sglang_control_clients()] == [API_DYNAMO]


def test_native_route_url_has_no_engine_or_control_prefix():
    client = DynamoSGLangControlClient("http://engine:2048/", api=API_NATIVE)
    assert client.route_url("release_memory_occupation") == "http://engine:2048/release_memory_occupation"
    assert client.route_url(ROUTE_UPDATE_WEIGHTS_FROM_TENSOR) == "http://engine:2048/update_weights_from_tensor"
    assert client.route_url("control/update_weight_version") == "http://engine:2048/update_weight_version"
    with pytest.raises(DynamoSGLangControlError, match="call_tokenizer_manager"):
        client.route_url("call_tokenizer_manager")
    with pytest.raises(ValueError, match="api"):
        DynamoSGLangControlClient("http://engine:2048", api="grpc")


@pytest.mark.asyncio
async def test_native_plain_text_flush_is_success():
    session = _RawSession("Cache flushed.\nPlease check backend logs for more details.\n")
    client = _native_client(session)
    result = await client.flush_cache()
    assert result["status"] == "ok"
    assert session.calls == [("http://engine:2048/flush_cache", {})]
    assert await client.wait_ready(timeout_s=1, poll_s=0)


@pytest.mark.asyncio
async def test_native_plain_text_is_still_an_error_on_the_dynamo_dialect():
    client = _client(_RawSession("Cache flushed.\n"))
    with pytest.raises(DynamoSGLangControlError, match="non-JSON"):
        await client.flush_cache()


@pytest.mark.asyncio
async def test_native_flush_failure_status_raises():
    """SGLang answers 400 (plain text) when it refuses to flush with requests running."""
    client = _native_client(_RawSession("Flush cache failed.\n", status=400))
    with pytest.raises(DynamoSGLangControlError, match="HTTP 400"):
        await client.flush_cache()


@pytest.mark.asyncio
async def test_native_null_memory_occupation_is_success_and_pauses_first():
    session = _RawSession("null")
    client = _native_client(session)
    await client.release_memory_occupation(tags=["kv_cache"])
    await client.resume_memory_occupation(tags=["kv_cache"])
    assert session.calls == [
        ("http://engine:2048/pause_generation", {"mode": "abort"}),
        ("http://engine:2048/release_memory_occupation", {"tags": ["kv_cache"]}),
        ("http://engine:2048/resume_memory_occupation", {"tags": ["kv_cache"]}),
        ("http://engine:2048/continue_generation", {}),
    ]


@pytest.mark.asyncio
async def test_native_weight_update_wire_format_is_unchanged():
    session = _FakeSession(payload={"success": True, "message": "ok"})
    client = _native_client(session)
    blob = bytes(range(256))
    req = SimpleNamespace(serialized_named_tensors=[blob], load_format=None, flush_cache=False)
    await client.update_weights_from_tensor(req)
    url, body = session.calls[0]
    assert url == "http://engine:2048/update_weights_from_tensor"
    assert base64.b64decode(body["serialized_named_tensors"][0]) == blob
    # a {"success": false} body still fails on the native dialect
    client = _native_client(_FakeSession(payload={"success": False, "message": "deserialize failed"}))
    with pytest.raises(DynamoSGLangControlError, match="deserialize failed"):
        await client.update_weights_from_tensor(req)


@pytest.mark.asyncio
async def test_native_weight_readback_uses_get_weights_by_name():
    session = _FakeSession(payload=[0.5, -0.25])
    client = _native_client(session)
    assert await client.get_weights_by_name("model.embed_tokens.weight", truncate_size=2) == [0.5, -0.25]
    assert session.calls == [
        ("http://engine:2048/get_weights_by_name", {"name": "model.embed_tokens.weight", "truncate_size": 2})
    ]


@pytest.mark.asyncio
async def test_worker_weight_readback_keeps_the_legacy_passthrough():
    session = _FakeSession(payload={"result": [0.5]})
    client = _client(session)
    await client.get_weights_by_name("w", truncate_size=1)
    url, body = session.calls[0]
    assert url == "http://worker:11000/engine/call_tokenizer_manager"
    assert body["args"] == [{"io_struct.GetWeightsByNameReqInput": {"name": "w", "truncate_size": 1}}]


# --------------------------------------------------------------------------- #
# release/resume tag symmetry — regression for the M2 scheduler kill
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# shard index must come from the GLOBAL rank — regression for the TP=2 silent bug
# --------------------------------------------------------------------------- #


def _verl_replica_local_rank(global_rank: int, tp: int, dp: int = 1, pp: int = 1) -> int:
    """Reproduce verl sglang ServerAdapter's local_rank (see sglang_rollout.py).

    rollout_world_size = tp * dp * pp; rollout_rank = rank % rollout_world_size;
    local_rank = rollout_rank % local_world_size. The key property: it is
    REPLICA-relative, so it never exceeds the replica's world size.
    """
    rollout_world_size = tp * dp * pp
    return global_rank % rollout_world_size


@pytest.mark.parametrize("local_world_size,tp", [(8, 2), (8, 4), (4, 2)])
def test_shard_index_uses_global_rank_not_replica_local_rank(local_world_size, tp):
    """Deriving the shard from verl's local_rank collapses every rank onto shard 0.

    Observed on 8xH100 TP=2 (job 16232079): all eight ranks logged shard=0 while
    four shards were registered, so every rank posted its CUDA-IPC handles to
    shard 0 and the other three engines kept stale weights. Nothing raised.
    At tp=1 the wrong formula happens to give the right answer, which is why six
    earlier green runs never caught it.
    """
    n_shards = local_world_size // tp
    shards_from_global = set()
    shards_from_replica_local = set()
    for global_rank in range(local_world_size):
        node_local = global_rank % local_world_size
        shards_from_global.add(node_local // tp)
        shards_from_replica_local.add(_verl_replica_local_rank(global_rank, tp) // tp)

    assert shards_from_global == set(range(n_shards)), "global-rank derivation must cover every shard"
    # The buggy derivation cannot reach past shard 0 whenever tp == replica width.
    assert shards_from_replica_local == {0}
    assert len(shards_from_global) > 1, "parametrisation must actually exercise >1 shard"


def test_single_shard_hides_the_bug_regardless_of_tp():
    """Pin down what actually hid the bug: ONE shard, not tp=1.

    The first explanation ("tp=1 makes both formulas agree") was wrong, and this
    test caught it. At tp=1 verl's rollout_world_size is 1, so its local_rank is 0
    for *every* rank — the buggy formula collapses to shard 0 there too. It only
    looked correct because the earlier runs used a single GPU, where shard 0 is the
    only shard that exists. An 8-GPU tp=1 run (8 shards) would have failed just as
    the tp=2 run did.
    """
    # one GPU -> one shard: both formulas trivially agree, bug invisible
    for tp in (1,):
        assert (0 % 1) // tp == _verl_replica_local_rank(0, tp) // tp

    # 8 GPUs, tp=1 -> 8 shards: the buggy formula still collapses to shard 0
    local_world_size, tp = 8, 1
    buggy = {_verl_replica_local_rank(r, tp) // tp for r in range(local_world_size)}
    correct = {(r % local_world_size) // tp for r in range(local_world_size)}
    assert buggy == {0}, "verl local_rank is 0 for every rank when tp*dp*pp == 1"
    assert correct == set(range(8))


@pytest.mark.parametrize("local_world_size,tp", [(8, 2), (8, 4), (8, 8), (4, 2)])
def test_shard_and_tp_group_agree(local_world_size, tp):
    """The shard a rank talks to must own the GPUs of its TP group."""
    for global_rank in range(local_world_size):
        node_local = global_rank % local_world_size
        shard = node_local // tp
        tp_group_src = (global_rank // tp) * tp
        # the TP group's first rank must sit at the start of that shard's GPU slice
        assert tp_group_src % local_world_size == shard * tp


# --------------------------------------------------------------------------- #
# engine dispatch lives in ONE place (rollout.name=dynamo + engine=...)
# --------------------------------------------------------------------------- #


def test_engine_dispatch_reads_config_not_rollout_name():
    """Selecting the engine must be a single config key, like every other verl backend.

    An earlier revision registered a second rollout name (``dynamo_sglang``) with its
    own trainer yaml and entry point, so the choice had to be repeated in three
    places that could disagree — and the extra name silently missed verl's own
    ``rollout.name == "sglang"`` special-cases.
    """
    from recipe.dynamo.dynamo_rollout import _dynamo_engine

    assert _dynamo_engine(None) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs=None)) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={})) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={"dynamo": {}})) == "vllm"
    assert _dynamo_engine(SimpleNamespace(engine_kwargs={"dynamo": {"engine": "sglang"}})) == "sglang"


def test_registry_exposes_exactly_one_dynamo_name():
    """No second rollout name for the sglang engine."""
    import recipe.dynamo.register  # noqa: F401  (registers on import)

    from verl.workers.rollout.base import _ROLLOUT_REGISTRY

    dynamo_names = {name for (name, _mode) in _ROLLOUT_REGISTRY if name.startswith("dynamo")}
    assert dynamo_names == {"dynamo"}, f"unexpected dynamo rollout names: {dynamo_names}"


@pytest.mark.parametrize("nnodes,local_world_size,tp", [(2, 8, 2), (2, 8, 4), (4, 8, 2)])
def test_node_rank_uses_global_rank_not_replica_relative(nnodes, local_world_size, tp):
    """Every rank must resolve the DynamoHttpServer actor on ITS OWN node.

    verl's node_rank is rollout_rank // local_world_size with
    rollout_rank = rank % (tp*dp*pp); for tp<local_world_size that inner value
    never reaches local_world_size, so node_rank collapses to 0 and every rank on
    every node targets node 0's actor. Ranks on other nodes then post CUDA-IPC
    handles for their own GPUs to node 0's engines, and sglang rejects them with
    "Invalid device_uuid=..." while killing the scheduler (observed on 2x8 H100,
    job 16253641). Invisible on one node, where 0 is the only right answer.
    """
    world = nnodes * local_world_size
    correct = {r: r // local_world_size for r in range(world)}
    buggy = {r: _verl_replica_local_rank(r, tp) // local_world_size for r in range(world)}

    assert set(correct.values()) == set(range(nnodes)), "global derivation must span all nodes"
    assert set(buggy.values()) == {0}, "the replica-relative form cannot leave node 0"
    for r in range(world):
        # the actor a rank talks to must own the GPU that rank runs on
        assert correct[r] == r // local_world_size


def test_missing_token_ids_is_loud_not_silent(caplog):
    """The 1-token fallback must announce itself.

    Regression for the failure that cost a full 2-node 30B retool run (job
    16270139): the frontend returned text but no token ids, so every sample was
    scored as a single EOS. response_length/mean==1.0, grad_norm==0, reward
    pinned at its floor — and the job exited 0 with a complete metric set. The
    text in the rollout dump was 2384 characters of coherent reasoning, which is
    what finally separated "generation is broken" from "the length accounting is
    broken". Nothing in the logs distinguished the two for three wrong
    hypotheses, so silence here is the actual defect.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server.model_config = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=151645))

    with caplog.at_level("ERROR"):
        token_ids = server._fallback_token_ids()

    assert token_ids == [151645]
    assert "NO TOKEN IDS" in caplog.text
    assert "request_completion_token_ids" in caplog.text, (
        "the log must name the flag that fixes it, not merely report the symptom"
    )


def _completion_response(token_ids, usage_tokens, finish_reason="stop"):
    return {
        "choices": [{"text": "4", "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 30, "completion_tokens": usage_tokens},
        "nvext": {"completion_token_ids": list(token_ids)},
    }


def _token_output_server():
    server = _make_server({"engine": "sglang", "request_completion_token_ids": True})
    server.model_config.tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    # instance state that __init__ normally sets and the response path reads
    server._logged_engine_data_token_ids = False
    server._logged_missing_engine_data = False
    server.global_steps = 0
    return server


def test_token_count_mismatch_is_loud():
    """Guard for ai-dynamo/dynamo#14302 (dynamo>=1.4.1 strips the stop token from
    completion_token_ids while usage still counts it). Off by one here means the
    trainer never sees EOS and nothing else in the metrics says so."""
    server = _token_output_server()
    out = server._completion_response_to_token_output(_completion_response([19, 151645], 2))
    assert list(out.token_ids) == [19, 151645]
    with pytest.raises(RuntimeError, match="14302"):
        server._completion_response_to_token_output(_completion_response([19], 2))
    # aborted responses are legitimately partial -- no guard
    out = server._completion_response_to_token_output(_completion_response([19], 2, finish_reason="abort"))
    assert list(out.token_ids) == [19]
    # no usage in the response -> nothing to compare against
    resp = _completion_response([19], 2)
    del resp["usage"]
    assert list(server._completion_response_to_token_output(resp).token_ids) == [19]


def test_fallback_keeps_logging_on_a_long_run(caplog):
    """Log the first few hits and then periodically — never just once.

    A single line at hit #1 scrolls away in minutes on a 100-step run and the
    remaining thousands of corrupted samples look clean.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server.model_config = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))

    with caplog.at_level("ERROR"):
        for _ in range(205):
            server._fallback_token_ids()

    assert server._fallback_token_id_hits == 205
    # 3 early + hits 100 and 200
    assert caplog.text.count("NO TOKEN IDS") == 5


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_frees_weights_at_every_level(level, monkeypatch):
    """vLLM sleep(level=1) already takes weights off the GPU, so the sglang
    equivalent must release BOTH tags.

    Regression for the OOM in job 16280143: mapping level 1 -> ["kv_cache"] left
    ~31 GB of TP=2 Qwen3-30B weights resident on each H100 through the training
    step, and the trainer's FSDP forward died with 92 MiB free
    (48.0 GiB trainer + 30.9 GiB engine on 79.1 GiB). The two APIs use different
    vocabularies for the same thing; matching them by label instead of by effect
    on GPU memory is what produced the bug.
    """
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    released = []

    async def fake_release(tags):
        released.append(list(tags))

    monkeypatch.setattr(server, "_is_sglang", lambda: True, raising=False)
    monkeypatch.setattr(server, "_free_engine_on_train", lambda: True, raising=False)
    monkeypatch.setattr(server, "sglang_release", fake_release, raising=False)

    asyncio.run(server.sleep(level=level))

    assert released == [["kv_cache", "weights"]], (
        f"level={level} must free weights; leaving them resident OOMs the trainer"
    )


class _FakeControlClient:
    """Stand-in for DynamoSGLangControlClient: records calls, can fail on demand."""

    def __init__(self, idx: int):
        self.base_url = f"http://shard{idx}"
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail_next: dict[str, int] = {}

    async def _call(self, method, tags):
        await asyncio.sleep(0)  # force a yield, as the real HTTP fan-out does
        self.calls.append((method, tuple(tags)))
        if self.fail_next.get(method, 0) > 0:
            self.fail_next[method] -= 1
            raise RuntimeError(f"{self.base_url} {method} failed")
        return {"success": True}

    async def release_memory_occupation(self, tags):
        return await self._call("release_memory_occupation", tags)

    async def resume_memory_occupation(self, tags):
        return await self._call("resume_memory_occupation", tags)


def _server_with_shards(n: int = 1, released: set[str] | None = None) -> DynamoHttpServer:
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    server._sglang_clients = [_FakeControlClient(i) for i in range(n)]
    server._engine_control_endpoints = [c.base_url for c in server._sglang_clients]
    server._sglang_released_by_shard = None if released is None else [set(released) for _ in range(n)]
    server._sglang_released_tags = set(released or ())
    server._sglang_tag_lock = None
    return server


def test_release_is_idempotent_per_tag():
    """A second release of a still-released tag must not reach the engine.

    Regression for job 16283764. verl releases before every weight sync but
    resumes "weights" and "kv_cache" at different points, so the real trace was:

        released ['kv_cache','weights']   -> released={kv_cache, weights}
        resumed  ['weights']              -> released={kv_cache}
        released ['kv_cache','weights']   -> kv_cache released a SECOND time

    Double-releasing under torch_memory_saver unbinds an already-unbound region,
    and the pool's storage stops being GPU-backed. The failure then surfaces three
    layers away as HTTP 500 "Failed to fold completions stream", with a Triton
    "cpu tensor?" ValueError in the shard log and nothing naming memory release.
    """
    server = _server_with_shards(1)

    # One event loop for the whole sequence: the lock is created lazily and binds
    # to the loop it was made in, which is exactly how the Ray async actor runs it.
    async def main():
        await server.sglang_release(["kv_cache", "weights"])
        await server.sglang_resume(["weights"])
        await server.sglang_release(["kv_cache", "weights"])

    asyncio.run(main())

    assert server._sglang_clients[0].calls == [
        ("release_memory_occupation", ("kv_cache", "weights")),
        ("resume_memory_occupation", ("kv_cache", "weights")),  # widened
        ("release_memory_occupation", ("kv_cache", "weights")),
    ]
    assert server._sglang_released_tags == {"kv_cache", "weights"}


def test_release_and_resume_guards_are_symmetric():
    """Both directions filter against the same state.

    The resume guard was added first (for a KeyError that killed the scheduler)
    and the release guard was not — one bug fixed, its sibling left in place.
    This asserts the pair, so neither can regress alone.
    """
    server = _server_with_shards(1)
    calls = server._sglang_clients[0].calls

    async def main():
        await server.sglang_resume(["weights"])  # never released -> no call
        assert calls == []
        await server.sglang_release(["weights"])
        await server.sglang_release(["weights"])  # already released -> no call

    asyncio.run(main())
    assert [m for m, _ in calls] == ["release_memory_occupation"]


def test_concurrent_release_reaches_engine_once():
    """Concurrent callers must not each perform the release.

    Root cause of the sglang arm's HTTP 500 storm (jobs 16283764 / 16439516).
    Every shard adapter on a node calls sglang_release/resume in parallel, and a
    Ray async actor runs them on one event loop. The tag filter alone is a
    check-then-act across an await, so all four coroutines passed the check
    before any wrote back the state — the log showed four "resumed ['weights']"
    lines in the same millisecond, each still reporting weights as released.

    Four real release_memory_occupation calls unbind an already-unbound region in
    torch_memory_saver, and the next prefill dies in a Triton kernel with
    "Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)".
    """
    server = _server_with_shards(1)

    async def main():
        await asyncio.gather(*[server.sglang_release(["kv_cache", "weights"]) for _ in range(4)])
        await asyncio.gather(*[server.sglang_resume(["weights"]) for _ in range(4)])

    asyncio.run(main())

    assert server._sglang_clients[0].calls == [
        ("release_memory_occupation", ("kv_cache", "weights")),
        # widened: a partial resume would put the shard back in the routing pool
        # with its kv_cache still released
        ("resume_memory_occupation", ("kv_cache", "weights")),
    ]
    assert server._sglang_released_tags == set()


def test_partial_resume_is_widened_to_every_released_tag():
    """A weights-only resume must also restore kv_cache.

    Dynamo's sglang handler re-registers the shard into discovery on the first
    resume. verl resumes "weights" before the weight sync and "kv_cache" only
    after it, so a faithful partial resume advertises a shard whose KV pool is
    still released. Job 16441143 spent ~9 s in that state and the scheduler died
    in write_req_to_token_pool_triton with "cpu tensor?", surfacing to the trainer
    as 257 unexplained HTTP 500s.
    """
    server = _server_with_shards(1, released={"kv_cache", "weights"})
    asyncio.run(server.sglang_resume(["weights"]))

    assert server._sglang_clients[0].calls == [("resume_memory_occupation", ("kv_cache", "weights"))]
    assert server._sglang_released_tags == set()


def test_partial_fanout_failure_is_tracked_per_shard():
    """A release that fails on one shard must not be re-sent to the shards it reached.

    The engine state is per shard; an all-or-nothing node-level record would leave
    the successful shards marked "not released" and the retry would release them a
    second time — the same torch_memory_saver double-unbind as the idempotency test,
    reached through a transient HTTP failure instead of a resume ordering quirk.
    """
    server = _server_with_shards(4)
    server._sglang_clients[3].fail_next["release_memory_occupation"] = 1

    async def main():
        with pytest.raises(RuntimeError, match="failed on shard"):
            await server.sglang_release(["kv_cache", "weights"])
        # shards 0-2 succeeded and are recorded; only shard 3 is still resident
        assert [len(c.calls) for c in server._sglang_clients] == [1, 1, 1, 1]
        assert server._sglang_released_by_shard[0] == {"kv_cache", "weights"}
        assert server._sglang_released_by_shard[3] == set()
        # retry touches ONLY the failed shard
        await server.sglang_release(["kv_cache", "weights"])
        assert [len(c.calls) for c in server._sglang_clients] == [1, 1, 1, 2]
        assert all(s == {"kv_cache", "weights"} for s in server._sglang_released_by_shard)
        # resume fans out to every shard exactly once and clears them all
        await server.sglang_resume(["weights"])
        assert [c.calls[-1] for c in server._sglang_clients] == [
            ("resume_memory_occupation", ("kv_cache", "weights"))
        ] * 4
        assert server._sglang_released_tags == set()

    asyncio.run(main())


def test_sglang_refuses_memory_saver_flag_split():
    """free_cache_engine=true with enable_sleep_mode=false must be rejected at launch.

    --enable-memory-saver follows enable_sleep_mode, the adapter's release calls
    follow free_cache_engine. Split, release_memory_occupation still deregisters the
    worker from discovery but frees nothing, and the trainer OOMs with no log line
    naming the cause.
    """
    server = _make_server(
        {"engine": "sglang", "request_completion_token_ids": True},
        enable_sleep_mode=False,
        free_cache_engine=True,
    )
    server.replica_rank = 0
    server.node_rank = 0
    server._cuda_visible_devices = "0"
    server._worker_specs = None
    with pytest.raises(ValueError, match="enable_sleep_mode"):
        server._start_engine_workers()


def test_facade_imports_without_vllm(monkeypatch):
    """recipe.dynamo.dynamo_rollout must import on a vLLM-free image.

    rollout.name=dynamo resolves through this one module for BOTH engines, so a
    hard top-level `from verl...vllm_rollout import ServerAdapter` makes the
    sglang path unusable on the official verlai/verl:sgl* images — verl's
    vllm_rollout/__init__.py raises PackageNotFoundError (not ImportError) when
    vLLM is absent. Job 16510923 died that way after a fully healthy bootstrap.
    """
    import importlib.metadata
    import sys

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def no_vllm(name, *args, **kwargs):
        if name.startswith("verl.workers.rollout.vllm_rollout"):
            raise importlib.metadata.PackageNotFoundError("vllm missing (simulated)")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m.startswith("recipe.dynamo.dynamo_rollout")]:
        del sys.modules[mod]
    monkeypatch.setattr("builtins.__import__", no_vllm)

    mod = importlib.import_module("recipe.dynamo.dynamo_rollout")
    assert mod.ServerAdapter is not None
    # the vLLM subclass still exists, but instantiating it must say why it cannot work
    with pytest.raises(RuntimeError, match="needs the vLLM package"):
        mod.VllmDynamoServerAdapter()


# --------------------------------------------------------------------------- #
# native /generate data plane (engine_kwargs.dynamo.generate_api=generate)
# --------------------------------------------------------------------------- #


def _native_server(dynamo_cfg=None, **rollout_cfg):
    cfg = {"engine": "sglang", "generate_api": "generate", "sglang": {"launch_mode": "sidecar"}}
    cfg.update(dynamo_cfg or {})
    rollout = {"prompt_length": 100, "response_length": 50}
    rollout.update(rollout_cfg)
    server = _make_server(cfg, **rollout)
    server.global_steps = 3
    server.model_config.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    return server


def _sglang_response(output_ids, finish="stop", logprobs=True, **meta):
    meta_info = {"finish_reason": finish if isinstance(finish, dict) else {"type": finish}}
    meta_info["completion_tokens"] = len(output_ids)
    if logprobs:
        meta_info["output_token_logprobs"] = [[-0.5 - i, tok, None] for i, tok in enumerate(output_ids)]
    meta_info.update(meta)
    return {"output_ids": list(output_ids), "meta_info": meta_info}


def test_generate_api_defaults_to_completions_and_is_validated():
    assert _make_server({"engine": "sglang"})._generate_api() == "completions"
    assert _native_server()._use_native_generate() is True
    with pytest.raises(ValueError, match="generate_api"):
        _make_server({"engine": "sglang", "generate_api": "chat"})._generate_api()
    # vLLM's /inference/v1/generate is a different body; refuse rather than mis-speak it
    with pytest.raises(ValueError, match="engine=sglang only"):
        _make_server({"engine": "vllm", "generate_api": "generate"})._generate_api()


def test_native_payload_mirrors_verl_sglang_request():
    server = _native_server()
    payload = server._build_frontend_generate_payload(
        [1, 2, 3],
        {
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "logprobs": True,
            "max_tokens": 20,
            "prompt_logprobs": None,
            "nvext": {"extra_fields": ["engine_data"]},
            "stop_token_ids": [7],
            "seed": None,
        },
        "traj-1-abc",
    )
    assert payload == {
        "rid": "traj-1-abc",
        "input_ids": [1, 2, 3],
        "sampling_params": {
            "max_new_tokens": 20,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "stop_token_ids": [7],
        },
        "return_logprob": True,
        "stream": False,
    }


@pytest.mark.parametrize(
    "prompt_len,sampling,expected",
    [
        (10, {}, 50),  # default: response_length
        (120, {}, 30),  # default bounded by the remaining budget (100 + 50 - 120)
        (120, {"max_tokens": 40}, 30),  # explicit budget bounded too
        (10, {"max_new_tokens": 7, "max_tokens": 40}, 7),  # SGLang's own name wins
        (200, {}, 1),  # never below 1, as on the completions path
    ],
)
def test_native_max_new_tokens_matches_the_completions_budget(prompt_len, sampling, expected):
    server = _native_server()
    payload = server._build_frontend_generate_payload(list(range(prompt_len)), dict(sampling), "r")
    assert payload["sampling_params"]["max_new_tokens"] == expected
    assert "max_tokens" not in payload["sampling_params"]


def test_native_response_keeps_engine_ids_and_checked_logprobs():
    out = _native_server()._sglang_generate_response_to_token_output(_sglang_response([5, 6, 2]), True)
    assert out.token_ids == [5, 6, 2]
    assert out.log_probs == [-0.5, -1.5, -2.5]
    assert out.stop_reason == "completed"
    assert out.extra_fields == {"global_steps": 3}
    length = _native_server()._sglang_generate_response_to_token_output(
        _sglang_response([5, 6], finish={"type": "length", "length": 2}, logprobs=False), False
    )
    assert length.log_probs is None and length.stop_reason == "completed"


def test_native_plain_abort_keeps_partial_tokens_for_resume():
    """pause_generation(mode=abort) during a weight sync: keep what was sampled."""
    server = _native_server()
    out = server._sglang_generate_response_to_token_output(
        _sglang_response([5, 6], finish={"type": "abort", "message": "Abort before"}), True
    )
    assert out.token_ids == [5, 6] and out.stop_reason == "aborted"
    empty = server._sglang_generate_response_to_token_output(_sglang_response([], finish="abort"), True)
    assert empty.token_ids == [] and empty.log_probs == [] and empty.stop_reason == "aborted"
    assert empty.extra_fields == {}


def test_native_error_abort_raises():
    finish = {"type": "abort", "message": "input too long", "status_code": 400, "err_type": "BadRequest"}
    with pytest.raises(RuntimeError, match="input too long"):
        _native_server()._sglang_generate_response_to_token_output(_sglang_response([5], finish=finish), False)


def test_native_empty_completion_is_loud_not_a_fake_eos():
    with pytest.raises(RuntimeError, match="no output_ids"):
        _native_server()._sglang_generate_response_to_token_output(_sglang_response([], finish="stop"), False)


def test_native_truncated_sequence_is_refused():
    data = _sglang_response([5, 6], finish="stop", completion_tokens=3)
    with pytest.raises(RuntimeError, match="completion_tokens=3"):
        _native_server()._sglang_generate_response_to_token_output(data, False)


def test_native_logprobs_must_align_with_output_ids():
    data = _sglang_response([5, 6], finish="stop")
    data["meta_info"]["output_token_logprobs"][1][1] = 9
    with pytest.raises(RuntimeError, match="does not match"):
        _native_server()._sglang_generate_response_to_token_output(data, True)
    short = _sglang_response([5, 6], finish="stop")
    short["meta_info"]["output_token_logprobs"] = short["meta_info"]["output_token_logprobs"][:1]
    with pytest.raises(RuntimeError, match="does not cover"):
        _native_server()._sglang_generate_response_to_token_output(short, True)
    # an aborted response with inconsistent logprobs is dropped for a retry, not trained on
    short["meta_info"]["finish_reason"] = {"type": "abort"}
    out = _native_server()._sglang_generate_response_to_token_output(short, True)
    assert out.token_ids == [] and out.stop_reason == "aborted"


@pytest.mark.parametrize("output_ids", [["5"], [5.0], [True], "56"])
def test_native_non_integer_output_ids_are_rejected(output_ids):
    data = _sglang_response([], finish="stop", logprobs=False)
    data["output_ids"] = output_ids
    with pytest.raises(RuntimeError, match="not a list of token ids"):
        _native_server()._sglang_generate_response_to_token_output(data, False)


def test_native_unknown_finish_reason_is_rejected():
    with pytest.raises(RuntimeError, match="unknown finish_reason"):
        _native_server()._sglang_generate_response_to_token_output(_sglang_response([5], finish=None), False)


@pytest.mark.asyncio
async def test_generate_dispatches_to_native_route_with_same_headers():
    server = _native_server()
    server._server_address = "10.0.0.1"
    server._server_port = 8000
    server._generation_resumed = asyncio.Event()
    server._generation_resumed.set()
    posts = []

    async def fake_post(payload, request_id, http_request_id=None, url=None):
        posts.append((payload, request_id, http_request_id, url))
        return 200, json.dumps(_sglang_response([5, 2]))

    server._frontend_post = fake_post
    out = await server.generate([1, 2, 3], {"logprobs": True, "temperature": 1.0}, "traj-7")
    assert out.token_ids == [5, 2] and out.log_probs == [-0.5, -1.5]
    payload, request_id, http_request_id, url = posts[0]
    assert url == "http://10.0.0.1:8000/generate"
    # trajectory id stays the affinity/session key; each request gets a unique id
    assert request_id == "traj-7" and http_request_id.startswith("traj-7-")
    assert payload["rid"] == http_request_id and payload["input_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_generate_keys_affinity_on_session_id_not_per_turn_request_id():
    # verl's LLMServerManager sends request_id=uuid4() per turn plus
    # session_id=<trajectory key>; affinity must follow the trajectory.
    server = _native_server()
    server.config.engine_kwargs["dynamo"]["router_session_affinity_ttl_secs"] = 600
    server._server_address = "10.0.0.1"
    server._server_port = 8000
    server._generation_resumed = asyncio.Event()
    server._generation_resumed.set()
    headers = []

    async def fake_post(payload, request_id, http_request_id=None, url=None):
        headers.append(server._frontend_headers(request_id))
        return 200, json.dumps(_sglang_response([5]))

    server._frontend_post = fake_post
    for turn in ("turn-uuid-1", "turn-uuid-2"):
        await server.generate([1, 2], {"logprobs": True}, turn, session_id="traj-9")
    assert [h["x-dynamo-session-id"] for h in headers] == ["traj-9", "traj-9"]


@pytest.mark.asyncio
async def test_native_frontend_error_during_pause_is_reported_as_aborted():
    server = _native_server()
    server._server_address = "10.0.0.1"
    server._server_port = 8000
    server._generation_resumed = asyncio.Event()
    server._generation_resumed.set()

    async def failing_post(payload, request_id, http_request_id=None, url=None):
        server._generation_resumed.clear()  # abort_all_requests ran while in flight
        return 503, "engine paused"

    server._frontend_post = failing_post
    out = await server.generate([1], {"logprobs": True}, "traj-8")
    assert out.token_ids == [] and out.stop_reason == "aborted"


def test_frontend_mounts_native_generate_only_when_selected(monkeypatch, tmp_path):
    started = []

    class _Proc:
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            started.append(env)

    monkeypatch.setattr("recipe.dynamo.dynamo_async_server.subprocess.Popen", _Proc)
    monkeypatch.setenv("VERL_DYNAMO_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("DYN_SGLANG_ENABLE_GENERATE", raising=False)
    for cfg, expected in (({}, None), ({"generate_api": "generate"}, "1")):
        server = _make_server({"engine": "sglang", **cfg})
        server.replica_rank = 0
        server._frontend_process = None
        server._frontend_port = 8000
        server._router_mode = "round-robin"
        server._namespace = "ns"
        server._allocated_tcp_ports = set()
        server._server_address = "127.0.0.1"
        server._dynamo_env_vars = lambda: {}
        server._frontend_router_args = lambda: []
        server._start_frontend()
        server._frontend_log_fp.close()
        assert started[-1].get("DYN_SGLANG_ENABLE_GENERATE") == expected


# --------------------------------------------------------------------------- #
# rollout routing replay (R3) on the native /generate path
# --------------------------------------------------------------------------- #

_R3_LAYERS, _R3_TOPK = 3, 2


def _r3_server(dynamo_cfg=None, **rollout_cfg):
    server = _native_server(dynamo_cfg, enable_rollout_routing_replay=True, **rollout_cfg)
    server.model_config.hf_config = SimpleNamespace(num_hidden_layers=_R3_LAYERS, num_experts_per_tok=_R3_TOPK)
    return server


def _routing(rows):
    """Deterministic (rows, layers, topk) routing and its SGLang wire form (base64 int32)."""
    import numpy as np

    arr = np.arange(rows * _R3_LAYERS * _R3_TOPK, dtype=np.int32).reshape(rows, _R3_LAYERS, _R3_TOPK)
    return arr, base64.b64encode(arr.tobytes()).decode("utf-8")


def test_r3_requires_sidecar_and_native_generate():
    _r3_server()._check_routing_replay_support()  # sidecar + /generate: accepted
    _native_server()._check_routing_replay_support()  # R3 off: nothing to check
    refused = (
        {"generate_api": "completions"},  # /v1/completions cannot carry routed experts
        {"sglang": {"launch_mode": "worker"}},  # dynamo.sglang worker passthrough unverified
        {"direct_generate": True},  # bypasses the frontend
    )
    for cfg in refused:
        with pytest.raises(ValueError, match="enable_rollout_routing_replay"):
            _r3_server(cfg)._check_routing_replay_support()
    with pytest.raises(ValueError, match="enable_rollout_routing_replay"):
        _make_server({"engine": "vllm"}, enable_rollout_routing_replay=True)._check_routing_replay_support()


def test_r3_engine_flag_and_request_field_follow_the_rollout_flag():
    assert "--enable-return-routed-experts" in _sidecar_engine_cmd(enable_rollout_routing_replay=True)
    assert "--enable-return-routed-experts" not in _sidecar_engine_cmd()
    assert _r3_server()._build_frontend_generate_payload([1, 2], {}, "r")["return_routed_experts"] is True
    assert "return_routed_experts" not in _native_server()._build_frontend_generate_payload([1, 2], {}, "r")


def test_r3_full_sequence_routing_decodes_like_verl_native():
    """prompt 4 + output 3 -> positions [0, 6): SGLang has not forwarded the last token."""
    import numpy as np

    expected, wire = _routing(4 + 3 - 1)
    out = _r3_server()._sglang_generate_response_to_token_output(
        _sglang_response([5, 6, 2], routed_experts=wire), True, prompt_len=4
    )
    assert out.token_ids == [5, 6, 2] and out.stop_reason == "completed"
    assert out.routed_experts.shape == (6, _R3_LAYERS, _R3_TOPK)
    assert out.routed_experts.dtype == np.int32
    np.testing.assert_array_equal(out.routed_experts, expected)
    # same bytes verl's native server gets from sglang's extract_routed_experts_from_meta_info
    native = np.frombuffer(base64.b64decode(wire.encode("utf-8")), dtype=np.int32)
    np.testing.assert_array_equal(out.routed_experts, native.reshape(-1, _R3_LAYERS, _R3_TOPK))


def test_r3_off_ignores_routing_in_the_response():
    _, wire = _routing(6)
    out = _native_server()._sglang_generate_response_to_token_output(
        _sglang_response([5, 6, 2], routed_experts=wire), True, prompt_len=4
    )
    assert out.routed_experts is None


def test_r3_reads_text_config_for_multimodal_wrappers():
    server = _r3_server()
    server.model_config.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=_R3_LAYERS, num_experts_per_tok=_R3_TOPK)
    )
    assert server._routed_experts_shape() == (_R3_LAYERS, _R3_TOPK)
    server.model_config.hf_config = SimpleNamespace(num_hidden_layers=4)  # dense model
    with pytest.raises(AttributeError, match="num_experts_per_tok"):
        server._routed_experts_shape()


@pytest.mark.parametrize(
    "meta,match",
    [
        ({}, "no meta_info.routed_experts"),  # engine launched without the flag
        ({"routed_experts": _routing(5)[1]}, "expected prompt\\+output-1"),  # one row short
        ({"routed_experts": base64.b64encode(b"\x00" * 4 * 7).decode()}, "not a multiple"),
    ],
)
def test_r3_completed_turn_without_exact_routing_raises(meta, match):
    with pytest.raises(RuntimeError, match=match):
        _r3_server()._sglang_generate_response_to_token_output(_sglang_response([5, 6, 2], **meta), True, prompt_len=4)


def test_r3_aborted_partial_keeps_tokens_only_with_routing():
    """A kept partial must carry its rows, or the client's next append lands after a gap."""
    server = _r3_server()
    abort = {"type": "abort", "message": "Abort before"}
    _, wire = _routing(4 + 2 - 1)
    kept = server._sglang_generate_response_to_token_output(
        _sglang_response([5, 6], finish=abort, routed_experts=wire), True, prompt_len=4
    )
    assert kept.token_ids == [5, 6] and kept.stop_reason == "aborted"
    assert kept.routed_experts.shape == (5, _R3_LAYERS, _R3_TOPK)
    dropped = server._sglang_generate_response_to_token_output(
        _sglang_response([5, 6], finish=abort), True, prompt_len=4
    )
    assert dropped.token_ids == [] and dropped.stop_reason == "aborted" and dropped.routed_experts is None


@pytest.mark.asyncio
async def test_r3_generate_checks_rows_against_the_sent_prompt():
    import numpy as np

    server = _r3_server()
    server._server_address = "10.0.0.1"
    server._server_port = 8000
    server._generation_resumed = asyncio.Event()
    server._generation_resumed.set()
    expected, wire = _routing(3 + 2 - 1)
    sent = []

    async def fake_post(payload, request_id, http_request_id=None, url=None):
        sent.append(payload)
        return 200, json.dumps(_sglang_response([5, 2], routed_experts=wire))

    server._frontend_post = fake_post
    out = await server.generate([1, 2, 3], {"logprobs": True}, "traj-r3")
    assert sent[0]["return_routed_experts"] is True
    np.testing.assert_array_equal(out.routed_experts, expected)


# --------------------------------------------------------------------------- #
# frontend readiness on ai-dynamo 1.7 (/health without an instance list)
# --------------------------------------------------------------------------- #

_WORKER_STATE = (
    'dynamo_frontend_router_worker_state{{model="m",reason="none",router_worker_id="{wid}",'
    'state="{state}",target_component="backend",target_endpoint="{ep}",target_namespace="ns",'
    'worker_type="decode"}} {value}'
)


def test_available_generate_workers_are_counted_from_router_metrics():
    text = "\n".join(
        [
            "# HELP dynamo_frontend_router_worker_state worker state",
            _WORKER_STATE.format(wid="1", state="available", ep="generate", value=1),
            _WORKER_STATE.format(wid="1", state="available", ep="generate", value=1),  # same worker twice
            _WORKER_STATE.format(wid="2", state="available", ep="generate", value=1),
            _WORKER_STATE.format(wid="3", state="busy", ep="generate", value=1),
            _WORKER_STATE.format(wid="4", state="available", ep="clear_kv_blocks", value=1),
            _WORKER_STATE.format(wid="5", state="available", ep="generate", value=0),
            'dynamo_component_router_ttft_seconds_count{worker_id="6"} 3',
        ]
    )
    assert DynamoHttpServer._count_available_generate_workers(text) == 2
    assert DynamoHttpServer._count_available_generate_workers("") == 0


@pytest.mark.asyncio
async def test_healthcheck_reads_router_metrics_when_health_has_no_instances(monkeypatch):
    server = _make_server({"engine": "sglang"})
    server._frontend_port = 8000
    server._raise_if_subprocess_died = lambda: None
    metrics = "\n".join(_WORKER_STATE.format(wid=w, state="available", ep="generate", value=1) for w in "12")
    seen = []

    def fake_get(url, timeout):
        seen.append(url)
        if url.endswith("/health"):
            return SimpleNamespace(status_code=200, json=lambda: {"status": "healthy", "stage": "ready"})
        return SimpleNamespace(status_code=200, text=metrics)

    monkeypatch.setattr("recipe.dynamo.dynamo_async_server.requests.get", fake_get)
    await server._healthcheck_frontend(expected_workers=2)
    assert seen == ["http://localhost:8000/health", "http://localhost:8000/metrics"]


def _flush_server(failures: list):
    server = DynamoHttpServer.__new__(DynamoHttpServer)
    calls = []

    async def control_all(method, *args, **kwargs):
        calls.append(method)
        if failures:
            raise failures.pop(0)
        return []

    server._sglang_control_all = control_all
    return server, calls


def test_flush_cache_retries_while_aborted_requests_drain():
    busy = RuntimeError("dynamo.sglang control call 'flush_cache' failed on [(url, 'HTTP 400: Flush cache failed.')]")
    server, calls = _flush_server([busy, busy])
    asyncio.run(server._sglang_flush_cache_all(timeout_s=5, poll_s=0))
    assert calls == ["flush_cache"] * 3


def test_flush_cache_raises_other_errors_and_gives_up_at_deadline():
    server, calls = _flush_server([RuntimeError("HTTP 500: boom")])
    with pytest.raises(RuntimeError, match="HTTP 500"):
        asyncio.run(server._sglang_flush_cache_all(timeout_s=5, poll_s=0))
    assert calls == ["flush_cache"]
    server, calls = _flush_server([RuntimeError("HTTP 400: busy")] * 50)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        asyncio.run(server._sglang_flush_cache_all(timeout_s=0, poll_s=0))
    assert calls == ["flush_cache"]
