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

import asyncio
import importlib.metadata
import inspect
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from recipe.dynamo import dynamo_async_server, dynamo_thunderagent, register
from recipe.dynamo.dynamo_agent_loop import (
    DynamoAgentLoopWorker,
    DynamoFullyAsyncLLMServerClient,
    DynamoLLMServerManager,
    DynamoServerManager,
)
from recipe.dynamo.dynamo_async_server import DynamoHttpServer
from recipe.dynamo.dynamo_thunderagent import (
    DynamoThunderAgentHttpServer,
    DynamoThunderAgentReplica,
)

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

RECIPE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RECIPE_ROOT.parent


def _make_http_server(thunderagent: dict | None = None) -> DynamoThunderAgentHttpServer:
    server = object.__new__(DynamoThunderAgentHttpServer)
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"thunderagent": thunderagent or {"enabled": True}}})
    server.model_config = SimpleNamespace(local_path="/models/test-model")
    server._namespace = "verl_dynamo"
    server._served_model_name = "test-model"
    server._router_mode = "round-robin"
    server.replica_rank = 0
    server._frontend_process = None
    server._thunderagent_process = None
    server._thunderagent_log_fp = None
    return server


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        result = self.function(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


class _FakeServer:
    def __init__(self):
        self.generate_calls = []
        self.finalize_calls = []
        self.generate = _RemoteMethod(self._generate)
        self.finalize_program = _RemoteMethod(self._finalize_program)

    def _generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return kwargs

    def _finalize_program(self, session_id: str) -> None:
        self.finalize_calls.append(session_id)


def _generate_kwargs(prompt_id: int) -> dict:
    return {
        "request_id": None,
        "prompt_ids": [prompt_id],
        "sampling_params": {"max_tokens": 1},
    }


@pytest.mark.asyncio
async def test_client_reuses_program_id_for_all_turns_and_finalizes_once() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=True)

    async with manager.program_scope():
        first = await manager.generate(**_generate_kwargs(1))
        second = await manager.generate(**_generate_kwargs(2))

    session_id = first["thunderagent_session_id"]
    assert session_id
    assert second["thunderagent_session_id"] == session_id
    assert server.finalize_calls == [session_id]


@pytest.mark.asyncio
async def test_client_isolates_concurrent_programs() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=True)

    async def run(prompt_id: int) -> str:
        async with manager.program_scope():
            output = await manager.generate(**_generate_kwargs(prompt_id))
            return output["thunderagent_session_id"]

    first_id, second_id = await asyncio.gather(run(1), run(2))
    assert first_id != second_id
    assert sorted(server.finalize_calls) == sorted([first_id, second_id])


@pytest.mark.asyncio
async def test_enabled_client_fails_closed_without_program_scope() -> None:
    manager = DynamoServerManager([("frontend:8000", _FakeServer())], thunderagent_enabled=True)

    with pytest.raises(RuntimeError, match="active ThunderAgent program"):
        await manager.generate(**_generate_kwargs(1))


@pytest.mark.asyncio
async def test_disabled_client_preserves_pr110_request_path() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    async with manager.program_scope():
        await manager.generate(**_generate_kwargs(1))

    assert "thunderagent_session_id" not in server.generate_calls[0]
    assert server.finalize_calls == []


@pytest.mark.asyncio
async def test_client_omits_empty_audio_arguments_unsupported_by_pr110() -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    await manager.generate(
        **_generate_kwargs(1),
        audio_data=None,
        mm_processor_kwargs=None,
    )

    assert "audio_data" not in server.generate_calls[0]
    assert "mm_processor_kwargs" not in server.generate_calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unsupported_argument", "value"),
    [("audio_data", [b"audio"]), ("mm_processor_kwargs", {"sampling_rate": 16_000})],
)
async def test_client_rejects_unsupported_audio_arguments(unsupported_argument, value) -> None:
    server = _FakeServer()
    manager = DynamoServerManager([("frontend:8000", server)], thunderagent_enabled=False)

    with pytest.raises(RuntimeError, match="does not support audio inputs"):
        await manager.generate(**_generate_kwargs(1), **{unsupported_argument: value})

    assert server.generate_calls == []


@pytest.mark.asyncio
async def test_agent_loop_worker_wraps_parent_run_in_program_scope(monkeypatch) -> None:
    class ScopeClient:
        active = False
        exits = 0

        @asynccontextmanager
        async def program_scope(self):
            self.active = True
            try:
                yield
            finally:
                self.active = False
                self.exits += 1

    client = ScopeClient()
    worker = object.__new__(DynamoAgentLoopWorker)
    worker.llm_client = client

    async def parent_run(self, *args, **kwargs):
        assert self.llm_client.active
        assert args == ("sampling", "trajectory")
        assert kwargs == {"agent_name": "tool_agent"}
        return "output"

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent_run)

    assert await worker._run_agent_loop("sampling", "trajectory", agent_name="tool_agent") == "output"
    assert not client.active
    assert client.exits == 1


@pytest.mark.asyncio
async def test_agent_loop_worker_closes_scope_when_parent_fails(monkeypatch) -> None:
    class ScopeClient:
        exits = 0

        @asynccontextmanager
        async def program_scope(self):
            try:
                yield
            finally:
                self.exits += 1

    client = ScopeClient()
    worker = object.__new__(DynamoAgentLoopWorker)
    worker.llm_client = client

    async def parent_run(_self, *_args, **_kwargs):
        raise ValueError("agent failed")

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent_run)

    with pytest.raises(ValueError, match="agent failed"):
        await worker._run_agent_loop("sampling", "trajectory", agent_name="tool_agent")
    assert client.exits == 1


@pytest.mark.asyncio
async def test_server_manager_returns_direct_thunderagent_client() -> None:
    # The V1 migration removed the recipe's no-op _init_global_load_balancer
    # override: the BASE load balancer is required so client_cls callers get
    # retry/version-aggregation semantics.
    assert "_init_global_load_balancer" not in DynamoLLMServerManager.__dict__

    def make_manager(*, use_v1: bool, thunderagent: bool = True) -> DynamoLLMServerManager:
        manager = object.__new__(DynamoLLMServerManager)
        manager.server_addresses = ["frontend:8000"]
        manager.server_handles = [_FakeServer()]
        manager.global_load_balancer = object()
        manager.rollout_config = SimpleNamespace(engine_kwargs={"dynamo": {"thunderagent": {"enabled": thunderagent}}})
        manager.config = SimpleNamespace(trainer={"use_v1": use_v1})
        return manager

    # Legacy V0 callers (no client_cls) keep the direct ThunderAgent manager.
    client = make_manager(use_v1=False).get_client()
    assert isinstance(client, DynamoServerManager)

    # V1 async trainers pass client_cls and get the affinity-aware subclass
    # wired with this pool's handles and auto-finalize.
    v1_client = make_manager(use_v1=True).get_client(client_cls=FullyAsyncLLMServerClient)
    assert isinstance(v1_client, DynamoFullyAsyncLLMServerClient)
    assert v1_client._auto_finalize is True
    assert len(v1_client._dynamo_server_handles) == 1

    # Without thunderagent the requested class is honored unchanged.
    plain = make_manager(use_v1=True, thunderagent=False).get_client(client_cls=FullyAsyncLLMServerClient)
    assert type(plain) is FullyAsyncLLMServerClient

    # V1 sync mode (no client_cls) has no ProgramScope provider: fail fast.
    with pytest.raises(ValueError, match="only supported under V1"):
        make_manager(use_v1=True).get_client()


def test_thunderagent_command_derives_endpoint_model_and_block_size() -> None:
    server = _make_http_server()

    assert server._build_thunderagent_cmd() == [
        sys.executable,
        "-m",
        "dynamo.thunderagent_router",
        "--endpoint",
        "verl_dynamo.backend.generate",
        "--model-name",
        "test-model",
        "--model-path",
        "/models/test-model",
        "--router-block-size",
        "16",
    ]


def _check_installed_dynamo_parser(module: str, args: list[str]) -> None:
    try:
        importlib.metadata.version("ai-dynamo-runtime")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Dynamo runtime is not installed")
    # Pytest can import this recipe directory as `dynamo`, shadowing the
    # installed namespace package. Validate the real CLI in a fresh process.
    script = "import importlib,sys; module=sys.argv.pop(1); importlib.import_module(module).parse_args()"
    result = subprocess.run([sys.executable, "-c", script, module, *args], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_thunderagent_command_is_accepted_by_dynamo_parser() -> None:
    _check_installed_dynamo_parser("dynamo.thunderagent_router.args", _make_http_server()._build_thunderagent_cmd()[3:])


def test_thunderagent_frontend_options_are_accepted_by_dynamo_parser() -> None:
    server = _make_http_server()
    _check_installed_dynamo_parser(
        "dynamo.frontend.main", ["--router-mode", "round-robin", *server._frontend_router_args()]
    )


def test_thunderagent_command_passes_configured_router_options() -> None:
    server = _make_http_server(
        {
            "enabled": True,
            "router_block_size": 32,
            "extra_args": ["--pause-threshold", "0.9"],
        }
    )

    command = server._build_thunderagent_cmd()

    assert command[command.index("--router-block-size") + 1] == "32"
    assert command[-2:] == ["--pause-threshold", "0.9"]


def test_thunderagent_extra_args_must_be_a_list() -> None:
    server = _make_http_server({"enabled": True, "extra_args": "--pause-threshold 0.9"})

    with pytest.raises(TypeError, match="extra_args must be a list"):
        server._build_thunderagent_cmd()


def test_vllm_and_thunderagent_share_block_size(monkeypatch) -> None:
    server = _make_http_server({"enabled": True, "router_block_size": 32})
    monkeypatch.setattr(
        DynamoHttpServer,
        "_build_vllm_cmd",
        lambda _self, _model, _tp, kv_events_config_json=None: ["python", "-m", "dynamo.vllm"],
    )

    command = server._build_vllm_cmd("test-model", 1, "{}")

    assert command[-2:] == ["--block-size", "32"]


def test_thunderagent_backend_workers_use_internal_model_name(monkeypatch) -> None:
    server = _make_http_server()
    base_calls = []

    def build_base_command(_self, served_model_name, _tp, kv_events_config_json=None):
        base_calls.append((served_model_name, kv_events_config_json))
        return ["python", "-m", "dynamo.vllm", "--served-model-name", served_model_name]

    monkeypatch.setattr(DynamoHttpServer, "_build_vllm_cmd", build_base_command)

    command = server._build_vllm_cmd("test-model", 1, "{}")
    thunderagent_command = server._build_thunderagent_cmd()

    assert base_calls == [("test-model--verl-thunderagent-backend", "{}")]
    # enable_kv_events=false: the launcher passes no config and the override must not
    # invent one (same contract as the sglang override).
    server._build_vllm_cmd("test-model", 1)
    assert base_calls[-1] == ("test-model--verl-thunderagent-backend", None)
    assert command[command.index("--served-model-name") + 1] == "test-model--verl-thunderagent-backend"
    assert thunderagent_command[thunderagent_command.index("--model-name") + 1] == "test-model"
    assert server._served_model_name == "test-model"


def test_thunderagent_frontend_payload_leaves_replica_selection_to_router() -> None:
    server = _make_http_server()
    server.model_config.tokenizer = SimpleNamespace(eos_token_id=2)
    server.config.prompt_length = 64
    server.config.response_length = 64

    payload = server._build_frontend_completion_payload([1], {"max_tokens": 1, "logprobs": True}, "request-a")

    assert payload["model"] == "test-model"
    assert "dp_rank" not in payload["nvext"]
    assert "backend_instance_id" not in payload["nvext"]
    assert "detailed_finish_reason" in payload["nvext"]["extra_fields"]
    assert payload["logprobs"] == 0


def test_frontend_start_starts_thunderagent_first(monkeypatch) -> None:
    server = _make_http_server()
    events = []
    monkeypatch.setattr(server, "_start_thunderagent", lambda: events.append("thunderagent"))
    monkeypatch.setattr(DynamoHttpServer, "_start_frontend", lambda _self: events.append("frontend"))

    server._start_frontend()

    assert events == ["thunderagent", "frontend"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "expected_workers"), [(True, 5), (False, 4)])
async def test_frontend_health_waits_for_router_and_workers(monkeypatch, enabled, expected_workers) -> None:
    server = _make_http_server({"enabled": enabled})
    observed = []

    async def healthcheck(_self, workers):
        observed.append(workers)

    async def finalize(_session_id):
        pass

    monkeypatch.setattr(DynamoHttpServer, "_healthcheck_frontend", healthcheck)
    monkeypatch.setattr(server, "finalize_program", finalize)

    await server._healthcheck_frontend(4)

    assert observed == [expected_workers]


@pytest.mark.asyncio
async def test_generation_adds_program_header(monkeypatch) -> None:
    server = _make_http_server()
    monkeypatch.setattr(server, "_use_direct_generate", lambda: False)

    async def base_generate(self, *_args, **_kwargs):
        return self._frontend_headers("request-a")

    monkeypatch.setattr(DynamoHttpServer, "generate", base_generate)

    headers = await server.generate(
        prompt_ids=[1],
        sampling_params={"max_tokens": 1},
        request_id="request-a",
        thunderagent_session_id="program-a",
    )

    assert headers == {
        "X-Request-Id": "request-a",
        "X-Dynamo-Session-ID": "program-a",
    }


def test_session_affinity_header_survives_on_the_thunderagent_class() -> None:
    """register.py always serves through DynamoThunderAgentHttpServer, so the base
    router_session_affinity_ttl_secs header must reach the wire from here too. An active
    ThunderAgent program overrides it with one spelling of the header, not two."""
    server = _make_http_server({"enabled": False})
    server.config.engine_kwargs["dynamo"]["router_session_affinity_ttl_secs"] = 600
    assert server._frontend_headers("traj-1") == {"X-Request-Id": "traj-1", "x-dynamo-session-id": "traj-1"}
    token = dynamo_thunderagent._REQUEST_PROGRAM.set(("program-a", False))
    try:
        headers = server._frontend_headers("traj-1")
    finally:
        dynamo_thunderagent._REQUEST_PROGRAM.reset(token)
    assert headers == {"X-Request-Id": "traj-1", "X-Dynamo-Session-ID": "program-a"}


@pytest.mark.asyncio
async def test_generation_requires_program_and_rejects_direct_bypass(monkeypatch) -> None:
    server = _make_http_server()

    with pytest.raises(RuntimeError, match="session ID"):
        await server.generate(prompt_ids=[1], sampling_params={}, request_id="request-a")

    monkeypatch.setattr(server, "_use_direct_generate", lambda: True)
    with pytest.raises(RuntimeError, match="direct_generate"):
        await server.generate(
            prompt_ids=[1],
            sampling_params={},
            request_id="request-a",
            thunderagent_session_id="program-a",
        )


@pytest.mark.asyncio
async def test_finalize_retries_and_accepts_empty_choices(monkeypatch) -> None:
    server = _make_http_server(
        {
            "enabled": True,
            "finalize_max_attempts": 2,
            "finalize_retry_delay_s": 0,
        }
    )
    responses = [(503, "temporarily unavailable"), (200, json.dumps({"choices": []}))]
    calls = []

    async def frontend_post(_payload, request_id):
        calls.append(server._frontend_headers(request_id))
        return responses.pop(0)

    monkeypatch.setattr(server, "_frontend_post", frontend_post)

    await server.finalize_program("program-a")

    assert len(calls) == 2
    assert calls[-1]["X-Dynamo-Session-ID"] == "program-a"
    assert calls[-1]["X-Dynamo-Session-Final"] == "true"


@pytest.mark.asyncio
async def test_finalize_rejects_nonempty_choices_as_router_bypass(monkeypatch) -> None:
    server = _make_http_server({"enabled": True, "finalize_max_attempts": 1})

    async def frontend_post(_payload, _request_id):
        return 200, json.dumps({"choices": [{"text": "model answered"}]})

    monkeypatch.setattr(server, "_frontend_post", frontend_post)

    with pytest.raises(RuntimeError, match="bypassed"):
        await server.finalize_program("program-a")


def test_watchdog_detects_thunderagent_exit(monkeypatch) -> None:
    server = _make_http_server()
    server._thunderagent_process = SimpleNamespace(poll=lambda: 9, returncode=9)
    monkeypatch.setattr(DynamoHttpServer, "_raise_if_subprocess_died", lambda _self: None)

    with pytest.raises(RuntimeError, match="ThunderAgent.*rc=9"):
        server._raise_if_subprocess_died()


@pytest.mark.asyncio
async def test_shutdown_stops_frontend_then_thunderagent_then_base(monkeypatch) -> None:
    server = _make_http_server()
    server._frontend_process = object()
    server._thunderagent_process = object()
    events = []

    monkeypatch.setattr(server, "_stop_one", lambda _process, name, _timeout: events.append(name))

    async def base_shutdown(_self):
        events.append("base-workers-and-infra")

    monkeypatch.setattr(DynamoHttpServer, "shutdown", base_shutdown)

    await server.shutdown()

    assert events == ["frontend", "ThunderAgent", "base-workers-and-infra"]


def test_registered_replica_uses_thunderagent_server_class() -> None:
    assert DynamoThunderAgentReplica.__name__ == "DynamoThunderAgentReplica"


def test_recipe_registry_loads_thunderagent_replica() -> None:
    assert register._load_dynamo() is DynamoThunderAgentReplica


@pytest.mark.asyncio
async def test_llm_server_manager_constructs_thunderagent_replica(monkeypatch) -> None:
    constructed = []

    class FakeReplica:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._server_handle = object()
            self._server_address = "frontend:8000"

        async def init_hybrid_worker_pool(self, worker_group):
            assert worker_group == "worker-group"

    def reject_base_replica(**_kwargs):
        raise AssertionError("base PR #110 replica bypasses ThunderAgent")

    monkeypatch.setattr(dynamo_thunderagent, "DynamoThunderAgentReplica", FakeReplica)
    monkeypatch.setattr(dynamo_async_server, "DynamoReplica", reject_base_replica)
    manager = object.__new__(DynamoLLMServerManager)
    manager.worker_group = "worker-group"
    manager.start_rank = 0
    manager.rollout_config = SimpleNamespace(
        n_gpus_per_node=8,
        prometheus=SimpleNamespace(enable=False),
        name="dynamo",
    )
    manager.model_config = SimpleNamespace(local_path="test-model")

    await manager._initialize_llm_servers()

    assert constructed == [
        {
            "replica_rank": 0,
            "config": manager.rollout_config,
            "model_config": manager.model_config,
            "gpus_per_node": 8,
        }
    ]


@pytest.mark.asyncio
async def test_llm_server_manager_standalone_pool(monkeypatch) -> None:
    calls = []

    class FakeReplica:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["replica_rank"]))
            self._server_handle = object()
            self._server_address = "frontend:8001"

        async def init_standalone_pool(self):
            calls.append(("standalone", None))

        async def init_hybrid_worker_pool(self, worker_group):
            raise AssertionError("standalone manager must not use the hybrid worker pool")

    monkeypatch.setattr(dynamo_thunderagent, "DynamoThunderAgentReplica", FakeReplica)
    manager = object.__new__(DynamoLLMServerManager)
    manager.worker_group = None
    manager.start_rank = 1  # offset past the hybrid pool (separate_async)
    manager.rollout_config = SimpleNamespace(
        n_gpus_per_node=1,
        prometheus=SimpleNamespace(enable=False),
        name="dynamo",
    )
    manager.model_config = SimpleNamespace(local_path="test-model")

    await manager._initialize_llm_servers()

    assert calls == [("init", 1), ("standalone", None)]
    assert manager.server_addresses == ["frontend:8001"]


def test_recipe_config_enables_thunderagent_agent_loop() -> None:
    # ThunderAgent defaults live in the shared fragment (hydra.searchpath is
    # only legal in PRIMARY configs, so the deltas were split out).
    base = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_base.yaml").read_text())
    assert base["actor_rollout_ref"]["rollout"]["engine_kwargs"]["dynamo"]["thunderagent"] == {
        "enabled": True,
        "router_block_size": 16,
    }

    # The legacy entry keeps the V0 manager AND pins use_v1=false: the legacy
    # manager violates the V1 TransferQueue contract.
    legacy = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_trainer.yaml").read_text())
    assert legacy["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_manager_class"] == (
        "recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager"
    )
    assert legacy["trainer"]["use_v1"] is False


def test_v1_presets_compose_expectations() -> None:
    colocate = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_trainer_v1_colocate.yaml").read_text())
    assert colocate["trainer"]["use_v1"] is True
    assert colocate["trainer"]["v1"]["trainer_mode"] == "colocate_async"
    assert colocate["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_manager_class"] is None
    assert colocate["actor_rollout_ref"]["rollout"]["free_cache_engine"] is True
    assert colocate["actor_rollout_ref"]["rollout"]["engine_kwargs"]["dynamo"]["thunderagent"]["enabled"] is False

    separate = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_trainer_v1_separate.yaml").read_text())
    assert separate["trainer"]["v1"]["trainer_mode"] == "separate_async"
    assert separate["actor_rollout_ref"]["rollout"]["checkpoint_engine"]["backend"] == "nccl"
    assert separate["actor_rollout_ref"]["rollout"]["nnodes"] >= 1
    assert separate["actor_rollout_ref"]["actor"]["strategy"] == "fsdp2"

    # Both V1 presets must be PRIMARY configs (own hydra.searchpath) and pull
    # the shared fragment via defaults; the fragment must stay hydra-free.
    for name in ("dynamo_trainer_v1_colocate.yaml", "dynamo_trainer_v1_separate.yaml", "dynamo_trainer.yaml"):
        raw = yaml.safe_load((RECIPE_ROOT / "config" / name).read_text())
        assert "searchpath" in raw.get("hydra", {}), name
        assert "dynamo_base" in raw.get("defaults", []), name
    assert "hydra" not in yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_base.yaml").read_text())


def test_recipe_pins_tested_verl_and_dynamo_revisions() -> None:
    required_verl = (RECIPE_ROOT / "REQUIRED_VERL.txt").read_text()
    readme = (RECIPE_ROOT / "README.md").read_text()
    repository_readme = (REPOSITORY_ROOT / "README.md").read_text()

    assert "MODE=pinned_commit" in required_verl
    # Pin carried forward from the sglang branch (V1 interfaces verified there).
    assert "COMMIT=6cbca9ce7208100d11b4d1b06eccf098cc9e76aa" in required_verl
    assert "COMMIT=d82d2777" not in required_verl
    assert "8c5a73723109058f96c15fff3fc912231d65ad6e" in readme
    assert "dynamo/REQUIRED_VERL.txt" in repository_readme


def test_legacy_manager_rejects_v1_before_initialization(monkeypatch) -> None:
    from recipe.dynamo.dynamo_agent_loop import DynamoAgentLoopManager

    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    def unexpected_init(*args, **kwargs):
        pytest.fail("V1 must be rejected before initializing the V0 manager")

    monkeypatch.setattr(AgentLoopManager, "__init__", unexpected_init)
    config = SimpleNamespace(trainer=SimpleNamespace(use_v1=True))
    with pytest.raises(ValueError, match="does not write TransferQueue"):
        DynamoAgentLoopManager(config=config, llm_client=None)


def test_legacy_manager_keeps_v0_client_contract(monkeypatch) -> None:
    from recipe.dynamo.dynamo_agent_loop import DynamoAgentLoopManager

    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    calls = []
    monkeypatch.setattr(AgentLoopManager, "__init__", lambda self, config, **kwargs: calls.append((config, kwargs)))
    config = SimpleNamespace(trainer=SimpleNamespace(use_v1=False))
    client = object()
    DynamoAgentLoopManager(config=config, llm_client=client)
    assert calls == [(config, {"llm_client": client})]


# --------------------------------------------------------------------------- #
# V1 migration behavior tests: generation gate, logprob strictness, finalize
# recovery. CPU-only — servers/clients are constructed bare (object.__new__).
# --------------------------------------------------------------------------- #


def _make_bare_server() -> DynamoHttpServer:
    server = object.__new__(DynamoHttpServer)
    server.config = SimpleNamespace(engine_kwargs={})
    server.model_config = SimpleNamespace(
        local_path="/models/test-model",
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=False: [0]),
    )
    server.global_steps = 7
    server._generation_resumed = asyncio.Event()
    server._generation_resumed.set()
    server._control_endpoints = []
    server._logged_engine_data_token_ids = False
    server._logged_missing_engine_data = False
    server.node_rank = 0
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_kwargs", [{}, {"reject_request": False}, {"reject_request": True}])
async def test_generation_gate_returns_untagged_aborted_empty(abort_kwargs) -> None:
    server = _make_bare_server()

    aborted = await server.abort_all_requests(**abort_kwargs)
    assert aborted["paused"] is False  # no sidecars in this bare setup
    assert not server._generation_resumed.is_set()

    out = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 4}, request_id="r1")
    assert out.stop_reason == "aborted"
    assert out.token_ids == []
    # Empty attempts must NOT carry a version tag (trajectory-span metrics).
    assert "global_steps" not in out.extra_fields

    await server.resume_generation()
    assert server._generation_resumed.is_set()


def test_normalize_log_probs_rejects_mismatch_and_none() -> None:
    with pytest.raises(RuntimeError, match="refusing to pad"):
        DynamoHttpServer._normalize_log_probs([0.1], 2)
    with pytest.raises(RuntimeError, match="None logprob"):
        DynamoHttpServer._normalize_log_probs([0.1, None], 2)
    assert DynamoHttpServer._normalize_log_probs([0.125, 0.25], 2) == [0.125, 0.25]


def test_extract_log_probs_rejects_cross_provenance() -> None:
    choice = {
        "nvext": {"engine_data": {"completion_token_ids": [1, 2]}},
        "logprobs": {"token_logprobs": [-0.1, -0.2]},
    }
    with pytest.raises(RuntimeError, match="cross-provenance"):
        DynamoHttpServer._extract_completion_log_probs(choice, 2, {})


def test_missing_logprob_source_raises_at_response_boundary() -> None:
    server = _make_bare_server()
    data = {"choices": [{"text": "hi", "finish_reason": "stop", "logprobs": {"token_ids": [1, 2]}}]}
    with pytest.raises(RuntimeError, match="no usable logprob"):
        server._completion_response_to_token_output(data, include_log_probs=True)


def test_aborted_responses_degrade_or_keep_partials() -> None:
    server = _make_bare_server()

    # cancelled (dynamo's normalization of vLLM "abort") + untrusted logprobs
    # degrades to an aborted-empty retry instead of crashing the trajectory.
    broken = {"choices": [{"text": "x", "finish_reason": "cancelled", "logprobs": {"token_ids": [1]}}]}
    out = server._completion_response_to_token_output(broken, include_log_probs=True)
    assert out.stop_reason == "aborted" and out.token_ids == []

    # cancelled with a consistent logprob source keeps the partial tokens.
    good = {
        "choices": [
            {
                "text": "ab",
                "finish_reason": "cancelled",
                "logprobs": {"token_ids": [1, 2], "token_logprobs": [-0.1, -0.2]},
            }
        ]
    }
    out = server._completion_response_to_token_output(good, include_log_probs=True)
    assert out.stop_reason == "aborted"
    assert out.token_ids == [1, 2]
    assert out.log_probs == [-0.1, -0.2]


class _FakeLoadBalancer:
    def __init__(self, server, fail_times: int = 0):
        self._server = server
        self.fail_times = fail_times
        self.acquire_keys: list[str] = []
        self.acquire_server = _RemoteMethod(self._acquire)
        self.release_server = SimpleNamespace(remote=lambda **_kwargs: None)

    def _acquire(self, request_id: str):
        self.acquire_keys.append(request_id)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("lb transient failure")
        return ("srv0", self._server)


class _FailingFinalizeServer(_FakeServer):
    def _finalize_program(self, session_id: str) -> None:
        raise RuntimeError("finalize transport down")


def _make_fully_async_client(lb, handles, threshold: int = 50) -> DynamoFullyAsyncLLMServerClient:
    return DynamoFullyAsyncLLMServerClient(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="dynamo"))),
        load_balancer_handle=lb,
        dynamo_server_handles=handles,
        auto_finalize=True,
        finalize_leak_threshold=threshold,
        finalize_timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_finalize_routes_by_request_id_not_session_id() -> None:
    server = _FakeServer()
    lb = _FakeLoadBalancer(server)
    client = _make_fully_async_client(lb, [server])

    await client.finalize_program("session-A", routing_key="request-B")

    assert lb.acquire_keys == ["request-B"]  # sticky lookup uses the routing key
    assert server.finalize_calls == ["session-A"]  # program key stays the session


@pytest.mark.asyncio
async def test_finalize_recovery_retries_then_succeeds() -> None:
    server = _FakeServer()
    lb = _FakeLoadBalancer(server, fail_times=2)
    client = _make_fully_async_client(lb, [server])

    await client._finalize_with_recovery("session-A", routing_key="session-A")

    assert server.finalize_calls == ["session-A"]
    assert client._unresolved_finalize_leaks == 0


@pytest.mark.asyncio
async def test_broadcast_fallback_counts_as_unconfirmed_leak() -> None:
    # A "successful" broadcast reaches only THIS pool's frontends; the session
    # may have been served by the other pool (separate_async), where an
    # unknown-session finalize no-ops with 2xx — cleanup is unconfirmed and
    # must count as a leak, not reset the counter.
    server = _FakeServer()
    lb = _FakeLoadBalancer(server, fail_times=99)
    client = _make_fully_async_client(lb, [server])

    await client._finalize_with_recovery("session-A", routing_key="session-A")

    assert server.finalize_calls == ["session-A"]  # broadcast dispatched
    assert client._unresolved_finalize_leaks == 1  # ...but unconfirmed


@pytest.mark.asyncio
async def test_leak_counter_is_cumulative_across_successes() -> None:
    # Leaked programs are permanent: an interleaved successful cleanup for a
    # DIFFERENT session must not reset the counter (the old consecutive
    # counter let alternating fail/success accumulate unbounded leaks).
    failing = _FailingFinalizeServer()
    working = _FakeServer()
    lb_fail = _FakeLoadBalancer(failing, fail_times=99)
    client = _make_fully_async_client(lb_fail, [failing], threshold=2)

    await client._finalize_with_recovery("s1", routing_key="s1")
    assert client._unresolved_finalize_leaks == 1

    client._load_balancer = _FakeLoadBalancer(working)
    client._dynamo_server_handles = [working]
    await client._finalize_with_recovery("s-ok", routing_key="s-ok")  # confirmed success
    assert client._unresolved_finalize_leaks == 1  # NOT reset

    client._load_balancer = _FakeLoadBalancer(failing, fail_times=99)
    client._dynamo_server_handles = [failing]
    with pytest.raises(RuntimeError, match="cumulative unconfirmed ThunderAgent finalize"):
        await client._finalize_with_recovery("s2", routing_key="s2")


@pytest.mark.asyncio
async def test_manual_finalize_uses_recorded_routing_key() -> None:
    # auto_finalize=false multi-turn callers invoke finalize_program with only
    # the session id; the client must route via the request_id recorded at
    # generate time, not the session id.
    server = _FakeServer()
    lb = _FakeLoadBalancer(server)
    client = _make_fully_async_client(lb, [server])
    client._auto_finalize = False

    client._remember_routing_key("sess-X", "req-Y")
    await client.finalize_program("sess-X")

    assert lb.acquire_keys == ["req-Y"]
    assert server.finalize_calls == ["sess-X"]
    assert "sess-X" not in client._session_routing_keys  # popped after success


class _TokenServer(_FakeServer):
    """Fake frontend returning canned TokenOutputs (aborts first, then serves)."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = list(outputs)

    def _generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self._outputs.pop(0)


class _SwitchingLoadBalancer:
    """First acquire lands on the hybrid server; separate_async then removes
    it from the pool, so every later acquire (the aborted retry, and any
    sticky lookup) re-selects the standalone server — verl's real LB deletes
    the stale sticky entry and re-selects exactly like this."""

    def __init__(self, hybrid, standalone):
        self._hybrid = hybrid
        self._standalone = standalone
        self.acquire_keys: list[str] = []
        self.acquire_server = _RemoteMethod(self._acquire)
        self.release_server = SimpleNamespace(remote=lambda **_kwargs: None)
        self.require_acquire_fields = _RemoteMethod(lambda: [])
        self.require_release_fields = _RemoteMethod(lambda: [])

    def _acquire(self, request_id: str):
        self.acquire_keys.append(request_id)
        if len(self.acquire_keys) == 1:
            return ("hybrid", self._hybrid)
        return ("standalone", self._standalone)


@pytest.mark.asyncio
async def test_cross_pool_retry_finalizes_all_served_frontends() -> None:
    # final_review 9.2: hybrid serves the first attempt (program created
    # there), switch_to_trainer removes it from the LB and aborts the
    # request, the retry re-routes to standalone — finalize must reach BOTH
    # frontends, not just the one the sticky entry ends up pointing at.
    from verl.workers.rollout.replica import TokenOutput

    hybrid = _TokenServer([TokenOutput(token_ids=[], log_probs=[], num_preempted=0, stop_reason="aborted")])
    standalone = _TokenServer([TokenOutput(token_ids=[7], log_probs=[-0.1], num_preempted=0, stop_reason="stop")])
    lb = _SwitchingLoadBalancer(hybrid, standalone)
    client = _make_fully_async_client(lb, [standalone])

    output = await client.generate("traj-1", prompt_ids=[1], sampling_params={"max_tokens": 4})

    assert output.token_ids == [7]
    assert lb.acquire_keys == ["traj-1", "traj-1"]  # same routing key, re-routed
    assert hybrid.finalize_calls == ["traj-1"]
    assert standalone.finalize_calls == ["traj-1"]
    assert client._unresolved_finalize_leaks == 0
    assert "traj-1" not in client._session_servers  # dropped after confirm-all


@pytest.mark.asyncio
async def test_unconfirmed_served_frontend_counts_individual_leak() -> None:
    # One of the served frontends never acks its finalize: that program is a
    # permanent frontend-local leak (broadcasting to OTHER frontends cannot
    # free it) and must count toward the cumulative threshold — while the
    # confirmable frontend is still cleaned up.
    failing = _FailingFinalizeServer()
    working = _FakeServer()
    client = _make_fully_async_client(_FakeLoadBalancer(working), [working])
    client._record_served_server("s1", "hybrid", failing)
    client._record_served_server("s1", "standalone", working)

    await client._finalize_with_recovery("s1", routing_key="s1")

    assert working.finalize_calls == ["s1"]
    assert client._unresolved_finalize_leaks == 1
    assert "s1" not in client._session_servers


@pytest.mark.asyncio
async def test_manual_finalize_covers_all_served_frontends() -> None:
    # auto_finalize=false multi-turn callers: the manual finalize must use the
    # servers recorded at generate time — not the LB sticky entry, which after
    # a switch points only at the last pool.
    hybrid = _FakeServer()
    standalone = _FakeServer()
    lb = _FakeLoadBalancer(standalone)
    client = _make_fully_async_client(lb, [standalone])
    client._auto_finalize = False
    client._record_served_server("sess-M", "hybrid", hybrid)
    client._record_served_server("sess-M", "standalone", standalone)

    await client.finalize_program("sess-M")

    assert hybrid.finalize_calls == ["sess-M"]
    assert standalone.finalize_calls == ["sess-M"]
    assert lb.acquire_keys == []  # sticky lookup not consulted
    assert "sess-M" not in client._session_servers


@pytest.mark.asyncio
async def test_finalize_rpc_is_time_bounded() -> None:
    # A hung frontend (e.g. paused hybrid whose router stopped answering)
    # must not park the trajectory for the full request_timeout_s.
    class _HungServer(_FakeServer):
        async def _hang(self, **_kwargs):
            await asyncio.sleep(600)

        def __init__(self):
            super().__init__()
            self.finalize_program = _RemoteMethod(self._hang)

    hung = _HungServer()
    client = _make_fully_async_client(_FakeLoadBalancer(hung), [hung])
    client._finalize_timeout_s = 0.05
    client._record_served_server("s-hung", "hybrid", hung)

    await client._finalize_with_recovery("s-hung", routing_key="s-hung")

    assert client._unresolved_finalize_leaks == 1  # timed out -> unconfirmed


@pytest.mark.asyncio
async def test_probe_deadline_caps_hung_requests() -> None:
    server = _make_bare_server()
    server.config = SimpleNamespace(
        engine_kwargs={"dynamo": {"logprob_probe_timeout_s": 1}},
        calculate_log_probs=True,
    )

    async def hung_generate(**_kwargs):
        await asyncio.sleep(600)

    server.generate = hung_generate
    with pytest.raises(RuntimeError, match="probe timed out"):
        await server.probe_logprob_channel()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_request", [False, True])
async def test_sglang_abort_closes_gate_then_aborts_and_flushes(reject_request) -> None:
    # V1 partial rollout on sglang: pause-equivalent semantics = close the
    # engine-agnostic gate FIRST (nothing slips in between abort and weight
    # sync), then abort in-flight via tokenizer_manager, then flush the radix
    # cache (sglang's flush no-ops while requests run, so order matters).
    server = _make_bare_server()
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"engine": "sglang"}})
    calls = []

    async def control_all(method, **kwargs):
        # The gate must already be closed when the engine-side abort fires.
        assert not server._generation_resumed.is_set()
        calls.append(method)

    server._sglang_control_all = control_all

    result = await server.abort_all_requests(reject_request=reject_request)

    assert result["paused"] is True
    # pause_generation, NOT abort_request: sglang's abort_request is sync and
    # dynamo's passthrough awaits results unconditionally (500 on NoneType).
    assert calls == ["pause_generation", "flush_cache"]
    assert not server._generation_resumed.is_set()

    # Retried/new generate() calls answer aborted-empty at the actor without
    # touching the frontend, and carry no version tag.
    out = await server.generate(prompt_ids=[1], sampling_params={"max_tokens": 4}, request_id="r1")
    assert out.stop_reason == "aborted" and out.token_ids == []
    assert "global_steps" not in out.extra_fields

    # resume_generation reopens engine intake (continue_generation) and the
    # gate; no ZMQ sidecars on the sglang path.
    server._control_endpoints = []
    await server.resume_generation()
    assert calls == ["pause_generation", "flush_cache", "continue_generation"]
    assert server._generation_resumed.is_set()


@pytest.mark.asyncio
async def test_sglang_abort_skips_flush_when_not_requested() -> None:
    server = _make_bare_server()
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"engine": "sglang"}})
    calls = []

    async def control_all(method, **kwargs):
        calls.append(method)

    server._sglang_control_all = control_all

    result = await server.abort_all_requests(reset_prefix_cache=False)

    assert calls == ["pause_generation"]
    assert result["paused"] is True


def test_v1_sglang_preset_hard_requirements() -> None:
    # The sglang engine refuses to start without these three switches; the
    # preset must carry them so users cannot hit the runtime rejections.
    preset = yaml.safe_load((RECIPE_ROOT / "config" / "dynamo_trainer_v1_colocate_sglang.yaml").read_text())
    assert preset["trainer"]["use_v1"] is True
    assert preset["trainer"]["v1"]["trainer_mode"] == "colocate_async"
    rollout = preset["actor_rollout_ref"]["rollout"]
    assert rollout["enable_sleep_mode"] is True  # --enable-memory-saver
    assert rollout["free_cache_engine"] is True
    dynamo_kwargs = rollout["engine_kwargs"]["dynamo"]
    assert dynamo_kwargs["engine"] == "sglang"
    assert dynamo_kwargs["request_completion_token_ids"] is True
    assert dynamo_kwargs["enable_worker_system_metrics"] is True
    assert dynamo_kwargs["thunderagent"]["enabled"] is False
    # Must be a PRIMARY config (own hydra.searchpath) pulling the shared fragment.
    assert "searchpath" in preset.get("hydra", {})
    raw_defaults = preset.get("defaults", [])
    assert "dynamo_base" in raw_defaults


def test_control_actor_naming_standalone_override() -> None:
    # V1 separate_async: standalone CheckpointEngineWorkers resolve the
    # control actor by the POOL's replica_rank (offset past the hybrid
    # pool); hybrid keeps the shared_pool key. Both engine adapters delegate
    # here (importing the sglang adapter would pull the sglang package, so
    # the naming contract is tested at its single source of truth).
    from recipe.dynamo.dynamo_naming import control_actor_name

    engine_kwargs = {"dynamo": {"engine": "sglang"}}
    assert control_actor_name(engine_kwargs, 1) == "dynamo_server_0_1"
    assert control_actor_name(engine_kwargs, 1, replica_rank=3) == "dynamo_server_3_1"
    # shared_pool key only applies when no explicit replica_rank is given
    pooled = {"dynamo": {"shared_pool_replica_rank": 2}}
    assert control_actor_name(pooled, 0) == "dynamo_server_2_0"
    assert control_actor_name(pooled, 0, replica_rank=5) == "dynamo_server_5_0"


def test_thunderagent_sglang_workers_use_internal_model_name(monkeypatch) -> None:
    # With ThunderAgent on, sglang workers must register under the suffixed
    # name so the ONLY handler advertising the public model is the TA router —
    # otherwise generation silently bypasses program affinity (the vLLM path
    # has had this rename since PR #126; the sglang path was missing it).
    server = _make_http_server()
    base_calls = []

    def build_base(_self, served_model_name, _tp, nccl_port=None, **kwargs):
        base_calls.append((served_model_name, kwargs))
        return ["python", "-m", "dynamo.sglang", "--served-model-name", served_model_name]

    monkeypatch.setattr(DynamoHttpServer, "_build_sglang_cmd", build_base)

    command = server._build_sglang_cmd("test-model", 1)
    assert base_calls == [("test-model--verl-thunderagent-backend", {})]
    assert command[-1] == "test-model--verl-thunderagent-backend"
    # The launcher passes kv_events_config_json by keyword; the override must forward it,
    # otherwise every sglang launch with enable_kv_events (the default) dies with TypeError.
    server._build_sglang_cmd("test-model", 1, kv_events_config_json="{}")
    assert base_calls[-1] == ("test-model--verl-thunderagent-backend", {"kv_events_config_json": "{}"})

    # Explicit page_size contradicting the router block size fails fast.
    server.config = SimpleNamespace(
        engine_kwargs={
            "dynamo": {
                "engine": "sglang",
                "sglang": {"page_size": 32},
                "thunderagent": {"enabled": True, "router_block_size": 16},
            }
        }
    )
    with pytest.raises(ValueError, match="block sizes must match"):
        server._build_sglang_cmd("test-model", 1)

    # TA off: no rename, no check.
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"engine": "sglang", "thunderagent": {"enabled": False}}})
    base_calls.clear()
    server._build_sglang_cmd("test-model", 1, kv_events_config_json="{}")
    assert base_calls == [("test-model", {"kv_events_config_json": "{}"})]


# --------------------------------------------------------------------------- #
# Frontend request identity: one X-Request-Id per request, one session key per
# trajectory.
# --------------------------------------------------------------------------- #


def _make_posting_server(responses):
    """A bare server whose frontend POST is captured instead of sent."""
    server = _make_bare_server()
    server.posts = []
    replies = list(responses)

    async def post(payload, request_id, http_request_id=None):
        server.posts.append((payload.get("request_id"), request_id, http_request_id))
        return replies.pop(0)

    server._frontend_post = post
    server._build_frontend_completion_payload = lambda prompt_ids, sampling_params, request_id: {
        "request_id": request_id
    }
    server._completion_response_to_token_output = lambda body, include_log_probs=False: body
    return server


def _ok_body() -> str:
    return json.dumps({"choices": [{"text": "", "finish_reason": "stop"}]})


@pytest.mark.asyncio
async def test_concurrent_turns_of_one_trajectory_get_distinct_frontend_request_ids() -> None:
    # verl reuses one request_id for every turn of a trajectory; Claude Code
    # subagents make those turns concurrent, so the frontend must still see a
    # unique id per request while the session key stays the trajectory's.
    server = _make_posting_server([(200, _ok_body()), (200, _ok_body())])

    await asyncio.gather(
        server.generate(prompt_ids=[1], sampling_params={}, request_id="traj-1"),
        server.generate(prompt_ids=[2], sampling_params={}, request_id="traj-1"),
    )

    payload_ids, session_keys, http_ids = zip(*server.posts, strict=True)
    assert session_keys == ("traj-1", "traj-1")
    assert http_ids[0] != http_ids[1]
    assert payload_ids == http_ids
    assert all(http_id.startswith("traj-1-") for http_id in http_ids)


@pytest.mark.asyncio
async def test_extension_field_retry_is_the_same_frontend_request() -> None:
    server = _make_posting_server([(400, "Unsupported parameter: nvext"), (200, _ok_body())])

    await server.generate(prompt_ids=[1], sampling_params={}, request_id="traj-1")

    assert len(server.posts) == 2
    assert server.posts[0][2] == server.posts[1][2]


@pytest.mark.asyncio
async def test_frontend_post_keeps_the_session_key_and_overrides_the_request_id() -> None:
    server = _make_bare_server()
    server.config = SimpleNamespace(engine_kwargs={"dynamo": {"router_session_affinity_ttl_secs": 60}})
    server._server_address, server._server_port = "127.0.0.1", 8000
    sent = {}

    class _Response:
        status = 200

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, *, json, headers, timeout):
            sent.update(headers)
            return _Response()

    async def get_session():
        return _Session()

    server._get_http_session = get_session

    await server._frontend_post({}, "traj-1", "traj-1-0001")

    assert sent["x-dynamo-session-id"] == "traj-1"
    assert sent["X-Request-Id"] == "traj-1-0001"
