# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import threading
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from verl_tinker.config_utils import _validate_config
from verl_tinker.state_tracker import ModelStateTracker
from verl_tinker.tinker_router import (
    TINKER_COOKBOOK_COMPAT_LORA_RANK,
    FifoReadWriteGate,
    RequestStatus,
    ScheduledOpKind,
    ServerStatus,
    TinkerServer,
    app,
)


def _router_class():
    for cls in TinkerServer.func_or_class.__mro__:
        if cls.__module__ == "verl_tinker.tinker_router" and cls.__name__ == "TinkerServer":
            return cls
    raise AssertionError("Original TinkerServer class not found under Ray Serve wrapper")


def _init_future_tracking(server):
    if not hasattr(server, "_status"):
        server._status = ServerStatus.INITIALIZED
    server._futures = {}
    server._pending = {}
    server._errors = {}
    server._request_status = {}
    server._retrieved_request_status_archive = {}


def _init_state_tracker(server, actor_model="actor", actor_identifiers=None, teacher_models=()):
    server._state_tracker = ModelStateTracker(
        actor_model_identifiers=actor_identifiers or (actor_model,),
        teacher_models=teacher_models,
    )


def test_critic_routes_are_not_registered():
    paths = {route.path for route in app.routes}

    assert "/api/v1/compute_values" not in paths
    assert "/api/v1/compute_advantages" not in paths
    assert "/api/v1/update_critic" not in paths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [ServerStatus.INITIALIZING, ServerStatus.ERROR, ServerStatus.SHUTTING_DOWN, ServerStatus.SHUTDOWN_COMPLETE],
)
async def test_normal_endpoints_require_fully_initialized_server(status):
    server = object.__new__(_router_class())
    server._status = status
    _init_state_tracker(server)

    with pytest.raises(HTTPException) as exc_info:
        await server.get_sampler("missing")

    assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert status.value in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_sampler_uses_tracker_without_actor_engine():
    server = object.__new__(_router_class())
    server._status = ServerStatus.INITIALIZED
    _init_state_tracker(server)
    server._state_tracker.register_actor_sampler("actor", base_model="actor", legal_rollout_ids={0})

    assert await server.get_sampler("actor") == {
        "sampler_id": "actor",
        "base_model": "actor",
        "model_path": None,
    }


@pytest.mark.asyncio
async def test_get_session_returns_only_currently_valid_samplers():
    server = object.__new__(_router_class())
    server._status = ServerStatus.INITIALIZED
    _init_state_tracker(server)
    server._state_tracker.register_actor_sampler("stale", base_model="actor", legal_rollout_ids={0})
    server._state_tracker.actor_updated()
    server._state_tracker.rollout_synchronized()
    server._state_tracker.register_actor_sampler("current", base_model="actor", legal_rollout_ids={1})

    response = await server.get_session("session")

    assert response["sampler_ids"] == ["current"]


def test_no_rollout_config_requires_backend_runtime_sections():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"path": "/models/qwen"}},
        }
    )

    with patch("verl_tinker.config_utils._validate_supported_verl_config") as mock_validate:
        errors = _validate_config(config)

    assert "algorithm config is required" in errors
    assert "trainer.nnodes is required" in errors
    assert "trainer.n_gpus_per_node is required" in errors
    mock_validate.assert_not_called()


def test_create_model_metadata_is_full_model_training_even_with_lora_request():
    server = object.__new__(_router_class())
    req = SimpleNamespace(
        base_model="Qwen/Qwen3-1.7B",
        lora_config=SimpleNamespace(rank=128, train_unembed=True, train_mlp=True, train_attn=True),
    )

    assert server._metadata_from_create_model(req) == {
        "base_model": "Qwen/Qwen3-1.7B",
        "is_lora": False,
        "lora_rank": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_config", "expected_model_name"),
    [
        ({"model_name": "configured-model"}, "configured-model"),
    ],
)
async def test_get_info_always_uses_configured_model_for_model_and_tokenizer(server_config, expected_model_name):
    server = object.__new__(_router_class())
    server._engine = SimpleNamespace(
        config=OmegaConf.create(
            {
                "server": server_config,
                "actor_rollout_ref": {"model": {"path": "/models/original-model"}},
            }
        )
    )
    server._status = ServerStatus.INITIALIZED
    server._shutdown_started = False
    server._model_to_base_model = {"verl-remote-actor-model": "client-requested-model"}

    response = await server.get_info(SimpleNamespace())

    assert response["model_data"]["model_name"] == expected_model_name
    assert response["model_data"]["tokenizer_id"] == expected_model_name
    assert response["model_name"] == expected_model_name


@pytest.mark.asyncio
async def test_create_sampling_session_routes_exact_teacher_model_to_unique_sampler():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = SimpleNamespace(
        config=OmegaConf.create({"server": {}, "actor_rollout_ref": {"model": {"path": "actor"}}})
    )
    server._teacher_backend = SimpleNamespace(
        resolve=lambda *values: "teacher" if "teacher-model" in values else None,
        get_model_path=lambda key: "teacher-model",
    )
    _init_state_tracker(server, teacher_models=(("teacher-model", "teacher-model"),))

    response = await server.create_sampling_session(
        SimpleNamespace(
            session_id="session",
            sampling_session_seq_id=3,
            base_model="teacher-model",
            model_path=None,
        )
    )

    sampler_id = response.sampling_session_id
    assert sampler_id == "session:sample:3"
    binding = server._state_tracker.get_sampler(sampler_id)
    assert binding.teacher_model_path == "teacher-model"
    assert binding.base_model == "teacher-model"
    assert binding.model_path == "teacher-model"


@pytest.mark.asyncio
async def test_create_sampling_session_rejects_teacher_checkpoint_path():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = SimpleNamespace(
        config=OmegaConf.create({"server": {}, "actor_rollout_ref": {"model": {"path": "actor"}}})
    )
    aliases = {"teacher-name": "teacher"}
    server._teacher_backend = SimpleNamespace(
        resolve=lambda *values: next((aliases[value] for value in values if value in aliases), None),
        get_model_path=lambda key: "/models/teacher",
    )
    _init_state_tracker(server, teacher_models=(("teacher-name", "/models/teacher"),))

    with pytest.raises(HTTPException) as exc_info:
        await server.create_sampling_session(
            SimpleNamespace(
                session_id="session",
                sampling_session_seq_id=4,
                base_model="teacher-name",
                model_path="tinker://teacher/checkpoint",
            )
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert "frozen" in exc_info.value.detail


@pytest.mark.asyncio
async def test_asample_routes_teacher_sampler_to_teacher_client():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)

    class TeacherClient:
        def __init__(self):
            self.calls = []

        async def generate(self, request_id, *, prompt_ids, sampling_params):
            self.calls.append((prompt_ids, sampling_params))
            return SimpleNamespace(
                token_ids=[3],
                log_probs=[-0.2],
                stop_reason="stop",
                extra_fields={},
            )

    teacher_client = TeacherClient()
    server._teacher_backend = SimpleNamespace(
        get_client=lambda key: teacher_client,
    )
    _init_state_tracker(server)
    server._state_tracker.register_teacher_sampler(
        "teacher-sampler",
        teacher_model_path="teacher-model",
        base_model="teacher-model",
    )
    req = SimpleNamespace(
        sampling_session_id="teacher-sampler",
        prompt=SimpleNamespace(to_ints=lambda: [1, 2]),
        sampling_params=SimpleNamespace(max_tokens=1, temperature=1, top_p=1, top_k=-1, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )

    response = await server.asample(req)
    await asyncio.gather(*server._pending.values())

    assert teacher_client.calls[0][0] == [1, 2]
    assert server._futures[response["request_id"]]["sequences"][0]["tokens"] == [3]


@pytest.mark.asyncio
async def test_asample_rejects_unknown_non_actor_sampler():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._teacher_backend = None
    _init_state_tracker(server)

    with pytest.raises(HTTPException) as exc_info:
        await server.asample(SimpleNamespace(sampling_session_id="unknown"))

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_asample_without_session_resolves_direct_actor(monkeypatch):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    server._teacher_backend = None
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)
    _init_state_tracker(server)
    sample = AsyncMock(return_value={"type": "sample", "sequences": []})
    monkeypatch.setitem(server.asample.__globals__, "tinker_sample", sample)

    response = await server.asample(
        SimpleNamespace(
            sampling_session_id=None,
            base_model="actor",
            model_path=None,
        )
    )
    await asyncio.gather(*server._pending.values())

    assert server._futures[response["request_id"]] == {"type": "sample", "sequences": []}
    sample.assert_awaited_once()


@pytest.mark.asyncio
async def test_asample_without_session_rejects_unknown_direct_model():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._teacher_backend = None
    _init_state_tracker(server)

    with pytest.raises(HTTPException) as exc_info:
        await server.asample(
            SimpleNamespace(
                sampling_session_id=None,
                base_model="unknown",
                model_path=None,
            )
        )

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_asample_rejects_actor_sampler_after_rollout_diverges():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    server._teacher_backend = None
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)
    _init_state_tracker(server)
    server._state_tracker.register_actor_sampler(
        "actor-sampler",
        base_model="actor",
        legal_rollout_ids={0},
    )
    server._state_tracker.actor_updated()
    server._state_tracker.rollout_synchronized()

    response = await server.asample(SimpleNamespace(sampling_session_id="actor-sampler"))
    await asyncio.gather(*server._pending.values())

    request_id = response["request_id"]
    assert server._request_status[request_id] is RequestStatus.ERROR
    assert "409" in server._errors[request_id]
    assert "no longer resident" in server._errors[request_id]


@pytest.mark.asyncio
async def test_named_sampler_save_binds_path_to_current_rollout(monkeypatch):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    _init_state_tracker(server)
    server._state_tracker.actor_updated()
    saved_path = "tinker://verl-tinker/weights/saved"
    update_weights = AsyncMock(
        return_value={
            "type": "save_weights_for_sampler",
            "path": saved_path,
            "sampling_session_id": None,
        }
    )
    monkeypatch.setitem(
        server._run_save_weights_for_sampler.__globals__,
        "tinker_save_weights_for_sampler",
        update_weights,
    )

    result = await server._run_save_weights_for_sampler(
        SimpleNamespace(path="saved", sampling_session_seq_id=None)
    )
    response = await server.create_sampling_session(
        SimpleNamespace(
            session_id="session",
            sampling_session_seq_id=7,
            base_model="actor",
            model_path=saved_path,
        )
    )

    assert result["path"] == saved_path
    assert server._state_tracker.rollout_id == 1
    binding = server._state_tracker.get_sampler(response.sampling_session_id)
    assert binding.model_path == saved_path
    assert binding.legal_rollout_ids == frozenset({1})


@pytest.mark.asyncio
async def test_create_sampling_session_accepts_actor_name_or_load_path():
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._teacher_backend = None
    _init_state_tracker(
        server,
        actor_model="client-name",
        actor_identifiers=("client-name", "/models/actor"),
    )

    for sequence_id, base_model in enumerate(("client-name", "/models/actor")):
        response = await server.create_sampling_session(
            SimpleNamespace(
                session_id="session",
                sampling_session_seq_id=sequence_id,
                base_model=base_model,
                model_path=None,
            )
        )
        binding = server._state_tracker.get_sampler(response.sampling_session_id)
        assert binding.base_model == "client-name"


@pytest.mark.asyncio
async def test_unnamed_sampler_saves_return_distinct_sampler_ids(monkeypatch):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    _init_state_tracker(server)
    update_weights = AsyncMock(
        side_effect=[
            {"type": "save_weights_for_sampler", "path": None, "sampling_session_id": "legacy"},
            {"type": "save_weights_for_sampler", "path": None, "sampling_session_id": "legacy"},
        ]
    )
    monkeypatch.setitem(
        server._run_save_weights_for_sampler.__globals__,
        "tinker_save_weights_for_sampler",
        update_weights,
    )

    first = await server._run_save_weights_for_sampler(
        SimpleNamespace(path=None, sampling_session_seq_id=2)
    )
    second = await server._run_save_weights_for_sampler(
        SimpleNamespace(path=None, sampling_session_seq_id=3)
    )

    assert first["sampling_session_id"] == "verl-remote-actor:sample:2"
    assert second["sampling_session_id"] == "verl-remote-actor:sample:3"
    assert first["sampling_session_id"] != second["sampling_session_id"]
    assert server._state_tracker.get_sampler(first["sampling_session_id"]).legal_rollout_ids == frozenset({0})


@pytest.mark.asyncio
async def test_load_skips_engine_when_checkpoint_actor_is_already_loaded(tmp_path, monkeypatch):
    server = object.__new__(_router_class())
    _init_state_tracker(server)
    uri = "tinker://verl-tinker/state/already-loaded"
    local_dir = tmp_path / "already-loaded"
    local_dir.mkdir()
    server._checkpoint_root = str(tmp_path)
    server._saved_state_paths = {uri: str(local_dir)}
    server._state_tracker.state_saved(uri)
    engine_load = AsyncMock()
    monkeypatch.setitem(server._run_load_state.__globals__, "load_state", engine_load)

    result = await server._run_load_state({"path": uri, "optimizer": True})

    assert result == {"type": "load_weights", "path": uri}
    engine_load.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_failure_marks_actor_unknown_without_shutting_down(tmp_path, monkeypatch):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    _init_state_tracker(server)
    uri = "tinker://verl-tinker/state/broken"
    local_dir = tmp_path / "broken"
    local_dir.mkdir()
    server._checkpoint_root = str(tmp_path)
    server._saved_state_paths = {uri: str(local_dir)}
    monkeypatch.setitem(
        server._run_load_state.__globals__,
        "load_state",
        AsyncMock(side_effect=RuntimeError("load exploded")),
    )

    with pytest.raises(RuntimeError, match="load exploded"):
        await server._run_load_state({"path": uri, "optimizer": True})

    assert server._state_tracker.actor_id == 1
    assert server._state_tracker.rollout_id == 0
    assert server._status is ServerStatus.INITIALIZED
    assert server._shutdown_started is False


@pytest.mark.asyncio
async def test_rollout_sync_failure_marks_rollout_unknown_without_shutting_down(monkeypatch):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._engine = MagicMock()
    _init_state_tracker(server)
    monkeypatch.setitem(
        server._run_save_weights_for_sampler.__globals__,
        "tinker_save_weights_for_sampler",
        AsyncMock(side_effect=RuntimeError("sync exploded")),
    )

    with pytest.raises(RuntimeError, match="sync exploded"):
        await server._run_save_weights_for_sampler(
            SimpleNamespace(path=None, sampling_session_seq_id=9)
        )

    assert server._state_tracker.rollout_id is None
    assert server._status is ServerStatus.INITIALIZED
    assert server._shutdown_started is False


def test_weights_info_metadata_uses_cookbook_lora_compat_shape():
    server = object.__new__(_router_class())

    assert server._weights_info_compat_metadata(
        {
            "base_model": "Qwen/Qwen3-1.7B",
            "is_lora": False,
            "lora_rank": None,
            "extra": "kept",
        }
    ) == {
        "base_model": "Qwen/Qwen3-1.7B",
        "is_lora": True,
        "lora_rank": TINKER_COOKBOOK_COMPAT_LORA_RANK,
        "train_unembed": True,
        "train_mlp": True,
        "train_attn": True,
        "extra": "kept",
    }


@pytest.mark.asyncio
async def test_scheduled_samples_run_concurrently():
    server = object.__new__(_router_class())
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)

    events = []
    both_started = asyncio.Event()
    active = 0

    async def sample(name: str):
        nonlocal active
        events.append(f"start:{name}")
        active += 1
        if active == 2:
            both_started.set()
        await both_started.wait()
        active -= 1
        events.append(f"end:{name}")
        return {"type": name}

    server._schedule("first", sample("first"), ScheduledOpKind.SAMPLE)
    server._schedule("second", sample("second"), ScheduledOpKind.SAMPLE)
    tasks = list(server._pending.values())

    await asyncio.gather(*tasks)

    assert events[:2] == ["start:first", "start:second"]
    assert server._futures == {"first": {"type": "first"}, "second": {"type": "second"}}


@pytest.mark.asyncio
async def test_scheduled_exclusive_operations_run_in_received_order():
    server = object.__new__(_router_class())
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)

    events = []

    async def op(name: str, delay_s: float):
        events.append(f"start:{name}")
        await asyncio.sleep(delay_s)
        events.append(f"end:{name}")
        return {"type": name}

    server._schedule("first", op("first", 0.01))
    server._schedule("second", op("second", 0.0))
    tasks = list(server._pending.values())

    await asyncio.gather(*tasks)

    assert events == ["start:first", "end:first", "start:second", "end:second"]
    assert server._futures == {"first": {"type": "first"}, "second": {"type": "second"}}
    assert server._request_status == {"first": RequestStatus.COMPLETED, "second": RequestStatus.COMPLETED}


@pytest.mark.asyncio
async def test_retrieve_unknown_future_reports_whether_it_was_seen_before():
    server = object.__new__(_router_class())
    _init_future_tracking(server)
    server._futures = {"seen": {"type": "done"}}
    server._request_status = {"seen": RequestStatus.COMPLETED}

    assert await server.retrieve_future(SimpleNamespace(request_id="seen")) == {"type": "done"}
    assert server._request_status["seen"] is RequestStatus.RETRIEVED

    seen_again = await server.retrieve_future(SimpleNamespace(request_id="seen"))
    never_seen = await server.retrieve_future(SimpleNamespace(request_id="never-seen"))

    assert seen_again["category"] == "Application"
    assert seen_again["seen_before"] is True
    assert seen_again["seen_status"] == "seen_before"
    assert "already retrieved" in seen_again["error"]
    assert seen_again["request_id"] == "seen"

    assert never_seen["category"] == "Application"
    assert never_seen["seen_before"] is False
    assert never_seen["seen_status"] == "never_seen"
    assert "never seen" in never_seen["error"]
    assert never_seen["request_id"] == "never-seen"


@pytest.mark.asyncio
async def test_retrieve_error_marks_retrieved_and_clears_error_payload():
    server = object.__new__(_router_class())
    _init_future_tracking(server)
    server._errors = {"failed": "RuntimeError('boom')"}
    server._request_status = {"failed": RequestStatus.ERROR}

    response = await server.retrieve_future(SimpleNamespace(request_id="failed"))

    assert response == {"error": "RuntimeError('boom')", "category": "Application"}
    assert server._errors == {}
    assert server._request_status == {"failed": RequestStatus.RETRIEVED}


def test_request_status_compaction_keeps_only_retrieved_archive_batch(monkeypatch):
    server = object.__new__(_router_class())
    _init_future_tracking(server)
    monkeypatch.setitem(server._compact_request_status_if_needed.__globals__, "MAX_REQUEST_STATUS_ENTRIES", 4)
    server._retrieved_request_status_archive = {"old": RequestStatus.RETRIEVED}
    server._request_status = {
        "pending": RequestStatus.PENDING,
        "completed": RequestStatus.COMPLETED,
        "retrieved-1": RequestStatus.RETRIEVED,
        "retrieved-2": RequestStatus.RETRIEVED,
    }

    server._compact_request_status_if_needed()

    assert server._request_status == {
        "pending": RequestStatus.PENDING,
        "completed": RequestStatus.COMPLETED,
    }
    assert server._retrieved_request_status_archive == {
        "retrieved-1": RequestStatus.RETRIEVED,
        "retrieved-2": RequestStatus.RETRIEVED,
    }


def test_request_status_compaction_does_not_drop_operational_states(monkeypatch):
    server = object.__new__(_router_class())
    _init_future_tracking(server)
    monkeypatch.setitem(server._compact_request_status_if_needed.__globals__, "MAX_REQUEST_STATUS_ENTRIES", 2)
    server._retrieved_request_status_archive = {"old": RequestStatus.RETRIEVED}
    server._request_status = {
        "pending": RequestStatus.PENDING,
        "completed": RequestStatus.COMPLETED,
    }

    server._compact_request_status_if_needed()

    assert server._request_status == {
        "pending": RequestStatus.PENDING,
        "completed": RequestStatus.COMPLETED,
    }
    assert server._retrieved_request_status_archive == {"old": RequestStatus.RETRIEVED}


@pytest.mark.asyncio
async def test_scheduled_writer_blocks_later_samples_without_blocking_earlier_samples():
    server = object.__new__(_router_class())
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)

    events = []
    first_samples_started = asyncio.Event()
    release_first_samples = asyncio.Event()
    active_first_samples = 0

    async def first_sample(name: str):
        nonlocal active_first_samples
        events.append(f"start:{name}")
        active_first_samples += 1
        if active_first_samples == 2:
            first_samples_started.set()
        await release_first_samples.wait()
        events.append(f"end:{name}")
        return {"type": name}

    async def writer():
        events.append("start:writer")
        events.append("end:writer")
        return {"type": "writer"}

    async def later_sample():
        events.append("start:later")
        events.append("end:later")
        return {"type": "later"}

    server._schedule("sample-1", first_sample("sample-1"), ScheduledOpKind.SAMPLE)
    server._schedule("sample-2", first_sample("sample-2"), ScheduledOpKind.SAMPLE)
    await first_samples_started.wait()

    server._schedule("writer", writer())
    server._schedule("later", later_sample(), ScheduledOpKind.SAMPLE)
    await asyncio.sleep(0)

    assert events == ["start:sample-1", "start:sample-2"]

    release_first_samples.set()
    tasks = list(server._pending.values())
    await asyncio.gather(*tasks)

    assert events == [
        "start:sample-1",
        "start:sample-2",
        "end:sample-1",
        "end:sample-2",
        "start:writer",
        "end:writer",
        "start:later",
        "end:later",
    ]


@pytest.mark.asyncio
async def test_shutdown_rejects_new_work_and_drains_existing_tasks_before_engine_teardown():
    server = object.__new__(_router_class())
    server._status = ServerStatus.INITIALIZED
    server._error = None
    server._op_gate = FifoReadWriteGate()
    _init_future_tracking(server)
    server._shutdown_started = False
    server._init_future = asyncio.get_running_loop().create_future()
    server._init_future.set_result(None)
    server._gpu_executor = MagicMock()

    events = []
    release_task = asyncio.Event()

    class DummyEngine:
        def shutdown(self):
            events.append("engine:shutdown")

    server._engine = DummyEngine()

    async def existing_task():
        events.append("task:start")
        await release_task.wait()
        events.append("task:end")
        return {"type": "done"}

    async def rejected_task():
        events.append("new:start")

    server._schedule("existing", existing_task())
    await asyncio.sleep(0)

    server._begin_shutdown()

    with pytest.raises(HTTPException) as exc_info:
        server._schedule("new", rejected_task())
    assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    shutdown_task = asyncio.create_task(server._shutdown_process())
    await asyncio.sleep(0.2)

    assert events == ["task:start"]
    assert server._engine is not None

    release_task.set()
    await shutdown_task

    assert events == ["task:start", "task:end", "engine:shutdown"]
    assert server._futures == {"existing": {"type": "done"}}
    assert server._engine is None
    server._gpu_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)


@pytest.mark.asyncio
async def test_healthz_responds_while_shutdown_cleanup_is_blocked():
    server = object.__new__(_router_class())
    server._status = ServerStatus.INITIALIZED
    server._error = None
    _init_future_tracking(server)
    server._shutdown_started = False
    server._init_future = asyncio.get_running_loop().create_future()
    server._init_future.set_result(None)
    server._gpu_executor = MagicMock()

    shutdown_started = threading.Event()
    release_shutdown = threading.Event()

    class BlockingEngine:
        def shutdown(self):
            shutdown_started.set()
            assert release_shutdown.wait(timeout=5)

    server._engine = BlockingEngine()
    server._begin_shutdown()

    shutdown_task = asyncio.create_task(server._shutdown_process())
    assert await asyncio.to_thread(shutdown_started.wait, 2)

    assert await server.healthz() == {"status": ServerStatus.SHUTTING_DOWN.value}
    assert not shutdown_task.done()

    release_shutdown.set()
    await shutdown_task

    assert await server.healthz() == {"status": ServerStatus.SHUTDOWN_COMPLETE.value}
    assert server._engine is None
    server._gpu_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)


@pytest.mark.asyncio
async def test_gate_skips_cancelled_queued_writer():
    gate = FifoReadWriteGate()
    events = []
    release_reader = asyncio.Event()

    async def hold_reader():
        async with gate.hold(ScheduledOpKind.SAMPLE):
            events.append("reader:start")
            await release_reader.wait()
            events.append("reader:end")

    async def queued_writer():
        async with gate.hold(ScheduledOpKind.EXCLUSIVE):
            events.append("writer")

    async def later_sample():
        async with gate.hold(ScheduledOpKind.SAMPLE):
            events.append("later")

    reader_task = asyncio.create_task(hold_reader())
    await asyncio.sleep(0)
    writer_task = asyncio.create_task(queued_writer())
    await asyncio.sleep(0)
    writer_task.cancel()
    await asyncio.gather(writer_task, return_exceptions=True)

    later_task = asyncio.create_task(later_sample())
    await asyncio.sleep(0)
    assert events == ["reader:start", "later"]

    release_reader.set()
    await asyncio.gather(reader_task, later_task)

    assert events == ["reader:start", "later", "reader:end"]


@pytest.mark.asyncio
async def test_gate_skips_cancelled_queued_sample():
    gate = FifoReadWriteGate()
    events = []
    release_writer = asyncio.Event()

    async def hold_writer():
        async with gate.hold(ScheduledOpKind.EXCLUSIVE):
            events.append("writer:start")
            await release_writer.wait()
            events.append("writer:end")

    async def queued_sample():
        async with gate.hold(ScheduledOpKind.SAMPLE):
            events.append("sample")

    async def later_sample():
        async with gate.hold(ScheduledOpKind.SAMPLE):
            events.append("later")

    writer_task = asyncio.create_task(hold_writer())
    await asyncio.sleep(0)
    sample_task = asyncio.create_task(queued_sample())
    await asyncio.sleep(0)
    sample_task.cancel()
    await asyncio.gather(sample_task, return_exceptions=True)

    later_task = asyncio.create_task(later_sample())
    await asyncio.sleep(0)
    assert events == ["writer:start"]

    release_writer.set()
    await asyncio.gather(writer_task, later_task)

    assert events == ["writer:start", "writer:end", "later"]


@pytest.mark.asyncio
async def test_gate_limits_concurrent_samples():
    gate = FifoReadWriteGate(max_readers=2)
    events = []
    release_samples = asyncio.Event()

    async def sample(name: str):
        async with gate.hold(ScheduledOpKind.SAMPLE):
            events.append(f"start:{name}")
            await release_samples.wait()
            events.append(f"end:{name}")

    tasks = [asyncio.create_task(sample(str(i))) for i in range(3)]
    await asyncio.sleep(0)

    assert events == ["start:0", "start:1"]

    release_samples.set()
    await asyncio.gather(*tasks)

    assert events[:2] == ["start:0", "start:1"]
    assert "start:2" in events[2:]
    assert sorted(events) == ["end:0", "end:1", "end:2", "start:0", "start:1", "start:2"]
