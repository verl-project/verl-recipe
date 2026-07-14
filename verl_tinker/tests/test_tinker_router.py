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
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from verl_tinker.config_utils import _validate_config
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
    server._futures = {}
    server._pending = {}
    server._errors = {}
    server._request_status = {}
    server._retrieved_request_status_archive = {}


def test_critic_routes_are_not_registered():
    paths = {route.path for route in app.routes}

    assert "/api/v1/compute_values" not in paths
    assert "/api/v1/compute_advantages" not in paths
    assert "/api/v1/update_critic" not in paths


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
        ({}, "/models/original-model"),
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
