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

"""Ray Serve deployment exposing the Tinker-compatible server API."""

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from ray import serve
from tinker.types import (
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    FutureRetrieveRequest,
    GetInfoRequest,
    OptimStepRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
)

from .backends.colocated import ColocatedBackend
from .backends.teacher import TeacherInferenceBackend
from .model_resource_manager import (
    ModelResourceManager,
    ModelResourceManagerError,
    SamplingResource,
    StaleSamplerError,
    UnknownSamplerError,
)
from .schemas import StatusResponse, TinkerForwardBackwardRequest, TinkerForwardRequest
from .tinker_ops import (
    forward as tinker_forward,
)
from .tinker_ops import (
    forward_backward as tinker_forward_backward,
)
from .tinker_ops import (
    get_configured_model_name,
    get_supported_models,
    load_state,
    load_state_metadata,
    save_state,
    state_path_to_local,
)
from .tinker_ops import (
    optim_step as tinker_optim_step,
)
from .tinker_ops import (
    sample as tinker_sample,
)
from .tinker_ops import (
    sample_prompt_logprobs as tinker_sample_prompt_logprobs,
)
from .tinker_ops import (
    save_weights_for_sampler as tinker_save_weights_for_sampler,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="VeRL Tinker Server", version="0.1.0")

TINKER_COOKBOOK_COMPAT_LORA_RANK = 1
MAX_REQUEST_STATUS_ENTRIES = 100_000


class ServerStatus(str, Enum):
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    ERROR = "error"


class ScheduledOpKind(str, Enum):
    SAMPLE = "sample"
    EXCLUSIVE = "exclusive"


class RequestStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    RETRIEVED = "retrieved"
    ERROR = "error"


class FifoReadWriteGate:
    """
    FIFO gate where consecutive SAMPLE operations may run concurrently.

    Every non-SAMPLE operation is exclusive and acts as a barrier:

        SAMPLE, SAMPLE, EXCLUSIVE, SAMPLE
        └── run together ──┘          └─ waits for EXCLUSIVE
    """

    def __init__(self, max_readers: int = 16):
        max_readers = int(max_readers)
        if max_readers <= 0:
            raise ValueError(f"max_readers must be positive, got {max_readers}")

        self._max_readers = max_readers

        # Protects the queue and active-operation counters.
        self._lock = asyncio.Lock()

        # Each queued request gets an Event that is set when it may enter.
        self._queue: deque[tuple[ScheduledOpKind, asyncio.Event]] = deque()

        self._active_samples = 0
        self._exclusive_active = False

    def _grant_ready_locked(self) -> None:
        """Grant the FIFO prefix that can currently run.

        The caller must hold self._lock.
        """
        if self._exclusive_active:
            return

        while self._queue:
            kind, ready = self._queue[0]

            if kind is not ScheduledOpKind.SAMPLE:
                # A non-sample request is an exclusive barrier. It cannot start
                # until all earlier samples finish, and later requests cannot
                # pass it.
                if self._active_samples > 0:
                    return

                self._queue.popleft()
                self._exclusive_active = True
                ready.set()
                return

            # Samples may run concurrently only while they form the continuous
            # prefix at the front of the queue.
            if self._active_samples >= self._max_readers:
                return

            self._queue.popleft()
            self._active_samples += 1
            ready.set()

    @asynccontextmanager
    async def hold(
        self,
        kind: ScheduledOpKind,
    ) -> AsyncIterator[None]:
        ready = asyncio.Event()
        waiter = (kind, ready)

        # The order in which requests are appended here defines FIFO order.
        async with self._lock:
            self._queue.append(waiter)
            self._grant_ready_locked()

        try:
            await ready.wait()
            yield
        finally:
            async with self._lock:
                if ready.is_set():
                    # The request had already been granted, so release its slot.
                    if kind is ScheduledOpKind.SAMPLE:
                        self._active_samples -= 1
                    else:
                        self._exclusive_active = False
                else:
                    # The caller was cancelled while it was still waiting.
                    self._queue.remove(waiter)

                self._grant_ready_locked()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=1024,
    graceful_shutdown_wait_loop_s=5.0,
    graceful_shutdown_timeout_s=5.0,
)
@serve.ingress(app)
class TinkerServer:
    """Single Ray Serve/FastAPI route owner for the Tinker server."""

    def __init__(self, config):
        self._status = ServerStatus.INITIALIZING
        self._error: Optional[str] = None
        self._engine: Optional[ColocatedBackend] = None
        self._teacher_backend: Optional[TeacherInferenceBackend] = None
        self._gpu_executor = ThreadPoolExecutor(max_workers=1)
        self._op_gate = FifoReadWriteGate(
            max_readers=config["server"].get("max_concurrent_samples", 16),
        )

        self._futures: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._errors: dict[str, str] = {}
        self._request_status: dict[str, RequestStatus] = {}
        self._retrieved_request_status_archive: dict[str, RequestStatus] = {}
        self._shutdown_started = False
        self._shutdown_task: Optional[asyncio.Task] = None
        self._model_to_base_model: dict[str, str] = {}
        self._saved_state_paths: dict[str, str] = {}
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._saved_state_metadata: dict[str, dict[str, Any]] = {}
        self._step_counter = 0
        self._checkpoint_root = config["server"].get("checkpoint_dir", "/tmp/tinker-checkpoints")

        actor_base_model = str(config["server"]["model_name"])
        actor_model_path = str(config["actor_rollout_ref"]["model"]["path"])
        self._model_resource_manager = ModelResourceManager(
            actor_model_identifiers=(actor_base_model, actor_model_path),
        )

        self.config = config

        loop = asyncio.get_running_loop()
        self._init_future: asyncio.Future = loop.run_in_executor(self._gpu_executor, self._init_engine)

    async def _exec(self, op_name: str, fn, *args, **kwargs):
        """Run a blocking engine operation in the serialized GPU executor."""
        self._require_ready()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._gpu_executor, partial(fn, *args, **kwargs))
        except Exception as e:
            logger.exception(f"{op_name} failed: {e}")
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)) from e

    def _get_engine(self) -> ColocatedBackend:
        self._require_ready()
        if self._engine is None:
            raise RuntimeError("Server is initialized without an actor engine")
        return self._engine

    def _begin_shutdown(self) -> bool:
        if getattr(self, "_shutdown_started", False):
            return False
        self._shutdown_started = True
        self._status = ServerStatus.SHUTTING_DOWN
        self._error = "Server shutdown requested"
        return True

    def _require_ready(self) -> None:
        if getattr(self, "_status", None) != ServerStatus.INITIALIZED:
            status = getattr(self, "_status", ServerStatus.ERROR)
            status_value = getattr(status, "value", str(status))
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Server not ready: status={status_value}",
            )

    def _close_unawaited(self, awaitable) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close rejected scheduled coroutine")

    def _metadata_from_create_model(self, req: CreateModelRequest) -> dict[str, Any]:
        return {
            "base_model": req.base_model,
            "is_lora": False,
            "lora_rank": None,
        }

    def _current_model_metadata(self, model_id: str) -> dict[str, Any]:
        metadata = self._model_metadata.get(model_id)
        if metadata is None:
            raise HTTPException(
                HTTPStatus.CONFLICT,
                f"No metadata for model {model_id!r}; create_model must run first",
            )
        return metadata

    def _weights_info_compat_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # Tinker cookbook's public checkpoint reload path currently assumes LoRA
        # metadata before it calls load_weights. The local VeRL backend is still a
        # full-model server; this compatibility shape is only for /weights_info.
        return {
            **metadata,
            "is_lora": True,
            "lora_rank": TINKER_COOKBOOK_COMPAT_LORA_RANK,
            "train_unembed": True,
            "train_mlp": True,
            "train_attn": True,
        }

    async def _run_optim_step(self, req: OptimStepRequest) -> dict:
        try:
            result = await tinker_optim_step(self._get_engine(), req.adam_params)
        except BaseException:
            # An optimizer failure may have partially changed weights. Give that
            # state a fresh identity so it can never compare equal to rollout.
            self._model_resource_manager.actor_updated()
            raise
        self._model_resource_manager.actor_updated()
        return result

    async def _run_save_state(self, req: dict) -> dict:
        result = await save_state(
            self._get_engine(),
            self._checkpoint_root,
            self._saved_state_paths,
            self._saved_state_metadata,
            self._current_model_metadata(req.get("model_id", "")),
            self._step_counter,
            req.get("path"),
        )
        self._model_resource_manager.state_saved(result["path"])
        return result

    def _resolve_load_path(self, uri: str) -> str:
        if not uri or not uri.startswith("tinker://"):
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"Invalid checkpoint path: {uri!r}")
        local_dir = self._saved_state_paths.get(uri) or state_path_to_local(self._checkpoint_root, uri)
        if not Path(local_dir).exists():
            raise HTTPException(HTTPStatus.NOT_FOUND, f"Unknown checkpoint path: {uri}")
        return local_dir

    async def _run_load_state(self, req: dict) -> dict:
        uri = req.get("path", "")
        self._resolve_load_path(uri)
        if self._model_resource_manager.should_skip_state_load(uri):
            logger.info(
                "[model_resource_manager] skipping load path=%s actor_id=%s reason=already_loaded",
                uri,
                self._model_resource_manager.actor_id,
            )
            return {"type": "load_weights", "path": uri}

        try:
            result = await load_state(
                self._get_engine(),
                self._checkpoint_root,
                self._saved_state_paths,
                uri,
                load_optimizer=req.get("optimizer", True),
            )
        except BaseException:
            self._model_resource_manager.state_load_failed()
            raise

        self._model_resource_manager.state_loaded(uri)
        return result

    async def _run_save_weights_for_sampler(self, req: SaveWeightsForSamplerRequest) -> dict:
        named = req.path is not None
        session_id: str | None = None
        if not named and req.sampling_session_seq_id is None:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                "Unnamed save_weights_for_sampler requires sampling_session_seq_id",
            )
        if not named:
            session_id = self._model_resource_manager.session_id_for_model(str(req.model_id))

        try:
            result = await tinker_save_weights_for_sampler(self._get_engine(), named=named)
        except BaseException:
            self._model_resource_manager.rollout_synchronization_failed()
            raise

        rollout_id = self._model_resource_manager.rollout_synchronized()
        if named:
            self._model_resource_manager.sampler_path_saved(result["path"])
            return result

        assert session_id is not None
        sampler_id = self._model_resource_manager.sampler_id(session_id, req.sampling_session_seq_id)
        self._model_resource_manager.register_actor_sampler(
            sampler_id,
            base_model=self._model_resource_manager.actor_base_model,
            actor_id=rollout_id,
        )
        result["sampling_session_id"] = sampler_id
        return result

    def _init_engine(self):
        try:
            self._engine = ColocatedBackend(self.config)
            self._model_resource_manager.configure_resources(
                rollout_enabled=not self._engine.no_rollout_deployment,
                reference_enabled=self._engine.capabilities.use_reference_policy,
            )
            if TeacherInferenceBackend.is_enabled(self.config):
                self._teacher_backend = TeacherInferenceBackend(self.config)
                self._model_resource_manager.configure_teacher_models(self._teacher_backend.sampling_targets)
            if self._shutdown_started:
                logger.info("Tinker server initialization complete after shutdown was requested")
            else:
                self._status = ServerStatus.INITIALIZED
                logger.info("Tinker server initialization complete")
        except Exception as e:
            logger.exception(f"Initialization failed: {e}")
            self._shutdown_current_engine("failed initialization")
            if not self._shutdown_started:
                self._status = ServerStatus.ERROR
                self._error = str(e)

    async def _wait_for_engine_initialization(self):
        init_future = getattr(self, "_init_future", None)
        if init_future is not None and not init_future.done():
            logger.info("Shutdown requested: waiting for engine initialization to finish")
            await asyncio.shield(init_future)

    async def _drain_pending_tasks(self):
        while self._pending:
            tasks = tuple(self._pending.values())
            logger.info("Shutdown requested: waiting for %d scheduled task(s) to finish", len(tasks))
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    async def _shutdown_process(self):
        """Drain local work and mark this deployment ready for supervisor shutdown."""
        await asyncio.sleep(1)
        self._begin_shutdown()
        logger.info("Shutdown requested: draining scheduled Tinker server work before stopping")
        await self._drain_pending_tasks()
        await self._wait_for_engine_initialization()
        await asyncio.to_thread(self._shutdown_current_engine, "shutdown")
        await asyncio.to_thread(self._gpu_executor.shutdown, wait=True, cancel_futures=False)
        self._status = ServerStatus.SHUTDOWN_COMPLETE
        self._error = None
        logger.info("Shutdown requested: cleanup complete; supervisor may stop Tinker server deployment")

    def _shutdown_current_engine(self, reason: str) -> None:
        teacher_backend = getattr(self, "_teacher_backend", None)
        if teacher_backend is not None:
            try:
                teacher_backend.shutdown()
            except Exception:
                logger.exception("%s requested: teacher shutdown failed", reason.capitalize())
            finally:
                if self._teacher_backend is teacher_backend:
                    self._teacher_backend = None

        engine = self._engine
        if engine is None:
            logger.info("%s requested with no initialized engine to shut down", reason.capitalize())
            return

        logger.info(
            "%s requested: shutting down current engine before supervisor lifecycle action", reason.capitalize()
        )
        try:
            engine.shutdown()
        except Exception:
            logger.exception(
                "%s requested: engine.shutdown() failed; proceeding with supervisor lifecycle action",
                reason.capitalize(),
            )
        else:
            logger.info("%s requested: engine.shutdown() completed", reason.capitalize())
        finally:
            if self._engine is engine:
                self._engine = None

    def _stash(self, payload: dict) -> str:
        request_id = uuid.uuid4().hex
        self._request_status[request_id] = RequestStatus.COMPLETED
        self._futures[request_id] = payload
        return request_id

    def _future_envelope(self, request_id: str, model_id: Optional[str] = None) -> dict:
        return {"request_id": request_id, "model_id": model_id}

    def _schedule(self, request_id: str, coro, kind: ScheduledOpKind = ScheduledOpKind.EXCLUSIVE):
        try:
            self._require_ready()
        except HTTPException:
            self._close_unawaited(coro)
            raise

        self._request_status[request_id] = RequestStatus.PENDING
        logger.info(f"[tinker_router] scheduling task rid={request_id}")

        self._compact_request_status_if_needed()

        async def _run():
            try:
                async with self._op_gate.hold(kind):
                    try:
                        self._require_ready()
                    except HTTPException:
                        self._close_unawaited(coro)
                        raise
                    result = await coro
                logger.info(f"[tinker_router] task rid={request_id} succeeded")
                self._futures[request_id] = result
                self._request_status[request_id] = RequestStatus.COMPLETED
            except BaseException as e:
                logger.exception(f"[tinker_router] task rid={request_id} failed: {e!r}")
                self._errors[request_id] = repr(e)
                self._request_status[request_id] = RequestStatus.ERROR
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                self._pending.pop(request_id, None)

        self._pending[request_id] = asyncio.create_task(_run())

    def _compact_request_status_if_needed(self) -> None:
        if len(self._request_status) < MAX_REQUEST_STATUS_ENTRIES:
            return

        retrieved_status = {
            request_id: status
            for request_id, status in self._request_status.items()
            if status is RequestStatus.RETRIEVED
        }
        if not retrieved_status:
            return

        self._retrieved_request_status_archive = retrieved_status
        for request_id in retrieved_status:
            del self._request_status[request_id]

    @app.get("/api/v1/healthz")
    async def healthz(self) -> dict[str, str]:
        """Compatibility health endpoint for existing launch helpers."""
        if self._status == ServerStatus.INITIALIZED:
            return {"status": "ready"}
        if self._status == ServerStatus.ERROR:
            return {"status": self._status.value, "message": self._error or ""}
        return {"status": self._status.value}

    @app.post("/api/v1/shutdown")
    async def shutdown(self, request: Request) -> StatusResponse:
        """Compatibility shutdown endpoint for existing launch helpers."""
        self._begin_shutdown()
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self._shutdown_process())
        return StatusResponse(status="accepted")

    @app.api_route("/api/v1/get_server_capabilities", methods=["GET", "POST"])
    async def get_server_capabilities(self):
        return get_supported_models(self._get_engine(), getattr(self, "_teacher_backend", None))

    @app.post("/api/v1/client/config")
    async def client_config(self):
        self._require_ready()
        return {
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "sample_dispatch_bytes_semaphore_size": 0,
            "inflight_response_bytes_semaphore_size": 0,
            "parallel_fwdbwd_chunks": False,
        }

    @app.post("/api/v1/get_info")
    async def get_info(self, req: GetInfoRequest):
        # Always describe the model/tokenizer that this server actually loaded.
        # A client-provided base_model from create_model is request metadata and
        # must not override the configured model identity returned by get_info.
        model_name = get_configured_model_name(self._get_engine())
        return {
            "type": "get_info",
            "model_data": {"arch": None, "model_name": model_name, "tokenizer_id": model_name},
            "model_id": req.model_id,
            "is_lora": False,
            "lora_rank": None,
            "model_name": model_name,
        }

    @app.post("/api/v1/create_session")
    async def create_session(self, _: dict = None):
        self._require_ready()
        return {
            "type": "create_session",
            "session_id": self._model_resource_manager.create_session(),
            "info_message": None,
            "warning_message": None,
            "error_message": None,
        }

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(self, session_id: str):
        self._require_ready()
        try:
            return {
                "training_run_ids": self._model_resource_manager.model_ids_for_session(session_id),
                "sampler_ids": self._model_resource_manager.valid_sampler_ids_for_session(session_id),
            }
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc

    @app.post("/api/v1/sessions")
    async def list_sessions(self):
        self._require_ready()
        return {"sessions": self._model_resource_manager.session_ids()}

    @app.post("/api/v1/create_model")
    async def create_model(self, req: CreateModelRequest):
        self._require_ready()
        try:
            model_id = self._model_resource_manager.register_model(req.session_id, req.model_seq_id)
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if req.base_model:
            self._model_to_base_model[model_id] = req.base_model
        self._model_metadata[model_id] = self._metadata_from_create_model(req)
        request_id = self._stash({"type": "create_model", "model_id": model_id})
        return self._future_envelope(request_id, model_id=model_id)

    @app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
    async def create_sampling_session(self, req: CreateSamplingSessionRequest):
        self._require_ready()
        try:
            sampler_id = self._model_resource_manager.sampler_id(req.session_id, req.sampling_session_seq_id)
            self._model_resource_manager.resolve_sampler_target(
                sampler_id=sampler_id,
                base_model=req.base_model,
                model_path=req.model_path,
            )
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        return CreateSamplingSessionResponse(
            type="create_sampling_session",
            sampling_session_id=sampler_id,
        )

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(self, sampler_id: str):
        self._require_ready()
        try:
            binding = self._model_resource_manager.get_sampler(sampler_id)
        except UnknownSamplerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        return {
            "sampler_id": sampler_id,
            "base_model": binding.base_model,
            "model_path": binding.model_path,
        }

    @app.post("/api/v1/weights_info")
    async def weights_info(self, req: dict):
        self._require_ready()
        tinker_path = req.get("tinker_path") or req.get("path")
        if not tinker_path:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "weights_info requires tinker_path")

        metadata = load_state_metadata(self._checkpoint_root, self._saved_state_metadata, tinker_path)
        if metadata is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"Unknown checkpoint path: {tinker_path}")
        return self._weights_info_compat_metadata(metadata)

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(self, body: dict = None):
        self._require_ready()
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    async def telemetry(self, body: dict = None):
        self._require_ready()
        return {"status": "accepted"}

    @app.post("/api/v1/forward")
    async def forward(self, req: TinkerForwardRequest):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, tinker_forward(self._get_engine(), req.forward_input.data))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/forward_backward")
    async def forward_backward(self, req: TinkerForwardBackwardRequest):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            tinker_forward_backward(
                self._get_engine(),
                req.forward_backward_input.data,
                req.forward_backward_input.loss_fn,
                req.forward_backward_input.loss_fn_config,
            ),
        )
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/optim_step")
    async def optim_step(self, req: OptimStepRequest):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_optim_step(req))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(self, req: SaveWeightsForSamplerRequest):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_save_weights_for_sampler(req))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights")
    async def save_weights(self, req: dict):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._step_counter += 1
        self._schedule(request_id, self._run_save_state(req))
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    @app.post("/api/v1/load_weights")
    async def load_weights(self, req: dict):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_load_state(req))
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    @app.post("/api/v1/asample")
    async def asample(self, req: SampleRequest):
        self._require_ready()
        try:
            binding = self._model_resource_manager.resolve_sampling_request(
                sampling_session_id=req.sampling_session_id,
                base_model=getattr(req, "base_model", None),
                model_path=getattr(req, "model_path", None),
            )
        except UnknownSamplerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc

        scalar_prompt_logprobs = (
            int(getattr(req, "num_samples", 1)) == 1
            and getattr(getattr(req, "sampling_params", None), "max_tokens", None) == 1
            and getattr(req, "prompt_logprobs", None) is True
            and int(getattr(req, "topk_prompt_logprobs", 0)) == 0
        )

        async def _sample_with_late_binding():
            try:
                resource = self._model_resource_manager.resolve_resource(
                    binding,
                    scalar_prompt_logprobs=scalar_prompt_logprobs,
                )
            except StaleSamplerError as exc:
                raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc

            if resource is SamplingResource.TEACHER:
                if self._teacher_backend is None or binding.teacher_model_path is None:
                    raise RuntimeError("Teacher sampler binding has no initialized teacher backend")
                return await tinker_sample(self._teacher_backend.get_client(binding.teacher_model_path), req)
            if resource is SamplingResource.ROLLOUT:
                return await tinker_sample(self._get_engine(), req)
            if resource is SamplingResource.ACTOR:
                return await tinker_sample_prompt_logprobs(self._get_engine(), req, reference=False)
            if resource is SamplingResource.REFERENCE:
                return await tinker_sample_prompt_logprobs(self._get_engine(), req, reference=True)
            raise RuntimeError(f"Unsupported sampling resource: {resource!r}")

        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            _sample_with_late_binding(),
            ScheduledOpKind.SAMPLE,
        )
        return self._future_envelope(request_id, model_id=None)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(self, req: FutureRetrieveRequest):
        self._require_ready()
        request_id = req.request_id
        task = self._pending.get(request_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except TimeoutError:
                return {"type": "try_again", "message": "future not yet completed"}

            # we are seeing that sometimes we double request for the same id, leading to error
            return {"type": "try_again", "message": "future completed; retry to retrieve result"}

        if request_id in self._futures:
            payload = self._futures.pop(request_id)
            self._request_status[request_id] = RequestStatus.RETRIEVED
            return payload

        if request_id in self._errors:
            error = self._errors.pop(request_id)
            self._request_status[request_id] = RequestStatus.RETRIEVED
            return {"error": error, "category": "Application"}

        if request_id in self._request_status or request_id in self._retrieved_request_status_archive:
            logger.warning(
                f"[tinker_router] retrieve_future for id={request_id}; but future likely already retrieved"
                f"pending={len(self._pending)} futures={len(self._futures)} errors={len(self._errors)}"
            )
            return {
                "error": "future id result already retrieved",
                "category": "Application",
                "request_id": request_id,
                "seen_before": True,
                "seen_status": "seen_before",
                "pending_count": len(self._pending),
                "ready_count": len(self._futures),
                "error_count": len(self._errors),
            }
        else:
            # we have never seen this request id so we will have to have raise
            logger.warning(
                f"[tinker_router] retrieve_future for unknown rid={request_id}; "
                f"pending={len(self._pending)} "
                f"futures={len(self._futures)} errors={len(self._errors)}"
            )
            return {
                "error": "future id was never seen by this server",
                "category": "Application",
                "request_id": request_id,
                "seen_before": False,
                "seen_status": "never_seen",
                "pending_count": len(self._pending),
                "ready_count": len(self._futures),
                "error_count": len(self._errors),
            }
