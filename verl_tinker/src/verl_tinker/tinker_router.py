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
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from ray import serve
from tinker.types import (
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    ForwardBackwardRequest,
    ForwardRequest,
    FutureRetrieveRequest,
    GetInfoRequest,
    OptimStepRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
)

from .backends.colocated import ColocatedBackend
from .schemas import StatusResponse
from .tinker_ops import (
    GLOBAL_MODEL_ID,
    GLOBAL_SAMPLING_SESSION_ID,
    GLOBAL_SESSION_ID,
    get_configured_model_name,
    get_supported_models,
    load_state,
    load_state_metadata,
    save_state,
)
from .tinker_ops import (
    forward as tinker_forward,
)
from .tinker_ops import (
    forward_backward as tinker_forward_backward,
)
from .tinker_ops import (
    optim_step as tinker_optim_step,
)
from .tinker_ops import (
    sample as tinker_sample,
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
        self._sampling_to_model_path: dict[str, str] = {}
        self._sampling_to_base_model: dict[str, str] = {}
        self._model_to_base_model: dict[str, str] = {}
        self._saved_state_paths: dict[str, str] = {}
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._saved_state_metadata: dict[str, dict[str, Any]] = {}
        self._step_counter = 0
        self._checkpoint_root = config["server"].get("checkpoint_dir", "/tmp/tinker-checkpoints")

        self.config = config

        loop = asyncio.get_running_loop()
        self._init_future: asyncio.Future = loop.run_in_executor(self._gpu_executor, self._init_engine)

    async def _exec(self, op_name: str, fn, *args, **kwargs):
        """Run a blocking engine operation in the serialized GPU executor."""
        self._reject_if_shutting_down()
        if self._status != ServerStatus.INITIALIZED:
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "Server not ready")
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._gpu_executor, partial(fn, *args, **kwargs))
        except Exception as e:
            logger.exception(f"{op_name} failed: {e}")
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)) from e

    def _get_engine(self) -> ColocatedBackend:
        self._reject_if_shutting_down()
        if self._status != ServerStatus.INITIALIZED or self._engine is None:
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "Server not ready")
        return self._engine

    def _begin_shutdown(self) -> bool:
        if getattr(self, "_shutdown_started", False):
            return False
        self._shutdown_started = True
        self._status = ServerStatus.SHUTTING_DOWN
        self._error = "Server shutdown requested"
        return True

    def _is_shutting_down(self) -> bool:
        return getattr(self, "_shutdown_started", False) or getattr(self, "_status", None) == ServerStatus.SHUTTING_DOWN

    def _reject_if_shutting_down(self) -> None:
        if self._is_shutting_down():
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "Server shutting down")

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

    def _current_model_metadata(self) -> dict[str, Any]:
        metadata = self._model_metadata.get(GLOBAL_MODEL_ID)
        if metadata is None:
            raise HTTPException(HTTPStatus.CONFLICT, "No model metadata available; create_model must run first")
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

    def _init_engine(self):
        try:
            self._engine = ColocatedBackend(self.config)
            if self._shutdown_started:
                logger.info("Tinker server initialization complete after shutdown was requested")
            else:
                self._status = ServerStatus.INITIALIZED
                logger.info("Tinker server initialization complete")
        except Exception as e:
            logger.exception(f"Initialization failed: {e}")
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
        if self._is_shutting_down():
            self._close_unawaited(coro)
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "Server shutting down")

        self._request_status[request_id] = RequestStatus.PENDING
        logger.info(f"[tinker_router] scheduling task rid={request_id}")

        self._compact_request_status_if_needed()

        async def _run():
            try:
                async with self._op_gate.hold(kind):
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
        return get_supported_models(self._get_engine())

    @app.post("/api/v1/client/config")
    async def client_config(self):
        self._reject_if_shutting_down()
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
            "model_id": GLOBAL_MODEL_ID,
            "is_lora": False,
            "lora_rank": None,
            "model_name": model_name,
        }

    @app.post("/api/v1/create_session")
    async def create_session(self, _: dict = None):
        self._reject_if_shutting_down()
        # dummy endpoint mostly for compatibility
        return {
            "type": "create_session",
            "session_id": GLOBAL_SESSION_ID,
            "info_message": None,
            "warning_message": None,
            "error_message": None,
        }

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(self, session_id: str):
        self._reject_if_shutting_down()
        return {"training_run_ids": [GLOBAL_MODEL_ID], "sampler_ids": [GLOBAL_SAMPLING_SESSION_ID]}

    @app.post("/api/v1/sessions")
    async def list_sessions(self):
        self._reject_if_shutting_down()
        return {"sessions": [GLOBAL_SESSION_ID]}

    @app.post("/api/v1/create_model")
    async def create_model(self, req: CreateModelRequest):
        self._reject_if_shutting_down()
        if req.base_model:
            self._model_to_base_model[GLOBAL_MODEL_ID] = req.base_model
        self._model_metadata[GLOBAL_MODEL_ID] = self._metadata_from_create_model(req)
        request_id = self._stash({"type": "create_model", "model_id": GLOBAL_MODEL_ID})
        return self._future_envelope(request_id, model_id=GLOBAL_MODEL_ID)

    @app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
    async def create_sampling_session(self, req: CreateSamplingSessionRequest):
        self._reject_if_shutting_down()
        if req.model_path:
            self._sampling_to_model_path[GLOBAL_SAMPLING_SESSION_ID] = req.model_path
        if req.base_model:
            self._sampling_to_base_model[GLOBAL_SAMPLING_SESSION_ID] = req.base_model
        return CreateSamplingSessionResponse(
            type="create_sampling_session",
            sampling_session_id=GLOBAL_SAMPLING_SESSION_ID,
        )

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(self, sampler_id: str):
        engine = self._get_engine()
        base_model = self._sampling_to_base_model.get(sampler_id, get_configured_model_name(engine))
        return {
            "sampler_id": sampler_id,
            "base_model": base_model,
            "model_path": self._sampling_to_model_path.get(sampler_id, str(engine.config.actor_rollout_ref.model.path)),
        }

    @app.post("/api/v1/weights_info")
    async def weights_info(self, req: dict):
        self._reject_if_shutting_down()
        tinker_path = req.get("tinker_path") or req.get("path")
        if not tinker_path:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "weights_info requires tinker_path")

        metadata = load_state_metadata(self._checkpoint_root, self._saved_state_metadata, tinker_path)
        if metadata is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"Unknown checkpoint path: {tinker_path}")
        return self._weights_info_compat_metadata(metadata)

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(self, body: dict = None):
        self._reject_if_shutting_down()
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    async def telemetry(self, body: dict = None):
        self._reject_if_shutting_down()
        return {"status": "accepted"}

    @app.post("/api/v1/forward")
    async def forward(self, req: ForwardRequest):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, tinker_forward(self._get_engine(), req.forward_input.data))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/forward_backward")
    async def forward_backward(self, req: ForwardBackwardRequest):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            tinker_forward_backward(
                self._get_engine(),
                req.forward_backward_input.data,
                req.forward_backward_input.loss_fn,
            ),
        )
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/optim_step")
    async def optim_step(self, req: OptimStepRequest):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, tinker_optim_step(self._get_engine(), req.adam_params))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(self, req: SaveWeightsForSamplerRequest):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, tinker_save_weights_for_sampler(self._get_engine(), named=req.path is not None))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights")
    async def save_weights(self, req: dict):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._step_counter += 1
        self._schedule(
            request_id,
            save_state(
                self._get_engine(),
                self._checkpoint_root,
                self._saved_state_paths,
                self._saved_state_metadata,
                self._current_model_metadata(),
                self._step_counter,
                req.get("path"),
            ),
        )
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    @app.post("/api/v1/load_weights")
    async def load_weights(self, req: dict):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            load_state(
                self._get_engine(),
                self._checkpoint_root,
                self._saved_state_paths,
                req.get("path", ""),
                load_optimizer=req.get("optimizer", True),
            ),
        )
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    @app.post("/api/v1/asample")
    async def asample(self, req: SampleRequest):
        self._reject_if_shutting_down()
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            tinker_sample(self._get_engine(), req),
            ScheduledOpKind.SAMPLE,
        )
        return self._future_envelope(request_id, model_id=None)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(self, req: FutureRetrieveRequest):
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
