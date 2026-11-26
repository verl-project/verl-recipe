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
"""HTTP control-plane client for ``dynamo.sglang`` workers.

Unlike the vLLM path (which needs ``_dynamo_vllm_with_control.py``, a verl-private
ZMQ REP sidecar, because bare ``dynamo.vllm`` exposes no hook onto its AsyncLLM),
``dynamo.sglang`` already registers a full RL control plane of its own:

``components/src/dynamo/sglang/request_handlers/handler_base.py::register_engine_routes``
registers, unconditionally::

    control/start_profile              control/stop_profile
    pause_generation                  continue_generation
    release_memory_occupation         resume_memory_occupation
    control/update_weights_from_disk   control/update_weights_from_tensor
    control/update_weights_from_distributed
    control/update_weights_from_ipc    control/update_weight_version

The recipe also explicitly registers this tokenizer-manager route::

    flush_cache    # --engine-route flush_cache:tm

Those land on the worker's system-status server, which mounts ``/engine/{*path}``
(``lib/runtime/src/system_status_server.rs``) on ``$DYN_SYSTEM_PORT``. So the whole
control plane is plain HTTP POST against
``http://<worker-host>:<DYN_SYSTEM_PORT>/engine/<registered-key>``.

Two consequences drive the design here:

1. ``DYN_SYSTEM_PORT`` is **mandatory** for the sglang backend (it is merely a
   metrics nicety for vLLM). Dynamo's Rust runtime parses it as i16, so the port
   must stay below 32768 — see ``_allocate_stable_node_port`` on the actor side.
2. Dynamo PR #13951 removed ``call_tokenizer_manager`` and separated discovery
   pause/resume from memory release/restore. Pause before releasing memory and
   continue only after restoring it; readiness uses the configured native flush.

The native lifecycle is validated with Dynamo 8c5a73723109 and SGLang 0.5.19.
Older Dynamo workers exposing only the legacy control routes are not supported
by this version of the client.

**Sidecar mode** (``engine_kwargs.dynamo.sglang.launch_mode=sidecar``). The engine is
a stock ``sglang.launch_server`` and ``dynamo.sglang.sidecar`` registers it with
Dynamo over SGLang's native gRPC API. The sidecar publishes no ``/engine/*`` routes,
so the control plane is SGLang's own HTTP server instead
(``python/sglang/srt/entrypoints/http_server.py``): the same operations, served at
``http://<engine-host>:<sglang --port>/<name>`` with no ``/engine/`` or ``control/``
prefix. Two native responses are not JSON objects: ``/flush_cache`` answers plain
text and ``/release_memory_occupation`` / ``/resume_memory_occupation`` answer
``null``; failures there are signalled only by the HTTP status. ``api=native``
selects that dialect.

**Robustness note.** A ``serialized_named_tensors`` payload that fails to
deserialize does not come back as an error — it kills the worker process outright
(observed as ``RemoteDisconnected``, then connection-refused for everything after;
M0c, job 16215105). So a malformed weight sync is an engine restart, not a retryable
failure, and the caller cannot distinguish "bad payload" from "worker crashed for an
unrelated reason". Keep the wire format exactly as SGLang specifies it.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# Registered engine-route keys. Mirrors handler_base.register_engine_routes;
# kept as constants so a dynamo-side rename fails loudly in one place.
ROUTE_PAUSE_GENERATION = "pause_generation"
ROUTE_CONTINUE_GENERATION = "continue_generation"
ROUTE_RELEASE_MEMORY = "release_memory_occupation"
ROUTE_RESUME_MEMORY = "resume_memory_occupation"
ROUTE_UPDATE_WEIGHTS_FROM_TENSOR = "control/update_weights_from_tensor"
ROUTE_UPDATE_WEIGHTS_FROM_IPC = "control/update_weights_from_ipc"
ROUTE_UPDATE_WEIGHT_VERSION = "control/update_weight_version"
ROUTE_START_PROFILE = "control/start_profile"
ROUTE_STOP_PROFILE = "control/stop_profile"
ROUTE_CALL_TOKENIZER_MANAGER = "call_tokenizer_manager"
ROUTE_FLUSH_CACHE = "flush_cache"

# Control-plane dialects. ``dynamo``: the dynamo.sglang worker's /engine/* routes.
# ``native``: SGLang's own HTTP server, used when a sidecar fronts the engine.
API_DYNAMO = "dynamo"
API_NATIVE = "native"
_APIS = (API_DYNAMO, API_NATIVE)

# Launch modes for engine_kwargs.dynamo.sglang.launch_mode.
LAUNCH_MODE_WORKER = "worker"
LAUNCH_MODE_SIDECAR = "sidecar"
_LAUNCH_MODES = (LAUNCH_MODE_WORKER, LAUNCH_MODE_SIDECAR)


def sglang_launch_mode(sglang_cfg: Optional[dict]) -> str:
    """Validated ``engine_kwargs.dynamo.sglang.launch_mode`` (default ``worker``).

    Shared by the node actor (which launches the processes) and the trainer-side
    adapter (which builds its own control client), so the two cannot disagree on
    which control dialect the engine speaks.
    """
    mode = str((sglang_cfg or {}).get("launch_mode") or LAUNCH_MODE_WORKER).strip().lower()
    if mode not in _LAUNCH_MODES:
        raise ValueError(
            f"rollout.engine_kwargs.dynamo.sglang.launch_mode must be one of {_LAUNCH_MODES}, got {mode!r}"
        )
    return mode


def control_api_for_launch_mode(mode: str) -> str:
    return API_NATIVE if mode == LAUNCH_MODE_SIDECAR else API_DYNAMO


class DynamoSGLangControlError(RuntimeError):
    """Raised when a control route returns non-200 or a ``status: error`` body."""


class DynamoSGLangControlClient:
    """Async HTTP client for one ``dynamo.sglang`` worker's ``/engine/*`` routes.

    One instance per DP shard. The node-level ``DynamoHttpServer`` actor holds a
    list of them (one per local shard) and fans out; a trainer-side ServerAdapter
    holds exactly the one that matches its own shard, so CUDA-IPC handles are only
    ever posted to the engine that shares its GPUs.
    """

    def __init__(self, base_url: str, timeout_s: float = 600.0, api: str = API_DYNAMO):
        """
        Args:
            base_url: ``http://host:port`` of the worker's system-status server
                (``api=dynamo``) or of SGLang's own HTTP server (``api=native``).
            timeout_s: default per-request timeout.
            api: control dialect, ``dynamo`` or ``native`` (see module docstring).
        """
        if api not in _APIS:
            raise ValueError(f"api must be one of {_APIS}, got {api!r}")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.api = api
        self._session = None

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #

    async def _get_session(self):
        if self._session is not None and not self._session.closed:
            return self._session
        import aiohttp

        self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def route_url(self, route: str) -> str:
        if self.api == API_NATIVE:
            if route == ROUTE_CALL_TOKENIZER_MANAGER:
                raise DynamoSGLangControlError(
                    "call_tokenizer_manager is a dynamo.sglang worker route; SGLang's native "
                    "HTTP server (sidecar mode) has no generic tokenizer-manager passthrough"
                )
            # dynamo.sglang namespaces some keys under control/; the native names are bare.
            return f"{self.base_url}/{route.removeprefix('control/')}"
        return f"{self.base_url}/engine/{route}"

    async def post(
        self,
        route: str,
        body: Optional[dict] = None,
        timeout_s: Optional[float] = None,
        raise_on_error_status: bool = True,
    ) -> dict:
        """POST one control request and return the decoded JSON body.

        ``/engine/{*path}`` is mounted with axum's ``any(...)``, so the verb is
        not load-bearing; POST is used for every route for uniformity. Every
        native SGLang control route used here accepts POST as well.

        On the native dialect a 200 whose body is not a JSON object (plain-text
        ``/flush_cache``, ``null`` from memory occupation) is a success: SGLang
        reports those routes' failures only through the status code. It comes
        back as ``{"status": "ok", "raw": <body>}``.
        """
        import aiohttp

        session = await self._get_session()
        url = self.route_url(route)
        timeout = aiohttp.ClientTimeout(total=timeout_s if timeout_s is not None else self.timeout_s)
        async with session.post(url, json=body or {}, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise DynamoSGLangControlError(f"{url} -> HTTP {resp.status}: {text[:2000]!r}")
            try:
                data = await resp.json(content_type=None)
            except Exception as e:
                if self.api == API_NATIVE:
                    return {"status": "ok", "raw": text}
                raise DynamoSGLangControlError(f"{url} -> non-JSON body: {text[:2000]!r}") from e
            if data is None and self.api == API_NATIVE:
                return {"status": "ok", "raw": None}

        if raise_on_error_status and isinstance(data, dict):
            # Handlers answer either {"status": "ok"|"error", ...} or
            # {"success": bool, "message": str}; treat both as failures.
            if data.get("status") == "error":
                raise DynamoSGLangControlError(f"{url} -> {data.get('message')!r}")
            if data.get("success") is False:
                raise DynamoSGLangControlError(f"{url} -> {data.get('message')!r}")
        return data

    # ------------------------------------------------------------------ #
    # memory occupation (sleep / wake)
    # ------------------------------------------------------------------ #

    async def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict:
        """Release GPU memory for ``tags`` (``["kv_cache"]`` / ``["kv_cache","weights"]``).

        Requires the worker to have been started with ``--enable-memory-saver``;
        without it SGLang's torch_memory_saver is inactive and this is a no-op at
        best. Dynamo PR #13951 separates native pause and memory control. Pause
        first to drain requests and remove the worker from discovery, then free
        memory. A failed release leaves the worker paused.
        """
        body: dict[str, Any] = {}
        if tags is not None:
            body["tags"] = list(tags)
        await self.pause_generation(mode="abort")
        return await self.post(ROUTE_RELEASE_MEMORY, body)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict:
        """Restore every released tag, then resume generation and discovery.

        The node actor widens resumes to its complete per-shard released-tag set.
        Keep that contract: generation must remain paused until all memory is live.
        """
        body: dict[str, Any] = {}
        if tags is not None:
            body["tags"] = list(tags)
        result = await self.post(ROUTE_RESUME_MEMORY, body)
        await self.continue_generation()
        return result

    # ------------------------------------------------------------------ #
    # weight sync
    # ------------------------------------------------------------------ #

    async def update_weights_from_tensor(self, req, timeout_s: Optional[float] = None) -> dict:
        """POST one ``UpdateWeightsFromTensorReqInput`` worth of CUDA-IPC handles.

        ``req.serialized_named_tensors`` is ``list[bytes]`` (one entry per SGLang TP
        rank, each a ``MultiprocessingSerializer`` blob of CUDA-IPC handles). JSON has
        no bytes type, so it goes on the wire base64-encoded.

        Base64 is not a workaround here, it is SGLang's documented contract for this
        field — ``UpdateWeightsFromTensorReqInput.serialized_named_tensors`` is typed
        ``List[Union[str, bytes]]`` and ``MultiprocessingSerializer.deserialize``
        b64-decodes any ``str`` it is handed. Dynamo's engine route passes the JSON
        body through untouched, which is exactly right. Verified end-to-end against an
        unmodified dynamo 1.3.0 + sglang 0.5.14 (M0c, job 16215105).

        Do NOT "fix" this by decoding server-side: a payload that fails to
        deserialize takes the whole worker process down (see the class docstring),
        so the encoding must match what SGLang expects, not what looks symmetric.
        """
        body = {
            "serialized_named_tensors": [base64.b64encode(b).decode("utf-8") for b in req.serialized_named_tensors],
            "load_format": req.load_format,
            "flush_cache": req.flush_cache,
        }
        return await self.post(ROUTE_UPDATE_WEIGHTS_FROM_TENSOR, body, timeout_s=timeout_s)

    async def update_weight_version(self, new_version: str, abort_all_requests: bool = False) -> dict:
        return await self.post(
            ROUTE_UPDATE_WEIGHT_VERSION,
            {"new_version": str(new_version), "abort_all_requests": bool(abort_all_requests)},
        )

    # ------------------------------------------------------------------ #
    # legacy tokenizer_manager passthrough and explicit native routes
    # ------------------------------------------------------------------ #

    async def call_tokenizer_manager(
        self,
        method: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> dict:
        """Legacy-only passthrough, unavailable after Dynamo PR #13951.

        Retained for callers of the old optional weight-readback diagnostic.
        New control operations must use explicitly registered engine routes.
        The standard readiness, refit and generation lifecycle do not call it.
        """
        return await self.post(
            ROUTE_CALL_TOKENIZER_MANAGER,
            {"method": method, "args": args or [], "kwargs": kwargs or {}},
            timeout_s=timeout_s,
        )

    async def get_weights_by_name(self, name: str, truncate_size: int, timeout_s: Optional[float] = None) -> Any:
        """Read back a truncated parameter (the weight-sync verification probe).

        Native dialect: SGLang's ``/get_weights_by_name`` route. Dynamo dialect:
        the legacy ``call_tokenizer_manager`` passthrough, which needs the
        ``{"io_struct.X": {...}}`` wrapper because tokenizer_manager takes a request
        object, not kwargs. That route was removed by Dynamo PR #13951, so on
        current workers this raises and the caller logs "verification unavailable".
        """
        if self.api == API_NATIVE:
            return await self.post(
                "get_weights_by_name",
                {"name": name, "truncate_size": int(truncate_size)},
                timeout_s=timeout_s,
                raise_on_error_status=False,
            )
        return await self.call_tokenizer_manager(
            "get_weights_by_name",
            args=[{"io_struct.GetWeightsByNameReqInput": {"name": name, "truncate_size": int(truncate_size)}}],
            timeout_s=timeout_s,
        )

    async def flush_cache(self, timeout_s: Optional[float] = None) -> dict:
        """Flush the explicitly registered ``--engine-route flush_cache:tm``.

        PR13951 removed the generic call_tokenizer_manager route. The same
        native flush is required before weight updates and for readiness. On the
        native dialect this is SGLang's own ``/flush_cache``, which answers 400 when
        requests are still running.
        """
        return await self.post(ROUTE_FLUSH_CACHE, {}, timeout_s=timeout_s)

    async def abort_request(self, rid: str = "", abort_all: bool = False) -> dict:
        """DO NOT use for abort-all: tokenizer_manager.abort_request is a SYNC
        method returning None, and dynamo's call_tokenizer_manager handler
        awaits the result unconditionally -> HTTP 500 "object NoneType can't
        be used in 'await' expression" (observed on ai-dynamo 1.3.0.post1,
        first sglang colocate_async run). Kept for per-rid aborts should a
        future dynamo build stop awaiting sync results; the working abort-all
        path is pause_generation(mode="abort") below.
        """
        return await self.call_tokenizer_manager("abort_request", kwargs={"rid": rid, "abort_all": bool(abort_all)})

    async def pause_generation(self, mode: str = "abort") -> dict:
        """Pause native SGLang generation and synchronize Dynamo discovery."""
        return await self.post(ROUTE_PAUSE_GENERATION, {"mode": mode})

    async def continue_generation(self) -> dict:
        """Counterpart to pause_generation; reopens engine intake."""
        return await self.post(ROUTE_CONTINUE_GENERATION, {})

    # ------------------------------------------------------------------ #
    # profiling
    # ------------------------------------------------------------------ #

    async def start_profile(self, **kwargs) -> dict:
        return await self.post(ROUTE_START_PROFILE, dict(kwargs))

    async def stop_profile(self) -> dict:
        return await self.post(ROUTE_STOP_PROFILE, {})

    # ------------------------------------------------------------------ #
    # health
    # ------------------------------------------------------------------ #

    async def wait_ready(self, timeout_s: float = 1800.0, poll_s: float = 5.0) -> bool:
        """Block until this worker's ``/engine/*`` routes are actually serving.

        The obvious readiness check — "does the port answer" — is wrong twice over,
        and both failure modes were observed on the first real run (M0b, job 16213723):

        1. The system-status server binds ``DYN_SYSTEM_PORT`` within ~20s of process
           start, but the Python engine routes are only registered once the model has
           finished loading. In between, every ``/engine/*`` path 404s while the port
           happily accepts connections. Probing too early "succeeds" against a server
           that cannot do anything.
        2. A 404 is a perfectly good HTTP response, so any check that only asks
           "did I get a reply" passes during that window.

        So gate on a **200** from the configured ``flush_cache`` route: it proves the engine
        routes are registered, the tokenizer_manager is alive, and ``--engine-route flush_cache:tm``
        actually took — all three of which the weight-sync path needs.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        last: str = "no attempt made"
        while loop.time() < deadline:
            try:
                await self.flush_cache(timeout_s=15.0)
                return True
            except Exception as e:  # noqa: BLE001 - 404/transport both mean not-ready-yet
                last = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(poll_s)
        if self.api == API_NATIVE:
            hint = (
                "The SGLang HTTP server never answered /flush_cache; check the engine log "
                "(sglang.launch_server) for a model-load failure."
            )
        else:
            hint = (
                "If this is a 404 the worker is up but never finished loading the model; check "
                "--engine-route flush_cache:tm and the native tokenizer-manager result."
            )
        raise DynamoSGLangControlError(
            f"sglang control routes at {self.base_url} (api={self.api}) not serving after "
            f"{timeout_s}s. Last: {last}. {hint}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DynamoSGLangControlClient({self.base_url!r}, api={self.api!r})"


__all__ = [
    "API_DYNAMO",
    "API_NATIVE",
    "LAUNCH_MODE_SIDECAR",
    "LAUNCH_MODE_WORKER",
    "DynamoSGLangControlClient",
    "DynamoSGLangControlError",
    "control_api_for_launch_mode",
    "sglang_launch_mode",
]
