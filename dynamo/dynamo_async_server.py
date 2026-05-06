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
"""DynamoHttpServer + DynamoReplica (recipe-side).

m2: same Ray actor name prefix delta as vLLM (``dynamo_*``); m2 e2e
    pipeline is functionally identical to vLLM colocate.

m3 (this file, gated by VERL_DYNAMO_ENABLE_RUNTIME=1): after vLLM
    AsyncLLM is up, additionally:
      * create_runtime(discovery_backend="file") same-process
      * register_model(ModelInput.Tokens, ...) so dynamo-side
        components (router, KVBM, frontend in m4) can discover this
        worker via /tmp/dynamo file discovery
      * KvEventPublisher tap on vLLM's KV cache events (when prefix
        caching + kv events are enabled) so the KV router can compute
        overlap scores

m3 does NOT start a dynamo Frontend; verl's generate path still goes
through the existing vLLM AsyncLLM (server_handle.<m>.remote()). m4
flips generate to the Frontend HTTP API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer as _VllmHttpServer,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMReplica as _VllmReplica,
)

logger = logging.getLogger(__name__)

# Opt-in. Default off so existing m2 deployments on vllm017_latest.sqsh
# (which lacks the dynamo SDK) keep working. Set to "1" only after
# switching to dynamo_vllm_*.sqsh, which ships dynamo + vllm together.
_DYNAMO_RUNTIME_ENV = "VERL_DYNAMO_ENABLE_RUNTIME"
_DYNAMO_FRONTEND_ENV = "VERL_DYNAMO_ENABLE_FRONTEND"
# m5: route generate calls through dynamo runtime's KV-aware client.
# Implies m3 (need register_model + serve_endpoint to make peer replicas reachable
# via the dynamo runtime's TCP request plane). Independent of m4 (frontend HTTP):
# m5 uses endpoint.client(RouterMode.KV) directly, bypasses the OpenAI HTTP layer.
_DYNAMO_ROUTER_ENV = "VERL_DYNAMO_ROUTE_THROUGH_ROUTER"

# Where DistributedRuntime registers worker endpoints when discovery is
# "file" (matches dynamo's default; frontend in m4 reads from the same path
# unless DYN_FILE_KV is overridden in its env).
_DEFAULT_FILE_KV = "/tmp/dynamo/file_kv"

# Frontend health-check budget: dynamo Rust runtime startup + axum bind +
# file_kv discovery sync.
_FRONTEND_READY_TIMEOUT_S = 120
_FRONTEND_READY_POLL_S = 1.0


def _dynamo_runtime_enabled() -> bool:
    return os.environ.get(_DYNAMO_RUNTIME_ENV, "0") == "1"


def _dynamo_frontend_enabled() -> bool:
    # m4 implies m3 (frontend has nothing to discover without registered workers).
    return (
        _dynamo_runtime_enabled()
        and os.environ.get(_DYNAMO_FRONTEND_ENV, "0") == "1"
    )


def _dynamo_router_enabled() -> bool:
    # m5 implies m3 (need handler+register so peer replicas dispatchable).
    return (
        _dynamo_runtime_enabled()
        and os.environ.get(_DYNAMO_ROUTER_ENV, "0") == "1"
    )


class DynamoHttpServer(_VllmHttpServer):
    """Per-node Ray actor.

    m2: identical to vLLMHttpServer.
    m3: after super().run_server() finishes, optionally bring up a
        same-process dynamo DistributedRuntime and register this worker.
    """

    # m3 state, populated only when the runtime is enabled.
    _dynamo_runtime: Optional[Any] = None
    _dynamo_kv_publishers: Optional[list] = None
    _dynamo_generate_endpoint: Optional[Any] = None
    # m4 state, populated only when the frontend is enabled (master replica only).
    _dynamo_frontend_proc: Optional[subprocess.Popen] = None
    _vllm_server_port: Optional[int] = None  # original vLLM uvicorn port, kept for diagnostics
    # m5 state, populated only when route-through-router is enabled.
    _dynamo_endpoint_serve_task: Optional[asyncio.Task] = None
    _dynamo_kv_client: Optional[Any] = None  # dynamo Client with RouterMode.KV
    _dynamo_served_model_name: Optional[str] = None  # cached for handler use

    async def run_server(self, args):  # noqa: D401 — match parent signature
        await super().run_server(args)
        if not _dynamo_runtime_enabled():
            return
        try:
            await self._setup_dynamo_runtime(args)
        except Exception:
            # Use warning so the message survives Ray's default actor log
            # filter (info gets dropped). m3/m4 must never break m2 generate;
            # on failure the worker keeps serving via the vLLM OpenAI server
            # that super() already started.
            logger.warning(
                "[m3/m4] dynamo runtime setup failed for replica=%s node=%s; "
                "continuing on vLLM-native path. Traceback follows.",
                self.replica_rank,
                self.node_rank,
                exc_info=True,
            )
            self._dynamo_runtime = None

    async def _setup_dynamo_runtime(self, args):
        """Bring up an in-process DistributedRuntime + register_model.

        Pre: ``self.engine`` is a live ``vllm.AsyncLLM`` (set by parent
        ``run_server``). ``self.replica_rank`` and ``self.node_rank`` are
        set. Every replica registers its own ``worker_{r}_{n}/generate``
        endpoint so the Frontend's KV router has the full set of workers
        to route between. The master replica then launches the Frontend
        (m4); non-master replicas redirect their advertised server
        address to the master Frontend (see ``_connect_to_master_frontend``).
        """
        # Imports are local so the module loads in environments that
        # don't ship the dynamo SDK (e.g. m2 baseline on vllm017).
        from dynamo.common.utils.runtime import create_runtime
        from dynamo.llm import (
            KvEventPublisher,
            ModelInput,
            ModelRuntimeConfig,
            ModelType,
            register_model,
        )

        os.makedirs(_DEFAULT_FILE_KV, exist_ok=True)
        os.environ.setdefault("DYN_DISCOVERY_BACKEND", "file")
        os.environ.setdefault("DYN_FILE_KV", _DEFAULT_FILE_KV)

        runtime, _loop = create_runtime(
            discovery_backend=os.environ["DYN_DISCOVERY_BACKEND"],
            request_plane="tcp",
            event_plane="zmq",
            use_kv_events=True,
        )
        self._dynamo_runtime = runtime

        # Endpoint naming differs by mode:
        # * m3 / m4: per-replica unique paths so each appears as a distinct
        #   entry in file_kv. Frontend (m4) lists all of them; only registry
        #   metadata is exchanged.
        # * m5 (router): all replicas share ONE endpoint name and become
        #   distinct *instances* of it. RouterMode.KV in the client picks
        #   among instances using KV-overlap heuristics. Cross-replica
        #   dispatch only works when this collapse happens.
        if _dynamo_router_enabled():
            endpoint_path = "verl_dynamo.shared.generate"
        else:
            endpoint_path = (
                f"verl_dynamo.worker_{self.replica_rank}_{self.node_rank}.generate"
            )
        gen_endpoint = runtime.endpoint(endpoint_path)
        self._dynamo_generate_endpoint = gen_endpoint

        # Pull the runtime numbers vLLM computed at engine init.
        cache_cfg = self.engine.vllm_config.cache_config
        sched_cfg = self.engine.vllm_config.scheduler_config
        rt_cfg = ModelRuntimeConfig()
        rt_cfg.total_kv_blocks = cache_cfg.num_gpu_blocks or 0
        rt_cfg.max_num_seqs = sched_cfg.max_num_seqs
        rt_cfg.max_num_batched_tokens = sched_cfg.max_num_batched_tokens

        served_model_name = getattr(args, "served_model_name", None) or args.model

        # Plan B: every replica registers its own endpoint so the Frontend
        # discovers all 8 workers (one per replica) and KV router has real
        # routing space. Same logical model name, distinct endpoint paths
        # (worker_{replica_rank}_{node_rank}/generate); dynamo's
        # discovery joins them under one model card.
        await register_model(
            ModelInput.Tokens,  # KV router requirement
            ModelType.Chat | ModelType.Completions,
            gen_endpoint,
            args.model,
            served_model_name,
            context_length=self.engine.vllm_config.model_config.max_model_len,
            kv_cache_block_size=cache_cfg.block_size,
            runtime_config=rt_cfg,
        )
        logger.warning(
            "[m3] replica=%s registered model=%s endpoint=%s blocks=%s",
            self.replica_rank,
            served_model_name,
            endpoint_path,
            rt_cfg.total_kv_blocks,
        )
        self._dynamo_served_model_name = served_model_name

        # m5: serve handler on the endpoint + create KV-routed client.
        # Without this, register_model only writes metadata — frontends/clients
        # have nowhere to dispatch when they pick this worker. With it, peer
        # replicas (and frontends) can actually invoke vLLM AsyncLLM.generate
        # on this replica via dynamo's TCP request plane.
        if _dynamo_router_enabled():
            from dynamo._core import RouterMode

            # serve_endpoint returns a PyO3 Future (not a coroutine), so we
            # wrap it in a Python coroutine before create_task — without
            # the wrapper asyncio.create_task raises TypeError.
            async def _serve_loop():
                try:
                    await gen_endpoint.serve_endpoint(self._dynamo_generate_handler)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "[m5] serve_endpoint raised on replica=%s; this replica "
                        "stops serving but other replicas remain dispatchable.",
                        self.replica_rank,
                    )

            self._dynamo_endpoint_serve_task = asyncio.create_task(_serve_loop())
            self._dynamo_kv_client = await gen_endpoint.client(router_mode=RouterMode.KV)
            try:
                instance_ids = await asyncio.wait_for(
                    self._dynamo_kv_client.wait_for_instances(), timeout=60
                )
                logger.warning(
                    "[m5] replica=%s KV-router client ready, %d instances visible: %s",
                    self.replica_rank,
                    len(instance_ids),
                    instance_ids,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[m5] replica=%s wait_for_instances timed out (60s); "
                    "client may have empty instance list initially.",
                    self.replica_rank,
                )

        # KV-event tap. Optional: only meaningful when prefix caching is on
        # and vLLM was started with kv_events_config.endpoint set. We
        # don't synthesize a kv_events_config here — that's a vLLM startup
        # concern (engine_kwargs.kv_events_config). If absent, KV router
        # falls back to approximate routing.
        kv_events_cfg = getattr(self.engine.vllm_config, "kv_events_config", None)
        if (
            cache_cfg.enable_prefix_caching
            and kv_events_cfg is not None
            and getattr(kv_events_cfg, "enable_kv_cache_events", False)
        ):
            try:
                publisher = KvEventPublisher(
                    endpoint=gen_endpoint,
                    kv_block_size=cache_cfg.block_size,
                    zmq_endpoint=kv_events_cfg.endpoint.replace("*", "127.0.0.1"),
                    zmq_topic="",
                    enable_local_indexer=False,
                    dp_rank=0,
                )
                self._dynamo_kv_publishers = [publisher]
                logger.info(
                    "[m3] KvEventPublisher attached at %s",
                    kv_events_cfg.endpoint,
                )
            except Exception:
                logger.exception(
                    "[m3] KvEventPublisher init failed; router will fall back "
                    "to approximate routing."
                )
                self._dynamo_kv_publishers = None
        else:
            logger.info(
                "[m3] kv_events not configured on vLLM (prefix_caching=%s, "
                "kv_events_config=%s); KV router will approximate.",
                cache_cfg.enable_prefix_caching,
                kv_events_cfg,
            )

        # m4: master replica launches Frontend and rebinds its own
        # server_port to it. Non-master replicas poll the master Ray actor
        # and rebind their own (server_address, server_port) to the master
        # Frontend, so verl's per-replica server_handle.get_server_address
        # all return the same Frontend endpoint -> all generate traffic
        # goes through the KV router.
        if not _dynamo_frontend_enabled():
            return
        if self.replica_rank == 0 and self.node_rank == 0:
            await self._launch_dynamo_frontend()
        else:
            await self._connect_to_master_frontend()

    async def _launch_dynamo_frontend(self):
        """Spawn ``python -m dynamo.frontend --router-mode kv`` and rebind
        ``self._server_port`` to it so verl's agent loop reaches the Frontend
        instead of vLLM's own OpenAI server.

        Pre: ``self._server_port`` is the vLLM uvicorn port (set by
        ``run_uvicorn`` in parent ``run_server``); ``self._dynamo_runtime``
        is alive and at least one ``register_model`` has fired (this one)
        so the Frontend has a worker to route to.
        """
        from verl.utils.net_utils import get_free_port

        # Pick a port and immediately release the holding socket so the
        # Frontend subprocess can bind it. There's a tiny TOCTOU window;
        # if it bites us in practice we'll switch to UDS or fixed port.
        fe_port, holder = get_free_port(self._server_address)
        if holder is not None:
            holder.close()

        env = {
            **os.environ,
            "DYN_DISCOVERY_BACKEND": os.environ.get("DYN_DISCOVERY_BACKEND", "file"),
            "DYN_FILE_KV": os.environ.get("DYN_FILE_KV", _DEFAULT_FILE_KV),
        }
        cmd = [
            "python",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            "kv",
            "--http-port",
            str(fe_port),
        ]
        # Redirect Frontend stdout/stderr to a file so router decision logs
        # (KV overlap scores, worker selection) survive after the actor exits
        # and so the buffer doesn't fill up with no reader.
        fe_log_path = os.path.join("/tmp/dynamo", "frontend.log")
        os.makedirs("/tmp/dynamo", exist_ok=True)
        fe_log_fp = open(fe_log_path, "w")  # closed in shutdown via Popen.terminate

        logger.warning(
            "[m4] spawning dynamo Frontend: %s (env DYN_FILE_KV=%s, log=%s)",
            cmd,
            env["DYN_FILE_KV"],
            fe_log_path,
        )
        self._dynamo_frontend_proc = subprocess.Popen(  # noqa: S603 — controlled cmd
            cmd,
            env=env,
            stdout=fe_log_fp,
            stderr=subprocess.STDOUT,
        )

        await self._wait_frontend_ready(fe_port, _FRONTEND_READY_TIMEOUT_S)

        self._vllm_server_port = self._server_port
        self._server_port = fe_port
        logger.warning(
            "[m4] dynamo Frontend ready at %s:%s (was vLLM uvicorn at port %s); "
            "verl agent loop now routes through KV router.",
            self._server_address,
            fe_port,
            self._vllm_server_port,
        )

    async def is_dynamo_frontend_ready(self) -> bool:
        """Master-side query; non-master replicas poll this before
        rebinding their server address to the master Frontend."""
        return (
            self._dynamo_frontend_proc is not None
            and self._dynamo_frontend_proc.poll() is None
            and self._vllm_server_port is not None  # set after _server_port swap
        )

    async def _connect_to_master_frontend(self):
        """Non-master: wait for master to publish its Frontend, then
        redirect this replica's server address to the master Frontend.

        After the rebind, ``self.get_server_address()`` (inherited from
        vLLMHttpServer) returns ``(master_host, master_frontend_port)``,
        which is what ``vLLMReplica.launch_servers`` advertises to verl.
        """
        # NOTE: this prefix is the one DynamoReplica._get_server_name_prefix
        # returns; DynamoHttpServer instances don't carry that method, so
        # we hardcode here to avoid a cross-class lookup.
        master_name = "dynamo_server_0_0"
        deadline = time.monotonic() + _FRONTEND_READY_TIMEOUT_S
        master = None
        last_err: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                if master is None:
                    master = ray.get_actor(master_name)
                ready = await master.is_dynamo_frontend_ready.remote()
                if ready:
                    addr, port = await master.get_server_address.remote()
                    self._vllm_server_port = self._server_port
                    self._server_port = port
                    self._server_address = addr
                    logger.warning(
                        "[m4] replica=%s redirected to master Frontend at %s:%s "
                        "(was vLLM port %s)",
                        self.replica_rank,
                        addr,
                        port,
                        self._vllm_server_port,
                    )
                    return
            except ValueError as e:
                # ray.get_actor before master is registered.
                last_err = f"ValueError: {e}"
            except Exception as e:  # noqa: BLE001 — diagnostic only
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_FRONTEND_READY_POLL_S)
        raise TimeoutError(
            f"[m4] master Frontend not advertised within {_FRONTEND_READY_TIMEOUT_S}s "
            f"(last error: {last_err})"
        )

    async def _wait_frontend_ready(self, port: int, timeout: float):
        """Poll ``GET /v1/models`` until 200 or timeout. Raises on timeout
        (the calling try/except in ``run_server`` will fall back to m3-only
        behavior, leaving generate on the vLLM port)."""
        import aiohttp

        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{port}/v1/models"
        last_err: Optional[str] = None
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if self._dynamo_frontend_proc.poll() is not None:
                    raise RuntimeError(
                        f"[m4] Frontend exited early rc={self._dynamo_frontend_proc.returncode}"
                    )
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return
                        last_err = f"HTTP {resp.status}"
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(_FRONTEND_READY_POLL_S)
        raise TimeoutError(
            f"[m4] dynamo Frontend at port {port} not ready within {timeout}s "
            f"(last error: {last_err})"
        )

    # ──────────────── m5: route generates through dynamo KV router ────────────────

    async def _dynamo_generate_handler(self, request, context):
        """Handler invoked when a peer (or this) replica's KV-routed Client
        dispatches a request to *this* replica.

        Wire format (dynamo internal protocol) — request dict:
            token_ids: list[int]                    # the prompt
            sampling_options: dict (temperature, top_p, top_k, ...)
            stop_conditions: dict (max_tokens, stop_token_ids, ...)
            output_options: dict (logprobs, prompt_logprobs, ...)
            model: str                              # served model name
            routing: dict (optional)                # priority, dp_rank
        We yield chunks shaped as { token_ids, log_probs?, finish_reason?, ... }
        — same shape as dynamo.vllm.handlers.generate_tokens, so peer Client
        unwraps them transparently.
        """
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import SamplingParams

        token_ids = list(request.get("token_ids") or [])
        sampling_options = request.get("sampling_options") or {}
        stop_conditions = request.get("stop_conditions") or {}
        output_options = request.get("output_options") or {}

        # [m5-route] trace: per-replica counter `n=` makes every line unique
        # (defeats Ray log dedup so we can see which replica handled each
        # request). pfx hashes the first 64 tokens — enough to escape the
        # ~20-token chat template. sfx hashes the last 32 tokens — disambiguates
        # turn-1 vs turn-2 of the same conversation (turn-2 shares pfx with
        # turn-1 but has different sfx).
        self._m5_route_counter = getattr(self, "_m5_route_counter", 0) + 1
        if token_ids:
            pfx = hash(tuple(token_ids[:64])) & 0xFFFFFF
            sfx = hash(tuple(token_ids[-32:])) & 0xFFFF
        else:
            pfx = sfx = 0
        logger.warning(
            "[m5-route] replica=%s n=%d len=%d pfx=%06x sfx=%04x",
            self.replica_rank,
            self._m5_route_counter,
            len(token_ids),
            pfx,
            sfx,
        )

        # Build vLLM SamplingParams (subset; full mapping lives in
        # dynamo.vllm.handlers.build_sampling_params, we'd rather not import
        # that wholesale and pull in dynamo.common.* stack here).
        sp_kwargs: dict[str, Any] = {"detokenize": False}
        for k, v in sampling_options.items():
            if v is not None and hasattr(SamplingParams, k):
                sp_kwargs[k] = v
        for k, v in stop_conditions.items():
            if v is None or k == "stop":
                continue
            if hasattr(SamplingParams, k):
                sp_kwargs[k] = v
        if output_options.get("logprobs"):
            try:
                sp_kwargs["logprobs"] = int(output_options["logprobs"])
            except (TypeError, ValueError):
                pass

        try:
            sampling_params = SamplingParams(**sp_kwargs)
        except Exception as e:
            yield {"finish_reason": f"error: SamplingParams: {e}", "token_ids": []}
            return

        prompt = TokensPrompt(prompt_token_ids=token_ids)
        request_id = context.id() if context is not None else uuid4().hex
        num_emitted = 0
        try:
            async for output in self.engine.generate(
                prompt=prompt, sampling_params=sampling_params, request_id=request_id
            ):
                if not output.outputs:
                    continue
                res = output.outputs[0]
                new_tokens = list(res.token_ids[num_emitted:])
                if new_tokens:
                    yield {"token_ids": new_tokens}
                    num_emitted = len(res.token_ids)
                if res.finish_reason:
                    yield {"finish_reason": res.finish_reason, "token_ids": []}
                    return
        except Exception as e:
            logger.exception("[m5] handler raised on replica=%s", self.replica_rank)
            yield {"finish_reason": f"error: {type(e).__name__}: {e}", "token_ids": []}

    async def generate(  # type: ignore[override]
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
        priority: int = 0,
    ):
        """When m5 is on, dispatch via the KV-routed Client instead of calling
        ``self.engine.generate`` directly. This is what makes the KV router
        actually engage — verl's agent loop calls ``server.generate.remote()``
        unchanged, but inside this method the request now flows through the
        dynamo runtime, gets routed by KV overlap, and lands on whatever
        replica the router chose.
        """
        if not _dynamo_router_enabled() or self._dynamo_kv_client is None:
            return await super().generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
            )

        # Pop max_tokens / max_new_tokens from sampling_params (caller's mutable
        # dict) the same way parent generate() does, to keep downstream behavior
        # consistent if we ever fall back.
        sp = dict(sampling_params)  # shallow copy so we don't mutate caller's
        max_tokens = sp.pop("max_tokens", None) or sp.pop("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = max(
                0,
                min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_ids),
                ),
            )

        # Build dynamo internal-protocol request dict (matches the shape
        # build_sampling_params + _build_prompt_from_request expect on the
        # handler side).
        sampling_options = {
            k: sp[k]
            for k in ("temperature", "top_p", "top_k", "repetition_penalty", "n", "seed")
            if k in sp and sp[k] is not None
        }
        stop_conditions: dict[str, Any] = {"max_tokens": max_tokens}
        if "stop_token_ids" in sp:
            stop_conditions["stop_token_ids"] = sp["stop_token_ids"]
        output_options: dict[str, Any] = {}
        if sp.get("logprobs") is not None:
            output_options["logprobs"] = sp["logprobs"]

        request = {
            "token_ids": list(prompt_ids),
            "sampling_options": sampling_options,
            "stop_conditions": stop_conditions,
            "output_options": output_options,
            "model": self._dynamo_served_model_name or "",
            "routing": {"priority": int(priority)},
        }

        all_tokens: list[int] = []
        log_probs: list[float] = []
        finish_reason: Optional[str] = None
        try:
            # client.generate() is a PyO3 async fn — await it first to obtain
            # the AsyncIterator, then iterate. Bare `async for` on the call
            # raises TypeError: got _asyncio.Future.
            stream = await self._dynamo_kv_client.generate(request)
            async for raw in stream:
                # When `annotated=True` (default), each stream item is wrapped
                # in an envelope; the actual handler-yielded payload is under
                # .data(). Mirrors dynamo.global_router.handler usage:
                #   data = output.data() if hasattr(output, "data") else output
                chunk = raw.data() if hasattr(raw, "data") else raw
                if not isinstance(chunk, dict):
                    continue
                tok = chunk.get("token_ids")
                if tok:
                    all_tokens.extend(tok)
                lp = chunk.get("log_probs")
                if lp:
                    log_probs.extend(lp)
                fr = chunk.get("finish_reason")
                if fr:
                    finish_reason = fr
                    break
        except Exception:
            logger.exception(
                "[m5] KV-routed client.generate failed on replica=%s; "
                "falling back to local engine.",
                self.replica_rank,
            )
            return await super().generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                priority=priority,
            )

        # Map dynamo finish_reason → verl stop_reason.
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        # Defer TokenOutput import to avoid coupling at module load (older
        # verl trees may not have it on older import paths).
        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(
            token_ids=all_tokens,
            log_probs=log_probs if log_probs else None,
            stop_reason=stop_reason,
            extra_fields={"global_steps": getattr(self, "global_steps", 0)},
        )

    async def shutdown(self, *args, **kwargs):  # type: ignore[override]
        # m5: cancel the endpoint serve task so the runtime can drain.
        if self._dynamo_endpoint_serve_task is not None:
            try:
                self._dynamo_endpoint_serve_task.cancel()
                try:
                    await self._dynamo_endpoint_serve_task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception:
                logger.exception("[m5] endpoint serve task cancel raised; ignoring.")
            self._dynamo_endpoint_serve_task = None
            self._dynamo_kv_client = None

        # m4: stop the Frontend before tearing down the runtime; otherwise
        # in-flight HTTP requests would race against runtime shutdown.
        if self._dynamo_frontend_proc is not None:
            try:
                self._dynamo_frontend_proc.terminate()
                try:
                    self._dynamo_frontend_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._dynamo_frontend_proc.kill()
                    self._dynamo_frontend_proc.wait(timeout=5)
            except Exception:
                logger.exception("[m4] dynamo Frontend shutdown raised; ignoring.")
            self._dynamo_frontend_proc = None
        # Tear down dynamo runtime so its background tasks unwind
        # cleanly while the vLLM engine is still alive.
        if self._dynamo_runtime is not None:
            try:
                self._dynamo_runtime.shutdown()
            except Exception:
                logger.exception("[m3] dynamo runtime shutdown raised; ignoring.")
            self._dynamo_runtime = None
            self._dynamo_kv_publishers = None
            self._dynamo_generate_endpoint = None
        # vLLMHttpServer may or may not define shutdown — call only if it does.
        parent_shutdown = getattr(_VllmHttpServer, "shutdown", None)
        if callable(parent_shutdown):
            result = parent_shutdown(self, *args, **kwargs)
            if asyncio.iscoroutine(result):
                await result


class DynamoReplica(_VllmReplica):
    """Replica for the dynamo backend.

    Reuses the entire vLLMReplica launch pipeline but binds the Ray actor
    name to ``dynamo_*`` so ServerAdapter (which reads ``config.name``)
    lands on these actors and not on someone else's vllm replicas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebind to the recipe-side server class so subclass hooks (m3)
        # take effect. vLLMReplica.__init__ already set this to
        # ``ray.remote(vLLMHttpServer)``; we replace it.
        self.server_class = ray.remote(DynamoHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"


__all__ = ["DynamoHttpServer", "DynamoReplica"]
