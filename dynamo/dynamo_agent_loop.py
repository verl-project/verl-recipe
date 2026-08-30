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
"""Dynamo-specific AgentLoopManager.

Dynamo exposes one logical rollout endpoint through a master dynamo.frontend.
The worker-level routing happens inside Dynamo's KV router, not in verl's
GlobalRequestLoadBalancer. This module keeps the AgentLoop execution model but
replaces the generic server manager with a direct Dynamo server manager.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import update_prometheus_config

from .thunderagent import current_program
from .thunderagent import program_scope as bind_program

logger = logging.getLogger(__name__)


class DynamoServerManager:
    """Direct manager for the shared Dynamo frontend actor.

    Unlike AsyncLLMServerManager, this class intentionally does not acquire a
    server from GlobalRequestLoadBalancer. Dynamo owns routing behind its
    frontend, so verl should only call the single shared Dynamo actor.
    """

    def __init__(
        self,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        *,
        thunderagent_enabled: bool = False,
    ):
        if len(servers) != 1:
            raise ValueError(f"DynamoServerManager expects exactly one shared server, got {len(servers)}")
        self.server_address, self.server = servers[0]
        self.thunderagent_enabled = thunderagent_enabled

    @asynccontextmanager
    async def program_scope(self):
        """Bind all turns in one agent-loop run to one Dynamo program."""
        if not self.thunderagent_enabled:
            yield
            return
        async with bind_program(uuid4().hex, self._finalize_program):
            yield

    async def _finalize_program(self, session_id: str) -> None:
        await self.server.finalize_program.remote(session_id=session_id)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if audio_data is not None or mm_processor_kwargs:
            raise RuntimeError("Dynamo frontend generate does not support audio inputs or processor options")

        generate_kwargs = dict(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )
        if not self.thunderagent_enabled:
            output = await self.server.generate.remote(**generate_kwargs)
            return self._tag_weight_versions(output)

        scope = current_program()
        if scope is None:
            raise RuntimeError("Dynamo generation requires an active ThunderAgent program")
        async with scope.request():
            output = await self.server.generate.remote(
                **generate_kwargs,
                thunderagent_session_id=scope.session_id,
            )
            return self._tag_weight_versions(output)

    @staticmethod
    def _tag_weight_versions(output: TokenOutput) -> TokenOutput:
        """Match LLMServerClient.generate's min/max_global_steps contract.

        V1 sync-mode consumers read these keys unconditionally; leaving them
        unset crashes trainer staleness accounting downstream. Pass through
        outputs without extra_fields untouched (duck-typed test doubles).
        """
        extra_fields = getattr(output, "extra_fields", None)
        if extra_fields is None:
            return output
        global_steps = extra_fields.get("global_steps")
        output.extra_fields.setdefault("min_global_steps", global_steps)
        output.extra_fields.setdefault("max_global_steps", global_steps)
        return output


class DynamoFullyAsyncLLMServerClient(FullyAsyncLLMServerClient):
    """FullyAsyncLLMServerClient with ThunderAgent program affinity.

    ThunderAgent pins all turns of one trajectory to one worker via the
    ``thunderagent_session_id`` kwarg (the server turns it into an
    X-Dynamo-Session-ID routing header). Callers with a stable per-trajectory
    request_id — uni-agent uses its gateway session_id — get affinity with no
    caller-side changes: when the kwarg is absent we key the program by
    request_id.

    Program lifecycle: the router's ProgramTable only frees an entry on an
    explicit session-final request — there is NO passive expiry, so an
    unfinalized program leaks router capacity for the whole training run and
    eventually pauses admission. With ``auto_finalize`` (default) every
    generate call finalizes its program on the way out: correct for
    single-turn callers, at the cost of cross-turn affinity. Multi-turn
    callers should set engine_kwargs.dynamo.thunderagent.auto_finalize=false
    AND call :meth:`finalize_program` from their trajectory-end hook
    (e.g. uni-agent gateway finalize/abort).

    ProgramTables are frontend-local and the LB re-routes retries when a
    server leaves the pool (separate_async switch_to_trainer, sticky-cache
    eviction), so one session can hold programs on SEVERAL frontends. The
    client therefore records every server acquired for a session and
    finalizes each of them; cleanup counts as confirmed only when all ack.
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        dynamo_server_handles=None,
        auto_finalize=True,
        finalize_leak_threshold=50,
        finalize_timeout_s=60.0,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._dynamo_server_handles = list(dynamo_server_handles or [])
        self._auto_finalize = bool(auto_finalize)
        self._finalize_leak_threshold = int(finalize_leak_threshold)
        self._finalize_timeout_s = float(finalize_timeout_s)
        # CUMULATIVE unconfirmed-cleanup count. Leaked programs are permanent
        # (the router has no passive expiry), so this is never reset by later
        # successes — a consecutive counter would let alternating fail/success
        # accumulate unbounded leaks without ever tripping the threshold.
        self._unresolved_finalize_leaks = 0
        # session_id -> routing key actually used for generation, so manual
        # finalize_program(session_id) calls (auto_finalize=false multi-turn
        # callers) route through the same sticky entry the requests used.
        self._session_routing_keys: OrderedDict[str, str] = OrderedDict()
        # session_id -> {server_id: handle} for every server that served (or
        # may have served) a generation attempt. The LB sticky entry alone
        # cannot identify these: separate_async removes hybrid servers from
        # the LB mid-trajectory, so the aborted retry re-routes to the other
        # pool and the sticky entry ends up pointing ONLY at the last server —
        # finalizing just there leaks the program created on every earlier
        # frontend. ProgramTables are frontend-local; cleanup must reach each
        # server that ever admitted the session.
        self._session_servers: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # OUTER routing key -> session for requests currently inside
        # generate(), so the per-attempt _acquire_server hook can attribute
        # each acquired server to its session.
        self._inflight_routing_sessions: dict[str, str] = {}

    _SESSION_ROUTING_CACHE_MAX = 65536

    def _remember_routing_key(self, session_id: str, routing_key: str) -> None:
        cache = self._session_routing_keys
        cache[session_id] = routing_key
        cache.move_to_end(session_id)
        while len(cache) > self._SESSION_ROUTING_CACHE_MAX:
            cache.popitem(last=False)

    def _record_served_server(self, session_id: str, server_id: str, handle: Any) -> None:
        served = self._session_servers.get(session_id)
        if served is None:
            served = {}
            self._session_servers[session_id] = served
            while len(self._session_servers) > self._SESSION_ROUTING_CACHE_MAX:
                self._session_servers.popitem(last=False)
        self._session_servers.move_to_end(session_id)
        served[str(server_id)] = handle

    def _drop_session(self, session_id: str) -> None:
        self._session_servers.pop(session_id, None)
        self._session_routing_keys.pop(session_id, None)

    async def _acquire_server(self, request_id):
        # Every retry attempt of the parent FullyAsync loop acquires here with
        # the same outer routing key. Recording each acquisition (a superset
        # of "actually admitted" — a finalize for a session unknown to a
        # frontend is a cheap no-op) is what lets finalize reach frontends the
        # LB has since removed or re-routed away from.
        server_id, handle = await super()._acquire_server(request_id)
        session_id = self._inflight_routing_sessions.get(str(request_id))
        if session_id is not None:
            self._record_served_server(session_id, server_id, handle)
        return server_id, handle

    async def generate(self, request_id, **kwargs):
        session_id = str(kwargs.setdefault("thunderagent_session_id", str(request_id)))
        # The LB sticky cache is keyed by the OUTER request_id (the routing
        # key every generate attempt used), which may differ from an
        # explicitly passed session_id — remember it so both auto and manual
        # finalize route with the key the requests actually used.
        self._remember_routing_key(session_id, str(request_id))
        self._inflight_routing_sessions[str(request_id)] = session_id
        try:
            return await super().generate(request_id, **kwargs)
        finally:
            self._inflight_routing_sessions.pop(str(request_id), None)
            if self._auto_finalize:
                await self._finalize_with_recovery(session_id, routing_key=str(request_id))

    async def _finalize_on_handle(self, handle: Any, session_id: str) -> None:
        # Bound each finalize RPC: the target frontend may be a hybrid server
        # currently paused for the trainer phase — the router answers final
        # requests without the engine, but a hung frontend must not park the
        # trajectory for the full request_timeout_s (see the 8.4 probe fix).
        await asyncio.wait_for(
            handle.finalize_program.remote(session_id=session_id),
            timeout=self._finalize_timeout_s,
        )

    async def finalize_program(self, session_id: str, routing_key: str = None) -> None:
        """Release the ThunderAgent program for one trajectory.

        ProgramTables are frontend-local, so the finalize must reach EVERY
        frontend that served a generation attempt: with the servers recorded
        at acquire time, each one is finalized directly (the LB sticky entry
        is not consulted — after a separate_async switch it points only at
        the last server and would leak the earlier pool's program). For
        sessions with no recorded servers (manual callers that never
        generated through this client), falls back to the sticky lookup via
        the recorded routing key, then to broadcasting to this pool's static
        handles when no LB is wired.

        Raises if any served frontend could not be confirmed clean.
        """
        session_id = str(session_id)
        served = self._session_servers.get(session_id)
        if served:
            await asyncio.gather(
                *[self._finalize_on_handle(handle, session_id) for handle in dict(served).values()]
            )
            self._drop_session(session_id)
            return
        if routing_key is None:
            routing_key = self._session_routing_keys.get(session_id, session_id)
        key = str(routing_key)
        if self._load_balancer is not None:
            server_id, handle = await self._load_balancer.acquire_server.remote(request_id=key)
            try:
                await self._finalize_on_handle(handle, session_id)
            finally:
                self._load_balancer.release_server.remote(server_id=server_id)
            self._drop_session(session_id)
            return
        await asyncio.gather(
            *[self._finalize_on_handle(handle, session_id) for handle in self._dynamo_server_handles]
        )
        self._drop_session(session_id)

    def _raise_if_leak_threshold(self) -> None:
        if self._unresolved_finalize_leaks >= self._finalize_leak_threshold:
            raise RuntimeError(
                f"{self._unresolved_finalize_leaks} cumulative unconfirmed ThunderAgent finalize "
                "cleanups — the router ProgramTable is leaking toward admission pause. Investigate "
                "frontend/router health (threshold: "
                "engine_kwargs.dynamo.thunderagent.finalize_leak_threshold)."
            )

    async def _finalize_served_with_recovery(self, session_id: str, served: dict[str, Any]) -> None:
        """Finalize every served frontend; cleanup is confirmed only when ALL ack.

        Each frontend gets its own bounded retry loop; the ones that never
        ack are permanent frontend-local leaks (a broadcast to OTHER
        frontends cannot free them) and count toward the cumulative
        threshold individually.
        """

        async def finalize_one(server_id: str, handle: Any) -> bool:
            delay = 0.2
            for attempt in range(3):
                try:
                    await self._finalize_on_handle(handle, session_id)
                    return True
                except Exception:
                    logger.warning(
                        "finalize_program attempt %d/3 failed for session %s on server %s",
                        attempt + 1,
                        session_id,
                        server_id,
                        exc_info=(attempt == 2),
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
            return False

        results = await asyncio.gather(*[finalize_one(sid, handle) for sid, handle in served.items()])
        self._drop_session(session_id)
        unconfirmed = [sid for sid, confirmed in zip(served.keys(), results) if not confirmed]
        if not unconfirmed:
            return
        logger.error(
            "finalize_program for session %s is UNCONFIRMED on %d of %d served frontends (%s) — "
            "counting as leaked programs",
            session_id,
            len(unconfirmed),
            len(served),
            ", ".join(unconfirmed),
        )
        self._unresolved_finalize_leaks += len(unconfirmed)
        self._raise_if_leak_threshold()

    async def _finalize_with_recovery(self, session_id: str, routing_key: str) -> None:
        """Bounded per-frontend retries → cumulative leak threshold.

        Cleanup is CONFIRMED only when every frontend recorded as serving the
        session acks its finalize. For untracked sessions the legacy chain
        remains: sticky-routed retries, then a static-handle broadcast that is
        counted as UNCONFIRMED (it covers only this pool's frontends, and a
        finalize for an unknown session no-ops with 2xx — it cannot confirm
        anything). Unconfirmed cleanups count as permanent leaks; past the
        threshold (engine_kwargs.dynamo.thunderagent.finalize_leak_threshold)
        the run fails before router admission pauses. A single leak does not
        destroy an already-successful trajectory.
        """
        session_id = str(session_id)
        served = self._session_servers.get(session_id)
        if served:
            await self._finalize_served_with_recovery(session_id, dict(served))
            return
        delay = 0.2
        for attempt in range(3):
            try:
                await self.finalize_program(session_id, routing_key=routing_key)
                return  # confirmed: routed via the sticky entry the requests used
            except Exception:
                logger.warning(
                    "finalize_program attempt %d/3 failed for session %s",
                    attempt + 1,
                    session_id,
                    exc_info=(attempt == 2),
                )
                await asyncio.sleep(delay)
                delay *= 2
        try:
            await asyncio.gather(
                *[h.finalize_program.remote(session_id=str(session_id)) for h in self._dynamo_server_handles]
            )
            logger.error(
                "finalize_program for session %s fell back to a static-handle broadcast; "
                "cleanup is UNCONFIRMED (the serving frontend may be in another pool) — "
                "counting as a leaked program",
                session_id,
            )
        except Exception:
            logger.error(
                "finalize_program broadcast fallback failed for session %s; router entry leaks",
                session_id,
                exc_info=True,
            )
        self._drop_session(str(session_id))
        self._unresolved_finalize_leaks += 1
        self._raise_if_leak_threshold()


class DynamoLLMServerManager(LLMServerManager):
    """LLM server manager that launches Dynamo through its shared worker pool."""

    def _thunderagent_config(self) -> dict:
        dynamo_config = (self.rollout_config.engine_kwargs or {}).get("dynamo", {}) or {}
        return dynamo_config.get("thunderagent", {}) or {}

    def _thunderagent_enabled(self) -> bool:
        return bool(self._thunderagent_config().get("enabled", False))

    async def _initialize_llm_servers(self, start_rank: int = None):
        if start_rank is None:
            start_rank = self.start_rank

        from recipe.dynamo.dynamo_thunderagent import DynamoThunderAgentReplica

        replica = DynamoThunderAgentReplica(
            replica_rank=start_rank,
            config=self.rollout_config,
            model_config=self.model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
        )
        if self.worker_group is None:
            # separate_async standalone pool: own resource pool + one
            # CheckpointEngineWorker per rollout GPU (nccl first hop),
            # dynamo stack launched on the pool nodes.
            await replica.init_standalone_pool()
        else:
            await replica.init_hybrid_worker_pool(self.worker_group)

        self.rollout_replicas = [replica]
        self.server_handles = [replica._server_handle]
        self.server_addresses = [replica._server_address]
        print(f"DynamoLLMServerManager: {self.server_addresses}")

        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    def get_client(self, client_cls=None, **kwargs):
        """Return an LLM client for the shared Dynamo frontend.

        V1 trainers pass an explicit ``client_cls`` (FullyAsyncLLMServerClient
        for colocate_async / separate_async): delegate to the base manager so
        the client gets the GlobalRequestLoadBalancer (degenerate single-server
        pass-through — Dynamo's KV router still does the real routing), the
        abort/resume retry loop, and min/max_global_steps aggregation. With
        ThunderAgent enabled the client is upgraded to the affinity-aware
        subclass (the server hard-requires a session id per request).

        Legacy V0 callers (ray_trainer / DynamoAgentLoopManager) pass no
        ``client_cls`` and keep the direct DynamoServerManager, which carries
        the ThunderAgent program-affinity path via DynamoAgentLoopWorker.
        """
        if client_cls is not None:
            if self._thunderagent_enabled() and issubclass(DynamoFullyAsyncLLMServerClient, client_cls):
                # separate_async is covered too: the client records every
                # server that serves an attempt and finalizes each one, so
                # programs created on a pool the LB later removed (hybrid
                # switch) are still cleaned up.
                return super().get_client(
                    client_cls=DynamoFullyAsyncLLMServerClient,
                    dynamo_server_handles=self.server_handles,
                    auto_finalize=bool(self._thunderagent_config().get("auto_finalize", True)),
                    finalize_leak_threshold=int(self._thunderagent_config().get("finalize_leak_threshold", 50)),
                    finalize_timeout_s=float(self._thunderagent_config().get("finalize_timeout_s", 60.0)),
                    **kwargs,
                )
            return super().get_client(client_cls=client_cls, **kwargs)

        if self._thunderagent_enabled() and bool(self.config.trainer.get("use_v1", False)):
            # V1 sync mode reaches here (base get_client() passes no client_cls).
            # Its AgentLoopWorkerTQ never establishes a ProgramScope, so the
            # legacy DynamoServerManager below would fail on every generate.
            raise ValueError(
                "engine_kwargs.dynamo.thunderagent.enabled=true is only supported under V1 for "
                "trainer_mode colocate_async/separate_async (their clients are upgraded to "
                "DynamoFullyAsyncLLMServerClient). For trainer_mode=sync disable thunderagent, "
                "or use the legacy path with trainer.use_v1=false."
            )

        dynamo_config = (self.rollout_config.engine_kwargs or {}).get("dynamo", {}) or {}
        thunderagent_config = dynamo_config.get("thunderagent", {}) or {}
        servers = list(zip(self.server_addresses, self.server_handles, strict=True))
        return DynamoServerManager(
            servers,
            thunderagent_enabled=bool(thunderagent_config.get("enabled", False)),
        )


class DynamoAgentLoopWorker(AgentLoopWorker):
    """Bind each trajectory to one ThunderAgent program."""

    async def _run_agent_loop(self, *args, **kwargs):
        async with self.llm_client.program_scope():
            return await super()._run_agent_loop(*args, **kwargs)


class DynamoAgentLoopManager(AgentLoopManager):
    """AgentLoopManager compatible with the current verl LLMServerClient API."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DynamoAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        await super()._init_agent_loop_workers()


__all__ = [
    "DynamoAgentLoopManager",
    "DynamoAgentLoopWorker",
    "DynamoFullyAsyncLLMServerClient",
    "DynamoLLMServerManager",
    "DynamoServerManager",
]
