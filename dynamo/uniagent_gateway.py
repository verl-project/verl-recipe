"""Dynamo program lifecycle over Uni-Agent's agent-aware Gateway.

Uni-Agent owns header parsing, chain isolation and trajectory metadata. This
module maps AgentContext to TA programs and coordinates generation and cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from urllib.parse import quote

import ray
from fastapi import HTTPException, Request
from omegaconf import OmegaConf
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter, AgentFrameworkWorker
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import GatewayActor, _GatewayActor
from uni_agent.gateway.manager import GatewayManager
from uni_agent.gateway.session.types import AgentContext

from verl.utils.config import omega_conf_to_dataclass

from .dynamo_agent_loop import GenerationPaused

logger = logging.getLogger(__name__)

SUBAGENT_PROGRAMS = ("tree", "isolated", "off")


def _program_ids(rollout_id: str, context: AgentContext | None) -> tuple[str, str | None]:
    if context is None:
        return rollout_id, None

    def program(agent_id):
        return f"{rollout_id}/main" if agent_id is None else f"{rollout_id}/agent/{quote(agent_id, safe='')}"

    agent = context["agent_id"]
    parent = context["parent_agent_id"]
    # Claude omits the parent header for direct children of the main session.
    if parent == context["session_id"]:
        parent = None
    return program(agent), program(parent) if agent is not None else None


@dataclass
class _Program:
    program_id: str
    parent_id: str | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: bool = False
    generation: int = 0
    cleanup_error: Exception | None = None
    release: asyncio.Task | None = None


class _ProgramClient:
    supports_agent_context = True

    def __init__(self, actor, client):
        self.actor, self.client = actor, client

    async def generate(self, request_id, *, agent_context=None, **kwargs):
        program = self.actor._program(request_id, agent_context)
        program.active = True  # A failed/cancelled call may already have reached the frontend.
        program.generation += 1
        if getattr(self.client, "supports_agent_context", False) is True and agent_context is not None:
            kwargs["agent_context"] = agent_context
        return await self.client.generate(
            request_id=request_id,
            program_id=program.program_id,
            parent_program_id=program.parent_id if self.actor._subagent_programs == "tree" else None,
            **kwargs,
        )

    async def finalize_program(self, program_id):
        return await self.client.finalize_program(program_id)


async def _wait_for_cleanup(task):
    """Keep an owned operation alive across cancellation of its HTTP/Ray waiter."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class _DynamoGatewayActor(_GatewayActor):
    _finalize_attempts = 3
    _finalize_timeout = 60.0
    _retry_delay = 0.2
    _drain_timeout = 60.0
    _deferred_poll = 1.0
    _deferred_attempts = 3

    def __init__(self, config, backend, subagent_programs="tree"):
        if subagent_programs not in SUBAGENT_PROGRAMS:
            raise ValueError(f"thunderagent.subagent_programs must be one of {SUBAGENT_PROGRAMS}")
        self._subagent_programs = subagent_programs
        if not callable(getattr(backend, "finalize_program", None)):
            raise ValueError("Dynamo Gateway requires a client with finalize_program")
        if getattr(backend, "_auto_finalize", True):
            raise ValueError("Dynamo Gateway requires thunderagent.auto_finalize=false")
        super().__init__(
            replace(config, rollout_backend=config.rollout_backend or "vllm"), _ProgramClient(self, backend)
        )
        self._requests = {}
        self._programs = {}
        self._identities = {}
        self._hook_events = {}
        self._closing = {}
        self._close_requested = set()
        self._abort_requested = set()
        self._closed = OrderedDict()
        self._deferred = {}
        self._deferred_flush = None
        self._shutting_down = False

    def _register_routes(self):
        super()._register_routes()

        @self._app.post("/sessions/{session_id}/agent_events")
        async def agent_event(session_id: str, request: Request):
            return await self.handle_agent_event(session_id, await request.json())

    async def create_session(self, session_id, **kwargs):
        if session_id in self._close_requested or session_id in self._closed or session_id in self._programs:
            raise ValueError("Use a fresh session ID for every rollout")
        return await super().create_session(session_id, **kwargs)

    def _program(self, session_id, context):
        # Names deliberately omit the harness session: enforce one harness per rollout.
        identity = (context["provider"], context["session_id"]) if context else None
        if session_id in self._identities and self._identities[session_id] != identity:
            raise HTTPException(409, "Agent identity source changed within the rollout")
        self._identities[session_id] = identity
        program_id, parent_id = _program_ids(session_id, None if self._subagent_programs == "off" else context)
        programs = self._programs.setdefault(session_id, {})
        program = programs.get(program_id)
        if program is None:
            ancestor = parent_id
            while ancestor is not None:
                if ancestor == program_id:
                    raise HTTPException(409, "Agent parent relationship contains a cycle")
                ancestor = programs[ancestor].parent_id if ancestor in programs else None
            program = programs[program_id] = _Program(program_id, parent_id)
        elif program.parent_id != parent_id:
            raise HTTPException(409, "Agent parent changed within the rollout")
        return program

    async def _handle_request(self, handler, session_id, payload, agent_context=None):
        if session_id in self._close_requested or session_id in self._closed:
            raise HTTPException(409, "Rollout is closing")
        self._get_session(session_id)
        program = self._program(session_id, agent_context)

        async def generate():
            async with program.lock:
                if program.cleanup_error is not None:
                    raise HTTPException(503, "Program cleanup is unconfirmed; retry cleanup before resuming")
                # A resumed agent supersedes a deferred stop from its previous turn.
                self._deferred.pop((session_id, program.program_id), None)
                return await handler(session_id=session_id, payload=payload, agent_context=agent_context)

        request = asyncio.create_task(generate())
        pending = self._requests.setdefault(session_id, set())
        pending.add(request)
        try:
            return await _wait_for_cleanup(request)
        finally:
            pending.discard(request)
            if not pending:
                self._requests.pop(session_id, None)

    async def _handle_openai_chat_completions(self, *, session_id, payload, agent_context=None):
        return await self._handle_request(super()._handle_openai_chat_completions, session_id, payload, agent_context)

    async def _handle_openai_responses(self, *, session_id, payload, agent_context=None):
        return await self._handle_request(super()._handle_openai_responses, session_id, payload, agent_context)

    async def _handle_anthropic_messages(self, *, session_id, payload, agent_context=None):
        return await self._handle_request(super()._handle_anthropic_messages, session_id, payload, agent_context)

    async def _release_program(self, program_id):
        for attempt in range(self._finalize_attempts):
            try:
                released = await asyncio.wait_for(self._backend.finalize_program(program_id), self._finalize_timeout)
                if released is False:
                    raise GenerationPaused(program_id)
                return True
            except GenerationPaused:
                return False
            except Exception:
                if attempt + 1 == self._finalize_attempts:
                    logger.exception("ThunderAgent cleanup failed: program=%s", program_id)
                    raise
                await asyncio.sleep(self._retry_delay * 2**attempt)

    def _defer(self, session_id, program):
        key = (session_id, program.program_id)
        if key not in self._deferred or self._deferred[key][0] != program.generation:
            self._deferred[key] = (program.generation, self._deferred_attempts)
        if not self._shutting_down and (self._deferred_flush is None or self._deferred_flush.done()):
            self._deferred_flush = asyncio.create_task(self._flush_deferred())

    async def _close_program(self, session_id, program, expected_generation=None):
        async def release():
            async with program.lock:
                if expected_generation is not None and program.generation != expected_generation:
                    return
                if not program.active:
                    return
                try:
                    if not await self._release_program(program.program_id):
                        self._defer(session_id, program)
                        return
                except Exception as error:
                    program.cleanup_error = error
                    raise
                program.active = False
                program.cleanup_error = None
                self._deferred.pop((session_id, program.program_id), None)
                logger.info("ThunderAgent program finalized: program=%s", program.program_id)

        # Keep the shared lock until the RPC really settles, even if a close waiter
        # times out. A retry joins that release instead of racing a resumed turn.
        if program.release is None or program.release.done():
            program.release = asyncio.create_task(release())
            program.release.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        timeout = self._finalize_timeout * self._finalize_attempts + self._retry_delay * 2**self._finalize_attempts + 1
        await asyncio.wait_for(asyncio.shield(program.release), max(timeout, 0.01))
        if expected_generation is None and program.active and (session_id, program.program_id) not in self._deferred:
            # A joined hook release may have become stale while it waited for
            # the lock. Rollout close still owes cleanup of the latest turn.
            await self._close_program(session_id, program)

    async def _flush_deferred(self):
        while any(left > 0 for _, left in self._deferred.values()):
            await asyncio.sleep(self._deferred_poll)
            for key, (generation, left) in list(self._deferred.items()):
                if left <= 0 or self._deferred.get(key) != (generation, left):
                    continue
                session_id, program_id = key
                program = self._programs.get(session_id, {}).get(program_id)
                if program is None or program.generation != generation:
                    self._deferred.pop(key, None)
                    continue
                try:
                    await self._close_program(session_id, program, generation)
                except Exception:
                    if self._deferred.get(key) == (generation, left):
                        self._deferred[key] = (generation, left - 1)
                    logger.exception("Deferred TA cleanup failed; retries_left=%s program=%s", left - 1, program_id)
                self._forget_released_programs(session_id)

    def _forget_released_programs(self, session_id):
        if session_id not in self._sessions and not any(p.active for p in self._programs.get(session_id, {}).values()):
            self._programs.pop(session_id, None)

    async def handle_agent_event(self, session_id, payload):
        if session_id in self._closed:
            return {"status": "closed"}
        if session_id in self._close_requested:
            return {"status": "closing"}
        self._get_session(session_id)
        name = payload.get("hook_event_name")
        if name not in ("SubagentStart", "SubagentStop", "SessionEnd"):
            raise HTTPException(400, "Unsupported lifecycle event")
        identity = self._identities.get(session_id)
        root = payload.get("session_id")
        if not root or (identity is not None and identity != ("claude_code", root)):
            raise HTTPException(409, "Lifecycle event belongs to another Claude session")
        events = self._hook_events.setdefault(session_id, {})
        event_id = payload.get("event_id")
        previous = events.get(event_id) if event_id else None
        if previous is not None and previous[1]:
            return {"status": "duplicate"}
        generation = None
        if name == "SubagentStop" and self._subagent_programs != "off":
            agent = payload.get("agent_id")
            if not isinstance(agent, str) or not agent:
                raise HTTPException(400, "SubagentStop requires agent_id")
            program = self._programs.get(session_id, {}).get(f"{session_id}/agent/{quote(agent, safe='')}")
            if program is not None:
                generation = previous[0] if previous is not None else program.generation
                if event_id:
                    events[event_id] = (generation, False)
                await self._close_program(session_id, program, generation)
        if event_id:
            events[event_id] = (generation, True)
        return {"status": "ok"}

    async def _release_session_programs(self, session_id):
        programs = self._programs.get(session_id, {})

        def depth(program):
            level, parent = 0, program.parent_id
            while parent in programs:
                level, parent = level + 1, programs[parent].parent_id
            return level

        for program in sorted(programs.values(), key=depth, reverse=True):
            try:
                await self._close_program(session_id, program)
            except Exception:
                # Generation has drained: keep completed training samples while
                # cleanup retries independently, including explicit later closes.
                self._defer(session_id, program)
                logger.exception("Deferring unconfirmed TA cleanup: program=%s", program.program_id)

    async def _close_session(self, session_id):
        pending = set(self._requests.get(session_id, ()))
        if pending and session_id not in self._abort_requested:
            _, pending = await asyncio.wait(pending, timeout=self._drain_timeout)
        if pending:
            for request in pending:
                request.cancel()
            _, pending = await asyncio.wait(pending, timeout=self._drain_timeout)
        if pending:
            raise TimeoutError(f"Rollout {session_id} still has an active generation; program was not released")
        # Cleanup failure must not send the framework down its abort path and
        # discard completed trajectories. Keep failed programs for later retries.
        await self._release_session_programs(session_id)
        if session_id in self._abort_requested:
            await super().abort_session(session_id)
            result = None
        else:
            result = await super().finalize_session(session_id)
            result.sort(key=lambda t: (t.extra_fields.get("agent_context") or {}).get("agent_id") is None)
        self._closed[session_id] = result
        while len(self._closed) > 64:
            self._closed.popitem(last=False)
        self._forget_released_programs(session_id)
        self._identities.pop(session_id, None)
        self._hook_events.pop(session_id, None)
        self._close_requested.discard(session_id)
        self._abort_requested.discard(session_id)
        return result

    async def _finish(self, session_id, *, abort):
        if session_id in self._closed:
            await self._release_session_programs(session_id)
            self._forget_released_programs(session_id)
            return None if abort else self._closed[session_id]
        if session_id not in self._sessions:
            if session_id in self._programs:
                await self._release_session_programs(session_id)
                self._forget_released_programs(session_id)
            if abort:
                return None
            raise KeyError(session_id)
        self._close_requested.add(session_id)
        if abort:
            self._abort_requested.add(session_id)
            for request in self._requests.get(session_id, ()):
                request.cancel()
        task = self._closing.get(session_id)
        if task is None or task.done():
            task = asyncio.create_task(self._close_session(session_id))
            self._closing[session_id] = task
        try:
            result = await _wait_for_cleanup(task)
            return None if abort else result
        finally:
            if task.done() and self._closing.get(session_id) is task:
                self._closing.pop(session_id, None)

    async def finalize_session(self, session_id):
        return await self._finish(session_id, abort=False)

    async def abort_session(self, session_id):
        await self._finish(session_id, abort=True)

    async def get_cleanup_state(self):
        """Actor-level counters remain available after rollout sessions close."""
        programs = [program for tree in self._programs.values() for program in tree.values()]
        return {
            "active_programs": sum(program.active for program in programs),
            "pending_cleanup": len(self._deferred),
            "exhausted_cleanup": sum(left == 0 for _, left in self._deferred.values()),
            "running_releases": sum(program.release is not None and not program.release.done() for program in programs),
        }

    async def shutdown(self):
        self._shutting_down = True
        if self._deferred_flush is not None:
            self._deferred_flush.cancel()
            await asyncio.gather(self._deferred_flush, return_exceptions=True)
        try:
            results = await asyncio.gather(
                *(self.abort_session(sid) for sid in set(self._sessions) | set(self._programs)), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if any(program.active for programs in self._programs.values() for program in programs.values()):
                raise RuntimeError("Shutdown left unconfirmed ThunderAgent programs")
        finally:
            await super().shutdown()


DynamoGatewayActor = ray.remote(_DynamoGatewayActor)


class DynamoGatewayManager(GatewayManager):
    """Use the recipe actor while retaining upstream session routing methods."""

    def __init__(
        self, llm_client, *, gateway_count, gateway_actor_config, actor_cls=DynamoGatewayActor, **actor_kwargs
    ):
        if gateway_count <= 0:
            raise ValueError("gateway_count must be positive")
        nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not nodes:
            raise RuntimeError("No alive CPU nodes available for DynamoGatewayActor")
        self.gateways = [
            actor_cls.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=nodes[i % len(nodes)], soft=True
                )
            ).remote(gateway_actor_config, backend=llm_client, **actor_kwargs)
            for i in range(gateway_count)
        ]
        ray.get([gateway.start.remote() for gateway in self.gateways])
        self.gateway_count = len(self.gateways)
        self.active_sessions_per_gateway = [0 for _ in self.gateways]
        self._session_to_gateway_index = {}


def _build_gateway_manager(config, llm_client, *, thunderagent, engine, subagent_programs="tree"):
    """Mirror Uni-Agent's GatewayActorConfig wiring without replacing globals.

    Dynamo serves its engines behind its own rollout registry name, so the tool
    parser backend is the engine behind the frontend, not ``rollout.name``.
    """
    af = config.actor_rollout_ref.rollout.custom.agent_framework
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    allowed = af.get("allowed_request_sampling_param_keys")
    if allowed is not None:
        if not (isinstance(allowed, list | tuple) or OmegaConf.is_list(allowed)):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
        if any(not isinstance(key, str) for key in allowed):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
    actor_config = GatewayActorConfig(
        tokenizer=model.tokenizer,
        processor=model.processor,
        tool_parser_name=rollout.multi_turn.format,
        rollout_backend=engine,
        enable_tool_parser_cache=af.get("enable_tool_parser_cache", True),
        hf_model_type=getattr(model.hf_config, "model_type", None),
        apply_chat_template_kwargs=dict(config.data.get("apply_chat_template_kwargs", {})),
        mm_processor_kwargs=dict(config.data.get("mm_processor_kwargs", {})),
        prompt_length=rollout.prompt_length,
        response_length=rollout.response_length,
        enable_last_assistant_rollback=af.get("enable_last_assistant_rollback", True),
        allowed_request_sampling_param_keys=None if allowed is None else set(allowed),
        coalesce_reserved_exact_requests=af.get("coalesce_reserved_exact_requests", True),
    )
    return DynamoGatewayManager(
        llm_client,
        gateway_count=int(af["gateway_count"]),
        gateway_actor_config=actor_config,
        actor_cls=DynamoGatewayActor if thunderagent else GatewayActor,
        **({"subagent_programs": subagent_programs} if thunderagent else {}),
    )


class DynamoAgentFrameworkRolloutAdapter(AgentFrameworkRolloutAdapter):
    """V1 adapter for current Uni-Agent, without modifying its source tree."""

    @classmethod
    def create(cls, *, config, llm_client, teacher_client=None, reward_loop_worker_handles=None, **_):
        if teacher_client is not None:
            raise ValueError("Dynamo Uni-Agent Gateway does not support teacher_client")
        dynamo = config.actor_rollout_ref.rollout.engine_kwargs.dynamo
        engine = dynamo.get("engine", "vllm")
        if engine not in ("vllm", "sglang"):
            raise ValueError(f"Dynamo Uni-Agent Gateway supports the vllm and sglang engines, got {engine!r}")
        thunderagent = bool(dynamo.thunderagent.enabled)
        if thunderagent and dynamo.thunderagent.get("auto_finalize", True):
            raise ValueError("ThunderAgent with the Uni-Agent Gateway requires thunderagent.auto_finalize=false")
        ablation = {"subagent_programs": dynamo.thunderagent.get("subagent_programs", "tree")} if thunderagent else {}
        gateway_manager = _build_gateway_manager(
            config, llm_client, thunderagent=thunderagent, engine=engine, **ablation
        )
        instance = cls()
        instance.gateway_manager = gateway_manager
        instance.framework_worker = AgentFrameworkWorker.remote(
            config=config, gateway_manager=gateway_manager, reward_loop_worker_handles=reward_loop_worker_handles
        )
        return instance
