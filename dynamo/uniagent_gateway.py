"""ThunderAgent program lifecycle for the unmodified Uni-Agent Gateway API.

Select DynamoAgentFrameworkRolloutAdapter through agent_loop_manager_class. A session is one
trajectory but may be a tree of programs (Claude Code subagents): the Gateway terminates the identity
headers, so this module reads them, names each turn's program and releases the tree, subagents first,
when the session closes. Model calls are serial within a program; programs run concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from contextvars import ContextVar

import ray
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from omegaconf import OmegaConf
from starlette.datastructures import Headers
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter, AgentFrameworkWorker
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import GatewayActor, _GatewayActor
from uni_agent.gateway.manager import GatewayManager

from verl.utils.config import omega_conf_to_dataclass

from .agent_identity import AgentIdentity, agent_identity_from_headers
from .dynamo_agent_loop import GenerationPaused

logger = logging.getLogger(__name__)

# The identity the current request announced, visible down to the backend call.
_REQUEST_IDENTITY: ContextVar[AgentIdentity | None] = ContextVar("dynamo_gateway_agent_identity", default=None)


class _AgentIdentityMiddleware:
    """Bind each request's agent identity headers to its task (pure ASGI: same task as the route)."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self._app(scope, receive, send)
        token = _REQUEST_IDENTITY.set(agent_identity_from_headers(Headers(scope=scope)))
        try:
            return await self._app(scope, receive, send)
        finally:
            _REQUEST_IDENTITY.reset(token)


# The last finish reason a response carries, in either dialect; streams repeat it per chunk.
_FINISH_REASON = re.compile(r'"(?:finish_reason|stop_reason)":\s*"([a-z_]+)"')


def _turn_returned(body: bytes) -> bool:
    """True when the turn ended with ``stop`` / ``end_turn``: the agent's final answer, for a subagent the
    moment the Agent tool returns it. Tool calls, truncation and unknown reasons are not a return."""
    reasons = _FINISH_REASON.findall(body.decode("utf-8", "replace"))
    return bool(reasons) and reasons[-1] in ("stop", "end_turn")


def _observe_body(response, on_complete):
    """Call ``on_complete(body)`` with the response body once it has been produced."""
    if isinstance(response, StreamingResponse):
        inner = response.body_iterator

        async def tee():
            chunks = []
            async for chunk in inner:
                chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
                yield chunk
            on_complete(b"".join(chunks))

        response.body_iterator = tee()
    else:
        on_complete(response.body)


# How agent identities become programs; ``tree`` is the design, the other two are benchmark ablations.
SUBAGENT_PROGRAMS = ("tree", "isolated", "off")


def _program_of(session_id: str, identity: AgentIdentity | None) -> tuple[str, str | None]:
    """The program a turn belongs to and its parent; a turn without identity headers is the session."""
    if identity is None:
        return session_id, None
    return identity.session_id, identity.parent_session_id


def _routed_program(session_id: str, identity: AgentIdentity | None, mode: str) -> tuple[str, str | None]:
    """``tree``: the program and its parent; ``isolated``: the program alone; ``off``: the session
    (every agent's turns in one program, as before the recipe knew about subagents)."""
    if mode == "off":
        return session_id, None
    program_id, parent_program_id = _program_of(session_id, identity)
    return program_id, (parent_program_id if mode == "tree" else None)


class _ProgramClient:
    """The session's view of the client: a program per agent identity, with the session id kept as
    the routing key so a whole tree reaches one frontend."""

    def __init__(self, client, subagent_programs="tree"):
        self._client = client
        self._subagent_programs = subagent_programs

    async def generate(self, request_id, **kwargs):
        program_id, parent_program_id = _routed_program(request_id, _REQUEST_IDENTITY.get(), self._subagent_programs)
        return await self._client.generate(
            request_id, program_id=program_id, parent_program_id=parent_program_id, **kwargs
        )

    async def finalize_program(self, program_id):
        await self._client.finalize_program(program_id)


async def _wait_for_cleanup(task):
    """Finish cleanup even if the waiter is cancelled repeatedly; then propagate."""
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
    """Close the gateway session and every ThunderAgent program it created together."""

    _finalize_attempts = 3
    _finalize_timeout = 60.0
    _retry_delay = 0.2
    _drain_timeout = 60.0
    _deferred_poll = 1.0
    _deferred_retries = 3

    def __init__(self, config, backend, subagent_programs="tree"):
        if not callable(getattr(backend, "finalize_program", None)):
            raise ValueError("Dynamo Gateway requires a client with finalize_program")
        if getattr(backend, "_auto_finalize", True):
            raise ValueError("Dynamo Gateway requires thunderagent.auto_finalize=false")
        if subagent_programs not in SUBAGENT_PROGRAMS:
            raise ValueError(
                f"thunderagent.subagent_programs must be one of {SUBAGENT_PROGRAMS}, got {subagent_programs!r}"
            )
        self._subagent_programs = subagent_programs
        super().__init__(config, _ProgramClient(backend, subagent_programs))
        self._app.add_middleware(_AgentIdentityMiddleware)
        self._requests = {}
        # session -> {program: parent}, in creation order; a child is created after its parent.
        self._programs = {}
        # session -> releases of returned subagents still in flight.
        self._releases = {}
        # program -> retries left for releases deferred by a pause or a failure.
        self._deferred = {}
        self._deferred_flush = None
        self._closing = {}
        self._closed = OrderedDict()

    async def create_session(self, session_id, **kwargs):
        if session_id in self._closing or session_id in self._closed:
            raise ValueError("Use a fresh session ID for every trajectory")
        return await super().create_session(session_id, **kwargs)

    async def _handle_request(self, handler, session_id, payload):
        if session_id in self._closing or session_id in self._closed:
            raise HTTPException(status_code=409, detail="Session is closing")
        # off: one program per session, released at close
        identity = None if self._subagent_programs == "off" else _REQUEST_IDENTITY.get()
        program_id, parent_program_id = _program_of(session_id, identity)
        inflight = self._requests.setdefault(session_id, {})
        if program_id in inflight:
            raise HTTPException(status_code=409, detail="Dynamo Gateway requires serial calls within one program")
        inflight[program_id] = asyncio.current_task()
        try:
            self._programs.setdefault(session_id, {}).setdefault(program_id, parent_program_id)
            try:
                await self._release_returned_children(session_id, program_id)
            except Exception:
                pass  # logged by _release_program; the child stays in the tree for the close path
            # own task: a cancelled HTTP waiter must not abandon it; closing the session cancels it
            request = asyncio.create_task(handler(session_id=session_id, payload=payload))
            inflight[program_id] = request
            response = await _wait_for_cleanup(request)
            if parent_program_id is not None:
                _observe_body(response, lambda body: self._release_if_returned(session_id, program_id, body))
            return response
        finally:
            inflight.pop(program_id, None)
            if not inflight:
                self._requests.pop(session_id, None)

    async def _handle_openai_chat_completions(self, *, session_id, payload):
        return await self._handle_request(super()._handle_openai_chat_completions, session_id, payload)

    async def _handle_anthropic_messages(self, *, session_id, payload):
        return await self._handle_request(super()._handle_anthropic_messages, session_id, payload)

    async def _release_program(self, program_id):
        for attempt in range(self._finalize_attempts):
            try:
                await asyncio.wait_for(self._backend.finalize_program(program_id), self._finalize_timeout)
                logger.info("ThunderAgent program finalized: program=%s", program_id)
                return True
            except GenerationPaused:
                # engines paused by abort_all_requests: hand the trajectories back, release after resume
                self._defer_release(program_id)
                return False
            except Exception:
                if attempt + 1 == self._finalize_attempts:
                    logger.exception("ThunderAgent program cleanup failed: program=%s", program_id)
                    raise
                await asyncio.sleep(self._retry_delay * 2**attempt)

    def _defer_release(self, program_id, *, failed=False):
        if failed:
            left = self._deferred.get(program_id, self._deferred_retries) - 1
            if left <= 0:
                self._deferred.pop(program_id, None)
                logger.warning("ThunderAgent program leaked: release gave up after retries: program=%s", program_id)
                return
            logger.warning("ThunderAgent program release retried in the background: program=%s", program_id)
            self._deferred[program_id] = left
        elif program_id not in self._deferred:
            logger.info("ThunderAgent program release deferred until generation resumes: program=%s", program_id)
            self._deferred[program_id] = self._deferred_retries
        if self._deferred_flush is None or self._deferred_flush.done():
            self._deferred_flush = asyncio.create_task(self._flush_deferred())

    async def _flush_deferred(self):
        while self._deferred:
            await asyncio.sleep(self._deferred_poll)
            for program_id in list(self._deferred):
                try:
                    released = await self._release_program(program_id)
                except Exception:
                    self._defer_release(program_id, failed=True)
                    continue
                if not released:
                    break  # still paused
                self._deferred.pop(program_id, None)

    async def _release_returned_children(self, session_id, program_id):
        """Release the idle children of a program taking a new turn (it blocked until they returned).
        A child still mid-request is kept. Every child is attempted; the first failure is raised last."""
        tree = self._programs.get(session_id, {})
        failures = []
        for child in [child for child, parent in tree.items() if parent == program_id]:
            # re-checked per child: a release yields, and another path may have taken it
            if child not in tree or child in self._requests.get(session_id, {}):
                continue
            try:
                await self._release_tree_member(session_id, child)
            except Exception as error:
                failures.append(error)
        if failures:
            raise failures[0]

    def _release_if_returned(self, session_id, program_id, body):
        """Release a subagent as soon as its final answer is produced."""
        if not _turn_returned(body):
            return
        task = asyncio.create_task(self._release_subagent(session_id, program_id))
        pending = self._releases.setdefault(session_id, set())
        pending.add(task)
        task.add_done_callback(pending.discard)

    async def _release_subagent(self, session_id, program_id):
        if program_id not in self._programs.get(session_id, {}) or program_id in self._requests.get(session_id, {}):
            return
        try:
            await self._release_tree_member(session_id, program_id)
        except Exception:
            return  # logged by _release_program; the close path retries it

    async def _release_tree_member(self, session_id, program_id):
        """Release a program and its idle children. It leaves the tree before the await, so no other
        path finalizes it twice; a failed or cancelled release puts it back for the close path."""
        tree = self._programs.get(session_id, {})
        if program_id not in tree:
            return  # another release path took it out first and owns it
        parent = tree.pop(program_id)
        try:
            await self._release_returned_children(session_id, program_id)
            await self._release_program(program_id)
        except BaseException:
            tree[program_id] = parent
            raise

    def _session_programs(self, session_id):
        """Every program the session created, children before the root that dispatched them."""
        programs = list(self._programs.get(session_id, {}))
        return list(reversed(programs)) if programs else [session_id]

    async def _close_session(self, session_id, abort):
        requests = list(self._requests.get(session_id, {}).values())
        if requests:
            pending = set(requests)
            if not abort:
                # The agent has exited; a turn that is still finishing is drained briefly.
                _, pending = await asyncio.wait(requests, timeout=self._drain_timeout)
            if pending:
                # nobody will consume these turns; cancelling reaches the frontend, which rolls them back
                for request in pending:
                    request.cancel()
                logger.warning("Session %s closed with %d turn(s) in flight; cancelled", session_id, len(pending))
                _, pending = await asyncio.wait(pending, timeout=self._drain_timeout)
            if pending:
                self._closing.pop(session_id, None)  # a later close retries once the turn has ended
                raise TimeoutError(f"Session {session_id} still has an active generation; program was not released")
        if pending := self._releases.pop(session_id, None):
            await asyncio.gather(*pending, return_exceptions=True)
        # The same actor owns the client's served-frontend records. A driver
        # copy of the client would not know every frontend visited on retries.
        # a failed release must not strand the rest of the tree or fail the close (the
        # trajectories are complete); it is retried in the background
        for program_id in self._session_programs(session_id):
            try:
                await self._release_program(program_id)
            except Exception:
                self._defer_release(program_id, failed=True)  # logged by _release_program
        self._programs.pop(session_id, None)
        if abort:
            await super().abort_session(session_id)
            result = None
        else:
            result = await super().finalize_session(session_id)
        self._closed[session_id] = result
        self._closed.move_to_end(session_id)
        while len(self._closed) > 64:
            self._closed.popitem(last=False)
        self._closing.pop(session_id, None)
        return result

    async def _finish(self, session_id, *, abort):
        if session_id in self._closed:
            return None if abort else self._closed[session_id]
        if session_id not in self._sessions and session_id not in self._closing:
            if abort:
                return None
            raise KeyError(session_id)
        task = self._closing.get(session_id)
        if task is None:
            task = asyncio.create_task(self._close_session(session_id, abort))
            self._closing[session_id] = task
        # Reuse the task and its bounded retry budget, including failures.
        return await _wait_for_cleanup(task)

    async def finalize_session(self, session_id):
        return await self._finish(session_id, abort=False)

    async def abort_session(self, session_id):
        await self._finish(session_id, abort=True)

    async def shutdown(self):
        try:
            results = await asyncio.gather(
                *(self.abort_session(session_id) for session_id in tuple(self._sessions)), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            if self._deferred_flush is not None:
                self._deferred_flush.cancel()
            if self._deferred:
                logger.warning("%d ThunderAgent program release(s) still deferred at shutdown", len(self._deferred))
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
