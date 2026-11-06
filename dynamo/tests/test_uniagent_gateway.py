"""Exercise the pinned Uni-Agent Gateway with the recipe lifecycle adapter.

Requires the Uni-Agent pin in dynamo/requirements-uniagent.txt and verl v0.9.1.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("uni_agent.gateway.gateway")

from recipe.dynamo.uniagent_gateway import DynamoAgentFrameworkRolloutAdapter, _DynamoGatewayActor
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor

from verl.workers.rollout.replica import TokenOutput


class Tokenizer:
    eos_token_id = ord("\n")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"{m['role']}:{m.get('content') or ''}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(c) for c in text] if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(int(token)) for token in token_ids)


class Backend:
    _auto_finalize = False

    def __init__(self):
        self.ids = []
        self.finalize_program = AsyncMock()

    async def generate(self, request_id, **kwargs):
        self.ids.append(request_id)
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason="completed")


@pytest.fixture
def make_actor(monkeypatch):
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")

    def make(cls=_DynamoGatewayActor, **kwargs):
        backend = Backend()
        actor = cls(GatewayActorConfig(tokenizer=Tokenizer()), backend, **kwargs)
        actor._server_base_url = "http://test"
        actor._retry_delay = 0
        return actor, backend

    return make


async def create(actor, session="session"):
    return await actor.create_session(session, sampling_params={"logprobs": True})


async def generate(actor, session="session", messages=None):
    return await actor._handle_openai_chat_completions(
        session_id=session,
        payload={"model": "test", "messages": messages or [{"role": "user", "content": "question"}]},
    )


@pytest.mark.asyncio
async def test_upstream_session_close_does_not_release_thunderagent(make_actor):
    # Regression evidence: a plain upstream upgrade alone cannot remove cleanup.
    actor, backend = make_actor(_GatewayActor)
    await create(actor)
    await generate(actor)
    await actor.finalize_session("session")
    backend.finalize_program.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_turns_share_program_and_cleanup_preserves_tokens(make_actor):
    actor, backend = make_actor()
    await create(actor)
    first = await generate(actor)
    assistant = json.loads(first.body)["choices"][0]["message"]
    await generate(
        actor, messages=[{"role": "user", "content": "question"}, assistant, {"role": "user", "content": "continue"}]
    )
    backend.finalize_program.assert_not_awaited()
    trajectories = await actor.finalize_session("session")
    assert backend.ids == ["session", "session"]
    backend.finalize_program.assert_awaited_once_with("session")
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    # Uni-Agent counts conversation messages in num_turns; backend.ids above
    # establishes the actual number of model calls.
    assert sum(trajectory.response_mask) == 2
    assert len(trajectory.response_ids) == len(trajectory.response_mask) == len(trajectory.response_logprobs)
    assert not actor._sessions
    assert await actor.finalize_session("session") is trajectories
    await actor.abort_session("session")
    assert backend.finalize_program.await_count == 1


@pytest.mark.asyncio
async def test_transient_cleanup_failure_retries(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = [RuntimeError("busy"), TimeoutError(), None]
    await actor.finalize_session("session")
    assert backend.finalize_program.await_count == 3
    assert not actor._sessions


@pytest.mark.asyncio
async def test_repeated_waiter_cancellation_waits_for_cleanup(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await generate(actor)
    started, release = asyncio.Event(), asyncio.Event()

    async def finalize(_):
        started.set()
        await release.wait()

    backend.finalize_program.side_effect = finalize
    task = asyncio.create_task(actor.finalize_session("session"))
    await started.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    assert "session" in actor._sessions
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "session" not in actor._sessions
    assert backend.finalize_program.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_close_drains_or_cancels_the_complete_request(make_actor, abort):
    actor, backend = make_actor()
    await create(actor)
    started, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        try:
            await release.wait()
            return await original(*args, **kwargs)
        finally:
            stopped.set()

    async def finalized(_):
        assert stopped.is_set(), "program released while generation is still running"

    backend.generate = pending
    backend.finalize_program.side_effect = finalized
    request = asyncio.create_task(generate(actor))
    await started.wait()
    task = asyncio.create_task((actor.abort_session if abort else actor.finalize_session)("session"))
    if abort:
        await asyncio.wait_for(task, 1)
        assert stopped.is_set()
        assert not release.is_set()
    else:
        await asyncio.sleep(0)
        backend.finalize_program.assert_not_awaited()
        assert not stopped.is_set()
        release.set()
        await task
    result = await asyncio.gather(request, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError) == abort
    assert not actor._sessions
    backend.finalize_program.assert_awaited_once_with("session")


@pytest.mark.asyncio
async def test_drain_timeout_keeps_live_program_and_allows_later_abort(make_actor):
    actor, backend = make_actor()
    actor._drain_timeout = 0.01
    await create(actor)
    started, release = asyncio.Event(), asyncio.Event()

    async def uncooperative(**kwargs):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason="completed")

    backend.generate = uncooperative
    request = asyncio.create_task(generate(actor))
    await started.wait()
    with pytest.raises(TimeoutError, match="still has an active generation"):
        await actor.abort_session("session")
    backend.finalize_program.assert_not_awaited()
    assert not request.done()
    release.set()
    await request
    await actor.abort_session("session")
    backend.finalize_program.assert_awaited_once_with("session")
    assert not actor._sessions


@pytest.mark.asyncio
async def test_independent_sessions_overlap_and_same_program_serializes(make_actor):
    actor, backend = make_actor()
    for name in ("one", "two"):
        await create(actor, name)
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate
    calls = []

    async def pending(*args, **kwargs):
        calls.append(kwargs["request_id"])
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    requests = [asyncio.create_task(generate(actor, name)) for name in ("one", "two", "one")]
    while len(calls) < 2:
        await asyncio.sleep(0)
    assert calls == ["one", "two"]
    release.set()
    await asyncio.gather(*requests)
    assert calls == ["one", "two", "one"]
    await asyncio.gather(actor.finalize_session("one"), actor.finalize_session("two"))
    assert backend.finalize_program.await_count == 2


@pytest.mark.asyncio
async def test_anthropic_requests_share_the_lifecycle(make_actor):
    actor, backend = make_actor()
    await create(actor)
    await actor._handle_anthropic_messages(
        session_id="session",
        payload={
            "model": "test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "question"}],
        },
    )
    await actor.finalize_session("session")
    backend.finalize_program.assert_awaited_once_with("session")


@pytest.mark.asyncio
async def test_new_verl_routing_fields_reach_parent_client(monkeypatch):
    from collections import OrderedDict

    from recipe.dynamo.dynamo_agent_loop import _REQUEST_PROGRAM, DynamoFullyAsyncLLMServerClient

    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    handle = object()
    acquire = AsyncMock(return_value=("worker", handle))
    monkeypatch.setattr(FullyAsyncLLMServerClient, "_acquire_server", acquire)
    client = object.__new__(DynamoFullyAsyncLLMServerClient)
    client._session_servers = OrderedDict()
    token = _REQUEST_PROGRAM.set("session")
    try:
        server_id, wrapper = await client._acquire_server("request", prompt_ids=[1, 2])
    finally:
        _REQUEST_PROGRAM.reset(token)
    assert (server_id, wrapper._handle) == ("worker", handle)
    acquire.assert_awaited_once_with("request", prompt_ids=[1, 2])
    assert client._session_servers["session"]["worker"] is handle


@pytest.mark.asyncio
async def test_concurrent_siblings_record_their_own_frontends_under_one_routing_key(monkeypatch):
    from collections import OrderedDict

    from recipe.dynamo.dynamo_agent_loop import DynamoFullyAsyncLLMServerClient

    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    handles = [object(), object()]
    keys = []

    async def acquire(self, request_id, **kwargs):
        index = len(keys)
        keys.append(request_id)
        await asyncio.sleep(0)
        return f"worker-{index}", handles[index]

    async def generate(self, request_id, **kwargs):
        await self._acquire_server(request_id)
        await asyncio.sleep(0)
        return TokenOutput(token_ids=[1])

    monkeypatch.setattr(FullyAsyncLLMServerClient, "_acquire_server", acquire)
    monkeypatch.setattr(FullyAsyncLLMServerClient, "generate", generate)
    client = object.__new__(DynamoFullyAsyncLLMServerClient)
    client._session_servers = OrderedDict()
    client._session_routing_keys = OrderedDict()
    client._auto_finalize = False
    await asyncio.gather(
        client.generate("rollout", thunderagent_session_id="rollout/agent/a"),
        client.generate("rollout", thunderagent_session_id="rollout/agent/b"),
    )
    assert keys == ["rollout", "rollout"]
    assert client._session_servers == {
        "rollout/agent/a": {"worker-0": handles[0]},
        "rollout/agent/b": {"worker-1": handles[1]},
    }


@pytest.mark.asyncio
async def test_ray_generation_cancellation_is_acknowledged_before_cleanup():
    import ray
    from recipe.dynamo.dynamo_agent_loop import _await_or_cancel

    @ray.remote
    class Probe:
        def __init__(self):
            self.started = self.stopped = False

        async def generate(self):
            self.started = True
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.02)
                self.stopped = True

        async def state(self):
            return self.started, self.stopped

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    probe = Probe.remote()
    try:
        task = asyncio.create_task(_await_or_cancel(probe.generate.remote()))
        for _ in range(500):
            if (await probe.state.remote())[0]:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Ray generation did not start")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert await probe.state.remote() == (True, True)
    finally:
        ray.kill(probe)
        if owns_ray:
            ray.shutdown()


def _adapter_config(enabled: bool, auto_finalize: bool, engine: str = "vllm"):
    from omegaconf import OmegaConf

    thunderagent = {"enabled": enabled, "auto_finalize": auto_finalize}
    dynamo = {"engine": engine, "thunderagent": thunderagent}
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"engine_kwargs": {"dynamo": dynamo}}}})


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_adapter_builds_plain_gateways_for_each_engine_without_thunderagent(monkeypatch, engine) -> None:
    built = {}

    def build(config, llm_client, *, thunderagent, engine):
        built.update(thunderagent=thunderagent, engine=engine)
        return "manager"

    worker = type("W", (), {"remote": staticmethod(lambda **_: "worker")})
    monkeypatch.setattr("recipe.dynamo.uniagent_gateway._build_gateway_manager", build)
    monkeypatch.setattr("recipe.dynamo.uniagent_gateway.AgentFrameworkWorker", worker)

    config = _adapter_config(False, True, engine)
    adapter = DynamoAgentFrameworkRolloutAdapter.create(config=config, llm_client=object())

    assert built == {"thunderagent": False, "engine": engine}
    assert adapter.gateway_manager == "manager" and adapter.framework_worker == "worker"


def test_adapter_rejects_an_unknown_engine() -> None:
    with pytest.raises(ValueError, match="vllm and sglang"):
        DynamoAgentFrameworkRolloutAdapter.create(config=_adapter_config(False, True, "trtllm"), llm_client=object())


def test_adapter_requires_manual_finalize_with_thunderagent() -> None:
    with pytest.raises(ValueError, match="auto_finalize=false"):
        DynamoAgentFrameworkRolloutAdapter.create(config=_adapter_config(True, True), llm_client=object())


@pytest.mark.asyncio
async def test_framework_returns_completed_trajectory_when_cleanup_fails(make_actor):
    from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig
    from uni_agent.tasks import TaskResult

    actor, backend = make_actor()
    actor._deferred_poll = 0.01
    backend.finalize_program.side_effect = [RuntimeError("unreachable")] * 3 + [None]
    framework = object.__new__(GatewayAgentFramework)
    framework._log_dir = None
    framework.gateway_manager = actor
    framework._trajectory_postprocessor = None
    framework.reward_loop_worker_handles = None

    async def runner(*, session, **kwargs):
        await generate(actor, session.session_id)
        return TaskResult(reward=0.75, finished=True)

    framework._inline_runners = {"probe": runner}
    trajectories, _ = await framework._run_agent_episode(
        sample_fields={"raw_prompt": [{"role": "user", "content": "question"}], "uid": "task"},
        sample_index=0,
        session_index=0,
        global_steps=1,
        runner_name="probe",
        runner_config=_RunnerConfig("probe", {}, "inline_async", 0),
        sampling_params={"logprobs": True},
    )
    assert len(trajectories) == 1 and sum(trajectories[0].response_mask) == 1
    assert trajectories[0].reward_score == 0.75 and trajectories[0].finished
    assert not actor._sessions and actor._deferred
    await asyncio.wait_for(actor._deferred_flush, 1)
    assert not actor._programs and not actor._deferred
    assert backend.finalize_program.await_count == 4


@pytest.mark.asyncio
async def test_failed_cleanup_retains_samples_and_can_retry_after_budget_exhaustion(make_actor):
    actor, backend = make_actor()
    actor._deferred_poll = 0.01
    actor._finalize_attempts = 1
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = RuntimeError("unreachable")
    trajectories = await actor.finalize_session("session")
    await asyncio.wait_for(actor._deferred_flush, 1)
    assert len(trajectories) == 1
    assert list(actor._deferred.values()) == [(1, 0)]
    assert backend.finalize_program.await_count == 1 + actor._deferred_attempts
    assert await actor.get_cleanup_state() == {
        "active_programs": 1,
        "pending_cleanup": 1,
        "exhausted_cleanup": 1,
        "running_releases": 0,
    }
    backend.finalize_program.side_effect = None
    assert await actor.finalize_session("session") is trajectories
    assert not actor._deferred and not actor._programs
    assert not any((await actor.get_cleanup_state()).values())


@pytest.mark.asyncio
async def test_cleanup_timeout_returns_samples_and_shutdown_reports_unconfirmed_programs(make_actor):
    actor, backend = make_actor()
    actor._finalize_timeout = 0.01
    actor._deferred_poll = 60
    await create(actor)
    await generate(actor)

    async def hang(_):
        await asyncio.Future()

    backend.finalize_program.side_effect = hang
    trajectories = await asyncio.wait_for(actor.finalize_session("session"), 1)
    assert len(trajectories) == 1 and actor._deferred
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await actor.shutdown()
    backend.finalize_program.side_effect = None
    assert await actor.finalize_session("session") is trajectories
    assert not actor._programs and not actor._deferred


@pytest.mark.asyncio
async def test_finalize_timeout_waits_for_rpc_cancellation_before_releasing_lock():
    from types import SimpleNamespace

    from recipe.dynamo.dynamo_agent_loop import DynamoFullyAsyncLLMServerClient

    cancelled, acknowledged = asyncio.Event(), asyncio.Event()

    async def remote(**kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
            await acknowledged.wait()

    client = object.__new__(DynamoFullyAsyncLLMServerClient)
    client._finalize_timeout_s = 0.01
    handle = SimpleNamespace(finalize_program=SimpleNamespace(remote=remote))
    finalize = asyncio.create_task(client._finalize_on_handle(handle, "program"))
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not finalize.done()
    acknowledged.set()
    with pytest.raises(TimeoutError):
        await finalize


@pytest.mark.asyncio
async def test_transport_error_is_not_hidden_by_another_frontends_pause():
    from recipe.dynamo.dynamo_agent_loop import DynamoFullyAsyncLLMServerClient, GenerationPaused

    client = object.__new__(DynamoFullyAsyncLLMServerClient)

    async def finalize(handle, program_id):
        if handle == "paused":
            raise GenerationPaused(program_id)
        raise ConnectionError("unreachable")

    client._finalize_on_handle = finalize
    with pytest.raises(ConnectionError, match="unreachable"):
        await client._finalize_on_handles(["paused", "broken"], "program")
