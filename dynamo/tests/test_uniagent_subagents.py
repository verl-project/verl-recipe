"""Uni-Agent AgentContext through real routes and Dynamo program lifecycle."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from recipe.dynamo.tests.test_uniagent_gateway import Tokenizer
from recipe.dynamo.uniagent_gateway import _DynamoGatewayActor
from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.config import GatewayActorConfig

from verl.workers.rollout.replica import TokenOutput


class Backend:
    _auto_finalize = False

    def __init__(self):
        self.calls = []
        self.finalize_program = AsyncMock()
        self.gate = None

    async def generate(self, request_id, **kwargs):
        self.calls.append(dict(request_id=request_id, **kwargs))
        if self.gate:
            await self.gate.wait()
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason="completed")


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    backend = Backend()
    actor = _DynamoGatewayActor(GatewayActorConfig(tokenizer=Tokenizer()), backend)
    actor._server_base_url = "http://test"
    actor._retry_delay = 0
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=actor._app), base_url="http://test")
    return actor, backend, client


async def turn(client, agent=None, parent=None, messages=None, root="cc-root", outer="rollout"):
    headers = {"x-claude-code-session-id": root} if root else {}
    if agent:
        headers["x-claude-code-agent-id"] = agent
    if parent:
        headers["x-claude-code-parent-agent-id"] = parent
    return await client.post(
        f"/sessions/{outer}/v1/messages",
        headers=headers,
        json={
            "model": "test",
            "max_tokens": 10,
            "system": "identical system prompt",
            "messages": messages or [{"role": "user", "content": "inspect"}],
        },
    )


async def stop(client, agent, outer="rollout", event_id=None):
    return await client.post(
        f"/sessions/{outer}/agent_events",
        json={"hook_event_name": "SubagentStop", "session_id": "cc-root", "agent_id": agent, "event_id": event_id},
    )


@pytest.mark.asyncio
async def test_real_ids_isolate_identical_siblings_nested_and_rl_ownership(setup):
    actor, backend, client = setup
    await actor.create_session("rollout", sampling_params={"logprobs": True})
    async with client:
        for agent, parent in [(None, None), ("child1", None), ("child2", None), ("nested", "child1")]:
            response = await turn(client, agent, parent)
            assert response.status_code == 200, response.text
        assert len({c["program_id"] for c in backend.calls}) == 4
        assert backend.calls[-1]["parent_program_id"] == "rollout/agent/child1"
        assert (await stop(client, "child1")).status_code == 200
        backend.finalize_program.assert_awaited_once_with("rollout/agent/child1")
        trajectories = await actor.finalize_session("rollout")
        assert len(trajectories) == 4
        assert len({t.chain_id for t in trajectories}) == 4
        assert {call["request_id"] for call in backend.calls} == {"rollout"}
        assert len({t.extra_fields["agent_context"]["agent_id"] for t in trajectories}) == 4
        assert trajectories[-1].extra_fields["agent_context"]["agent_id"] is None
        # The framework attaches TaskResult after finalization.
        assert all(t.reward_score is None and t.finished is None for t in trajectories)
        assert sum(sum(t.response_mask) for t in trajectories) == 4
        assert all(len(t.response_ids) == len(t.response_mask) == len(t.response_logprobs) for t in trajectories)
        assert backend.finalize_program.await_count == 4
        # Exercise the unchanged Uni-Agent training serializer: all agent chains
        # share the prompt uid and rollout index, while retaining their TA IDs.
        framework = object.__new__(GatewayAgentFramework)
        framework._processor = None
        framework._mask_unfinished_episode = False
        for trajectory in trajectories:
            field, tag = framework._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields={"uid": "task-uid"},
                session_index=2,
                global_steps=3,
                uid="task-uid",
            )
            assert field["uid"] == tag["uid"] == "task-uid"
            assert field["session_id"] == 2
        assert await actor.finalize_session("rollout") is trajectories


@pytest.mark.asyncio
async def test_resume_retains_identity_context_and_refinalizes_only_new_epoch(setup):
    actor, backend, client = setup
    await actor.create_session("rollout", sampling_params={"logprobs": True})
    async with client:
        first = await turn(client, "child")
        assert first.status_code == 200
        assert (await stop(client, "child", event_id="stop-1")).status_code == 200
        assert (await stop(client, "child", event_id="stop-1")).json()["status"] == "duplicate"
        assert backend.finalize_program.await_count == 1
        messages = [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": first.json()["content"]},
            {"role": "user", "content": "continue"},
        ]
        assert (await turn(client, "child", messages=messages)).status_code == 200
        # A delayed retry of the earlier stop cannot close the resumed agent.
        assert (await stop(client, "child", event_id="stop-1")).json()["status"] == "duplicate"
        assert backend.finalize_program.await_count == 1
        trajectories = await actor.finalize_session("rollout")
        assert len(trajectories) == 1
        assert sum(trajectories[0].response_mask) == 2
        assert {c["program_id"] for c in backend.calls} == {"rollout/agent/child"}
        assert backend.finalize_program.await_count == 2


@pytest.mark.asyncio
async def test_children_overlap_while_same_agent_serializes_and_stop_drains(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    backend.gate = asyncio.Event()
    async with client:
        calls = [asyncio.create_task(turn(client, child)) for child in ("a", "b", "a")]
        for _ in range(100):
            if len(backend.calls) == 2:
                break
            await asyncio.sleep(0.005)
        assert len(backend.calls) == 2
        stopping = asyncio.create_task(stop(client, "a"))
        await asyncio.sleep(0.02)
        backend.finalize_program.assert_not_awaited()
        backend.gate.set()
        assert all(r.status_code == 200 for r in await asyncio.gather(*calls))
        assert (await stopping).status_code == 200
        assert len(backend.calls) == 3
        backend.finalize_program.assert_not_awaited()  # The queued newer turn superseded that stop.
        await actor.abort_session("rollout")
        assert {c.args[0] for c in backend.finalize_program.await_args_list} == {"rollout/agent/a", "rollout/agent/b"}


@pytest.mark.asyncio
async def test_abort_cancels_generation_without_waiting_for_backend_gate(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    backend.gate = asyncio.Event()
    async with client:
        request = asyncio.create_task(turn(client, "child"))
        while not backend.calls:
            await asyncio.sleep(0.005)
        request.cancel()
        abort = asyncio.create_task(actor.abort_session("rollout"))
        await asyncio.wait_for(abort, 1)
        assert not backend.gate.is_set()
        await asyncio.gather(request, return_exceptions=True)
        backend.finalize_program.assert_awaited_once_with("rollout/agent/child")
        assert not actor._sessions


@pytest.mark.asyncio
async def test_identity_failures_are_visible_and_no_fingerprint_fallback(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    async with client:
        assert (await turn(client, "child")).status_code == 200
        assert (await turn(client, "child", root="other")).status_code == 409
        assert (await turn(client, "child", parent="other-parent")).status_code == 409
        assert (await turn(client, "child", parent="child")).status_code == 400
        await actor.abort_session("rollout")


@pytest.mark.asyncio
async def test_cleanup_failure_is_surfaced_and_outer_abort_retries_all_lanes(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    async with client:
        await turn(client, "a")
        await turn(client, "b")
        backend.finalize_program.side_effect = RuntimeError("unreachable")
        with pytest.raises(RuntimeError, match="unreachable"):
            await stop(client, "a")
        assert actor._programs["rollout"]["rollout/agent/a"].cleanup_error
        backend.finalize_program.side_effect = None
        await actor.abort_session("rollout")
        assert not actor._sessions


@pytest.mark.asyncio
async def test_same_agent_ids_in_separate_rollouts_never_share_programs(setup):
    actor, backend, client = setup
    for outer in ("one", "two"):
        await actor.create_session(outer)
    async with client:
        await turn(client, "child", outer="one")
        await turn(client, "child", outer="two")
        assert len({c["program_id"] for c in backend.calls}) == 2
        await actor.abort_session("one")
        await actor.abort_session("two")


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["messages", "chat/completions", "responses"])
async def test_upstream_context_routes_and_escaped_program_ids(setup, endpoint):
    from uni_agent.gateway.session import GatewaySession

    actor, backend, client = setup
    await actor.create_session("rollout")
    assert type(actor._sessions["rollout"]) is GatewaySession
    payload = {"model": "test"}
    payload["input" if endpoint == "responses" else "messages"] = [{"role": "user", "content": "inspect"}]
    if endpoint == "messages":
        payload["max_tokens"] = 8
    async with client:
        response = await client.post(
            f"/sessions/rollout/v1/{endpoint}",
            json=payload,
            headers={"x-claude-code-session-id": "cc-root", "x-claude-code-agent-id": "a/b"},
        )
        assert response.status_code == 200, response.text
        assert backend.calls[0]["program_id"] == "rollout/agent/a%2Fb"
        assert backend.calls[0]["parent_program_id"] == "rollout/main"
        trajectories = await actor.finalize_session("rollout")
        assert trajectories[0].extra_fields["agent_context"]["agent_id"] == "a/b"
        assert trajectories[0].extra_fields["agent_context"]["parent_agent_id"] is None


@pytest.mark.asyncio
async def test_paused_cleanup_returns_trajectories_and_retries_after_resume(setup):
    from recipe.dynamo.dynamo_agent_loop import GenerationPaused

    actor, backend, client = setup
    actor._deferred_poll = 0.01
    await actor.create_session("rollout")
    resumed = False

    async def finalize(program_id):
        if not resumed:
            raise GenerationPaused(program_id)

    backend.finalize_program.side_effect = finalize
    async with client:
        await turn(client)
        await turn(client, "child")
        trajectories = await asyncio.wait_for(actor.finalize_session("rollout"), 1)
        assert len(trajectories) == 2
        assert not actor._sessions and actor._deferred
        resumed = True
        await asyncio.wait_for(actor._deferred_flush, 1)
        assert not actor._deferred and not actor._programs
        assert await actor.finalize_session("rollout") is trajectories


@pytest.mark.asyncio
async def test_resume_waits_for_running_release_and_closes_its_new_epoch(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    started, released = asyncio.Event(), asyncio.Event()

    async def finalize(_):
        started.set()
        await released.wait()

    backend.finalize_program.side_effect = finalize
    async with client:
        await turn(client, "child")
        stopping = asyncio.create_task(stop(client, "child", event_id="first-stop"))
        await started.wait()
        resumed = asyncio.create_task(turn(client, "child"))
        await asyncio.sleep(0.02)
        assert len(backend.calls) == 1
        released.set()
        assert (await stopping).status_code == (await resumed).status_code == 200
        assert len(backend.calls) == 2
        assert (await stop(client, "child", event_id="first-stop")).json()["status"] == "duplicate"
        assert backend.finalize_program.await_count == 1
        await actor.finalize_session("rollout")
        assert backend.finalize_program.await_count == 2


@pytest.mark.asyncio
async def test_resumed_agent_supersedes_a_paused_stop(setup):
    from recipe.dynamo.dynamo_agent_loop import GenerationPaused

    actor, backend, client = setup
    actor._deferred_poll = 60
    await actor.create_session("rollout")
    async with client:
        await turn(client, "child")
        backend.finalize_program.side_effect = GenerationPaused("paused")
        await stop(client, "child", event_id="paused-stop")
        assert actor._deferred
        backend.finalize_program.side_effect = None
        assert (await turn(client, "child")).status_code == 200
        assert not actor._deferred
        assert backend.finalize_program.await_count == 1
        assert (await stop(client, "child", event_id="paused-stop")).json()["status"] == "duplicate"
        await actor.finalize_session("rollout")
        assert backend.finalize_program.await_count == 2
        await actor.shutdown()


@pytest.mark.asyncio
async def test_exhausted_deferred_cleanup_is_retained_for_explicit_retry(setup):
    from recipe.dynamo.dynamo_agent_loop import GenerationPaused

    actor, backend, client = setup
    actor._deferred_poll = 0.01
    actor._finalize_attempts = 1
    await actor.create_session("rollout")
    async with client:
        await turn(client, "child")
        backend.finalize_program.side_effect = GenerationPaused("paused")
        trajectories = await actor.finalize_session("rollout")
        backend.finalize_program.side_effect = RuntimeError("unreachable")
        await asyncio.wait_for(actor._deferred_flush, 1)
        assert list(actor._deferred.values()) == [(1, 0)]
        assert backend.finalize_program.await_count == 1 + actor._deferred_attempts
        backend.finalize_program.side_effect = None
        assert await actor.finalize_session("rollout") is trajectories
        assert not actor._programs and not actor._deferred


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tree", "isolated", "off"])
async def test_ablation_changes_routing_without_merging_agent_trajectories(setup, mode):
    actor, backend, client = setup
    actor._subagent_programs = mode
    await actor.create_session("rollout")
    async with client:
        for agent, parent in [(None, None), ("a", None), ("b", "a")]:
            assert (await turn(client, agent, parent)).status_code == 200
        expected = (
            [("rollout", None)] * 3
            if mode == "off"
            else [
                ("rollout/main", None),
                ("rollout/agent/a", "rollout/main" if mode == "tree" else None),
                ("rollout/agent/b", "rollout/agent/a" if mode == "tree" else None),
            ]
        )
        assert [(c["program_id"], c["parent_program_id"]) for c in backend.calls] == expected
        trajectories = await actor.finalize_session("rollout")
        assert len(trajectories) == 3
        assert {t.extra_fields["agent_context"]["agent_id"] for t in trajectories} == {None, "a", "b"}
        assert backend.finalize_program.await_count == (1 if mode == "off" else 3)


@pytest.mark.asyncio
async def test_resumed_program_does_not_wait_for_unrelated_deferred_cleanup(setup):
    from recipe.dynamo.dynamo_agent_loop import GenerationPaused

    actor, backend, client = setup
    actor._deferred_poll = 0.01
    for outer in ("rollout", "other"):
        await actor.create_session(outer)
    started, release = asyncio.Event(), asyncio.Event()
    pause_other = True

    async def finalize(program_id):
        if program_id == "rollout/agent/a":
            started.set()
            await release.wait()
        elif pause_other:
            raise GenerationPaused(program_id)

    async with client:
        await turn(client, "a")
        await turn(client, "b", outer="other")
        backend.finalize_program.side_effect = GenerationPaused("paused")
        await stop(client, "a", event_id="paused-stop")
        await stop(client, "b", outer="other")
        backend.finalize_program.side_effect = finalize
        await asyncio.wait_for(started.wait(), 1)
        resumed = asyncio.create_task(turn(client, "a"))
        await asyncio.sleep(0.01)
        assert not resumed.done()
        release.set()
        assert (await asyncio.wait_for(resumed, 1)).status_code == 200
        assert ("other", "other/agent/b") in actor._deferred
        pause_other = False
        await asyncio.wait_for(actor._deferred_flush, 1)
        await actor.finalize_session("rollout")
        await actor.finalize_session("other")


@pytest.mark.asyncio
async def test_parent_next_turn_does_not_release_an_idle_subagent(setup):
    actor, backend, client = setup
    await actor.create_session("rollout")
    async with client:
        await turn(client)
        await turn(client, "a")
        await turn(client, messages=[{"role": "user", "content": "continue"}])
        backend.finalize_program.assert_not_awaited()
        await stop(client, "a", event_id="actual-stop")
        backend.finalize_program.assert_awaited_once_with("rollout/agent/a")
        await actor.finalize_session("rollout")
