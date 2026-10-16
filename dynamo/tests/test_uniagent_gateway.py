"""Exercise the unmodified Uni-Agent Gateway with the recipe lifecycle adapter.

Requires a Uni-Agent checkout and its verl submodule.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("uni_agent.gateway.gateway")

from fastapi import HTTPException
from recipe.dynamo.dynamo_agent_loop import GenerationPaused
from recipe.dynamo.uniagent_gateway import DynamoAgentFrameworkRolloutAdapter, _DynamoGatewayActor, _turn_returned
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
        self.programs = []
        self.finalize_program = AsyncMock()
        # "tool_calls" keeps the agent's chain running; "completed" is its final answer.
        self.stop_reason = "completed"

    async def generate(self, request_id, **kwargs):
        self.ids.append(request_id)
        self.programs.append((kwargs.get("program_id"), kwargs.get("parent_program_id")))
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason=self.stop_reason)


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
    assert backend.programs == [("session", None)] * 2  # no identity headers: the session is the program
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
@pytest.mark.parametrize("recovers", [True, False])
async def test_a_failed_release_at_close_keeps_the_trajectories_and_is_retried_in_the_background(
    make_actor, caplog, recovers
):
    # The trajectories are complete when the session closes; a router that cannot release the
    # program must not cost them. The close spends its retry budget (3), then the flusher retries
    # in the background: two more rounds of three before the program is given up as leaked.
    actor, backend = make_actor()
    actor._deferred_poll = 0
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = [RuntimeError("unreachable")] * (3 if recovers else 9) + [None]
    with caplog.at_level(logging.WARNING):
        assert len(await actor.finalize_session("session")) == 1
        await actor._deferred_flush
    assert "session" not in actor._sessions
    assert not actor._deferred
    assert backend.finalize_program.await_count == (4 if recovers else 9)
    assert ("ThunderAgent program leaked" in caplog.text) is not recovers


@pytest.mark.asyncio
async def test_release_is_deferred_while_generation_is_paused(make_actor):
    actor, backend = make_actor()
    actor._deferred_poll = 0
    await create(actor)
    await generate(actor)
    backend.finalize_program.side_effect = [GenerationPaused("session"), GenerationPaused("session"), None]
    trajectories = await actor.finalize_session("session")
    # The trajectories come back at once; the program waits for resume_generation.
    assert len(trajectories) == 1
    assert "session" not in actor._sessions
    assert set(actor._deferred) == {"session"}
    await actor._deferred_flush
    assert not actor._deferred
    assert backend.finalize_program.await_count == 3


@pytest.mark.asyncio
async def test_cleanup_timeout_is_bounded(make_actor):
    actor, backend = make_actor()
    actor._finalize_timeout = 0.01
    await create(actor)

    async def hang(_):
        await asyncio.Future()

    backend.finalize_program.side_effect = hang
    await actor.finalize_session("session")
    assert backend.finalize_program.await_count == 3
    assert "session" not in actor._sessions
    assert set(actor._deferred) == {"session"}  # retried in the background
    actor._deferred_flush.cancel()


@pytest.mark.asyncio
async def test_repeated_waiter_cancellation_waits_for_cleanup(make_actor):
    actor, backend = make_actor()
    await create(actor)
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
async def test_finalize_cancels_a_parked_turn_and_releases_the_program(make_actor):
    """A session whose agent exited mid-turn (timeout) must not leave its program behind."""
    actor, backend = make_actor()
    actor._drain_timeout = 0.01
    await create(actor)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def parked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    backend.generate = parked
    request = asyncio.create_task(generate(actor))
    await started.wait()
    await actor.finalize_session("session")
    assert cancelled.is_set()
    backend.finalize_program.assert_awaited_once_with("session")
    assert "session" not in actor._sessions
    with pytest.raises(asyncio.CancelledError):
        await request


@pytest.mark.asyncio
async def test_close_reports_failure_when_a_turn_survives_cancellation(make_actor):
    actor, backend = make_actor()
    actor._drain_timeout = 0.01
    await create(actor)
    started, release = asyncio.Event(), asyncio.Event()

    async def unstoppable(*args, **kwargs):
        started.set()
        while True:
            try:
                await release.wait()
                break
            except asyncio.CancelledError:
                pass
        return TokenOutput(token_ids=[65], log_probs=[-0.1], stop_reason="completed")

    backend.generate = unstoppable
    request = asyncio.create_task(generate(actor))
    await started.wait()
    with pytest.raises(TimeoutError, match="still has an active generation"):
        await actor.abort_session("session")
    backend.finalize_program.assert_not_awaited()
    assert "session" in actor._sessions
    # Once the turn does finish, closing again releases the program.
    release.set()
    await request
    await actor.abort_session("session")
    backend.finalize_program.assert_awaited_once_with("session")
    assert "session" not in actor._sessions


@pytest.mark.asyncio
async def test_independent_sessions_run_concurrently_but_same_session_is_rejected(make_actor):
    actor, backend = make_actor()
    await create(actor, "one")
    await create(actor, "two")
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    request = asyncio.create_task(generate(actor, "one"))
    await started.wait()
    with pytest.raises(HTTPException) as error:
        await generate(actor, "one")
    assert error.value.status_code == 409
    second = asyncio.create_task(generate(actor, "two"))
    await asyncio.sleep(0)
    assert set(actor._requests) == {"one", "two"}
    release.set()
    await asyncio.gather(request, second)
    await asyncio.gather(actor.finalize_session("one"), actor.finalize_session("two"))
    assert backend.finalize_program.await_count == 2


async def post(actor, session, headers, content="question", stream=False):
    """Drive the real HTTP route, so identity headers pass through the middleware."""
    transport = httpx.ASGITransport(app=actor._app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        return await client.post(
            f"/sessions/{session}/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": content}], "stream": stream},
            headers=headers,
        )


async def settled(actor):
    """Let background releases of returned subagents finish."""
    while any(actor._releases.values()):
        await asyncio.sleep(0)


ROOT = {"x-claude-code-session-id": "run-1"}


def subagent(agent_id):
    return {"x-claude-code-session-id": "run-1", "x-claude-code-agent-id": agent_id}


@pytest.mark.asyncio
async def test_claude_code_subagents_are_programs_of_their_own_and_run_in_parallel(make_actor):
    # Claude Code names the run in x-claude-code-session-id and each subagent in
    # x-claude-code-agent-id; the recipe derives the same tree Dynamo's frontend
    # would. Two subagents of one session overlap, a second turn of the SAME
    # program is still rejected, and the request id stays the routing key.
    actor, backend = make_actor()
    backend.stop_reason = "tool_calls"
    await create(actor)
    await post(actor, "session", ROOT)
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    first = asyncio.create_task(post(actor, "session", subagent("sub-a"), "task a"))
    await started.wait()
    second = asyncio.create_task(post(actor, "session", subagent("sub-b"), "task b"))
    await asyncio.sleep(0)
    assert set(actor._requests["session"]) == {"sub-a", "sub-b"}
    assert (await post(actor, "session", subagent("sub-a"), "task a")).status_code == 409
    release.set()
    assert {r.status_code for r in await asyncio.gather(first, second)} == {200}
    assert backend.ids == ["session"] * 3
    assert backend.programs[0] == ("run-1", None)
    assert sorted(backend.programs[1:]) == [("sub-a", "run-1"), ("sub-b", "run-1")]


@pytest.mark.asyncio
async def test_a_subagent_still_running_survives_the_dispatchers_turn_and_goes_at_close(make_actor):
    actor, backend = make_actor()
    backend.stop_reason = "tool_calls"
    await create(actor)
    await post(actor, "session", ROOT)
    started, release = asyncio.Event(), asyncio.Event()
    original = backend.generate

    async def pending(*args, **kwargs):
        if kwargs.get("program_id") == "sub-b":
            started.set()
            await release.wait()
        return await original(*args, **kwargs)

    backend.generate = pending
    await post(actor, "session", subagent("sub-a"), "task a")
    background = asyncio.create_task(post(actor, "session", subagent("sub-b"), "task b"))
    await started.wait()

    await post(actor, "session", ROOT)  # sub-a returned, sub-b is mid-request
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a"]

    release.set()
    await background
    await actor.finalize_session("session")
    # Children before the root that dispatched them.
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-b", "run-1"]


def test_turn_returned_reads_the_last_finish_reason_in_either_dialect():
    openai_json = b'{"choices":[{"index":0,"finish_reason":"tool_calls"}]}'
    anthropic_json = b'{"stop_reason": "end_turn", "stop_sequence": null}'
    sse = (
        b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    assert _turn_returned(openai_json) is False
    assert _turn_returned(anthropic_json) is True
    assert _turn_returned(sse) is True
    assert _turn_returned(b'{"choices":[{"finish_reason":"length"}]}') is False
    assert _turn_returned(b"{}") is False


@pytest.mark.asyncio
async def test_a_subagent_is_released_the_moment_its_final_answer_is_produced(make_actor):
    # The dispatcher only continues once every subagent has returned; a subagent
    # that already stopped calling tools would otherwise hold admission and
    # resident KV while its siblings finish.
    actor, backend = make_actor()
    await create(actor)
    backend.stop_reason = "tool_calls"
    await post(actor, "session", ROOT)
    await post(actor, "session", subagent("sub-a"), "task a")
    await settled(actor)
    backend.finalize_program.assert_not_awaited()

    backend.stop_reason = "completed"
    await post(actor, "session", subagent("sub-a"), "task a, done")
    await settled(actor)
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a"]
    assert "sub-a" not in actor._programs["session"]

    # The streaming dialect carries the finish reason in its last chunk.
    response = await post(actor, "session", subagent("sub-b"), "task b", stream=True)
    assert response.status_code == 200 and b"[DONE]" in response.content
    await settled(actor)
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-b"]

    await post(actor, "session", ROOT)
    await actor.finalize_session("session")
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-b", "run-1"]


async def session_with_two_subagents(actor, backend):
    """A root turn and two subagent turns, all mid-chain (their last turn called a tool)."""
    backend.stop_reason = "tool_calls"
    await create(actor)
    await post(actor, "session", ROOT)
    await post(actor, "session", subagent("sub-a"), "task a")
    await post(actor, "session", subagent("sub-b"), "task b")


@pytest.mark.asyncio
async def test_a_failed_release_at_close_still_releases_the_rest_of_the_tree(make_actor):
    # An agent that exits mid-chain leaves the whole tree to the close path; one
    # program the router cannot release must not strand its siblings and the root.
    actor, backend = make_actor()
    await session_with_two_subagents(actor, backend)

    async def finalize(program):
        if program == "sub-b":
            raise RuntimeError("unreachable")

    backend.finalize_program.side_effect = finalize
    actor._deferred_poll = 0.2  # the flusher's first retry lands after the close has finished
    assert len(await actor.finalize_session("session")) == 3
    # Children before the root, every program attempted, the retry budget not restarted.
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-b"] * 3 + ["sub-a", "run-1"]
    assert set(actor._deferred) == {"sub-b"}
    backend.finalize_program.side_effect = None
    await actor._deferred_flush
    assert not actor._deferred
    assert [call.args[0] for call in backend.finalize_program.await_args_list][5:] == ["sub-b"]


@pytest.mark.asyncio
async def test_a_failed_subagent_release_does_not_fail_the_dispatchers_turn(make_actor):
    # The dispatcher's next turn releases its idle subagents first; one the router
    # rejects stays in the tree for the close path instead of failing the turn.
    actor, backend = make_actor()
    await session_with_two_subagents(actor, backend)
    unreachable = {"sub-a"}

    async def finalize(program):
        if program in unreachable:
            raise RuntimeError("unreachable")

    backend.finalize_program.side_effect = finalize
    assert (await post(actor, "session", ROOT)).status_code == 200
    assert backend.ids == ["session"] * 4
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a"] * 3 + ["sub-b"]

    unreachable.clear()
    await actor.finalize_session("session")
    assert [call.args[0] for call in backend.finalize_program.await_args_list][4:] == ["sub-a", "run-1"]


@pytest.mark.asyncio
async def test_a_subagent_release_cancelled_by_abort_is_retried_at_close(make_actor):
    # Aborting the session cancels the dispatcher's turn while it is still releasing
    # a subagent; that subagent must go back to the tree, or nothing releases it.
    actor, backend = make_actor()
    await session_with_two_subagents(actor, backend)
    stalled = asyncio.Event()

    async def finalize(program):
        if program == "sub-a" and not stalled.is_set():
            stalled.set()
            await asyncio.Event().wait()  # the frontend has not answered yet

    backend.finalize_program.side_effect = finalize
    turn = asyncio.create_task(post(actor, "session", ROOT))
    await stalled.wait()
    await actor.abort_session("session")
    await asyncio.gather(turn, return_exceptions=True)
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-a", "sub-b", "run-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling", ["resumed", "returned"])
async def test_a_sibling_that_moves_on_during_a_release_is_left_to_its_own_path(make_actor, sibling):
    # Idle subagents are released one at a time. While sub-a's release is pending,
    # sub-b starts another turn (a background subagent) or returns and is released
    # by its own path; the dispatcher must neither release a program mid-request
    # nor fail on one that is already gone.
    actor, backend = make_actor()
    await session_with_two_subagents(actor, backend)
    stalled, answer = asyncio.Event(), asyncio.Event()

    async def finalize(program):
        if program == "sub-a":
            stalled.set()
            await answer.wait()

    backend.finalize_program.side_effect = finalize
    turn = asyncio.create_task(post(actor, "session", ROOT))
    await stalled.wait()
    if sibling == "resumed":
        started, release = asyncio.Event(), asyncio.Event()
        original = backend.generate

        async def pending(*args, **kwargs):
            if kwargs.get("program_id") == "sub-b":
                started.set()
                await release.wait()
            return await original(*args, **kwargs)

        backend.generate = pending
        background = asyncio.create_task(post(actor, "session", subagent("sub-b"), "task b, more"))
        await started.wait()
    else:
        backend.stop_reason = "completed"
        await post(actor, "session", subagent("sub-b"), "task b, done")
        await settled(actor)
    answer.set()
    assert (await turn).status_code == 200
    if sibling == "resumed":
        assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a"]
        release.set()
        assert (await background).status_code == 200
    await actor.finalize_session("session")
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-b", "run-1"]


def test_subagent_programs_must_be_a_known_mode(make_actor):
    with pytest.raises(ValueError, match="subagent_programs"):
        make_actor(subagent_programs="flat")


@pytest.mark.asyncio
async def test_subagent_programs_off_puts_every_agent_in_the_sessions_program(make_actor):
    # The ablation baseline: identity headers are ignored, so the router sees one
    # program per session and nothing is released before the session closes.
    actor, backend = make_actor(subagent_programs="off")
    await session_with_two_subagents(actor, backend)
    assert backend.programs == [("session", None)] * 3
    backend.stop_reason = "completed"
    await post(actor, "session", subagent("sub-a"), "task a, done")
    await settled(actor)
    backend.finalize_program.assert_not_awaited()
    await actor.finalize_session("session")
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["session"]


@pytest.mark.asyncio
async def test_subagent_programs_isolated_keeps_programs_and_releases_but_hides_the_parent(make_actor):
    # Middle rung of the ablation: a program per agent, released on return as in tree
    # mode, but the router is never told who dispatched it (no sibling co-location).
    actor, backend = make_actor(subagent_programs="isolated")
    await session_with_two_subagents(actor, backend)
    assert backend.programs == [("run-1", None), ("sub-a", None), ("sub-b", None)]
    backend.stop_reason = "completed"
    await post(actor, "session", subagent("sub-a"), "task a, done")
    await settled(actor)
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a"]
    await actor.finalize_session("session")
    assert [call.args[0] for call in backend.finalize_program.await_args_list] == ["sub-a", "sub-b", "run-1"]


@pytest.mark.asyncio
async def test_shutdown_attempts_all_sessions_even_if_one_cleanup_fails(make_actor):
    actor, backend = make_actor()
    for session in ["bad", "healthy"]:
        await create(actor, session)

    async def finalize(session):
        if session == "bad":
            raise RuntimeError("failed")

    backend.finalize_program.side_effect = finalize
    await actor.shutdown()
    assert not actor._sessions
    calls = [call.args[0] for call in backend.finalize_program.await_args_list]
    assert calls.count("healthy") == 1 and calls.count("bad") == 3


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

    from recipe.dynamo.dynamo_agent_loop import DynamoFullyAsyncLLMServerClient

    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    handle = object()
    acquire = AsyncMock(return_value=("worker", handle))
    monkeypatch.setattr(FullyAsyncLLMServerClient, "_acquire_server", acquire)
    client = object.__new__(DynamoFullyAsyncLLMServerClient)
    client._session_servers = OrderedDict()
    client._inflight_routing_sessions = {"request": ["session"]}
    server_id, acquired = await client._acquire_server("request", prompt_ids=[1, 2])
    assert server_id == "worker" and acquired._handle is handle
    acquire.assert_awaited_once_with("request", prompt_ids=[1, 2])
    assert client._session_servers["session"]["worker"] is handle


@pytest.mark.asyncio
async def test_cancelling_a_generate_awaiter_cancels_the_actor_task(monkeypatch):
    from recipe.dynamo.dynamo_agent_loop import _CancellingHandle

    ref = asyncio.get_running_loop().create_future()
    cancelled = []
    monkeypatch.setattr("recipe.dynamo.dynamo_agent_loop.ray.cancel", cancelled.append)

    class Method:
        def remote(self, **kwargs):
            return ref

    class Handle:
        generate = Method()
        finalize_program = "forwarded"

    handle = _CancellingHandle(Handle())
    assert handle.finalize_program == "forwarded"
    waiter = asyncio.create_task(handle.generate.remote(prompt_ids=[1]))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled == [ref]


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
