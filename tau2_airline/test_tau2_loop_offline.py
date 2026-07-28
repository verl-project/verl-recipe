"""CPU-only validation of the tau2<->veRL bridge (NO GPU, NO vLLM needed).

Exercises everything except policy token generation, which is the only part that
genuinely needs a GPU:

  Test A  bridge plumbing     start_trajectory / execute_tool_calls / user_respond
                              build well-formed tau2 messages with matched ids.
  Test B  reward fidelity     replaying a task's GOLD actions + communicate_info
                              through the real tau2 evaluator yields reward > 0;
                              an unfinished trajectory scores exactly 0 (tau2's
                              premature-termination guard). This is the crux:
                              it proves reward is wired faithfully.
  Test C  concurrency isolation  N trajectories driven concurrently keep private,
                              non-cross-contaminated envs, and the ContextVar
                              rollout scope resolves to the right one per task.

Run (CPU only -- no GPU, no API key, from this directory):
    python test_tau2_loop_offline.py
"""

import asyncio
import sys

import tau2.user.user_simulator as us_mod
from loguru import logger
from rollout_context import current_rollout, rollout_scope
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import TerminationReason
from tau2.user.user_simulator_base import STOP
from tau2_bridge import Tau2Bridge

# --------------------------------------------------------------------------
# Mock the user-simulator LLM: first call returns a generic request, later
# calls return STOP. This removes the need for a live usersim endpoint.
# --------------------------------------------------------------------------
_call_counts: dict[int, int] = {}


def install_scripted_user(stop_after: int = 1):
    def fake_generate(model=None, messages=None, tools=None, call_name=None, **kwargs):
        # Key by the system prompt object id so concurrent sims count independently.
        key = id(messages[0]) if messages else 0
        _call_counts[key] = _call_counts.get(key, 0) + 1
        if _call_counts[key] > stop_after:
            content = f"Great, thank you. {STOP}"
        else:
            content = "Hi, I need help with my airline reservation please."
        return AssistantMessage(role="assistant", content=content)

    us_mod.generate = fake_generate


def make_bridge() -> Tau2Bridge:
    # user_llm_args unused because we mocked generate(); keep a dummy endpoint.
    return Tau2Bridge(
        domain="airline",
        user_llm="openai/usersim",
        user_llm_args={"api_base": "http://127.0.0.1:18001/v1", "api_key": "dummy"},
        evaluation_type="all",
    )


def pick_task_with_actions(bridge: Tau2Bridge):
    """Find an airline task that has gold DB actions (so db reward is meaningful)."""
    for tid, task in bridge._tasks.items():
        ec = task.evaluation_criteria
        if ec is not None and ec.actions:
            # keep only assistant-requestor actions (airline DB mutations)
            acts = [a for a in ec.actions if getattr(a, "requestor", "assistant") == "assistant"]
            if acts:
                return tid, task, acts
    return None, None, None


# --------------------------------------------------------------------------
# Test A: bridge plumbing
# --------------------------------------------------------------------------
def test_a_plumbing(bridge: Tau2Bridge) -> None:
    print("\n=== Test A: bridge plumbing ===")
    tid = next(iter(bridge._tasks))
    traj = bridge.start_trajectory(tid)

    assert len(traj.tau2_messages) == 2, "seed = greeting + first user msg"
    assert traj.tau2_messages[0].role == "assistant"
    assert traj.tau2_messages[1].role == "user"
    init = bridge.initial_messages(traj)
    assert init[0]["role"] == "system" and init[0]["content"], "system = airline policy"
    print(
        f"  task {tid}: seeded {len(traj.tau2_messages)} msgs, system prompt "
        f"{len(init[0]['content'])} chars, {len(bridge.tool_schemas)} tools offered"
    )

    # Execute a read tool that every airline env supports, if present.
    tool_name = next(iter(bridge.tool_names))
    calls = [{"name": tool_name, "arguments": {}, "id": "call_0"}]
    resps = bridge.execute_tool_calls(traj, calls)
    assert len(resps) == 1
    # messages now: greeting, user, assistant(tool_calls), tool
    assert traj.tau2_messages[-2].is_tool_call(), "assistant tool-call message recorded"
    assert traj.tau2_messages[-1].role == "tool", "tool response recorded"
    assert traj.tau2_messages[-2].tool_calls[0].id == traj.tau2_messages[-1].id, (
        "tool_call id MUST match the following ToolMessage id (evaluator requires it)"
    )
    print(f"  executed tool '{tool_name}': id match OK, response error={resps[0]['error']}")

    # User turn + stop detection.
    reply = bridge.user_respond(traj, "Anything else I can help with?")
    print(f"  user_respond -> stop={reply['stop']}, content={reply['content'][:60]!r}")
    print("  Test A PASSED")


# --------------------------------------------------------------------------
# Test B: reward fidelity
# --------------------------------------------------------------------------
def test_b_reward(bridge: Tau2Bridge) -> None:
    print("\n=== Test B: reward fidelity ===")
    tid, task, acts = pick_task_with_actions(bridge)
    if task is None:
        print("  (no task with gold actions found; skipping strict reward check)")
        return
    print(f"  task {tid}: {len(acts)} gold assistant actions, reward_basis={task.evaluation_criteria.reward_basis}")

    # (1) Unfinished trajectory -> tau2 scores exactly 0 (premature termination).
    traj0 = bridge.start_trajectory(tid)
    r0 = bridge.compute_reward(traj0, TerminationReason.MAX_STEPS)
    assert r0["reward"] == 0.0, f"unfinished must score 0, got {r0}"
    print(f"  unfinished (MAX_STEPS) -> reward {r0['reward']} (correct: premature guard)")

    # (2) Replay GOLD actions as tool calls + convey communicate_info, then STOP.
    traj = bridge.start_trajectory(tid)
    gold_calls = [{"name": a.name, "arguments": dict(a.arguments or {}), "id": f"call_{i}"} for i, a in enumerate(acts)]
    bridge.execute_tool_calls(traj, gold_calls)
    comm = task.evaluation_criteria.communicate_info or []
    if comm:
        # put every required info string into one assistant utterance
        traj.tau2_messages.append(AssistantMessage(role="assistant", content=" ".join(str(c) for c in comm)))
    r1 = bridge.compute_reward(traj, TerminationReason.USER_STOP)
    print(f"  gold replay (USER_STOP) -> reward {r1['reward']}, breakdown={r1['reward_breakdown']}")
    assert r1["reward"] > r0["reward"], "gold trajectory must beat the unfinished one"
    print("  Test B PASSED" + ("" if r1["reward"] >= 1.0 else "  (note: <1.0 -- gold args may need env-specific IDs)"))


# --------------------------------------------------------------------------
# Test C: concurrency isolation
# --------------------------------------------------------------------------
async def _drive_one(bridge: Tau2Bridge, tid: str, tool_name: str) -> str:
    """Simulate one trajectory's async lifetime with a bound ContextVar scope."""
    loop = asyncio.get_event_loop()
    traj = await loop.run_in_executor(None, bridge.start_trajectory, tid)
    req_id = f"req_{tid}"
    with rollout_scope(req_id, payload=traj):
        # inside the async scope, the ContextVar resolves to THIS trajectory
        assert current_rollout().request_id == req_id
        assert current_rollout().payload is traj
        await loop.run_in_executor(
            None, bridge.execute_tool_calls, traj, [{"name": tool_name, "arguments": {}, "id": "c0"}]
        )
    # after two tool calls the DBs must still be independent objects
    return id(traj.env.tools.db)


def test_c_isolation(bridge: Tau2Bridge) -> None:
    print("\n=== Test C: concurrency isolation ===")
    tids = list(bridge._tasks)[:8]
    tool_name = next(iter(bridge.tool_names))

    async def run_all():
        return await asyncio.gather(*[_drive_one(bridge, t, tool_name) for t in tids])

    db_ids = asyncio.run(run_all())
    assert len(set(db_ids)) == len(db_ids), (
        f"each trajectory must own a distinct DB, got {len(set(db_ids))}/{len(db_ids)} unique"
    )
    print(f"  drove {len(tids)} concurrent trajectories; all {len(set(db_ids))} DBs distinct")
    print("  Test C PASSED")


def main() -> int:
    logger.remove()  # quiet tau2's verbose logging for a clean test report
    install_scripted_user(stop_after=1)
    bridge = make_bridge()
    test_a_plumbing(bridge)
    test_b_reward(bridge)
    test_c_isolation(bridge)
    print(
        "\nALL OFFLINE TESTS PASSED (CPU-only). GPU is needed only for policy "
        "token generation + the real usersim server."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
