"""Run the actual recipe loop with CPU stubs for veRL/tau2 imports and services.

These test control flow and output contracts, not GPU inference or Ray transport.
"""

import asyncio
import importlib.util
import pickle
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def recipe(monkeypatch):
    context_spec = importlib.util.spec_from_file_location(
        "rollout_context", Path(__file__).with_name("rollout_context.py")
    )
    context = importlib.util.module_from_spec(context_spec)
    monkeypatch.setitem(sys.modules, "rollout_context", context)
    context_spec.loader.exec_module(context)

    class Reason(Enum):
        MAX_STEPS = "max_steps"
        USER_STOP = "user_stop"
        TOO_MANY_ERRORS = "too_many_errors"

    class AgentData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.extra_fields = {}
            self.prompt_ids = []
            self.response_ids = []
            self.response_mask = []
            self.response_logprobs = []
            self.user_turns = 0
            self.assistant_turns = 0

    class Output(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.extra_fields = {}

    modules = {
        "tau2.data_model.simulation": {"TerminationReason": Reason},
        "tau2_bridge": {"Tau2Bridge": object},
        "verl.experimental.agent_loop.agent_loop": {"AgentLoopOutput": Output, "register": lambda _: lambda c: c},
        "verl.experimental.agent_loop.tool_agent_loop": {"AgentData": AgentData, "ToolAgentLoop": type("Base", (), {})},
        "verl.tools.schemas": {"OpenAIFunctionToolSchema": dict},
        "verl.utils.profiler": {"simple_timer": lambda *a: nullcontext()},
        "verl.utils.rollout_trace": {"rollout_trace_op": lambda f: f},
        "verl.workers.rollout.replica": {"TokenOutput": SimpleNamespace},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_tau2_loop_under_test", Path(__file__).with_name("tau2_agent_loop.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_loop(recipe, *, failure=None, logprobs=True, user_turn=False, tool_turn=False, max_turns=20):
    obj = recipe.Tau2AgentLoop.__new__(recipe.Tau2AgentLoop)
    stats = {"reward_calls": 0, "generations": 0, "users": 0}

    def fail(stage):
        if failure == stage:
            raise ValueError(f"injected {stage}")

    def start(task_id):
        fail("start")
        return SimpleNamespace(num_errors=0)

    def initial(traj):
        fail("initial")
        return [{"role": "system", "content": "policy"}]

    def reward(traj, reason):
        stats["reward_calls"] += 1
        fail("reward")
        return {
            "reward": float(reason == recipe.TerminationReason.USER_STOP),
            "reward_breakdown": {},
            "termination": reason.value,
        }

    def user(traj, text):
        fail("user")
        stats["users"] += 1
        return {"stop": not user_turn or stats["users"] > 1, "content": "continue"}

    def tools(traj, calls):
        fail("tools")
        return [{"content": "tool result"}]

    async def template(*args, **kwargs):
        fail("template")
        return [10, 11]

    async def generate(**kwargs):
        stats["generations"] += 1
        fail("generate")
        if failure == "second_generate" and stats["generations"] == 2:
            raise ValueError("injected second generation")
        if failure == "cancel":
            raise asyncio.CancelledError()
        tokens = [] if failure == "empty" else [20, 21]
        probs = [-0.2, -0.3] if logprobs else None
        if failure == "missing_probs":
            probs = None
        if failure == "short_probs":
            probs = [-0.2]
        return SimpleNamespace(token_ids=tokens, log_probs=probs)

    async def parse(*args):
        fail("parser")
        calls = (
            [SimpleNamespace(arguments={}, name="lookup", tool_call_id="t1")]
            if ((tool_turn or failure == "tools") and stats["generations"] == 1)
            else []
        )
        return "assistant", calls

    obj.bridge = SimpleNamespace(
        start_trajectory=start,
        initial_messages=initial,
        compute_reward=reward,
        user_respond=user,
        execute_tool_calls=tools,
        max_errors=10,
    )
    obj.server_manager = SimpleNamespace(generate=generate)
    obj.tool_parser = SimpleNamespace(stop_token_ids=[], extract_tool_calls=parse)
    obj.apply_chat_template = template
    obj.rollout_config = SimpleNamespace(calculate_log_probs=logprobs)
    obj.tool_schemas = []
    obj.tool_schema_objs = []
    obj.turn_separator = [12]
    obj.response_length = 100
    obj.max_assistant_turns = max_turns
    obj.max_user_turns = 20
    obj.max_parallel_calls = 1
    obj.max_tool_response_length = 100
    return obj, stats


def run(obj):
    async def invoke():
        obj.loop = asyncio.get_running_loop()
        try:
            return await obj.run({}, extra_info={"task_id": "0"})
        finally:
            assert sys.modules["rollout_context"].try_current_rollout() is None

    return asyncio.run(invoke())


@pytest.mark.parametrize(
    "failure",
    [
        "start",
        "initial",
        "template",
        "generate",
        "second_generate",
        "parser",
        "tools",
        "user",
        "reward",
        "empty",
        "missing_probs",
        "short_probs",
    ],
)
def test_errors_never_return_synthetic_training_samples(recipe, failure):
    obj, stats = make_loop(recipe, failure=failure, user_turn=failure == "second_generate")
    with pytest.raises(RuntimeError, match="tau2 trajectory failed for task 0") as exc:
        run(obj)
    assert exc.value.__cause__ is None
    assert str(pickle.loads(pickle.dumps(exc.value))) == str(exc.value)
    assert stats["reward_calls"] == (1 if failure == "reward" else 0)


@pytest.mark.parametrize("logprobs", [True, False])
@pytest.mark.parametrize("turn", ["direct", "user", "tool"])
def test_success_preserves_masks_and_logprobs(recipe, logprobs, turn):
    obj, _ = make_loop(recipe, logprobs=logprobs, user_turn=turn == "user", tool_turn=turn == "tool")
    output = run(obj)
    assert output.prompt_ids == [10, 11]
    assert output.reward_score == 1
    assert output.extra_fields["outcome_binary"] == 1
    expected_mask = [1, 1] if turn == "direct" else [1, 1, 0, 0, 0, 1, 1]
    assert output.response_mask == expected_mask
    assert len(output.response_ids) == len(expected_mask)
    if logprobs:
        expected = [-0.2, -0.3] if turn == "direct" else [-0.2, -0.3, 0.0, 0.0, 0.0, -0.2, -0.3]
        assert output.response_logprobs == expected
    else:
        assert output.response_logprobs is None


def test_normal_turn_limit_remains_valid_zero_reward(recipe):
    obj, _ = make_loop(recipe, max_turns=1)
    output = run(obj)
    assert output.reward_score == 0
    assert output.response_ids == [20, 21]
    assert output.response_logprobs == [-0.2, -0.3]
    assert output.extra_fields["tau2_termination"] == "max_steps"


def test_cancellation_propagates(recipe):
    obj, _ = make_loop(recipe, failure="cancel")
    with pytest.raises(asyncio.CancelledError):
        run(obj)


@pytest.mark.parametrize("failed_first", [False, True])
def test_mixed_batch_aborts_instead_of_returning_none_logprobs(recipe, failed_first):
    async def mixed():
        good, _ = make_loop(recipe)
        bad, _ = make_loop(recipe, failure="generate")
        good.loop = bad.loop = asyncio.get_running_loop()
        runs = [good.run({}, extra_info={"task_id": "0"}), bad.run({}, extra_info={"task_id": "1"})]
        if failed_first:
            runs.reverse()
        return await asyncio.gather(*runs)

    with pytest.raises(RuntimeError, match="task 1"):
        asyncio.run(mixed())
