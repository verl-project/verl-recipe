# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Offline tests for the E2B ReTool tool: a fake sandbox, no network, no GPU.

Run from the verl root (where this repository is checked out as ``recipe``):
``pytest recipe/retool/tests/test_e2b_sandbox_tool.py``
"""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from recipe.retool import e2b_sandbox_tool
from recipe.retool.e2b_sandbox_tool import TIMEOUT_TEXT, UNAVAILABLE_TEXT, CustomE2BSandboxTool

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.tool_registry import initialize_tools_from_config

CONFIG_PATH = Path(__file__).parents[1] / "e2b_tool_config.yaml"


class Gone(Exception):
    """Stands in for the E2B SDK's retryable errors."""


class FakeSandbox:
    """Records what the tool does to a sandbox. ``script`` holds the next results of running code."""

    created: list["FakeSandbox"] = []
    script: list = []
    create_times: list[float] = []
    kill_fails = False

    def __init__(self):
        self.sandbox_id = f"sbx{len(FakeSandbox.created)}"
        self.killed = self.expired = False
        self.running = self.max_running = 0
        self.commands = SimpleNamespace(run=self._run)
        self.files = SimpleNamespace(write=self._write)
        FakeSandbox.created.append(self)

    @classmethod
    async def create(cls, **kwargs):
        cls.create_times.append(time.monotonic())
        return cls()

    async def set_timeout(self, timeout):
        if self.expired:
            raise Gone("expired")

    async def kill(self):
        self.killed = True
        if self.kill_fails:
            raise RuntimeError("kill failed")

    async def _write(self, path, data, **kwargs):
        pass

    async def _run(self, cmd, **kwargs):
        if cmd.startswith("install"):
            return SimpleNamespace(stdout="")
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            step = FakeSandbox.script.pop(0) if FakeSandbox.script else "0 ready\nok\n"
            if isinstance(step, BaseException):
                raise step
            if step == "block":
                await asyncio.sleep(3600)
            await asyncio.sleep(0.01)
            return SimpleNamespace(stdout=step)
        finally:
            self.running -= 1


@pytest.fixture
def make_tool(monkeypatch):
    FakeSandbox.created, FakeSandbox.script, FakeSandbox.create_times = [], [], []
    FakeSandbox.kill_fails = False
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.setattr(e2b_sandbox_tool, "_load_e2b", lambda: (FakeSandbox, (e2b_sandbox_tool._Unusable, Gone)))

    def make(**overrides):
        tool = initialize_tools_from_config(str(CONFIG_PATH))[0]
        tool.create_interval = 0.0
        for key, value in overrides.items():
            setattr(tool, key, value)
        tool._slots = asyncio.Semaphore(tool.pool_size)
        return tool

    return make


async def call(tool, code="1 + 1"):
    """One tool call, the way verl's ToolAgentLoop makes it."""
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tools = {tool.name: tool}
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "middle"
    function_call = FunctionCall(name=tool.name, arguments=json.dumps({"code": code}))
    response, _, _ = await loop._call_tool(function_call, {}, SimpleNamespace())
    return response.text


def test_config_loads_through_verl_registry(make_tool):
    tool = make_tool()
    assert isinstance(tool, CustomE2BSandboxTool)
    assert tool.name == "code_interpreter"


def test_missing_api_key_fails_at_startup(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="E2B_API_KEY"):
        initialize_tools_from_config(str(CONFIG_PATH))


def test_sandbox_is_reused_across_calls(make_tool):
    tool = make_tool()

    async def run():
        return [await call(tool), await call(tool)]

    assert asyncio.run(run()) == ["ok\n", "ok\n"]
    assert len(FakeSandbox.created) == 1
    assert not FakeSandbox.created[0].killed


def test_one_sandbox_runs_one_call_at_a_time(make_tool):
    tool = make_tool(pool_size=2)

    async def run():
        return await asyncio.gather(*(call(tool) for _ in range(10)))

    assert asyncio.run(run()) == ["ok\n"] * 10
    assert len(FakeSandbox.created) == 2
    assert [sandbox.max_running for sandbox in FakeSandbox.created] == [1, 1]


def test_sandbox_failure_is_retried_on_a_replacement(make_tool):
    tool = make_tool()
    FakeSandbox.script = [Gone("vm died"), "0 ready\n42\n"]

    assert asyncio.run(call(tool)) == "42\n"
    first, second = FakeSandbox.created
    assert first.killed and not second.killed


def test_runner_that_never_started_is_retried(make_tool):
    tool = make_tool()
    FakeSandbox.script = ["1 \n", "0 ready\n42\n"]

    assert asyncio.run(call(tool)) == "42\n"
    assert FakeSandbox.created[0].killed


def test_exhausted_retries_return_a_fixed_text(make_tool):
    tool = make_tool()
    FakeSandbox.script = [Gone("secret detail"), Gone("secret detail")]

    assert asyncio.run(call(tool)) == UNAVAILABLE_TEXT
    assert [sandbox.killed for sandbox in FakeSandbox.created] == [True, True]


def test_unexpected_error_is_not_retried_or_shown(make_tool):
    tool = make_tool()
    FakeSandbox.script = [RuntimeError("secret detail")]

    assert asyncio.run(call(tool)) == UNAVAILABLE_TEXT
    assert len(FakeSandbox.created) == 1


def test_timeout_is_reported_like_the_local_interpreter(make_tool):
    tool = make_tool()
    FakeSandbox.script = ["137 ready\npartial\n"]

    assert asyncio.run(call(tool)) == f"partial\n{TIMEOUT_TEXT}"
    assert not FakeSandbox.created[0].killed


def test_cancelled_call_kills_its_sandbox(make_tool):
    tool = make_tool(pool_size=1)
    FakeSandbox.script = ["block"]

    async def run():
        task = asyncio.create_task(call(tool))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await call(tool)

    assert asyncio.run(run()) == "ok\n"
    first, second = FakeSandbox.created
    assert first.killed and not second.killed


def test_failed_kill_does_not_raise(make_tool):
    tool = make_tool()
    FakeSandbox.kill_fails = True
    FakeSandbox.script = [Gone("vm died"), Gone("vm died"), "block"]

    async def run():
        text = await call(tool)
        task = asyncio.create_task(call(tool))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await tool.release("unknown-instance")
        return text

    assert asyncio.run(run()) == UNAVAILABLE_TEXT
    assert all(sandbox.killed for sandbox in FakeSandbox.created)


def test_expired_idle_sandboxes_do_not_use_up_retries(make_tool):
    tool = make_tool(pool_size=3)

    async def run():
        await asyncio.gather(*(call(tool) for _ in range(3)))
        for sandbox in FakeSandbox.created:
            sandbox.expired = True
        tool._renewed.clear()
        return await call(tool)

    assert asyncio.run(run()) == "ok\n"
    assert [sandbox.killed for sandbox in FakeSandbox.created] == [True, True, True, False]


def test_sandbox_creation_is_paced(make_tool):
    tool = make_tool(pool_size=3, create_interval=0.1)

    async def run():
        await asyncio.gather(*(call(tool) for _ in range(3)))

    asyncio.run(run())
    first, second, third = FakeSandbox.create_times
    assert second - first >= 0.09 and third - second >= 0.09
