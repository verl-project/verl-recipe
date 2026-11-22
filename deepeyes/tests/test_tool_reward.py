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
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

import verl.tools.image_zoom_in_tool as core_zoom
from verl.tools.utils.tool_registry import initialize_tools_from_config


@pytest.fixture
def reward_module(monkeypatch):
    # Prevent module-level model discovery from contacting a judge service.
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"data": [{"id": "test-judge"}]}
        ),
    )
    path = Path(__file__).resolve().parents[1] / "deepeyes.py"
    spec = importlib.util.spec_from_file_location("deepeyes_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(module, trajectory, correct):
    module.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECT" if correct else "INCORRECT"))]
                )
            )
        )
    )
    return module.compute_score("vstar", trajectory + "<answer>answer</answer>", "ground truth")


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("correct", [True, False])
async def test_reward_with_actual_crop_response(reward_module, monkeypatch, success, correct):
    monkeypatch.setattr(core_zoom, "init_visual_execution_pool", lambda **kwargs: None)
    monkeypatch.setattr(core_zoom, "fetch_image", lambda payload: payload["image"])
    config = Path(__file__).resolve().parents[1] / "configs/image_zoom_in_tool_config.yaml"
    tool = initialize_tools_from_config(str(config))[0]
    instance, _ = await tool.create(image=Image.new("RGB", (400, 200)))
    response, _, metrics = await tool.execute(instance, {"bbox_2d": [0, 0, 100, 100] if success else [0, 0, 0, 0]})
    assert metrics["success"] is success
    trajectory = "<tool_call>crop</tool_call><tool_response>" + response.text + "</tool_response>"
    assert score(reward_module, trajectory, correct) == (0.8 + 1.2 * success if correct else 0.0)
    await tool.release(instance)


@pytest.mark.parametrize(
    "trajectory,expected",
    [
        ("", 0.8),
        ("<tool_call>crop</tool_call>", 0.8),
        ("Zoomed in on the image to the region [0, 0, 100, 100].", 0.8),
        ("<tool_response>Error processing crop</tool_response>", 0.8),
        ("<tool_response>Quoted: Zoomed in on the image to the region [0,0,100,100].</tool_response>", 0.8),
        ("<tool_response>Zoomed in on the image to the region [0,0,100,100].</tool_response>", 2.0),
        (
            "<tool_response><|vision_start|><|image_pad|><|vision_end|>"
            "Zoomed in on the image to the region [0,0,100,100].</tool_response>",
            2.0,
        ),
        (
            "<tool_response>Error</tool_response><tool_response>"
            "Zoomed in on the image to the region [0,0,100,100].</tool_response>"
            "<tool_response>Error</tool_response>",
            2.0,
        ),
    ],
)
def test_only_successful_tool_responses_earn_bonus(reward_module, trajectory, expected):
    assert score(reward_module, trajectory, True) == expected
