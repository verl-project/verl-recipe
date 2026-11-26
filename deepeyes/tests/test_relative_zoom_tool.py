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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from qwen_vl_utils import fetch_image
from recipe.deepeyes.tools.image_zoom_in_tool_rel import ImageZoomInToolRelative

import verl.tools.image_zoom_in_tool as core_zoom
from verl.tools.utils.tool_registry import initialize_tools_from_config

RECIPE = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool(monkeypatch):
    # Isolate Ray pool setup and external image loading; execute/crop/lifecycle
    # still use the real core tool implementation and ToolResponse schema.
    monkeypatch.setattr(core_zoom, "init_visual_execution_pool", lambda **kwargs: None)
    monkeypatch.setattr(core_zoom, "fetch_image", lambda payload: payload["image"].convert("RGB"))
    tools = initialize_tools_from_config(str(RECIPE / "configs/image_zoom_in_rel_tool_config.yaml"))
    assert len(tools) == 1
    assert isinstance(tools[0], ImageZoomInToolRelative)
    return tools[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,bbox,pixel_box",
    [
        ((2000, 1000), [250, 250, 750, 750], (500, 250, 1500, 750)),
        ((400, 200), [250, 250, 750, 750], (100, 50, 300, 150)),
        ((1000, 1000), [250, 250, 750, 750], (250, 250, 750, 750)),
        ((2000, 1200), [0, 0, 1000, 1000], (0, 0, 2000, 1200)),
        ((400, 200), [-100, -200, 1200, 1100], (0, 0, 400, 200)),
        ((500, 300), [125.5, 125.5, 875.5, 875.5], (62.75, 37.65, 437.75, 262.65)),
    ],
)
async def test_crop_matches_source_region(tool, size, bbox, pixel_box):
    image = Image.new("RGB", size, "black")
    image.paste("red", tuple(round(value) for value in pixel_box))
    instance, _ = await tool.create(image=image)
    response, reward, metrics = await tool.execute(instance, {"bbox_2d": bbox})
    assert metrics["success"] is True
    assert reward == 0.0
    expected = image.crop(pixel_box)
    assert response.image[0].size == expected.size
    assert response.image[0].tobytes() == expected.tobytes()
    await tool.release(instance)


@pytest.mark.asyncio
async def test_small_crop_is_enlarged_within_image(tool):
    instance, _ = await tool.create(image=Image.new("RGB", (400, 200)))
    response, _, metrics = await tool.execute(instance, {"bbox_2d": [0, 0, 10, 10]})
    assert metrics["success"] is True
    width, height = response.image[0].size
    assert min(width, height) >= 28
    assert width <= 400 and height <= 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [],
        [0, 0, 1000],
        "0,0,1000,1000",
        [1, 1, 1, 100],
        [500, 0, 100, 1000],
        [0, 0, float("nan"), 1000],
        [0, 0, float("inf"), 1000],
        [False, 0, 1000, 1000],
        ["0", 0, 1000, 1000],
        [0, 0, 1, 1000],
    ],
)
async def test_invalid_bbox_returns_error_without_image(tool, bbox):
    instance, _ = await tool.create(image=Image.new("RGB", (400, 200)))
    response, reward, metrics = await tool.execute(instance, {"bbox_2d": bbox})
    assert metrics["success"] is False
    assert reward < 0
    assert response.image is None
    assert response.text.startswith("Error")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["pil", "path"])
async def test_crop_uses_dimensions_after_real_image_loading(tool, monkeypatch, tmp_path, source):
    # Unlike the geometry tests, exercise the base tool's real image loader,
    # including its resizing, with both supported local image representations.
    monkeypatch.setattr(core_zoom, "fetch_image", fetch_image)
    image = Image.new("RGB", (513, 257), "black")
    image.paste("red", (128, 64, 385, 193))
    image_input = image
    if source == "path":
        image_path = tmp_path / "source.png"
        image.save(image_path)
        image_input = str(image_path)
    loaded = fetch_image({"image": image_input})
    assert loaded.size != image.size  # Ensure the regression exercises a resize.
    width, height = loaded.size
    expected = loaded.crop((width / 4, height / 4, width * 3 / 4, height * 3 / 4))

    instance, _ = await tool.create(create_kwargs={"image": image_input})
    response, _, metrics = await tool.execute(instance, {"bbox_2d": [250, 250, 750, 750]})
    assert metrics["success"] is True
    assert response.image[0].size == expected.size
    assert response.image[0].tobytes() == expected.tobytes()
    await tool.release(instance)


@pytest.mark.parametrize("relative", [False, True])
def test_launcher_selects_config_without_starting_training(tmp_path, relative):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.json"
    # Replace only the launched Python command; the real launcher still expands its arguments.
    command = bin_dir / "python3"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['ZOOM_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    command.chmod(0o755)
    (tmp_path / "logs").mkdir()
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], ZOOM_TEST_ARGS=str(capture))
    env.pop("TOOL_CONFIG_PATH", None)
    selected = "recipe/deepeyes/configs/image_zoom_in_tool_config.yaml"
    if relative:
        selected = str(RECIPE / "configs/image_zoom_in_rel_tool_config.yaml")
        env["TOOL_CONFIG_PATH"] = selected
    subprocess.run(
        ["bash", str(RECIPE / "run_deepeyes_grpo.sh")], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    args = json.loads(capture.read_text())
    assert f"actor_rollout_ref.rollout.multi_turn.tool_config_path={selected}" in args


@pytest.mark.asyncio
async def test_existing_pixel_config_keeps_pixel_semantics(tool):
    absolute_tool = initialize_tools_from_config(str(RECIPE / "configs/image_zoom_in_tool_config.yaml"))[0]
    assert absolute_tool.name == tool.name == "image_zoom_in_tool"
    image = Image.new("RGB", (2000, 1000))
    absolute_instance, _ = await absolute_tool.create(image=image)
    relative_instance, _ = await tool.create(image=image)
    parameters = {"bbox_2d": [250, 250, 750, 750]}
    absolute_response, _, absolute_metrics = await absolute_tool.execute(absolute_instance, parameters)
    relative_response, _, relative_metrics = await tool.execute(relative_instance, parameters)
    assert absolute_metrics["success"] is relative_metrics["success"] is True
    assert absolute_response.image[0].size == (500, 500)
    assert relative_response.image[0].size == (1000, 500)
    await absolute_tool.release(absolute_instance)
    await tool.release(relative_instance)
