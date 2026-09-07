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

"""CPU launcher contracts; the training process is replaced by an argument recorder."""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPOSITORY_ROOT / "dynamo" / "smoke_dynamo_v1.sh"


def test_readme_generation_smoke_command_resolves_to_script() -> None:
    readme = (REPOSITORY_ROOT / "dynamo" / "README.md").read_text()
    commands = re.findall(r"^bash (recipe/dynamo/\S*smoke_dynamo_v1\.sh)(?:\s|$)", readme, re.MULTILINE)
    assert len(commands) == 1
    script = REPOSITORY_ROOT / Path(commands[0]).relative_to("recipe")
    assert script.is_file(), f"Documented smoke script does not exist: {commands[0]}"
    assert script == SMOKE_SCRIPT
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("child_exit", [0, 23], ids=["success", "failure"])
def test_smoke_launcher_preserves_arguments_and_child_status(tmp_path, child_exit: int) -> None:
    argument_log = tmp_path / "arguments.txt"
    recorder = tmp_path / "python3"
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$DYNAMO_TEST_ARGV"\nexit "$DYNAMO_TEST_EXIT"\n')
    recorder.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "DYNAMO_TEST_ARGV": str(argument_log),
        "DYNAMO_TEST_EXIT": str(child_exit),
        "MODEL_PATH": "/models/test model",
        "TRAIN_FILE": "/data/train.parquet",
        "TEST_FILE": "/data/test.parquet",
        "NNODES": "1",
        "NGPUS_PER_NODE": "1",
    }
    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "trainer.total_training_steps=2"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    args = argument_log.read_text().splitlines()
    assert args[:2] == ["-m", "recipe.dynamo.main_dynamo"]
    assert "actor_rollout_ref.model.path=/models/test model" in args
    assert "trainer.val_only=True" in args
    assert "actor_rollout_ref.rollout.calculate_log_probs=False" in args
    assert args[-1] == "trainer.total_training_steps=2"
    assert result.returncode == child_exit
    assert ("PASS: Dynamo validation smoke completed" in result.stdout) == (child_exit == 0)
