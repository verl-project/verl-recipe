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
"""Check GPU smoke launcher arguments without starting training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra._internal.utils import get_args_parser

RECIPE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script", sorted((RECIPE_ROOT / "tests" / "smoke").glob("smoke_dynamo_v1_*.sh")), ids=lambda p: p.stem
)
@pytest.mark.parametrize("child_exit", [0, 7])
def test_v1_launchers_use_native_entrypoint_and_forward_arguments(script, child_exit, tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    # Replace only the child process. The real Bash launcher supplies all arguments.
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
        "with open(os.environ['DYNAMO_LAUNCH_CAPTURE'], 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], 'registry': os.environ['VERL_USE_EXTERNAL_MODULES']}, f)\n"
        "sys.exit(int(os.environ['DYNAMO_CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "VERL_USE_EXTERNAL_MODULES": "recipe.dynamo.register,other.module",
        "DYNAMO_LAUNCH_CAPTURE": str(capture),
        "DYNAMO_CHILD_EXIT": str(child_exit),
        "MODEL_PATH": "/models/model with spaces",
    }
    result = subprocess.run(
        ["bash", str(script), "trainer.total_training_steps=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == child_exit, result.stderr
    launched = json.loads(capture.read_text())
    argv = launched["argv"]
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    # Hydra cannot resume its positional override list after config options.
    parsed = get_args_parser().parse_args(argv[2:])
    assert parsed.config_name.startswith("dynamo_trainer_v1_")
    assert Path(argv[argv.index("--config-path") + 1]) == RECIPE_ROOT / "config"
    assert "actor_rollout_ref.model.path=/models/model with spaces" in argv
    assert argv[-1] == "trainer.total_training_steps=3"
    assert launched["registry"] == env["VERL_USE_EXTERNAL_MODULES"]
    assert (
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='" + env["VERL_USE_EXTERNAL_MODULES"] + "'"
    ) in argv
    if child_exit:
        assert "PASS:" not in result.stdout


@pytest.mark.parametrize(
    "knobs, config_name, expected",
    [
        ({}, "dynamo_trainer_v1_colocate_sglang", []),
        (
            {"PD_PREFILL": "1", "DP_ATTN": "4"},
            "dynamo_trainer_v1_colocate_sglang",
            [
                "++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.disaggregation.prefill_shards=1",
                '++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.extra_args=["--dp-size","4",'
                '"--enable-dp-attention","--ep-size","4"]',
            ],
        ),
        ({"R3": "1"}, "dynamo_trainer_v1_colocate_sglang_r3", []),
    ],
)
def test_sidecar_rl_launcher_args(tmp_path, knobs, config_name, expected):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
        "json.dump(sys.argv[1:], open(os.environ['DYNAMO_LAUNCH_CAPTURE'], 'w'))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        **knobs,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "DYNAMO_LAUNCH_CAPTURE": str(capture),
        "MODEL_PATH": "/models/m",
        "TRAIN_FILE": "/data/train.parquet",
    }
    result = subprocess.run(
        ["bash", str(RECIPE_ROOT / "run_sglang_sidecar_rl.sh"), "trainer.total_training_steps=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    assert get_args_parser().parse_args(argv[2:]).config_name == config_name
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.launch_mode=sidecar" in argv
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.generate_api=generate" in argv
    assert "data.val_files=/data/train.parquet" in argv
    for arg in expected:
        assert arg in argv
    assert not any("disaggregation" in a for a in argv) or "PD_PREFILL" in knobs
    assert argv[-1] == "trainer.total_training_steps=3"
