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


SGLANG_SMOKE = RECIPE_ROOT / "tests" / "smoke" / "smoke_dynamo_v1_colocate_sglang.sh"
GOOD_LOG = (
    "starting sglang sidecar shard 0/2\n"
    "step:1 - training/rollout_actor_probs_pearson_corr:0.9989 - training/rollout_probs_diff_mean:0.0040"
    " - critic/advantages/max:0.0 - critic/advantages/min:0.0\n"
    "step:2 - training/rollout_actor_probs_pearson_corr:0.9991 - training/rollout_probs_diff_mean:0.0042"
    " - critic/advantages/max:1.73 - critic/advantages/min:-0.58\n"
)


def _step_log(pearsons, diff_means):
    return "".join(
        f"step:{i} - training/rollout_actor_probs_pearson_corr:{p} - training/rollout_probs_diff_mean:{d}"
        " - critic/advantages/max:1.0 - critic/advantages/min:-1.0\n"
        for i, (p, d) in enumerate(zip(pearsons, diff_means, strict=True), 1)
    )


def _run_sglang_smoke(tmp_path, child_stdout, **env_overrides):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
        "with open(os.environ['DYNAMO_LAUNCH_CAPTURE'], 'w') as f:\n"
        "    json.dump(sys.argv[1:], f)\n"
        "sys.stdout.write(os.environ['DYNAMO_CHILD_STDOUT'])\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "DYNAMO_LAUNCH_CAPTURE": str(capture),
        "DYNAMO_CHILD_STDOUT": child_stdout,
        "SMOKE_LOG": str(tmp_path / "smoke.log"),
        **env_overrides,
    }
    result = subprocess.run(
        ["bash", str(SGLANG_SMOKE), "trainer.total_training_steps=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, json.loads(capture.read_text())


def test_sglang_smoke_sidecar_and_dp_attention_args(tmp_path):
    result, argv = _run_sglang_smoke(tmp_path, "", SIDECAR="1", DP_ATTN="4", ROLLOUT_TP="4")
    assert result.returncode == 0, result.stderr
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.launch_mode=sidecar" in argv
    assert "++actor_rollout_ref.rollout.engine_kwargs.dynamo.generate_api=generate" in argv
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in argv
    assert (
        '++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.extra_args=["--dp-size","4",'
        '"--enable-dp-attention","--ep-size","4"]'
    ) in argv
    # Caller overrides still come last so they win.
    assert argv[-1] == "trainer.total_training_steps=3"


def test_sglang_smoke_defaults_keep_worker_mode(tmp_path):
    result, argv = _run_sglang_smoke(tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert not any("launch_mode" in arg or "generate_api" in arg or "extra_args" in arg for arg in argv)


@pytest.mark.parametrize(
    "child_stdout, passes",
    [
        (GOOD_LOG, True),
        (GOOD_LOG.replace("0.9991", "0.12"), False),
        (GOOD_LOG.replace("0.0042", "0.31"), False),
        (
            GOOD_LOG.replace("training/rollout_actor_probs_pearson_corr:0.9989 - ", "").replace(
                "training/rollout_actor_probs_pearson_corr:0.9991 - ", ""
            ),
            False,
        ),
        (GOOD_LOG + "[logprobs] engine returned 3/4 usable logprobs\n", False),
        (GOOD_LOG.replace("starting sglang sidecar shard 0/2\n", ""), False),
        (GOOD_LOG.replace("max:1.73", "max:0.0").replace("min:-0.58", "min:0.0"), False),
        # The gate averages over steps: one low step does not sink an otherwise good run.
        (GOOD_LOG.replace("0.9989", "0.9829"), True),
    ],
    ids=["ok", "low-pearson", "high-diff", "no-pearson", "padded", "no-sidecar", "zero-advantage", "step-mean"],
)
def test_sglang_smoke_gate(tmp_path, child_stdout, passes):
    result, _ = _run_sglang_smoke(tmp_path, child_stdout, SIDECAR="1", GATE="1")
    assert (result.returncode == 0) is passes, result.stderr
    assert ("PASS:" in result.stdout) is passes


@pytest.mark.parametrize(
    "modules, expected",
    [
        (None, "recipe.dynamo.verl_logging_compat"),
        ("recipe.dynamo.register", "recipe.dynamo.verl_logging_compat"),
        ("a.b,recipe.dynamo.register,c.d", "a.b,c.d"),
    ],
    ids=["default", "register-only", "mixed"],
)
def test_sglang_smoke_native_baseline_args(tmp_path, modules, expected):
    env = {"ROLLOUT_NAME": "sglang"}
    if modules is not None:
        env["VERL_USE_EXTERNAL_MODULES"] = modules
    else:
        env["VERL_USE_EXTERNAL_MODULES"] = ""
    result, argv = _run_sglang_smoke(tmp_path, "", **env)
    assert result.returncode == 0, result.stderr
    assert "actor_rollout_ref.rollout.name=sglang" in argv
    assert not any("engine_kwargs.dynamo" in arg for arg in argv)
    # recipe.dynamo.register would swap in the Dynamo server manager and launch Dynamo anyway.
    assert f"ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='{expected}'" in argv


@pytest.mark.parametrize("env", [{"SIDECAR": "1"}, {"DP_ATTN": "4"}], ids=["sidecar", "dp-attn"])
def test_sglang_smoke_native_rejects_dynamo_only_switches(tmp_path, env):
    script_env = {**os.environ, "ROLLOUT_NAME": "sglang", **env}
    result = subprocess.run(["bash", str(SGLANG_SMOKE)], cwd=tmp_path, env=script_env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "need ROLLOUT_NAME=dynamo" in result.stderr


@pytest.mark.parametrize(
    "pearsons, diff_means, passes",
    [
        ([0.962, 0.970, 0.965], [0.0041, 0.0043, 0.0040], True),  # within 0.01 / 0.002 of the baseline
        ([0.950, 0.955, 0.952], [0.0041, 0.0043, 0.0040], False),  # pearson drops > 0.01
        ([0.966, 0.970, 0.968], [0.0070, 0.0068, 0.0071], False),  # diff mean rises > 0.002
    ],
    ids=["near-baseline", "pearson-drop", "diff-rise"],
)
def test_sglang_smoke_gate_relative_to_baseline(tmp_path, pearsons, diff_means, passes):
    baseline = tmp_path / "native.log"
    baseline.write_text(_step_log([0.966, 0.971, 0.969], [0.0040, 0.0042, 0.0041]))
    log = "starting sglang sidecar shard 0/2\n" + _step_log(pearsons, diff_means)
    # The absolute bar alone would fail all three; the baseline replaces it.
    result, _ = _run_sglang_smoke(tmp_path, log, SIDECAR="1", GATE="1", REF_SMOKE_LOG=str(baseline))
    assert (result.returncode == 0) is passes, result.stderr
    assert "baseline: pearson=0.968667" in result.stdout


AB_DRIVER = RECIPE_ROOT / "tests" / "smoke" / "run_sidecar_logprob_ab.sh"


def _run_ab_driver(tmp_path, fail_arm=""):
    capture = tmp_path / "arms.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("python3", "etcd", "nats-server", "ray"):
        (bin_dir / tool).write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / tool).chmod(0o755)
    # Stands in for the V1 colocate smoke: records what each arm was given.
    fake_smoke = tmp_path / "smoke.sh"
    fake_smoke.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {sys.executable} -c '\n"
        "import json, os, sys\n"
        'keys = ("ROLLOUT_NAME", "SIDECAR", "DP_ATTN", "GATE", "REF_SMOKE_LOG", "SMOKE_LOG", "TRAIN_FILE")\n'
        'rec = {k: os.environ.get(k) for k in keys} | {"argv": sys.argv[1:]}\n'
        'open(os.environ["AB_CAPTURE"], "a").write(json.dumps(rec) + "\\n")\n'
        'arm = os.environ["ROLLOUT_NAME"] + ":" + os.environ["DP_ATTN"]\n'
        'print("FAIL: gate" if arm == os.environ["AB_FAIL_ARM"] else "PASS: ok")\n'
        'sys.exit(arm == os.environ["AB_FAIL_ARM"])\n'
        '\' "$@"\n'
    )
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "MODEL_PATH": "/models/m",
        "TRAIN_FILE": "/data/train.parquet",
        "SMOKE": str(fake_smoke),
        "OUT": str(tmp_path / "out"),
        "ARM_COOLDOWN_S": "0",
        "AB_CAPTURE": str(capture),
        "AB_FAIL_ARM": fail_arm,
    }
    result = subprocess.run(
        ["bash", str(AB_DRIVER), "trainer.total_training_steps=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, [json.loads(line) for line in capture.read_text().splitlines()]


def test_sidecar_ab_driver_gates_dynamo_arms_against_native(tmp_path):
    result, arms = _run_ab_driver(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert [(a["ROLLOUT_NAME"], a["SIDECAR"], a["DP_ATTN"]) for a in arms] == [
        ("sglang", "0", "1"),
        ("dynamo", "1", "1"),
        ("dynamo", "1", "4"),
    ]
    assert all(a["GATE"] == "1" and a["TRAIN_FILE"] == "/data/train.parquet" for a in arms)
    # The baseline's own gated log is the bar for both Dynamo arms.
    assert arms[0]["REF_SMOKE_LOG"] is None
    assert arms[0]["SMOKE_LOG"] == str(out / "native.log")
    assert arms[1]["REF_SMOKE_LOG"] == arms[2]["REF_SMOKE_LOG"] == str(out / "native.log")
    for arm in arms:
        assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" in arm["argv"]
        assert arm["argv"][-1] == "trainer.total_training_steps=3"
    summary = (out / "summary.txt").read_text().splitlines()
    assert [line.split()[:2] for line in summary] == [["native", "PASS"], ["tp4", "PASS"], ["dpattn4", "PASS"]]


def test_sidecar_ab_driver_runs_every_arm_and_fails_if_one_fails(tmp_path):
    result, arms = _run_ab_driver(tmp_path, fail_arm="dynamo:4")
    assert result.returncode == 1
    assert len(arms) == 3
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "dpattn4 FAIL rc=1 FAIL: gate" in summary
    assert "tp4 PASS" in summary
