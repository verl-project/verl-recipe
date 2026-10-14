"""Execute real shell launchers with fake GPU/trainer commands; no training or API calls."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def sandbox(tmp_path):
    work = tmp_path / "work with spaces"
    work.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text(f"export PATH={shlex.quote(str(bin_dir))}:$PATH\n")
    record = tmp_path / "calls.jsonl"
    stub = r"""#!PYTHON
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['CALLS'], 'a') as f:
    env = {k:v for k,v in os.environ.items()
           if k.startswith(('GATE_', 'TAU2_')) or k == 'CUDA_VISIBLE_DEVICES'}
    f.write(json.dumps({'name':name, 'args':sys.argv[1:], 'env':env}) + '\n')
if name == 'nvidia-smi':
    print(0)
elif name == 'python3' and '-m' in sys.argv:
    if os.environ.get('FAIL_TRAIN'):
        sys.exit(17)
    if not os.environ.get('NO_METRIC'):
        print('val-core/gsm8k/acc/mean@1:np.float64(0.75) response_length/mean:100')
elif name in ('ray', 'pgrep', 'kill'):
    sys.exit('global process cleanup must not be invoked')
""".replace("PYTHON", sys.executable, 1)
    for name in ("python3", "nvidia-smi", "curl", "ray", "pgrep", "kill"):
        p = bin_dir / name
        p.write_text(stub)
        p.chmod(0o755)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("GATE_", "TAU2_", "USERSIM_")):
            env.pop(key)
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        CALLS=str(record),
        GATE_ROOT=str(work),
        ROOT=str(work),
        VERL_VENV=str(venv),
        VENV=str(venv),
        HF_MIRROR="0",
    )

    def run(script, **extra):
        result = subprocess.run(
            ["bash", str(HERE / script)], cwd=tmp_path, env={**env, **extra}, text=True, capture_output=True, timeout=30
        )
        calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        training = [c for c in calls if c["name"] == "python3" and "-m" in c["args"]]
        return result, training, calls, work

    return run


@pytest.mark.parametrize(
    ("script", "count"),
    [
        ("run_gate_chain.sh", 3),
        ("run_lambda_chain.sh", 6),
        ("run_phantom_chain.sh", 3),
    ],
)
def test_chains_complete_once_per_arm(sandbox, script, count):
    result, training, calls, work = sandbox(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(training) == count
    assert result.stdout.count("ARM_DONE") == count
    assert "CHAIN_COMPLETE" in result.stdout
    assert not any(c["name"] in ("ray", "pgrep", "kill") for c in calls)
    for call in training:
        args = call["args"]
        assert f"reward.custom_reward_function.path={HERE / 'gate_shaped_reward.py'}" in args
        assert "trainer.resume_mode=disable" in args
        mode = call["env"]["GATE_MODE"]
        lam = call["env"]["GATE_LAMBDA"]
        assert f"+ray_kwargs.ray_init.runtime_env.env_vars.GATE_LAMBDA='{lam}'" in args
        assert f"+ray_kwargs.ray_init.runtime_env.env_vars.GATE_MODE={mode}" in args
        exp = next(a.split("=", 1)[1] for a in args if a.startswith("trainer.experiment_name="))
        assert (work / f"{exp}.log").read_text().count("acc/mean@1:") == 1
        assert "0.75" in (work / "gate_results" / f"{exp}.md").read_text()
    if script == "run_lambda_chain.sh":
        assert [c["env"]["GATE_LAMBDA"] for c in training] == ["0.50"] * 3 + ["0.10"] * 3
    if script == "run_phantom_chain.sh":
        assert all(c["env"]["GATE_PHANTOM"] == "long" for c in training)


@pytest.mark.parametrize("script", ["run_gate_chain.sh", "run_lambda_chain.sh", "run_phantom_chain.sh"])
@pytest.mark.parametrize("failure", ["FAIL_TRAIN", "NO_METRIC"])
def test_failed_arm_stops_chain(sandbox, script, failure):
    result, training, _, _ = sandbox(script, **{failure: "1"})
    assert result.returncode != 0
    assert len(training) == 1
    assert "ARM_DONE" not in result.stdout
    assert "CHAIN_COMPLETE" not in result.stdout


def test_typo_mode_rejected(sandbox):
    result, training, _, _ = sandbox("run_gate_arm.sh", GATE_MODE="outcomme")
    assert result.returncode == 2
    assert not training
