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


@pytest.mark.parametrize("backend", ["local", "openrouter"])
def test_launch_defaults(sandbox, backend):
    result, training, calls, work = sandbox(
        "example/run_tau2_grpo_7b.sh", USERSIM_BACKEND=backend, OPENROUTER_API_KEY="fake-test-only"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(training) == 1
    call = training[0]
    for arg in [
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.5",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.seed=42",
        "data.seed=42",
        "trainer.total_training_steps=20",
        "data.train_batch_size=24",
        f"actor_rollout_ref.model.path={work}/models/qwen25-7b-sft-airline",
        f"trainer.default_local_dir={work}/ckpts/tau2_airline_7b",
    ]:
        assert arg in call["args"]
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert any(c["name"] == "curl" for c in calls) == (backend == "local")
    assert "fake-test-only" not in result.stdout + result.stderr
    if backend == "openrouter":
        assert call["env"]["TAU2_USER_LLM"].startswith("openrouter/")
        assert call["env"]["TAU2_USER_API_BASE"] == ""


def test_overrides(sandbox):
    result, training, _, _ = sandbox(
        "example/run_tau2_grpo_7b.sh",
        SEED="123",
        VAL_N="2",
        VAL_TEMP="0.7",
        GPU="3",
        TAU2_USER_LLM="openai/custom",
        TAU2_USER_API_BASE="http://localhost:9999/v1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    call = training[0]
    assert "data.seed=123" in call["args"]
    assert "actor_rollout_ref.rollout.val_kwargs.n=2" in call["args"]
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0.7" in call["args"]
    assert call["env"]["TAU2_USER_LLM"] == "openai/custom"
    assert call["env"]["TAU2_USER_API_BASE"] == "http://localhost:9999/v1"
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_failed_trainer_propagates(sandbox):
    result, training, _, _ = sandbox("example/run_tau2_grpo_7b.sh", FAIL_TRAIN="1")
    assert result.returncode == 17
    assert len(training) == 1


def test_usersim_uses_other_gpu(sandbox):
    result, training, _, _ = sandbox("example/serve_usersim_7b.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert training[0]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
