"""Exercise the Uni-Agent launchers without starting Ray or allocating GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra._internal.utils import get_args_parser
from omegaconf import OmegaConf

RECIPE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("custom_runner", [False, True])
@pytest.mark.parametrize("child_exit", [0, 7])
def test_uniagent_launchers_compose_real_config_and_forward_worker_environment(
    fully_async, custom_runner, child_exit, tmp_path, monkeypatch
):
    capture = tmp_path / "argv.json"
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "sys.exit(int(os.environ['CHILD_EXIT']))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "LAUNCH_CAPTURE": str(capture),
        "CHILD_EXIT": str(child_exit),
        "VERL_SRC": str(tmp_path),
        "MODEL_PATH": "/models/qwen with spaces",
        "TRAIN_FILE": "/data/train.parquet",
        "OUTPUT_DIR": str(tmp_path / "outputs"),
        "RUN_NAME": "test-uniagent",
        "TASK_CONFIG": "/data/task config.yaml",
        "AGENT_CONFIG": "/data/obsolete-agent.yaml",
        "AUTO_FINALIZE": "true",
        "AGENT_RUNNER_FQN": "custom.run" if custom_runner else "uni_agent.framework.task_runner.run_task",
        "AGENT_CONCURRENCY": "6",
        "GATEWAY_COUNT": "3",
        "RAY_CLUSTER_ADDRESS": "10.0.0.1:6379",
        "VERL_USE_EXTERNAL_MODULES": "recipe.dynamo.register,custom.reward",
    }
    # Installed packages are enough; source checkout overrides are optional.
    env.pop("DYNAMO_SRC", None)
    env.pop("UNIAGENT_ROOT", None)
    script = "run_uniagent_thunderagent_vllm" + ("_fully_async" if fully_async else "") + ".sh"
    result = subprocess.run(
        ["bash", str(RECIPE_ROOT / script), "trainer.total_training_steps=3", "data.train_batch_size=4"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == child_exit, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    parsed = get_args_parser().parse_args(argv[2:])
    assert parsed.config_name == "dynamo_uniagent_vllm"
    with initialize_config_dir(config_dir=parsed.config_path, version_base=None):
        cfg = compose(config_name=parsed.config_name, overrides=parsed.overrides)
    # Ray receives runtime_env before TaskRunner resolves the main config.
    # Resolve under the environment the worker actually receives, without the
    # driver's workload variables, to catch missing forwarding and lazy values.
    worker_env = OmegaConf.to_container(cfg.ray_kwargs.ray_init.runtime_env.env_vars, resolve=False)
    assert all(isinstance(v, str) and "${" not in v for v in worker_env.values())
    for key in env:
        if key not in worker_env:
            monkeypatch.delenv(key, raising=False)
    for key, value in worker_env.items():
        monkeypatch.setenv(key, value)
    OmegaConf.resolve(cfg)
    r = cfg.actor_rollout_ref.rollout
    assert cfg.trainer.v1.trainer_mode == ("separate_async" if fully_async else "colocate_async")
    assert cfg.actor_rollout_ref.hybrid_engine is not fully_async
    assert cfg.trainer.n_gpus_per_node == 8
    assert r.nnodes == int(fully_async)
    assert r.tensor_model_parallel_size == 4
    assert cfg.ray_kwargs.ray_init.address == ("10.0.0.1:6379" if fully_async else None)
    assert cfg.ray_kwargs.ray_init.num_cpus == (None if fully_async else 32)
    if fully_async:
        assert r.n_gpus_per_node == 8
        assert r.checkpoint_engine.backend == "nccl"
        assert cfg.data.train_batch_size == (
            cfg.trainer.v1.separate_async.parameter_sync_step * cfg.actor_rollout_ref.actor.ppo_mini_batch_size
        )
    modules = worker_env["VERL_USE_EXTERNAL_MODULES"].split(",")
    assert modules.count("recipe.dynamo.register") == 1
    assert modules.count("recipe.dynamo.verl_logging_compat") == 1
    assert "custom.reward" in modules
    assert r.engine_kwargs.dynamo.thunderagent.enabled is True
    assert r.engine_kwargs.dynamo.thunderagent.native is True
    assert r.engine_kwargs.dynamo.thunderagent.auto_finalize is False
    assert r.engine_kwargs.dynamo.router_mode == "kv"
    assert r.multi_turn.format == "qwen3_coder"
    assert r.max_model_len == cfg.data.max_prompt_length + cfg.data.max_response_length
    assert r.engine_kwargs.dynamo.thunderagent.subagent_programs == "tree"
    assert cfg.trainer.total_training_steps == 3
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == cfg.data.train_batch_size == 4
    assert cfg.actor_rollout_ref.model.path == "/models/qwen with spaces"
    assert cfg.data.val_files == cfg.data.train_files == "/data/train.parquet"
    assert r.agent.agent_loop_manager_class == "recipe.dynamo.uniagent_gateway.DynamoAgentFrameworkRolloutAdapter"
    assert r.agent.agent_loop_config_path is None
    framework = r.custom.agent_framework
    assert framework.gateway_count == 3
    assert framework.agent_runners.task.max_concurrent_sessions == 6
    assert ("task_config_path" in framework.agent_runners.task.runner_kwargs) is not custom_runner
    if not custom_runner:
        assert framework.agent_runners.task.runner_kwargs.task_config_path == "/data/task config.yaml"
        assert framework.agent_runners.task.runner_kwargs.model_name == cfg.actor_rollout_ref.model.path
