from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from verl_tinker.config_utils import (
    _validate_config,
    is_no_rollout_deployment,
    process_actor_rollout_ref_config,
)

_TINKER_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _minimal_tinker_config():
    return OmegaConf.create(
        {
            "server": {},
            "actor_rollout_ref": {
                "actor": {"strategy": "veomni"},
                "rollout": {},
                "ref": {"enable": False},
                "model": {"path": "/models/qwen"},
            },
            "algorithm": {"adv_estimator": "grpo"},
            "data": {"train_batch_size": 512},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 8},
        }
    )


def test_tinker_config_merges_verl_defaults_and_keeps_only_tinker_overrides():
    config = process_actor_rollout_ref_config(_minimal_tinker_config())

    assert set(config.keys()) == {"server", "actor_rollout_ref", "algorithm", "data", "trainer"}
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8000
    assert config.server.ray_address == "local"
    assert config.server.checkpoint_dir == "/tmp/tinker-checkpoints"
    assert config.server.max_concurrent_samples == 32
    assert config.server.enable_offload is True
    assert config.server.disable_config_fix is False
    assert config.actor_rollout_ref.actor._target_ == "verl.workers.config.VeOmniActorConfig"
    assert config.actor_rollout_ref.model._target_ == "verl.workers.config.HFModelConfig"
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.save_contents
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.load_contents


def test_tinker_config_preserves_explicit_server_values_over_defaults():
    config = _minimal_tinker_config()
    config.server = {
        "host": "127.0.0.1",
        "port": 9000,
        "checkpoint_dir": "/tmp/custom-checkpoints",
        "enable_offload": False,
    }

    config = process_actor_rollout_ref_config(config)

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 9000
    assert config.server.checkpoint_dir == "/tmp/custom-checkpoints"
    assert config.server.enable_offload is False
    assert config.server.disable_config_fix is False


def test_tinker_config_disables_verl_model_offload_flags():
    config = _minimal_tinker_config()
    config.actor_rollout_ref.actor.veomni = {"param_offload": True, "optimizer_offload": True}
    config.actor_rollout_ref.ref.veomni = {"param_offload": True, "optimizer_offload": True}

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.veomni.param_offload is False
    assert config.actor_rollout_ref.actor.veomni.optimizer_offload is False
    assert config.actor_rollout_ref.ref.veomni.param_offload is False
    assert config.actor_rollout_ref.ref.veomni.optimizer_offload is False


def test_tinker_config_does_not_keep_unsupported_ppo_sections():
    config = _minimal_tinker_config()
    config.reward = {"reward_model": {"enable": True}}
    config.critic = {"enable": False}
    config.distillation = {"enable": True}

    config = process_actor_rollout_ref_config(config)

    assert "reward" not in config
    assert "critic" not in config
    assert "distillation" not in config


def test_tinker_config_preserves_user_values_over_verl_defaults():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "actor_rollout_ref.actor.optim.lr", 2.0e-6, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 4, merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.optim.lr == 2.0e-6
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4


def test_tinker_config_enables_actor_profiler_when_global_profiler_tool_is_set():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "global_profiler.tool", "torch", merge=True)
    OmegaConf.update(config, "global_profiler.steps", [1], merge=True)
    OmegaConf.update(config, "global_profiler.save_path", "outputs/profile", merge=True)
    OmegaConf.update(config, "global_profiler.all_ranks", True, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.tool_config.torch.contents", ["cuda", "cpu"], merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.global_profiler.tool == "torch"
    assert list(config.global_profiler.steps) == [1]
    assert config.actor_rollout_ref.actor.profiler.enable is True
    assert config.actor_rollout_ref.actor.profiler.tool == "torch"
    assert config.actor_rollout_ref.actor.profiler.save_path == "outputs/profile"
    assert config.actor_rollout_ref.actor.profiler.all_ranks is True
    assert list(config.actor_rollout_ref.actor.profiler.tool_config.torch.contents) == ["cuda", "cpu"]


def test_tinker_config_preserves_explicit_actor_profiler_tool_and_save_path():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "global_profiler.tool", "torch", merge=True)
    OmegaConf.update(config, "global_profiler.steps", [1], merge=True)
    OmegaConf.update(config, "global_profiler.save_path", "outputs/profile", merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.tool", "torch_memory", merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.save_path", "/tmp/custom-profile", merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.profiler.enable is True
    assert config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
    assert config.actor_rollout_ref.actor.profiler.save_path == "/tmp/custom-profile"


def test_tinker_config_preserves_explicit_actor_profiler_enable_false():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "global_profiler.tool", "torch", merge=True)
    OmegaConf.update(config, "global_profiler.steps", [1], merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.enable", False, merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.profiler.enable is False


def test_tinker_config_propagates_global_profiler_ranks_when_actor_ranks_unset():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "global_profiler.tool", "torch", merge=True)
    OmegaConf.update(config, "global_profiler.ranks", [1, 3], merge=True)

    config = process_actor_rollout_ref_config(config)

    assert list(config.actor_rollout_ref.actor.profiler.ranks) == [1, 3]


def test_tinker_config_preserves_explicit_actor_profiler_rank_selection():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "global_profiler.tool", "torch", merge=True)
    OmegaConf.update(config, "global_profiler.all_ranks", True, merge=True)
    OmegaConf.update(config, "global_profiler.ranks", [1, 3], merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.all_ranks", False, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.profiler.ranks", [0], merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.profiler.all_ranks is False
    assert list(config.actor_rollout_ref.actor.profiler.ranks) == [0]


def test_sft_vexact_config_preserves_registration_external_libs():
    config = OmegaConf.load(_TINKER_CONFIG_DIR / "advance" / "qwen3_1b7_actor_rollout_vexact.yaml")

    config = process_actor_rollout_ref_config(config)

    assert list(config.actor_rollout_ref.model.external_lib) == [
        "vexact.integrations.verl.fsdp_enable_invariant",
        "vexact.integrations.verl.register",
    ]


def test_tinker_config_defaults_to_fsdp_when_actor_strategy_is_missing():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "actor_rollout_ref.actor.strategy", None, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.fsdp_config.model_dtype", "bfloat16", merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.actor._target_ == "verl.workers.config.FSDPActorConfig"
    assert config.actor_rollout_ref.actor.fsdp_config.model_dtype == "bfloat16"


def test_tinker_config_honors_explicit_unsupported_actor_strategy():
    config = _minimal_tinker_config()
    config.actor_rollout_ref.actor.strategy = "fsdp2"

    with pytest.raises(ValueError, match="actor_rollout_ref.actor.strategy='fsdp2' is not supported"):
        process_actor_rollout_ref_config(config)


def test_no_rollout_config_skips_verl_validation_when_ref_is_disabled():
    config = _minimal_tinker_config()
    config.actor_rollout_ref.rollout.enable = False
    config = process_actor_rollout_ref_config(config)

    with patch("verl_tinker.config_utils._validate_supported_verl_config") as mock_validate:
        errors = _validate_config(config)

    assert errors == []
    assert is_no_rollout_deployment(config)
    mock_validate.assert_not_called()


def test_tinker_config_rejects_enabled_critic_without_keeping_disabled_critic():
    config = _minimal_tinker_config()
    config.critic = {"enable": True}

    config = process_actor_rollout_ref_config(config)
    errors = _validate_config(config)

    assert errors == ["critic support has been removed from the Tinker server; set critic.enable=false"]
