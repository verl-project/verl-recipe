from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from verl_tinker.config_utils import (
    _validate_config,
    is_no_rollout_deployment,
    main,
    process_actor_rollout_ref_config,
    process_config,
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

    assert set(config.keys()) == {"server", "actor_rollout_ref", "algorithm", "data", "distillation", "trainer"}
    assert config.distillation.enabled is False
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8000
    assert config.server.ray_address == "local"
    assert config.server.checkpoint_dir == "/tmp/tinker-checkpoints"
    assert config.server.max_concurrent_samples == 32
    assert config.server.enable_offload is True
    assert config.server.auto_merge_verl_default_config is True
    assert config.actor_rollout_ref.actor._target_ == "verl.workers.config.VeOmniActorConfig"
    assert config.actor_rollout_ref.model._target_ == "verl.workers.config.HFModelConfig"
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.save_contents
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.load_contents


def test_config_utils_cli_validates_and_prints_processed_config(capsys):
    main(["--config", str(_TINKER_CONFIG_DIR / "quick_start" / "actor.yaml")])

    output = capsys.readouterr().out
    assert "Config validation succeeded. Final processed config:" in output
    assert "auto_merge_verl_default_config: true" in output
    assert "param_offload: false" in output


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
    assert config.server.auto_merge_verl_default_config is True


def test_disabling_verl_default_merge_still_applies_tinker_server_overrides():
    config = _minimal_tinker_config()
    config.server.auto_merge_verl_default_config = False
    config.actor_rollout_ref.actor.veomni = {"param_offload": True}
    config.actor_rollout_ref.actor.checkpoint = {
        "save_contents": ["model"],
        "load_contents": ["model"],
    }
    config.imported_verl_section = {"sentinel": True}

    with (
        patch("verl_tinker.config_utils._load_verl_section_defaults") as mock_load_defaults,
        patch("verl_tinker.config_utils._validate_config", return_value=[]),
    ):
        config = process_config(config)

    mock_load_defaults.assert_not_called()
    assert config.server.auto_merge_verl_default_config is False
    assert config.imported_verl_section.sentinel is True
    assert "_target_" not in config.actor_rollout_ref.actor
    assert config.actor_rollout_ref.actor.veomni.param_offload is False
    assert list(config.actor_rollout_ref.actor.checkpoint.save_contents) == [
        "model",
        "optimizer",
        "extra",
        "hf_model",
    ]
    assert list(config.actor_rollout_ref.actor.checkpoint.load_contents) == [
        "model",
        "optimizer",
        "extra",
        "hf_model",
    ]


def test_tinker_config_disables_verl_model_offload_flags():
    config = _minimal_tinker_config()
    config.actor_rollout_ref.actor.veomni = {"param_offload": True, "optimizer_offload": True}
    config.actor_rollout_ref.ref.veomni = {"param_offload": True, "optimizer_offload": True}

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.veomni.param_offload is False
    assert config.actor_rollout_ref.actor.veomni.optimizer_offload is False
    assert config.actor_rollout_ref.ref.veomni.param_offload is False
    assert config.actor_rollout_ref.ref.veomni.optimizer_offload is False


def test_tinker_config_keeps_distillation_but_not_other_unsupported_ppo_sections():
    config = _minimal_tinker_config()
    config.reward = {"reward_model": {"enable": True}}
    config.critic = {"enable": False}
    config.distillation = {"enabled": False}

    config = process_actor_rollout_ref_config(config)

    assert "reward" not in config
    assert "critic" not in config
    assert config.distillation.enabled is False


def test_tinker_config_preserves_and_validates_dedicated_teacher_config():
    config = _minimal_tinker_config()
    config.trainer.n_gpus_per_node = 4
    config.distillation = {
        "enabled": True,
        "nnodes": 1,
        "n_gpus_per_node": 4,
        "teacher_models": {
            "teacher_model": {
                "model_name": "Qwen/Qwen3-30B-A3B",
                "model_path": "Qwen/Qwen3-30B-A3B",
                "inference": {
                    "name": "vllm",
                    "tensor_model_parallel_size": 4,
                    "engine_kwargs": {"vllm": {"max_logprobs": 128}},
                },
            }
        },
    }

    processed = process_actor_rollout_ref_config(config)
    errors = _validate_config(processed)

    assert errors == []
    assert processed.distillation.enabled is True
    assert processed.distillation.teacher_models.teacher_model.model_name == "Qwen/Qwen3-30B-A3B"
    assert processed.distillation.teacher_models.teacher_model.model_path == "Qwen/Qwen3-30B-A3B"
    assert processed.distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size == 4


def test_multi_teacher_config_uses_dedicated_pools_and_validates_gpu_footprints():
    config = OmegaConf.load(_TINKER_CONFIG_DIR / "advance" / "qwen3_8b_actor_qwen3_32b_qwen3_235b_teachers.yaml")

    processed = process_actor_rollout_ref_config(config)
    errors = _validate_config(processed)

    assert errors == []
    assert processed.trainer.nnodes == 1
    assert processed.trainer.n_gpus_per_node == 4
    assert processed.distillation.nnodes == 2
    assert processed.distillation.n_gpus_per_node == 6
    assert processed.distillation.dedicated_resource_pools is True
    assert processed.distillation.teacher_models.deepmath_teacher.key == "deepmath"
    assert processed.distillation.teacher_models.deepmath_teacher.inference.tensor_model_parallel_size == 4
    assert processed.distillation.teacher_models.tulu3_teacher.key == "tulu3"
    assert processed.distillation.teacher_models.tulu3_teacher.inference.tensor_model_parallel_size == 8


@pytest.mark.parametrize(
    ("teacher_identifiers", "expected_name", "expected_path"),
    [
        ({"model_path": "Qwen/path"}, "Qwen/path", "Qwen/path"),
        ({"model_name": "Qwen/name"}, "Qwen/name", "Qwen/name"),
    ],
)
def test_teacher_model_name_and_path_default_to_each_other(teacher_identifiers, expected_name, expected_path):
    config = _minimal_tinker_config()
    config.trainer.n_gpus_per_node = 1
    config.distillation = {
        "enabled": True,
        "nnodes": 1,
        "n_gpus_per_node": 1,
        "teacher_models": {
            "teacher_model": {
                **teacher_identifiers,
                "inference": {"name": "vllm", "tensor_model_parallel_size": 1},
            }
        },
    }

    processed = process_actor_rollout_ref_config(config)

    assert _validate_config(processed) == []
    assert processed.distillation.teacher_models.teacher_model.model_name == expected_name
    assert processed.distillation.teacher_models.teacher_model.model_path == expected_path


def test_teacher_model_requires_name_or_path():
    config = _minimal_tinker_config()
    config.distillation = {
        "enabled": True,
        "teacher_models": {"teacher_model": {"inference": {"name": "vllm"}}},
    }

    processed = process_actor_rollout_ref_config(config)

    assert _validate_config(processed) == [
        "distillation.teacher_models.teacher_model requires at least one of model_name or model_path"
    ]


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
