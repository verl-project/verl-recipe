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

    assert config.actor_rollout_ref.actor._target_ == "verl.workers.config.VeOmniActorConfig"
    assert config.actor_rollout_ref.model._target_ == "verl.workers.config.HFModelConfig"
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.save_contents
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.load_contents


def test_tinker_config_preserves_user_values_over_verl_defaults():
    config = _minimal_tinker_config()
    OmegaConf.update(config, "actor_rollout_ref.actor.optim.lr", 2.0e-6, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 4, merge=True)

    config = process_actor_rollout_ref_config(config)

    assert config.actor_rollout_ref.actor.optim.lr == 2.0e-6
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4


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

    with patch("verl_tinker.config_utils.validate_config") as mock_validate:
        errors = _validate_config(config)

    assert errors == []
    assert is_no_rollout_deployment(config)
    mock_validate.assert_not_called()
