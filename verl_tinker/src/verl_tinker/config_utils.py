from importlib.resources import files
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config

_MISSING_VALUES = (None, "", "???")
_DEFAULT_MICRO_BATCH_SIZE_PER_GPU = 1
_ACTOR_CHECKPOINT_CONTENTS = ["model", "optimizer", "extra", "hf_model"]
_VERL_DEFAULT_CONFIG_BY_STRATEGY = {
    "dp": "_generated_ppo_trainer.yaml",
    "fsdp": "_generated_ppo_trainer.yaml",
    "veomni": "_generated_ppo_veomni_trainer.yaml",
    "megatron": "_generated_ppo_megatron_trainer.yaml",
    "torchtitan": "_generated_ppo_torchtitan_trainer.yaml",
}


REQUIRED_PATHS = [
    "server",
    "trainer",
    "actor_rollout_ref",
    "actor_rollout_ref.actor",
    "actor_rollout_ref.rollout",
    "actor_rollout_ref.ref",
    "actor_rollout_ref.model",
]


def is_enable_false(config: DictConfig, path: str) -> bool:
    node = OmegaConf.select(config, path)

    if not isinstance(node, DictConfig) or "enable" not in node:
        return False

    return node.enable is False


def is_no_rollout_deployment(config: DictConfig) -> bool:
    return is_enable_false(config, "actor_rollout_ref.rollout")


def process_actor_rollout_ref_config(config: DictConfig) -> DictConfig:
    """Merge a lightweight Tinker server config on top of Verl's defaults."""

    resolved_strategy = _resolve_actor_strategy(config)
    strategy_is_missing = _is_missing(OmegaConf.select(config, "actor_rollout_ref.actor.strategy"))
    user_set_actor_profiler_enable = has_path(config, "actor_rollout_ref.actor.profiler.enable")
    user_set_actor_profiler_tool = has_path(config, "actor_rollout_ref.actor.profiler.tool")
    user_set_actor_profiler_save_path = has_path(config, "actor_rollout_ref.actor.profiler.save_path")
    default_config = _load_verl_default_config(config)
    config = OmegaConf.merge(default_config, config)
    OmegaConf.set_struct(config, False)

    if strategy_is_missing:
        OmegaConf.update(config, "actor_rollout_ref.actor.strategy", resolved_strategy, merge=True)
    apply_tinker_server_overrides(
        config,
        user_set_actor_profiler_enable=user_set_actor_profiler_enable,
        user_set_actor_profiler_tool=user_set_actor_profiler_tool,
        user_set_actor_profiler_save_path=user_set_actor_profiler_save_path,
    )
    return config


def apply_tinker_server_overrides(
    config: DictConfig,
    *,
    user_set_actor_profiler_enable: bool = False,
    user_set_actor_profiler_tool: bool = False,
    user_set_actor_profiler_save_path: bool = False,
) -> None:
    """Apply only Tinker-server-specific config adjustments."""

    _apply_micro_batch_default(
        config,
        section="actor_rollout_ref.actor",
        dynamic_key="use_dynamic_bsz",
        micro_batch_key="ppo_micro_batch_size",
        micro_batch_per_gpu_key="ppo_micro_batch_size_per_gpu",
    )
    _apply_micro_batch_default(
        config,
        section="actor_rollout_ref.ref",
        dynamic_key="log_prob_use_dynamic_bsz",
        micro_batch_key="log_prob_micro_batch_size",
        micro_batch_per_gpu_key="log_prob_micro_batch_size_per_gpu",
    )
    _apply_micro_batch_default(
        config,
        section="actor_rollout_ref.rollout",
        dynamic_key="log_prob_use_dynamic_bsz",
        micro_batch_key="log_prob_micro_batch_size",
        micro_batch_per_gpu_key="log_prob_micro_batch_size_per_gpu",
    )
    _ensure_actor_checkpoint_contains_hf_model(config)
    _configure_actor_profiler_from_global_config(
        config,
        user_set_actor_profiler_enable=user_set_actor_profiler_enable,
        user_set_actor_profiler_tool=user_set_actor_profiler_tool,
        user_set_actor_profiler_save_path=user_set_actor_profiler_save_path,
    )


def _resolve_actor_strategy(config: DictConfig) -> str:
    strategy = _select(config, "actor_rollout_ref.actor.strategy")
    if not _is_missing(strategy):
        return str(strategy)

    model_engine = _select(config, "model_engine")
    if not _is_missing(model_engine):
        model_engine = str(model_engine)
        if model_engine in _VERL_DEFAULT_CONFIG_BY_STRATEGY:
            return model_engine

    actor = _select(config, "actor_rollout_ref.actor", {})
    if isinstance(actor, DictConfig):
        if "fsdp_config" in actor:
            return "fsdp"
        if "veomni" in actor:
            return "veomni"

    return "veomni"


def _load_verl_default_config(config: DictConfig) -> DictConfig:
    strategy = _resolve_actor_strategy(config)
    config_name = _VERL_DEFAULT_CONFIG_BY_STRATEGY.get(strategy)
    if config_name is None:
        supported = ", ".join(sorted(_VERL_DEFAULT_CONFIG_BY_STRATEGY))
        raise ValueError(
            f"actor_rollout_ref.actor.strategy={strategy!r} is not supported by the "
            f"Tinker server config merge helper. Supported strategies: {supported}."
        )

    return OmegaConf.load(files("verl.trainer.config").joinpath(config_name))


def _apply_micro_batch_default(
    config: DictConfig,
    *,
    section: str,
    dynamic_key: str,
    micro_batch_key: str,
    micro_batch_per_gpu_key: str,
) -> None:
    if bool(_select(config, f"{section}.{dynamic_key}", False)):
        return

    micro_batch = _select(config, f"{section}.{micro_batch_key}")
    micro_batch_per_gpu = _select(config, f"{section}.{micro_batch_per_gpu_key}")
    if _is_missing(micro_batch) and _is_missing(micro_batch_per_gpu):
        OmegaConf.update(
            config,
            f"{section}.{micro_batch_per_gpu_key}",
            _DEFAULT_MICRO_BATCH_SIZE_PER_GPU,
            merge=True,
        )


def _ensure_actor_checkpoint_contains_hf_model(config: DictConfig) -> None:
    for key in ("save_contents", "load_contents"):
        path = f"actor_rollout_ref.actor.checkpoint.{key}"
        contents = _select(config, path)
        if _is_missing(contents):
            OmegaConf.update(config, path, list(_ACTOR_CHECKPOINT_CONTENTS), merge=True)
            continue

        contents = list(contents)
        if "hf_model" not in contents:
            contents.append("hf_model")
            OmegaConf.update(config, path, contents, merge=True)


def _configure_actor_profiler_from_global_config(
    config: DictConfig,
    *,
    user_set_actor_profiler_enable: bool,
    user_set_actor_profiler_tool: bool,
    user_set_actor_profiler_save_path: bool,
) -> None:
    tool = _select(config, "global_profiler.tool")
    if _is_missing(tool):
        return

    if not user_set_actor_profiler_enable:
        OmegaConf.update(config, "actor_rollout_ref.actor.profiler.enable", True, merge=True)
    if not user_set_actor_profiler_tool:
        OmegaConf.update(config, "actor_rollout_ref.actor.profiler.tool", tool, merge=True)

    save_path = _select(config, "global_profiler.save_path")
    if not user_set_actor_profiler_save_path and not _is_missing(save_path):
        OmegaConf.update(config, "actor_rollout_ref.actor.profiler.save_path", save_path, merge=True)


def _select(config: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    return default if _is_missing(value) else value


def _is_missing(value: Any) -> bool:
    if isinstance(value, DictConfig | ListConfig):
        return False
    return value in _MISSING_VALUES


def _validate_config(config) -> list[str]:
    """Validate config before initialization. Returns list of error messages."""
    errors = []

    if not config.get("actor_rollout_ref", {}).get("model", {}).get("path"):
        errors.append("actor_rollout_ref.model.path is required")
    if "algorithm" not in config:
        errors.append("algorithm config is required")
    trainer_cfg = config.get("trainer", {})
    if trainer_cfg.get("nnodes") is None:
        errors.append("trainer.nnodes is required")
    if trainer_cfg.get("n_gpus_per_node") is None:
        errors.append("trainer.n_gpus_per_node is required")
    else:
        trainer_cfg.n_gpus_per_node = int(trainer_cfg.n_gpus_per_node)

    if bool(config.get("critic", {}).get("enable", False)):
        errors.append("critic support has been removed from the Tinker server; set critic.enable=false")

    if is_no_rollout_deployment(config):
        if not is_enable_false(config, "actor_rollout_ref.ref"):
            errors.append("no_rollout_deployment does not support reference policy")
    else:
        try:
            validate_config(config, need_reference_policy(config), False)
        except Exception as e:
            errors.append(f"VeRL config validation: {e}")

    return errors


def has_path(cfg: DictConfig, path: str) -> bool:
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, DictConfig) or part not in cur:
            return False
        cur = cur[part]
    return True


def process_config(config: DictConfig) -> DictConfig:
    """check and format the config so that it complies with our server's standard
    to the best of our ability. Note that it is still possible for uncaught errors
    to go down the pipeline"""
    # first check that we have the required fields:
    missing = [path for path in REQUIRED_PATHS if not has_path(config, path)]

    if missing:
        raise ValueError(
            "Missing required config fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
            + "\nPlease consult the quick_start configs for an example."
        )

    config = process_actor_rollout_ref_config(config)
    errors = _validate_config(config)
    if errors:
        raise ValueError(f"Config validation failed: {errors}")

    return config
