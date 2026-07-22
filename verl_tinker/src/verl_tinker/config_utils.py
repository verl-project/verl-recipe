import argparse
from importlib.resources import files
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import omega_conf_to_dataclass

_MISSING_VALUES = (None, "", "???")
_DEFAULT_MICRO_BATCH_SIZE_PER_GPU = 1
_ACTOR_CHECKPOINT_CONTENTS = ["model", "optimizer", "extra", "hf_model"]
_VERL_SECTION_DEFAULT_CONFIG_BY_STRATEGY = {
    "dp": "_generated_ppo_trainer.yaml",
    "fsdp": "_generated_ppo_trainer.yaml",
    "veomni": "_generated_ppo_veomni_trainer.yaml",
    "megatron": "_generated_ppo_megatron_trainer.yaml",
    "torchtitan": "_generated_ppo_torchtitan_trainer.yaml",
}
_VERL_STRATEGY_ALIASES = {
    "dp": "fsdp",
}
_DEFAULT_TOP_LEVEL_SECTIONS = (
    "actor_rollout_ref",
    "algorithm",
    "data",
    "distillation",
    "trainer",
)
_SUPPORTED_TOP_LEVEL_SECTIONS = (
    "server",
    "actor_rollout_ref",
    "algorithm",
    "data",
    "distillation",
    "trainer",
    "global_profiler",
    "external_libs",
)
DEFAULT_SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "ray_address": "local",
    "disk_check_path": "/tmp",
    "disk_check_min_free_gb": 0,
    "checkpoint_dir": "/tmp/tinker-checkpoints",
    "server_max_runtime": None,
    "max_concurrent_samples": 32,
    "enable_offload": True,
    "auto_merge_verl_default_config": True,
}
_VERL_OFFLOAD_FALSE_PATHS = (
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.fsdp_config.grad_offload",
    "actor_rollout_ref.actor.fsdp_config.offload_policy",
    "actor_rollout_ref.actor.veomni.param_offload",
    "actor_rollout_ref.actor.veomni.optimizer_offload",
    "actor_rollout_ref.actor.veomni.grad_offload",
    "actor_rollout_ref.actor.veomni.enable_fsdp_offload",
    "actor_rollout_ref.actor.megatron.param_offload",
    "actor_rollout_ref.actor.megatron.optimizer_offload",
    "actor_rollout_ref.actor.megatron.grad_offload",
    "actor_rollout_ref.actor.torchtitan.param_offload",
    "actor_rollout_ref.actor.torchtitan.optimizer_offload",
    "actor_rollout_ref.actor.torchtitan.grad_offload",
    "actor_rollout_ref.ref.fsdp_config.param_offload",
    "actor_rollout_ref.ref.fsdp_config.optimizer_offload",
    "actor_rollout_ref.ref.fsdp_config.grad_offload",
    "actor_rollout_ref.ref.fsdp_config.offload_policy",
    "actor_rollout_ref.ref.veomni.param_offload",
    "actor_rollout_ref.ref.veomni.optimizer_offload",
    "actor_rollout_ref.ref.veomni.grad_offload",
    "actor_rollout_ref.ref.veomni.enable_fsdp_offload",
    "actor_rollout_ref.ref.megatron.param_offload",
    "actor_rollout_ref.ref.megatron.optimizer_offload",
    "actor_rollout_ref.ref.megatron.grad_offload",
    "actor_rollout_ref.ref.torchtitan.param_offload",
    "actor_rollout_ref.ref.torchtitan.optimizer_offload",
    "actor_rollout_ref.ref.torchtitan.grad_offload",
)


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
    """Prepare the supported Verl sections and apply Tinker server overrides."""

    auto_merge_verl_defaults = bool(_select(config, "server.auto_merge_verl_default_config", True))
    resolved_strategy = _resolve_actor_strategy(config)
    strategy_is_missing = _is_missing(OmegaConf.select(config, "actor_rollout_ref.actor.strategy"))
    user_set_actor_profiler_enable = has_path(config, "actor_rollout_ref.actor.profiler.enable")
    user_set_actor_profiler_tool = has_path(config, "actor_rollout_ref.actor.profiler.tool")
    user_set_actor_profiler_save_path = has_path(config, "actor_rollout_ref.actor.profiler.save_path")
    user_set_actor_profiler_all_ranks = has_path(config, "actor_rollout_ref.actor.profiler.all_ranks")
    user_set_actor_profiler_ranks = has_path(config, "actor_rollout_ref.actor.profiler.ranks")
    if auto_merge_verl_defaults:
        default_config = _load_verl_section_defaults(resolved_strategy)
        user_config = _extract_supported_config_sections(config)
        config = OmegaConf.merge(default_config, user_config)
    OmegaConf.set_struct(config, False)

    if strategy_is_missing:
        OmegaConf.update(config, "actor_rollout_ref.actor.strategy", resolved_strategy, merge=True)
    apply_tinker_server_overrides(
        config,
        user_set_actor_profiler_enable=user_set_actor_profiler_enable,
        user_set_actor_profiler_tool=user_set_actor_profiler_tool,
        user_set_actor_profiler_save_path=user_set_actor_profiler_save_path,
        user_set_actor_profiler_all_ranks=user_set_actor_profiler_all_ranks,
        user_set_actor_profiler_ranks=user_set_actor_profiler_ranks,
    )
    _normalize_teacher_model_identifiers(config)
    return config


def apply_tinker_server_overrides(
    config: DictConfig,
    *,
    user_set_actor_profiler_enable: bool = False,
    user_set_actor_profiler_tool: bool = False,
    user_set_actor_profiler_save_path: bool = False,
    user_set_actor_profiler_all_ranks: bool = False,
    user_set_actor_profiler_ranks: bool = False,
) -> None:
    """Apply only Tinker-server-specific config adjustments."""

    _apply_server_defaults(config)
    _disable_verl_model_offload(config)
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
    _set_actor_checkpoint_contents(config)
    _configure_actor_profiler_from_global_config(
        config,
        user_set_actor_profiler_enable=user_set_actor_profiler_enable,
        user_set_actor_profiler_tool=user_set_actor_profiler_tool,
        user_set_actor_profiler_save_path=user_set_actor_profiler_save_path,
        user_set_actor_profiler_all_ranks=user_set_actor_profiler_all_ranks,
        user_set_actor_profiler_ranks=user_set_actor_profiler_ranks,
    )


def _apply_server_defaults(config: DictConfig) -> None:
    if "server" not in config:
        config.server = {}

    default_server = OmegaConf.create(DEFAULT_SERVER_CONFIG)
    config.server = OmegaConf.merge(default_server, config.server)
    for key, value in DEFAULT_SERVER_CONFIG.items():
        if _is_missing(config.server.get(key)):
            OmegaConf.update(config, f"server.{key}", value, merge=True)


def _disable_verl_model_offload(config: DictConfig) -> None:
    # The Tinker server owns the model offload/onload lifecycle. Disable Verl's
    # per-worker offload settings so ``server.enable_offload`` is the single knob
    # controlling this behavior and the two mechanisms cannot fight each other.
    for path in _VERL_OFFLOAD_FALSE_PATHS:
        if has_path(config, path):
            OmegaConf.update(config, path, False, merge=True)


def _resolve_actor_strategy(config: DictConfig) -> str:
    strategy = _select(config, "actor_rollout_ref.actor.strategy")
    if not _is_missing(strategy):
        return _VERL_STRATEGY_ALIASES.get(str(strategy), str(strategy))

    model_engine = _select(config, "model_engine")
    if not _is_missing(model_engine):
        model_engine = str(model_engine)
        if model_engine in _VERL_SECTION_DEFAULT_CONFIG_BY_STRATEGY:
            return _VERL_STRATEGY_ALIASES.get(model_engine, model_engine)

    actor = _select(config, "actor_rollout_ref.actor", {})
    if isinstance(actor, DictConfig):
        if "fsdp_config" in actor:
            return "fsdp"
        if "veomni" in actor:
            return "veomni"

    return "veomni"


def _load_verl_section_defaults(strategy: str) -> DictConfig:
    config_name = _VERL_SECTION_DEFAULT_CONFIG_BY_STRATEGY.get(strategy)
    if config_name is None:
        supported = ", ".join(sorted(_VERL_SECTION_DEFAULT_CONFIG_BY_STRATEGY))
        raise ValueError(
            f"actor_rollout_ref.actor.strategy={strategy!r} is not supported by the "
            f"Tinker server config merge helper. Supported strategies: {supported}."
        )

    generated = OmegaConf.load(files("verl.trainer.config").joinpath(config_name))
    defaults = OmegaConf.create({})
    for section in _DEFAULT_TOP_LEVEL_SECTIONS:
        if section in generated:
            defaults[section] = generated[section]
    return defaults


def _extract_supported_config_sections(config: DictConfig) -> DictConfig:
    supported = OmegaConf.create({})
    for section in _SUPPORTED_TOP_LEVEL_SECTIONS:
        if section in config:
            supported[section] = config[section]

    if bool(config.get("critic", {}).get("enable", False)):
        supported.critic = {"enable": True}

    return supported


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


def _set_actor_checkpoint_contents(config: DictConfig) -> None:
    # The Tinker server owns checkpoint recovery and must save everything needed
    # to recreate the complete actor after a load_state. Override both lists instead
    # of honoring a partial Verl checkpoint configuration that cannot be resumed.
    for key in ("save_contents", "load_contents"):
        path = f"actor_rollout_ref.actor.checkpoint.{key}"
        OmegaConf.update(config, path, list(_ACTOR_CHECKPOINT_CONTENTS), merge=True)


def _configure_actor_profiler_from_global_config(
    config: DictConfig,
    *,
    user_set_actor_profiler_enable: bool,
    user_set_actor_profiler_tool: bool,
    user_set_actor_profiler_save_path: bool,
    user_set_actor_profiler_all_ranks: bool,
    user_set_actor_profiler_ranks: bool,
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

    all_ranks = _select(config, "global_profiler.all_ranks")
    if not user_set_actor_profiler_all_ranks and not _is_missing(all_ranks):
        OmegaConf.update(config, "actor_rollout_ref.actor.profiler.all_ranks", bool(all_ranks), merge=True)

    ranks = _select(config, "global_profiler.ranks")
    if not user_set_actor_profiler_ranks and not _is_missing(ranks):
        ranks_list = list(ranks) if isinstance(ranks, (list, ListConfig)) else [ranks]
        OmegaConf.update(config, "actor_rollout_ref.actor.profiler.ranks", ranks_list, merge=True)


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

    disk_check_min_free_gb = config.get("server", {}).get("disk_check_min_free_gb", 0)
    try:
        disk_check_min_free_gb = float(disk_check_min_free_gb)
        if disk_check_min_free_gb < 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("server.disk_check_min_free_gb must be a non-negative number")

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

    if bool(config.get("distillation", {}).get("enabled", False)):
        teacher_identifier_errors = _normalize_teacher_model_identifiers(config)
        errors.extend(teacher_identifier_errors)
        try:
            if not teacher_identifier_errors:
                distillation_config = _to_verl_distillation_config(config.distillation)
                for teacher in distillation_config.teacher_models.values():
                    if teacher.inference.name not in {"vllm", "sglang"}:
                        raise ValueError(
                            f"teacher inference engine {teacher.inference.name!r} is unsupported; "
                            "use 'vllm' or 'sglang'"
                        )
        except Exception as e:
            errors.append(f"Teacher config validation: {e}")

    if is_no_rollout_deployment(config):
        if not is_enable_false(config, "actor_rollout_ref.ref"):
            errors.append("no_rollout_deployment does not support reference policy")
    else:
        try:
            _validate_supported_verl_config(config, need_reference_policy(config))
        except Exception as e:
            errors.append(f"VeRL config validation: {e}")

    return errors


def _normalize_teacher_model_identifiers(config: DictConfig) -> list[str]:
    """Fill missing teacher name/path aliases and report fully unidentified teachers."""
    if not bool(config.get("distillation", {}).get("enabled", False)):
        return []

    errors = []
    teacher_models = config.get("distillation", {}).get("teacher_models", {})
    for key, teacher in teacher_models.items():
        # VeRL merges a placeholder named ``teacher_model`` into every config
        # and removes it when named multi-teacher entries are present.
        if key == "teacher_model" and len(teacher_models) > 1:
            continue
        model_name = teacher.get("model_name")
        model_path = teacher.get("model_path")
        name_missing = _is_missing(model_name)
        path_missing = _is_missing(model_path)
        if name_missing and path_missing:
            errors.append(f"distillation.teacher_models.{key} requires at least one of model_name or model_path")
        elif name_missing:
            teacher.model_name = model_path
        elif path_missing:
            teacher.model_path = model_name
    return errors


def _to_verl_distillation_config(distillation_config: DictConfig):
    """Convert Tinker's extended teacher config to VeRL's path-only dataclass."""
    # Resolve while the node is still attached to the root config so VeRL's
    # cross-section interpolations (for example actor rollout lengths) retain
    # their values in the detached copy.
    verl_config = OmegaConf.create(OmegaConf.to_container(distillation_config, resolve=True))
    # Tinker can optionally give each teacher its own strictly packed Ray pool;
    # this is a server placement concern rather than part of VeRL's dataclass.
    verl_config.pop("dedicated_resource_pools", None)
    for teacher in verl_config.get("teacher_models", {}).values():
        teacher.pop("model_name", None)
    return omega_conf_to_dataclass(verl_config)


def _validate_supported_verl_config(config: DictConfig, use_reference_policy: bool) -> None:
    """Run the subset of Verl validation that applies to Tinker-supported roles."""

    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        rollout_n = OmegaConf.select(config, "actor_rollout_ref.rollout.n", default=1) or 1
        real_train_batch_size = config.data.train_batch_size * rollout_n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            _check_log_prob_micro_batch(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        _check_log_prob_micro_batch(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    if OmegaConf.select(config, "actor_rollout_ref.rollout.val_kwargs.do_sample", default=False):
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    lora_config = config.actor_rollout_ref.model.get("lora") or {}
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_supported_verl_config] All configuration checks passed successfully!")


def _check_log_prob_micro_batch(mbs: Any, mbs_per_gpu: Any, name: str) -> None:
    param = "log_prob_micro_batch_size"
    param_per_gpu = f"{param}_per_gpu"

    if mbs is None and mbs_per_gpu is None:
        raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

    if mbs is not None and mbs_per_gpu is not None:
        raise ValueError(
            f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
            f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
        )


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


def load_config(config_path: str | Path) -> DictConfig:
    """Load and resolve a Tinker YAML config using the server startup rules."""

    config_path = Path(config_path).expanduser()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"--config must point to a YAML file, got: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected top-level YAML mapping in {config_path}")

    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and render a processed verl_tinker config.")
    parser.add_argument("--config", required=True, help="Path to a YAML verl_tinker config.")
    args = parser.parse_args(argv)

    config = process_config(load_config(args.config))
    print("Config validation succeeded. Final processed config:\n")
    print(OmegaConf.to_yaml(config, resolve=True), end="")


if __name__ == "__main__":
    # utility entrypoint for people to validate their config and see how it will
    # be transformed
    main()
