"""Configuration expansion and fail-fast checks for the HACPO recipe."""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from .hacpo_core_algos import HacpoLossConfig


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """One policy's immutable public identity and expanded verl config."""

    agent_id: str
    config: DictConfig
    worker_key: str


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def build_agent_runtime_specs(config: DictConfig) -> tuple[AgentRuntimeSpec, ...]:
    """Expand the shared verl config into one full config per HACPO agent."""

    entries = config.hacpo.get("agents", None)
    if not entries:
        raise ValueError("hacpo.agents must declare at least one policy")

    base = OmegaConf.to_container(config, resolve=False)
    specs = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        agent_id = str(entry.get("id", "")).strip()
        model_path = entry.get("model_path", None)
        if not agent_id or not model_path or agent_id in seen_ids:
            raise ValueError(f"hacpo.agents[{index}] requires a unique id and model_path")
        seen_ids.add(agent_id)

        agent_config = OmegaConf.create(base)
        actor_rollout_ref_override = entry.get("actor_rollout_ref", {})
        with open_dict(agent_config):
            agent_config.actor_rollout_ref = OmegaConf.merge(
                agent_config.actor_rollout_ref,
                actor_rollout_ref_override,
            )
            agent_config.actor_rollout_ref.model.path = str(model_path)
            tokenizer_path = entry.get("tokenizer_path", None)
            if tokenizer_path:
                agent_config.actor_rollout_ref.model.tokenizer_path = str(tokenizer_path)
            apply_chat_template_kwargs = entry.get("apply_chat_template_kwargs", None)
            if apply_chat_template_kwargs is not None:
                agent_config.data.apply_chat_template_kwargs = _plain(apply_chat_template_kwargs)
        OmegaConf.resolve(agent_config)
        specs.append(
            AgentRuntimeSpec(
                agent_id=agent_id,
                config=agent_config,
                worker_key=f"policy{index}",
            )
        )
    return tuple(specs)


def build_loss_config(config: DictConfig) -> HacpoLossConfig:
    loss = config.hacpo.loss
    return HacpoLossConfig(
        self_clip_low=float(loss.self_clip_low),
        self_clip_high=float(loss.self_clip_high),
        cross_clip_delta=float(loss.cross_clip_delta),
        cross_clip_step=float(loss.cross_clip_step),
        alpha=float(loss.alpha),
        max_abs_log_ratio=(None if loss.get("max_abs_log_ratio", None) is None else float(loss.max_abs_log_ratio)),
    )


def select_validation_agent_specs(
    config: DictConfig,
    specs: tuple[AgentRuntimeSpec, ...],
) -> tuple[tuple[int, AgentRuntimeSpec], ...]:
    """Select validation policies without changing checkpoint topology."""

    requested = config.hacpo.get("validation_agent_ids", None)
    indexed_specs = {spec.agent_id: (index, spec) for index, spec in enumerate(specs)}
    if requested is None:
        return tuple(enumerate(specs))

    requested_ids = [str(agent_id).strip() for agent_id in requested]
    if (
        not requested_ids
        or len(set(requested_ids)) != len(requested_ids)
        or any(not agent_id or agent_id not in indexed_specs for agent_id in requested_ids)
    ):
        raise ValueError("hacpo.validation_agent_ids must contain unique, declared agent ids")
    return tuple(indexed_specs[agent_id] for agent_id in requested_ids)


def select_runtime_agent_specs(
    config: DictConfig,
    specs: tuple[AgentRuntimeSpec, ...],
) -> tuple[AgentRuntimeSpec, ...]:
    """Start only requested policies for validation-only jobs."""

    if not bool(config.trainer.get("val_only", False)):
        return specs
    return tuple(spec for _, spec in select_validation_agent_specs(config, specs))


def validate_hacpo_config(
    config: DictConfig,
    specs: tuple[AgentRuntimeSpec, ...],
) -> None:
    """Validate only the settings required by the shared HACPO runtime."""

    parameter_sync_step = int(config.trainer.v1.sync.get("parameter_sync_step", 1))
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync" or parameter_sync_step != 1:
        raise ValueError("HACPO requires the synchronous V1 trainer with parameter_sync_step=1")
    if (
        str(config.algorithm.adv_estimator).lower() != "grpo"
        or bool(config.critic.enable)
        or config.algorithm.use_kl_in_reward
        or config.reward.reward_model.enable
    ):
        raise ValueError("HACPO requires critic-free GRPO, rule-based rewards, and no KL reward penalty")
    if not bool(config.data.get("return_raw_chat", False)):
        raise ValueError("HACPO text exchange requires data.return_raw_chat=true")
    select_validation_agent_specs(config, specs)

    world_size = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    train_prompt_batch = int(config.data.train_batch_size)

    rollout_counts = set()
    for spec in specs:
        agent_config = spec.config
        actor = agent_config.actor_rollout_ref.actor
        rollout = agent_config.actor_rollout_ref.rollout
        model = agent_config.actor_rollout_ref.model
        if str(actor.strategy) != "fsdp" or str(rollout.name) not in ("vllm", "sglang") or str(rollout.mode) != "async":
            raise ValueError(f"agent {spec.agent_id!r} requires FSDP with an async vLLM or SGLang rollout")
        if (
            str(rollout.checkpoint_engine.backend) != "naive"
            or not rollout.free_cache_engine
            or not actor.fsdp_config.param_offload
            or not actor.fsdp_config.optimizer_offload
        ):
            raise ValueError(f"agent {spec.agent_id!r} requires the shared-GPU offload configuration")
        if str(actor.loss_agg_mode) != "seq-mean-token-mean":
            raise ValueError(f"agent {spec.agent_id!r} requires loss_agg_mode=seq-mean-token-mean")
        lora_rank = model.get("lora", {}).get("rank", 0) or model.get("lora_rank", 0)
        if (
            lora_rank > 0
            or model.get("lora_adapter_path") is not None
            or rollout.multi_turn.enable
            or rollout.disaggregation.enabled
        ):
            raise ValueError(f"agent {spec.agent_id!r} must use full-parameter, single-turn rollouts")
        rollout_n = int(rollout.n)
        if rollout_n <= 0:
            raise ValueError(f"agent {spec.agent_id!r} rollout.n must be positive")
        rollout_counts.add(rollout_n)
        source_batch = train_prompt_batch * rollout_n
        sequence_parallel_size = int(actor.fsdp_config.ulysses_sequence_parallel_size)
        if sequence_parallel_size <= 0 or world_size <= 0 or world_size % sequence_parallel_size != 0:
            raise ValueError(
                f"agent {spec.agent_id!r} has invalid Ulysses sequence parallel size "
                f"{sequence_parallel_size} for world size {world_size}"
            )
        data_parallel_size = world_size // sequence_parallel_size
        if source_batch % data_parallel_size != 0:
            raise ValueError(
                f"agent {spec.agent_id!r} source batch {source_batch} must be divisible by "
                f"data-parallel size {data_parallel_size}"
            )

    if len(rollout_counts) != 1:
        raise ValueError("all agents must use the same rollout.n in paper-exact HACPO mode")
    rollout_n = rollout_counts.pop()

    num_agents = len(specs)
    for spec in specs:
        target = spec.agent_id
        target_config = spec.config
        target_batch = train_prompt_batch * rollout_n * num_agents
        target_actor = target_config.actor_rollout_ref.actor
        mini_batch = int(target_actor.ppo_mini_batch_size) * rollout_n
        sequence_parallel_size = int(target_actor.fsdp_config.ulysses_sequence_parallel_size)
        data_parallel_size = world_size // sequence_parallel_size
        if bool(target_actor.shuffle):
            raise ValueError(f"agent {target!r} requires actor.shuffle=false for stepwise clipping")
        source_block = train_prompt_batch * rollout_n
        if mini_batch % data_parallel_size != 0 or target_batch % mini_batch != 0 or source_block % mini_batch != 0:
            raise ValueError(
                f"agent {target!r} has incompatible batch sizes: target={target_batch}, "
                f"source={source_block}, mini_batch={mini_batch}, data_parallel={data_parallel_size}"
            )

    build_loss_config(config)
    overflow = str(config.hacpo.trajectory.overflow)
    if overflow not in ("error", "truncate"):
        raise ValueError("hacpo.trajectory.overflow must be 'error' or 'truncate'")
    for field_name in ("target_max_prompt_length", "target_max_response_length"):
        value = config.hacpo.trajectory.get(field_name, None)
        if value is not None and int(value) <= 0:
            raise ValueError(f"hacpo.trajectory.{field_name} must be positive or null")


def public_agent_summary(specs: tuple[AgentRuntimeSpec, ...]) -> dict[str, Any]:
    """Stable checkpoint identity for the all-to-all HACPO runtime."""

    return {
        "agents": [
            {
                "id": spec.agent_id,
                "model_path": str(spec.config.actor_rollout_ref.model.path),
            }
            for spec in specs
        ]
    }
