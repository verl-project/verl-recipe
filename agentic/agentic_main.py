"""Agentic recipe: PPO training with an LLM proxy server for remote agents.

This recipe extends the standard PPO training flow by starting a
:class:`ProxyServerActor` (a Ray named actor on the head node) **after**
``trainer.init_workers()`` produces ``server_handles``.  The proxy
bridges OpenAI-compatible HTTP requests from remote agents to verl's
vLLM rollout servers.

Usage::

    python -m recipe.agentic.agentic_main
"""

import json
import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

# yaml 字段 → RemoteAgentLoop / proxy 使用的环境变量名的映射。
# 仅当 yaml 字段不为 None 且 shell 未设同名 env 时才注入，
# 以保证：命令行 override > shell 环境变量 > yaml 默认值。
_REMOTE_AGENT_ENV_MAP: dict[str, str] = {
    "agent_name": "REMOTE_AGENT_NAME",
    "agent_import_path": "REMOTE_AGENT_IMPORT_PATH",
    "model_name": "REMOTE_MODEL_NAME",
    "agent_kwargs": "REMOTE_AGENT_KWARGS",
    "max_retries": "REMOTE_AGENT_MAX_RETRIES",
    "retry_base_delay": "REMOTE_AGENT_RETRY_BASE_DELAY",
    "task_path_template": "REMOTE_AGENT_TASK_PATH_TEMPLATE",
    "environment_overrides": "REMOTE_AGENT_ENVIRONMENT_OVERRIDES",
    "environment_kwargs": "REMOTE_AGENT_ENVIRONMENT_KWARGS",
    "use_local_trial": "REMOTE_AGENT_USE_LOCAL_TRIAL",
    "proxy_server_url": "PROXY_SERVER_URL",
    "harbor_server_url": "HARBOR_SERVER_URL",
    "harbor_timeout": "HARBOR_TIMEOUT",
}
# 这些字段在 RemoteAgentConfig.from_env 中是 JSON 解码的，需要 dump 为字符串。
_REMOTE_AGENT_JSON_FIELDS = {
    "agent_kwargs",
    "environment_overrides",
    "environment_kwargs",
}


def _yaml_value_to_env_str(field: str, value) -> str:
    """将 yaml 中的值转成 RemoteAgentConfig.from_env 能识别的字符串。"""
    if field in _REMOTE_AGENT_JSON_FIELDS:
        # OmegaConf container -> plain python -> json
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def collect_yaml_env_overrides(config) -> dict[str, str]:
    """从 config 中收集需要注入的环境变量，仅覆盖 shell 中未设的项。”"""
    env_vars: dict[str, str] = {}

    proxy_cfg = config.get("proxy_server", {}) or {}
    llm_proxy_ip = (
        proxy_cfg.get("llm_proxy_ip", None) if hasattr(proxy_cfg, "get") else None
    )
    if llm_proxy_ip not in (None, "") and not os.environ.get("LLM_PROXY_IP"):
        env_vars["LLM_PROXY_IP"] = str(llm_proxy_ip)

    remote_agent_cfg = config.get("remote_agent", {}) or {}
    if hasattr(remote_agent_cfg, "get"):
        for field, env_name in _REMOTE_AGENT_ENV_MAP.items():
            value = remote_agent_cfg.get(field, None)
            if value is None:
                continue
            if os.environ.get(env_name):
                # shell 优先
                continue
            env_vars[env_name] = _yaml_value_to_env_str(field, value)

    return env_vars


def _materialize_harbor_datasets(config) -> None:
    """将 Harbor 本地任务目录物化为 parquet，并写回 config.data.train_files / val_files。

    仅在 ``config.data.train_harbor_dir`` / ``val_harbor_dir`` 任意一个被设置时生效。
    未设置的一侧保持原有 ``train_files`` / ``val_files`` 不变。
    """
    data_cfg = config.get("data", {}) or {}
    if not hasattr(data_cfg, "get"):
        return

    train_root = data_cfg.get("train_harbor_dir", None)
    val_root = data_cfg.get("val_harbor_dir", None)
    if not train_root and not val_root:
        return

    from recipe.agentic.dataset import build_verl_parquets

    cache_dir = data_cfg.get("harbor_cache_dir", "~/.cache/verl/agentic/harbor")
    cache_dir = os.path.expanduser(str(cache_dir))

    train_path, val_path = build_verl_parquets(
        train_root=os.path.expanduser(str(train_root)) if train_root else None,
        val_root=os.path.expanduser(str(val_root)) if val_root else None,
        cache_dir=cache_dir,
        data_source=data_cfg.get("harbor_data_source", "harbor"),
        ability=data_cfg.get("harbor_ability", "agent"),
        train_limit=data_cfg.get("harbor_train_limit", None),
        val_limit=data_cfg.get("harbor_val_limit", None),
        overwrite=bool(data_cfg.get("harbor_overwrite", False)),
    )

    if train_path is not None:
        print(f"[agentic] harbor train parquet -> {train_path}")
        OmegaConf.update(config, "data.train_files", [str(train_path)], merge=False)
    if val_path is not None:
        print(f"[agentic] harbor val parquet   -> {val_path}")
        OmegaConf.update(config, "data.val_files", [str(val_path)], merge=False)


@hydra.main(config_path="config", config_name="agentic_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # 将 yaml 中的 remote_agent / proxy_server.llm_proxy_ip 注入为 Ray worker 环境变量，
        # 同时设到当前 driver 进程，使 RemoteAgentConfig.from_env() 在任何进程都能读到。
        yaml_env = collect_yaml_env_overrides(config)
        if yaml_env:
            runtime_env_vars = dict(runtime_env_kwargs.get("env_vars", {}))
            for k, v in yaml_env.items():
                runtime_env_vars.setdefault(k, v)
                os.environ.setdefault(k, v)
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    taskrunner = TaskRunner()
    taskrunner.run(config)

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner(BaseTaskRunner):
    """Agentic TaskRunner: starts the proxy server after init_workers().

    Inherits the standard PPO ``TaskRunner`` and overrides ``run()`` to
    insert a ``start_proxy_server()`` call between ``init_workers()``
    and ``fit()``.  The proxy server receives the ``server_handles``
    produced by ``AgentLoopManager`` and exposes an HTTP endpoint that
    remote agents can use as an OpenAI-compatible ``base_url``.
    """

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 若配了 Harbor 本地任务目录，先生成 parquet 并覆盖 train_files / val_files。
        _materialize_harbor_datasets(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        # ----- Agentic-specific: start the LLM proxy server -----
        server_handles = trainer.async_rollout_manager.server_handles
        proxy_cfg = config.get("proxy_server", {})

        from recipe.agentic.proxyserver.ray_actor import start_proxy_server

        proxy_url = start_proxy_server(
            server_handles=server_handles,
            model_path=config.actor_rollout_ref.model.path,
            host=proxy_cfg.get("host", "0.0.0.0"),
            port=proxy_cfg.get("port", 0),
            tool_format=proxy_cfg.get("tool_format", "hermes"),
        )
        print(f"Proxy server started at {proxy_url}")

        trainer.fit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
