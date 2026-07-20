#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
"""Async Flow 配置类定义."""

from dataclasses import dataclass, field

from verl.base_config import BaseConfig


@dataclass
class AsyncFlowGRPOConfig(BaseConfig):
    """AsyncFlow GRPO 训练配置.

    完整的异步流式训练配置，包括批处理、staleness 控制。
    """

    # Staleness control
    staleness: int = 0
    # partial rollout related
    rollout_max_resume_attempts: int = 1
    wait_for_inflight_requests: bool = True
    # Transfer Queue 配置
    experience_topic: str = "experience"
    ref_experience_count: int = 4
    fwd_experience_count: int = 4
    reward_experience_count: int = 4

    def validate(self, config):
        """
        config: 全量配置
        """
        n_sample = config.actor_rollout_ref.rollout.n
        train_batch_size = config.data.train_batch_size
        total_num_per_batch = train_batch_size * n_sample

        # 验证 experience count 必须能整除 total_num_per_batch
        assert total_num_per_batch % self.ref_experience_count == 0, (
            f"train_batch_size * n_sample ({total_num_per_batch}) must divided by ({self.ref_experience_count=})"
        )
        assert total_num_per_batch % self.fwd_experience_count == 0, (
            f"train_batch_size * n_sample ({total_num_per_batch}) must divided by ({self.fwd_experience_count=})"
        )
        assert total_num_per_batch % self.reward_experience_count == 0, (
            f"train_batch_size * n_sample ({total_num_per_batch}) must divided by ({self.reward_experience_count=})"
        )

        assert self.reward_experience_count % n_sample == 0, (
            f"{self.reward_experience_count=}) must divided by ({n_sample=})"
        )

        # 验证 staleness 必须非负
        assert self.staleness >= 0, f"staleness ({self.staleness}) must >= 0"

        # 验证 experience_count 都必须大于 0
        assert self.ref_experience_count > 0, f"ref_experience_count ({self.ref_experience_count}) must > 0"
        assert self.fwd_experience_count > 0, f"fwd_experience_count ({self.fwd_experience_count}) must > 0"
        assert self.reward_experience_count > 0, f"reward_experience_count ({self.reward_experience_count}) must > 0"


@dataclass
class RoleResourceConfig:
    """单个角色的资源配置."""

    nnodes: int = 1
    n_gpus_per_node: int = 1
    use_gpu: bool = True
    num_cpus: int = 0

    def __init__(self, nnodes: int = 1, n_gpus_per_node: int = 1, use_gpu: bool = True, num_cpus: int = 0, **kwargs):
        self.nnodes = nnodes
        self.n_gpus_per_node = n_gpus_per_node
        self.use_gpu = use_gpu
        self.num_cpus = num_cpus


@dataclass
class AsyncFlowResourceConfig(BaseConfig):
    """异步流式训练资源配置.

    使用嵌套结构来定义每个角色的资源配置.
    """

    actor_fwd: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    ref_fwd: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    advantage: RoleResourceConfig = field(
        default_factory=lambda: RoleResourceConfig(use_gpu=False, num_cpus=4, n_gpus_per_node=0)
    )
    actor_train: RoleResourceConfig = field(default_factory=RoleResourceConfig)

    def __post_init__(self):
        # 确保 OmegaConf 中的嵌套配置被正确转换为 RoleResourceConfig
        for role_name in ["actor_fwd", "ref_fwd", "advantage", "actor_train"]:
            role_config = getattr(self, role_name)
            if role_config is not None and not isinstance(role_config, RoleResourceConfig):
                if hasattr(role_config, "__dict__"):
                    # OmegaConf 对象转字典
                    cfg_dict = dict(role_config)
                elif isinstance(role_config, dict):
                    cfg_dict = role_config
                else:
                    cfg_dict = {}
                setattr(self, role_name, RoleResourceConfig(**cfg_dict))

    def validate(self):
        """验证资源配置的合法性"""
        for role, cfg in [
            ("actor_fwd", self.actor_fwd),
            ("ref_fwd", self.ref_fwd),
            ("advantage", self.advantage),
            ("actor_train", self.actor_train),
        ]:
            assert cfg.nnodes >= 0, f"{role}.nnodes ({cfg.nnodes}) must >= 0"
            assert cfg.n_gpus_per_node >= 0, f"{role}.n_gpus_per_node ({cfg.n_gpus_per_node}) must >= 0"
            if isinstance(cfg, RoleResourceConfig):
                assert cfg.num_cpus > 0 if not cfg.use_gpu else True, (
                    f"{role}.num_cpus ({cfg.num_cpus}) must > 0 when use_gpu is False"
                )

        # 验证核心角色必须启用
        assert is_role_enabled(self, "actor_train"), "actor_train 角色必须启用（actor_train.nnodes > 0）"
        assert is_role_enabled(self, "actor_fwd"), "actor_fwd 角色必须启用（actor_fwd.nnodes > 0）"


def get_resource_pool_spec(config: AsyncFlowResourceConfig, role: str) -> list:
    """获取指定角色的资源池规范."""
    role_config = {
        "actor_fwd": config.actor_fwd,
        "ref_fwd": config.ref_fwd,
        "advantage": config.advantage,
        "actor_train": config.actor_train,
    }
    if role not in role_config:
        raise ValueError(f"Unknown role: {role}")
    role_cfg = role_config[role]
    return [role_cfg.n_gpus_per_node] * role_cfg.nnodes


def is_role_enabled(config: AsyncFlowResourceConfig, role: str) -> bool:
    """检查指定角色是否启用."""
    role_config = {
        "actor_fwd": config.actor_fwd,
        "ref_fwd": config.ref_fwd,
        "advantage": config.advantage,
        "actor_train": config.actor_train,
    }
    cfg = role_config.get(role)
    return cfg.nnodes > 0 if cfg else False
