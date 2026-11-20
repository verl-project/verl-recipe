# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Typed dataclass + DictConfig adapter for the DR.Kernel batch filter.

`BatchFilterConfig` mirrors the surface of `recipe.drkernel.filters.unified_filter
.PPOFilterConfig` and adds a single top-level `enable` flag that the trainer
hook checks before running the pipeline.

`build_ppo_filter_config(omega_cfg)` translates an OmegaConf `DictConfig`
(loaded from the yaml fragment under `algorithm.batch_filter`) into a
`PPOFilterConfig` instance ready for `PPOBatchFilter(...)`.

Why this lives here instead of in `verl/trainer/config/algorithm.py`:
    Mirrors the `async_training` pattern from fully_async_policy — the
    recipe owns its own typed config; verl core only needs the bare
    `algorithm.batch_filter` field as a passthrough so OmegaConf strict-
    mode validation accepts the new sub-tree.
"""

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from recipe.drkernel.filters.unified_filter import PPOFilterConfig

from verl.base_config import BaseConfig


@dataclass
class BatchFilterConfig(BaseConfig):
    """Schema for the `algorithm.batch_filter` config sub-tree.

    Field semantics match `PPOFilterConfig` with one addition: `enable`
    gates the entire pipeline. When False (the default), the trainer hook
    is a no-op and no filter pipeline is constructed.
    """

    # Master switch. When False, the trainer hook returns the batch unchanged.
    enable: bool = False

    # ---- mirrors PPOFilterConfig ----
    sample_selection_strategy: str = "efficiency_stochastic"
    target_group_size: int = 8
    min_group_size: Optional[int] = None
    target_num_groups: Optional[int] = None

    reward_threshold: Optional[float] = None
    max_response_length: Optional[int] = None
    reject_low_variance_groups: bool = True
    remove_clip: bool = False
    min_rollout_n: Optional[int] = None

    enable_two_gate_filter: bool = False
    gate1_enabled: bool = True
    gate1_bias_epsilon: float = 0.01
    gate2_enabled: bool = True
    gate2_instability_threshold: float = -15.0

    save_metrics: bool = True
    log_rejection_reasons: bool = False


# Fields that BatchFilterConfig adds on top of PPOFilterConfig — must be
# stripped before constructing the upstream dataclass.
_RECIPE_ONLY_FIELDS = frozenset({"enable"})


def build_ppo_filter_config(omega_cfg: Any) -> PPOFilterConfig:
    """Convert a config blob into a PPOFilterConfig.

    Accepts:
    - DictConfig (the typical case — e.g. `self.config.algorithm.batch_filter`)
    - dict
    - BatchFilterConfig
    - PPOFilterConfig (returned as-is)

    The `enable` flag is dropped — callers should check it themselves before
    invoking this function. Unknown keys are dropped with a warning rather
    than raising, so future yaml additions don't crash older recipe code.
    """
    if isinstance(omega_cfg, PPOFilterConfig):
        return omega_cfg

    if isinstance(omega_cfg, BatchFilterConfig):
        as_dict = {k: getattr(omega_cfg, k) for k in omega_cfg}
    elif isinstance(omega_cfg, DictConfig):
        as_dict = OmegaConf.to_container(omega_cfg, resolve=True)
    elif isinstance(omega_cfg, dict):
        as_dict = dict(omega_cfg)
    else:
        raise TypeError(
            f"build_ppo_filter_config expected DictConfig|dict|BatchFilterConfig|"
            f"PPOFilterConfig, got {type(omega_cfg).__name__}"
        )

    valid_keys = {f for f in PPOFilterConfig.__dataclass_fields__}
    kwargs = {}
    for key, val in as_dict.items():
        if key in _RECIPE_ONLY_FIELDS:
            continue
        if key in valid_keys:
            kwargs[key] = val
        # Silently drop unknown keys — yaml may carry forward-looking fields.

    return PPOFilterConfig(**kwargs)
