# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime patch for this recipe, in the style of ``nemo_gym/server_patch.py``.

Registered advantage estimators receive ``non_tensor_batch`` only for ``gdpo``
(`verl/trainer/ppo/ray_trainer.py`, the ``else`` branch of
``compute_advantage``), so the ``env_invalid`` flags this recipe produces never
reach the estimator. Until that passthrough exists upstream, this wrapper routes
GRPO through :mod:`group_stats` whenever the batch actually carries the flags,
and forwards every other batch to the original function untouched.

Nothing is patched unless a batch carries the flags, so other recipes and stock
GRPO runs are unaffected even when this module is loaded.

Applied by :func:`install`, which is idempotent. ``sitecustomize.py`` next to
this file calls it on interpreter start, because Ray workers and agent-loop
actors are separate processes that never see a driver-side import;
``submit_webvoyager.sh`` puts this directory on ``PYTHONPATH`` for that reason.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False


def _group_stats_module():
    """Import the sibling estimator both as a package member and as a top-level module."""
    try:
        from . import group_stats
    except ImportError:
        import group_stats
    return group_stats


def _env_invalid_flags(non_tensor_batch) -> object | None:
    """Return the per-sample flags, or None when this batch does not carry them."""
    import numpy as np

    raw = (non_tensor_batch or {}).get("env_invalid")
    if raw is None:
        return None
    return np.array([bool(flag) for flag in raw], dtype=bool)


def _patch_compute_advantage() -> None:
    import verl.trainer.ppo.ray_trainer as ray_trainer
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    original = ray_trainer.compute_advantage
    if getattr(original, "_nemo_gym_browser_patched", False):
        return

    group_stats = _group_stats_module()

    def compute_advantage(data, adv_estimator=None, *args, **kwargs):
        is_grpo = adv_estimator in (AdvantageEstimator.GRPO, "grpo", "grpo_env_aware")
        flags = _env_invalid_flags(data.non_tensor_batch) if is_grpo else None
        if flags is None:
            return original(data, adv_estimator, *args, **kwargs)

        advantages, returns = group_stats.compute_grpo_env_aware_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            config=kwargs.get("config"),
            non_tensor_batch=data.non_tensor_batch,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    compute_advantage._nemo_gym_browser_patched = True
    ray_trainer.compute_advantage = compute_advantage

    # `verl/trainer/ppo/v1/utils.py` does `from ...ray_trainer import compute_advantage`,
    # so a module already imported holds the original reference. Rebind it too.
    v1_utils = __import__("sys").modules.get("verl.trainer.ppo.v1.utils")
    if v1_utils is not None and hasattr(v1_utils, "compute_advantage"):
        v1_utils.compute_advantage = compute_advantage


def install() -> None:
    """Apply the patch. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("NEMO_GYM_BROWSER_DISABLE_PATCHES") == "1":
        return
    _patch_compute_advantage()
    _INSTALLED = True
    logger.info("nemo_gym/browser: env-aware advantage routing installed")
