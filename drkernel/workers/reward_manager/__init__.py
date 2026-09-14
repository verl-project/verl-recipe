# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

# Kernel-RL reward managers — DR.Kernel-native pattern.
#
# `AsyncKernelRewardManager` (in `kernel_async.py`) is DR.Kernel's
# per-trajectory reward manager verbatim. It owns the `__call__` that
# dispatches to a `compute_score` callable (= `compute_kernel_reward_batch`
# in this recipe) and assembles `reward_extra_info`.
#
# `KernelAsyncRewardManager` (in `kernel_async_adapter.py`) is the thin
# `RewardManagerBase` adapter the experimental `RewardLoopWorker` registry
# expects; it wraps `AsyncKernelRewardManager` and exposes
# `async run_single(data)`. Registered as `kernel_async` and loaded via
# importlib by `drkernel_kernel_trainer_native.yaml`.

from .kernel_async import AsyncKernelRewardManager  # noqa: F401
from .kernel_async_adapter import KernelAsyncRewardManager  # noqa: F401
