# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
DataProto <-> dict-of-tensors adapter for PPOBatchFilter.

PPOBatchFilter (ported from DR.Kernel) operates on a `dict[str, torch.Tensor]`
plus a `list[str]` of prompt UIDs. Current verl moves data around as
`verl.protocol.DataProto`. This module bridges the two so the filter can be
called with a single `filter_dataproto(batch, config)` and returns a filtered
`DataProto` of the same shape.

Tensor field mapping (verl key -> filter key):
- token_level_rewards (summed) -> rewards          [bs]
- response_mask                 -> response_mask   [bs, seq]
- old_log_probs (optional)      -> old_log_probs   [bs, seq]
- rollout_log_probs (optional)  -> rollout_log_probs [bs, seq]
- top_log_probs (optional)      -> top_log_probs   [bs, seq]
- prompt_index (optional)       -> prompt_index    [bs]

UIDs come from `non_tensor_batch["uid"]`. If absent, every sample is given a
unique UID (effectively disables group-level filtering — only individual
filters apply).
"""

from typing import Optional

import torch

from verl.protocol import DataProto

from .unified_filter import PPOBatchFilter, PPOFilterConfig

# Tensor keys consumed by PPOBatchFilter.filter_batch (besides 'rewards' which
# we derive from token_level_rewards). Optional — filter only uses what's
# present.
_FILTER_INPUT_TENSOR_KEYS = (
    "response_mask",
    "old_log_probs",
    "rollout_log_probs",
    "top_log_probs",
    "prompt_index",
)


def _extract_uids(batch: DataProto) -> list[str]:
    """Pull prompt UIDs out of non_tensor_batch['uid']; synthesize per-sample
    UIDs as a fallback so the filter can still run (group-level checks become
    no-ops because every group has size 1)."""
    if "uid" in batch.non_tensor_batch:
        return [str(uid) for uid in batch.non_tensor_batch["uid"]]
    bs = batch.batch["response_mask"].shape[0]
    return [f"_synthetic_{i}" for i in range(bs)]


def _build_filter_inputs(batch: DataProto) -> dict[str, torch.Tensor]:
    """Build the dict-of-tensors that PPOBatchFilter.filter_batch expects."""
    if "token_level_rewards" not in batch.batch:
        raise KeyError(
            "DataProto.batch is missing 'token_level_rewards'; the filter must run after reward computation."
        )
    if "response_mask" not in batch.batch:
        raise KeyError("DataProto.batch is missing 'response_mask'; cannot run filter.")

    inputs: dict[str, torch.Tensor] = {
        # Filter expects a 1D rewards tensor (already summed per sample).
        "rewards": batch.batch["token_level_rewards"].sum(dim=-1),
    }
    for key in _FILTER_INPUT_TENSOR_KEYS:
        if key in batch.batch:
            inputs[key] = batch.batch[key]
    return inputs


def filter_dataproto(
    batch: DataProto,
    config_or_filter: PPOFilterConfig | PPOBatchFilter,
    *,
    global_step: int = 0,
    data_config: Optional[dict] = None,
) -> tuple[DataProto, dict]:
    """Run PPOBatchFilter on a verl DataProto and return a filtered DataProto.

    The returned DataProto preserves all keys (tensor and non_tensor) and is
    indexed to the rows the filter selected. Empty batch is returned as-is
    with a warning logged by the filter (caller should detect and skip).

    Args:
        batch: input DataProto. Must contain `token_level_rewards` and
            `response_mask` in `batch`. Optional: `old_log_probs`,
            `rollout_log_probs`, `top_log_probs`, `prompt_index`. UIDs are
            read from `non_tensor_batch["uid"]` if present.
        config_or_filter: either a PPOFilterConfig (a fresh PPOBatchFilter is
            built and discarded) or a pre-built PPOBatchFilter (reused across
            calls — required when callers want stable RNG and accumulated
            metrics across steps).
        global_step: forwarded to PPOBatchFilter for any step-aware metrics.
        data_config: forwarded to PPOBatchFilter for seeded RNG. Ignored if
            a pre-built filter is supplied.

    Returns:
        (filtered_batch, metrics) where filtered_batch is a DataProto of the
        rows the filter selected, and metrics is the same dict the filter
        normally returns (use it to log selection rate, group rejections, etc).
    """
    inputs = _build_filter_inputs(batch)
    uids = _extract_uids(batch)

    if isinstance(config_or_filter, PPOBatchFilter):
        pipeline = config_or_filter
    else:
        pipeline = PPOBatchFilter(config_or_filter, data_config=data_config)

    selected_indices, metrics = pipeline.filter_batch(inputs, uids, global_step=global_step, return_indices=True)

    if selected_indices.numel() == 0:
        # Caller should detect via metrics['batch/critical_empty_batch'] and
        # decide what to do (skip step / fall back to unfiltered).
        return batch, metrics

    filtered = batch[selected_indices]
    return filtered, metrics
