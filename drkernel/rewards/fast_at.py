# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fast@X metric helper for the DR.Kernel recipe.

Fast@X = fraction of kernels that PASS correctness checks AND achieve at least
X-times speedup over the Torch reference. Per-trajectory we emit a 0/1 indicator
for each X; the mean across a batch (training) or across rollouts/prompts
(validation, via `process_validation_metrics`) recovers the Fast@X fraction.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# Default speedup thresholds (x-factor over Torch reference) used when no
# config override is provided. Override at runtime via:
#     reward_model.reward_kwargs.fast_at_thresholds=[0.5,1.0,1.5]   (native cfg)
#     reward_model.fast_at_thresholds=[0.5,1.0,1.5]                 (naive cfg)
FAST_AT_X_THRESHOLDS: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2)

# Tensorboard-tag prefix used for Fast@X keys in `reward_extra_info`. The
# trainer side discovers Fast@X columns by this prefix so it doesn't need
# its own copy of the threshold list.
FAST_AT_KEY_PREFIX = "Fast@"


def fast_at_key(threshold: float) -> str:
    """Stable tensorboard-friendly key for the Fast@<threshold> indicator."""
    # One-decimal format keeps the labels uniform (`Fast@1.0`, not `Fast@1`).
    # `@` and `.` are valid characters in tensorboard tag names.
    return f"{FAST_AT_KEY_PREFIX}{threshold:.1f}"


def resolve_thresholds(thresholds: Any) -> tuple[float, ...]:
    """Normalize a config-provided thresholds value into a tuple of floats.

    Accepts ``None`` (use defaults), an OmegaConf ``ListConfig``, or any
    iterable of numbers. Non-numeric / empty inputs fall back to
    ``FAST_AT_X_THRESHOLDS``.
    """
    if thresholds is None:
        return FAST_AT_X_THRESHOLDS
    if isinstance(thresholds, (str, bytes)):
        # A bare string is almost certainly a misconfig; don't try to parse.
        return FAST_AT_X_THRESHOLDS
    try:
        out = tuple(float(x) for x in thresholds)
    except (TypeError, ValueError):
        return FAST_AT_X_THRESHOLDS
    return out or FAST_AT_X_THRESHOLDS


def compute_fast_at_indicators(
    correctness: Any,
    speedup: Any,
    thresholds: Iterable[float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """Per-trajectory Fast@X indicators.

    Args:
        correctness: truthy iff the kernel passed correctness checks.
        speedup: x-factor over the Torch reference; ``None`` is treated as 0.
        thresholds: list of X values to emit. ``None`` -> ``FAST_AT_X_THRESHOLDS``.

    Returns:
        Mapping ``{"Fast@<X>": 1.0|0.0}`` for every X in ``thresholds``.
    """
    is_correct = bool(correctness)
    try:
        speedup_val = float(speedup) if speedup is not None else 0.0
    except (TypeError, ValueError):
        speedup_val = 0.0

    xs = resolve_thresholds(thresholds)
    return {fast_at_key(x): 1.0 if (is_correct and speedup_val >= x) else 0.0 for x in xs}
