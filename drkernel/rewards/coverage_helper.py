# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

from typing import Optional

import torch

import verl.utils.torch_functional as verl_F

SAFETY_BOUND = 20.0


def compute_rs_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for rejection sampling.

    This function calculates statistics for IS weights used in rejection sampling,
    balancing numerical stability (using clamped weights) and accuracy (using log-space
    for threshold checks).

    Args:
        rollout_is_weights: Clamped IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Rejection sampling aggregation level (matches compute_rollout_rejection_mask).
        rollout_rs_threshold: Upper threshold for valid IS weights.
        rollout_rs_threshold_lower: Lower threshold for valid IS weights.

    Returns:
        Dictionary of rejection sampling metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device

    # Precompute log thresholds for accurate threshold checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_rs in ["sequence", "geometric", "turn", "turn_geo"]:
        # Sequence/turn-level aggregation: use log-space for accurate max/min/threshold checks
        # True max/min (unclamped) converted with safety bounds
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_rs_min"] = torch.exp(log_min).item()

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower

        if rollout_rs in ["sequence", "turn"]:
            # Sequence/turn-level: all tokens in a group have the same weight
            metrics["rollout_rs_ratio_fraction_high"] = exceeds_upper.float().mean().item()
            metrics["rollout_rs_ratio_fraction_low"] = below_lower.float().mean().item()
        else:  # geometric/turn_geo
            # Broadcast threshold checks to match token dimensions
            exceeds_upper_expanded: torch.Tensor = exceeds_upper.expand_as(response_mask)
            below_lower_expanded: torch.Tensor = below_lower.expand_as(response_mask)
            metrics["rollout_rs_ratio_fraction_high"] = verl_F.masked_mean(
                exceeds_upper_expanded.float(), response_mask
            ).item()
            metrics["rollout_rs_ratio_fraction_low"] = verl_F.masked_mean(
                below_lower_expanded.float(), response_mask
            ).item()

    else:  # token-level
        # Token-level aggregation: compute directly from clamped weights
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_rs_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_rs_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_rs_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens first)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_rs_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_rs_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        # Clamp weights to threshold range to avoid squaring extreme values
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_rs_threshold_lower, max=rollout_rs_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        # Variance = E[X²] - (E[X])² (masked to valid tokens)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_rs_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_rs_std"] = 0.0

    # Compute Effective Sample Size (ESS) for IS weights
    # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_rs_threshold_lower, max=rollout_rs_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_rs_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        # Mean weight per sequence (masked to valid tokens)
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_rs_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_rs_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_rs_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_rs_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_rs_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_rs_seq_fraction_high"] = (seq_mean_weights > rollout_rs_threshold).float().mean().item()
        metrics["rollout_rs_seq_fraction_low"] = (seq_mean_weights < rollout_rs_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_rejection_mask(
    coverage_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    correctness: torch.Tensor,
    max_turns: int = 1,
    coverage_rs: str = "turn",
    coverage_rs_threshold: Optional[float] = None,
    coverage_rs_factor: Optional[float] = None,
    speedup: Optional[torch.Tensor] = None,
    speedup_threshold: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rejection mask based on coverage ratio for kernel RL training.

    This function computes a per-turn rejection mask based on coverage ratio (time or num),
    and only applies rejection sampling to correct samples. The probability of keeping a turn
    is calculated as: (coverage - threshold) / factor, clamped to [0, 1].

    Design:
    - Only correct samples are subject to rejection sampling
    - Incorrect samples always keep their original response_mask
    - Supports both per-turn and geometric (cross-turn) aggregation
    - Supports speedup-based OR condition: keep if EITHER coverage OR speedup meets threshold

    Args:
        coverage_ratio: Coverage ratio per turn, shape (batch_size, max_turns) or (batch_size * max_turns,)
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size * max_turns, seq_length)
        correctness: Whether each turn is correct (boolean tensor), shape (batch_size * max_turns,)
        max_turns: Maximum number of conversation turns
        coverage_rs: Rejection sampling aggregation level, must be one of:
            - "turn": Per-turn independent rejection sampling
            - "geometric": Geometric mean across all turns (sequence-level)
        coverage_rs_threshold: Lower threshold for coverage (e.g., 0.3)
            - Coverage >= threshold + factor: always keep (prob = 1.0)
            - Coverage <= threshold: never keep (prob = 0.0)
            - In between: linear interpolation
        coverage_rs_factor: Factor for probability calculation (e.g., 0.1)
            - Defines the range for linear interpolation
        speedup: Optional speedup values per turn, shape (batch_size, max_turns) or (batch_size * max_turns,)
            - If provided with speedup_threshold, enables OR logic with coverage
        speedup_threshold: Optional threshold for speedup (e.g., 1.5 for 1.5x speedup)
            - If speedup >= threshold, sample is always kept regardless of coverage
            - If None or < 0, speedup filtering is disabled

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with rejection applied,
                shape (batch_size * max_turns, seq_length)
            metrics: Dictionary of rejection sampling metrics (all scalars), including:
                - coverage_rs_masked_fraction: Overall rejection rate
                - coverage_rs_seq_masked_fraction: Sequence-level rejection rate
                - coverage_rs_correct_only_masked_fraction: Rejection rate for correct samples only
    """
    batch_size = response_mask.shape[0] // max_turns
    seq_length = response_mask.shape[1]

    # Reshape coverage_ratio to (batch_size, max_turns) if needed
    if coverage_ratio.dim() == 1:
        coverage_ratio = coverage_ratio.reshape(batch_size, max_turns)

    # Reshape correctness to (batch_size, max_turns)
    correctness_2d = correctness.reshape(batch_size, max_turns)

    # compute for coverage rejection mask
    if coverage_rs == "geometric":
        # Geometric mean across turns (only for valid turns)
        # Valid turn: any token in that turn is valid
        valid_turn_mask = response_mask.reshape(batch_size, max_turns, seq_length).any(
            dim=-1
        )  # (batch_size, max_turns)

        # Compute log of coverage (with clamping to avoid log(0))
        log_coverage = torch.log(coverage_ratio.clamp(min=1e-8))
        masked_log_coverage = log_coverage.masked_fill(~valid_turn_mask, 0)

        # Geometric mean: exp(mean(log(x)))
        num_valid_turns = valid_turn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_log_coverage = masked_log_coverage.sum(dim=1, keepdim=True) / num_valid_turns
        geometric_coverage = torch.exp(mean_log_coverage.clamp(min=-SAFETY_BOUND, max=SAFETY_BOUND))

        # Broadcast to all turns
        rollout_is_weights = geometric_coverage.expand(-1, max_turns)  # (batch_size, max_turns)
    else:
        # Per-turn coverage
        rollout_is_weights = coverage_ratio  # (batch_size, max_turns)

    # compute for speedup as mask
    rollout_is_weights_speedup = None
    if speedup is not None and speedup_threshold is not None:
        # Reshape speedup to match coverage dimensions (batch_size, max_turns)
        if speedup.dim() == 1:
            rollout_is_weights_speedup = speedup.reshape(batch_size, max_turns)
        else:
            rollout_is_weights_speedup = speedup

    # Compute mask probability: (coverage - threshold) / factor
    # mask_prob in [0, 1]
    if coverage_rs_factor is None or coverage_rs_factor == 0:
        # If factor is not provided or is 0, use threshold as a hard cutoff
        mask_prob = (rollout_is_weights >= coverage_rs_threshold).float()
    else:
        mask_prob = (rollout_is_weights - coverage_rs_threshold) / coverage_rs_factor
        mask_prob = mask_prob.clamp(min=0, max=1)  # (batch_size, max_turns)

    # Only apply RS to correct samples, incorrect samples keep original mask
    # For correct samples: sample from mask_prob
    # For incorrect samples: always keep (mask_prob = 1.0)
    mask_prob = torch.where(correctness_2d, mask_prob, torch.ones_like(mask_prob))

    # Apply speedup threshold as OR condition: keep if either coverage OR speedup meets threshold
    if rollout_is_weights_speedup is not None:
        speedup_mask = (rollout_is_weights_speedup >= speedup_threshold).float()
        # Use maximum to implement OR logic: keep if EITHER condition is satisfied
        mask_prob = torch.maximum(mask_prob, speedup_mask)

    # Sample from Bernoulli distribution
    turn_level_mask = torch.bernoulli(mask_prob).float()  # (batch_size, max_turns)

    # Expand to token level: (batch_size * max_turns, seq_length)
    turn_level_mask_flat = turn_level_mask.reshape(-1, 1)  # (batch_size * max_turns, 1)
    token_level_mask = turn_level_mask_flat.expand(-1, seq_length)  # (batch_size * max_turns, seq_length)

    # Apply to original response_mask
    modified_response_mask = response_mask * token_level_mask

    # Metrics
    metrics: dict[str, float] = {}
    valid_mask = response_mask > 0
    if valid_mask.any():
        # Overall rejection rate
        metrics["coverage_rs_masked_fraction"] = verl_F.masked_mean(1 - token_level_mask, valid_mask).item()

        # Sequence-level rejection rate
        if coverage_rs == "turn":
            # Turn-level aggregation: sequence is rejected if any turn is rejected
            seq_has_masked = (turn_level_mask.reshape(batch_size, max_turns) == 0).any(dim=1)
            metrics["coverage_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
        else:
            # Geometric: all turns in a sequence have the same mask
            # Check the first turn of each sequence
            first_turn_mask = turn_level_mask.reshape(batch_size, max_turns)[:, 0]
            metrics["coverage_rs_seq_masked_fraction"] = (first_turn_mask == 0).float().mean().item()

        # Rejection rate for correct samples only
        correct_mask = correctness.bool() & valid_mask.any(dim=-1)
        if correct_mask.any():
            correct_token_mask = correct_mask.unsqueeze(-1).expand_as(valid_mask) & valid_mask
            metrics["coverage_rs_correct_only_masked_fraction"] = verl_F.masked_mean(
                1 - token_level_mask, correct_token_mask
            ).item()

        # Coverage statistics (for debugging)
        metrics["coverage_rs_mean_coverage"] = rollout_is_weights.mean().item()
        metrics["coverage_rs_min_coverage"] = rollout_is_weights.min().item()
        metrics["coverage_rs_max_coverage"] = rollout_is_weights.max().item()

    return modified_response_mask, metrics


def compute_coverage_rejection_mask(
    time_coverage: torch.Tensor,
    num_coverage: torch.Tensor,
    response_mask: torch.Tensor,
    correctness: torch.Tensor,
    max_turns: int = 1,
    coverage_rs: str = "turn",
    coverage_rs_key: str = "time_coverage",
    coverage_rs_threshold: Optional[float] = 0.3,
    coverage_rs_factor: Optional[float] = 0.1,
    speedup: Optional[torch.Tensor] = None,
    speedup_threshold: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rejection mask for kernel coverage-based rejection sampling.

    This is the main entry point for coverage-based rejection sampling in kernel RL training.
    It selects the appropriate coverage metric (time or num) and applies rejection sampling
    only to correct samples. Supports OR logic with speedup threshold.

    Args:
        time_coverage: Time coverage ratio, shape (batch_size, max_turns) or (batch_size * max_turns,)
        num_coverage: Number of coverage ratio, shape (batch_size, max_turns) or (batch_size * max_turns,)
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size * max_turns, seq_length)
        correctness: Whether each turn is correct (boolean tensor), shape (batch_size * max_turns,)
        max_turns: Maximum number of conversation turns
        coverage_rs: Coverage ratio aggregation level, must be one of:
            - "turn": Per-turn independent rejection sampling
            - "geometric": Geometric mean across entire sequence
        coverage_rs_key: Which coverage metric to use, must be one of:
            - "time_coverage": Use time-based coverage (CUDA time ratio)
            - "num_coverage": Use count-based coverage (kernel count ratio)
        coverage_rs_threshold: Lower threshold for coverage (default: 0.3)
        coverage_rs_factor: Factor for probability calculation (default: 0.1)
        speedup: Optional speedup values per turn, shape (batch_size, max_turns) or (batch_size * max_turns,)
            - If provided with speedup_threshold, enables OR logic: keep if coverage OR speedup meets threshold
        speedup_threshold: Optional threshold for speedup (e.g., 1.5 for 1.5x speedup)
            - If None or < 0, speedup filtering is disabled

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with rejection applied,
                shape (batch_size * max_turns, seq_length)
            metrics: Dictionary of rejection sampling metrics with "coverage/" prefix
    """
    # Validate inputs
    assert coverage_rs_key in [
        "time_coverage",
        "num_coverage",
    ], f"Invalid coverage_rs_key: {coverage_rs_key}"
    assert coverage_rs in ["turn", "geometric"], f"Invalid coverage_rs: {coverage_rs}"

    if speedup_threshold is not None:
        if speedup_threshold < 0:
            speedup_threshold = None

    # Select coverage metric
    coverage_ratio: torch.Tensor = time_coverage if coverage_rs_key == "time_coverage" else num_coverage

    # Apply rejection sampling
    modified_response_mask, coverage_rs_metrics = compute_rollout_rejection_mask(
        coverage_ratio=coverage_ratio,
        response_mask=response_mask,
        correctness=correctness,
        max_turns=max_turns,
        coverage_rs=coverage_rs,
        coverage_rs_threshold=coverage_rs_threshold,
        coverage_rs_factor=coverage_rs_factor,
        speedup=speedup,
        speedup_threshold=speedup_threshold,
    )

    # Add prefix to metrics for clarity
    metrics: dict[str, float] = {}
    for key, value in coverage_rs_metrics.items():
        metrics[f"coverage/{key}"] = value

    return modified_response_mask, metrics
