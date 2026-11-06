#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""
Unified batch filter for PPO training.

This module provides a single, clean API for all filtering needs in PPO training:
- Rejection sampling (reward-based for individual samples, length-based for groups)
- Two-gate mismatch filtering (train/inference mismatch)
- Smart sample selection strategies (handles length preferences)
- Group management and metrics tracking

Key principle: Length filtering is applied at the GROUP level only:
- Always filters groups where ALL samples exceed max_response_length
- With remove_clip=True: also filters groups with insufficient short samples
Individual samples are NEVER filtered by length directly.

This is the ONLY filter interface that ray_trainer.py should use.

Ported from DR.Kernel (https://github.com/hkust-nlp/KernelGYM) at
verl_patch/trainer/code/filters/unified_filter.py. Logic unchanged; only an
unused `_compute_response_info` import was dropped.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Internal filter modules - not exposed to external users
from .two_gate_filter import FilterConfig as TwoGateConfig
from .two_gate_filter import TwoGateRejectionFilter

logger = logging.getLogger(__name__)

# Constants for better maintainability
HIGH_REJECTION_RATE_THRESHOLD = 0.8  # Warn if rejection rate exceeds 80%


@dataclass
class PPOFilterConfig:
    """
    Unified configuration for all PPO batch filtering.

    This is the only configuration class that external code needs to know about.
    All filter-specific configs are handled internally.

    Key design principles:
    - Groups with < min_group_size samples are rejected to limit padding overhead
    - min_group_size > target_group_size // 2 ensures padding overhead < 50% per group
    - Proportional allocation maintains natural positive/negative sample distribution
    - The filter only processes batches; oversampling is handled at DataLoader (prompt level)
      and generation (sample level)
    """

    # ========== Sample Selection Strategy ==========
    # Options: "uniform", "efficiency", "efficiency_stochastic"
    sample_selection_strategy: str = "efficiency_stochastic"

    # ========== Group Management ==========
    target_group_size: int = 8  # Target samples per group (rollout.n)
    min_group_size: Optional[int] = None  # Auto-set to target_group_size // 2 + 1 (must be > target_group_size // 2)
    target_num_groups: Optional[int] = None  # Target number of groups to select (train_batch_size)

    # ========== Filtering Thresholds ==========
    # Individual sample filtering
    reward_threshold: Optional[float] = None  # Min reward to keep individual sample

    # Group-level filtering (entire group is filtered if criteria met)
    max_response_length: Optional[int] = None  # Max length - filters groups with ALL samples over this
    reject_low_variance_groups: bool = True  # Reject groups with reward variance < 1e-3
    remove_clip: bool = False  # If True: also filter groups with < min_rollout_n samples under max_response_length
    min_rollout_n: Optional[int] = (
        None  # Min samples under max_response_length needed per group (when remove_clip=True)
    )

    # ========== Two-Gate Precision Filter ==========
    enable_two_gate_filter: bool = False

    # Gate 1: Systematic bias detection
    gate1_enabled: bool = True
    gate1_bias_epsilon: float = 0.01  # Max avg log-prob difference (1% tolerance)

    # Gate 2: Numerical instability detection
    gate2_enabled: bool = True
    gate2_instability_threshold: float = -15.0  # Min (sampled - max) log prob

    # ========== Metrics and Logging ==========
    save_metrics: bool = True
    log_rejection_reasons: bool = False

    def __post_init__(self):
        """Set derived parameters and validate config."""
        # Auto-set min_group_size to ensure padding overhead < 50% per group
        # min_group_size > target_group_size // 2 prevents excessive padding
        if self.min_group_size is None:
            self.min_group_size = self.target_group_size // 2 + 1

        # Validate user-provided min_group_size if specified
        if self.min_group_size <= self.target_group_size // 2:
            min_required = self.target_group_size // 2 + 1
            raise ValueError(
                f"min_group_size ({self.min_group_size}) must be > target_group_size // 2 ({self.target_group_size // 2}) "  # noqa: E501
                f"to avoid excessive padding overhead. Minimum required: {min_required}. "
                f"With min_group_size={self.min_group_size}, padding to target_group_size={self.target_group_size} "
                f"would result in >{(self.target_group_size - self.min_group_size) / self.target_group_size * 100:.0f}% padding per group."  # noqa: E501
            )

        # Validate strategy
        valid_strategies = {"uniform", "efficiency", "efficiency_stochastic"}
        if self.sample_selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.sample_selection_strategy}. Must be one of {valid_strategies}")


class PPOBatchFilter:
    """
    Unified batch filter for PPO training.

    This filter implements a multi-stage filtering pipeline:
    0. Two-gate precision filter (only with oversampling):
       - Removes samples with systematic bias or logit instability
       - May create incomplete groups (size < target_group_size)
    1. Group-level filtering:
       - Low variance groups (all rewards similar)
       - Groups where ALL samples exceed max_response_length
       - With remove_clip=True: groups with < min_rollout_n samples under max_response_length
    2. Individual sample filtering (reward threshold only, NO length filtering)
    3. Group selection and sample selection within groups (strategies handle length preferences)

    Key principle: Individual samples are NEVER filtered by length.
    This reduces skip rates while maintaining quality control.

    Example:
        config = PPOFilterConfig(
            sample_selection_strategy='efficiency_stochastic',  # Proportional allocation with exploration
            target_group_size=8,
            max_response_length=512,  # For group-level filtering
            remove_clip=True,  # Filter groups with < min_rollout_n short samples
            min_rollout_n=4
        )
        filter = PPOBatchFilter(config)

        # Apply all filtering
        selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)
    """

    def __init__(self, config: PPOFilterConfig, data_config: Optional[dict] = None) -> None:
        """Initialize the unified filter with given configuration."""
        self.config = config
        self.metrics: dict[str, float] = {}

        # Initialize random generator with seed from data config
        self._generator = torch.Generator()
        seed = 1  # default seed
        if data_config is not None:
            seed = data_config.get("seed", 1)
        self._generator.manual_seed(seed)
        self._device: Optional[torch.device] = None

        # Initialize two-gate filter if enabled (internal implementation)
        self._two_gate_filter = None
        if config.enable_two_gate_filter:
            two_gate_config = TwoGateConfig(
                enable_gate1=config.gate1_enabled,
                enable_gate2=config.gate2_enabled,
                bias_epsilon=config.gate1_bias_epsilon,
                instability_threshold=config.gate2_instability_threshold,
                save_rejection_stats=config.save_metrics,
                log_rejection_reasons=config.log_rejection_reasons,
            )
            self._two_gate_filter = TwoGateRejectionFilter(two_gate_config)

        logger.info(f"PPOBatchFilter initialized with config: {config}, seed: {seed}")

    def filter_batch(
        self, batch_data: dict[str, torch.Tensor], uids: list[str], global_step: int = 0, return_indices: bool = False
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, float]]:
        """
        Apply all configured filtering to the batch.

        Filtering stages in order:
        0. Two-gate precision filter (with oversampling only)
        1. Group-level filtering (low variance, all over-length, remove_clip logic)
        2. Individual sample filtering (reward threshold only, NO length filtering)
        3. Group selection and sample selection within groups (handles length preferences)

        Args:
            batch_data: Dictionary containing batch tensors:
                - rewards: Token-level rewards summed [batch_size]
                - response_mask: Valid token mask [batch_size, seq_len]
                - old_log_probs: FSDP log probs (optional) [batch_size, seq_len]
                - rollout_log_probs: vLLM log probs (optional) [batch_size, seq_len]
                - top_log_probs: Max log probs (optional) [batch_size, seq_len]
            uids: List of unique identifiers for each sample
            global_step: Current training step
            return_indices: If True, return selected indices; else return filtered data

        Returns:
            If return_indices=True: (selected_indices, metrics)
            If return_indices=False: (filtered_batch_data, metrics)

        Output guarantees:
            - batch_size ≤ target_num_groups × target_group_size (always holds)
            - groups_returned ≤ target_num_groups
            - Groups may be incomplete (min_group_size ≤ group_size ≤ target_group_size)
            - Incomplete groups are intentional (maximizes sample utilization)
        """
        # Validate inputs
        if not batch_data:
            raise ValueError("batch_data cannot be empty")
        if "rewards" not in batch_data:
            raise ValueError("batch_data must contain 'rewards' key")
        if len(uids) != batch_data["rewards"].shape[0]:
            raise ValueError(f"UIDs length ({len(uids)}) must match batch size ({batch_data['rewards'].shape[0]})")

        metrics: dict[str, float] = {}

        # Get batch dimensions
        total_samples = batch_data["rewards"].shape[0]

        # Get response length info
        response_lengths = batch_data["response_mask"].sum(-1).float()

        # Use dict.fromkeys() to preserve insertion order (first occurrence order)
        # This ensures deterministic and reproducible group processing order
        unique_uids = list(dict.fromkeys(uids))

        if total_samples == 0:
            self._log_empty_batch_warning()
            empty_indices = torch.tensor([], dtype=torch.long)
            empty_metrics = {
                "batch/total_samples_generated": 0,
                "batch/total_samples_selected": 0,
                "batch/selection_rate": 0,
                "batch/complete_groups_selected": 0,
                "batch/poor_groups_rejected": 0,
                "batch/single_sample_groups_rejected": 0,
                "batch/avg_group_size": 0,
            }
            self.metrics = empty_metrics
            return (empty_indices, empty_metrics) if return_indices else (batch_data, empty_metrics)

        # Build uid to indices mapping
        uid_to_indices: dict[str, list[int]] = {}
        for idx, uid in enumerate(uids):
            uid_to_indices.setdefault(uid, []).append(idx)

        # Initialize validity mask
        device = batch_data["rewards"].device
        self._device = device
        valid_mask = torch.ones(total_samples, dtype=torch.bool, device=device)

        # Check if we have oversampling by looking at group sizes
        # Oversampling means any group has more samples than target_group_size
        from collections import Counter

        group_sizes = Counter(uids)
        has_oversampling = max(group_sizes.values()) > self.config.target_group_size

        # ========== Stage 0: Two-Gate Precision Filtering (First, only with oversampling) ==========
        # Apply two-gate filter FIRST when we have oversampling and the filter is configured
        if self._two_gate_filter is not None and has_oversampling:
            # Apply two-gate filter to identify high-quality samples
            two_gate_indices, two_gate_metrics = self._two_gate_filter.filter_batch(batch_data, return_indices=True)

            # Create mask from indices
            two_gate_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
            two_gate_mask[two_gate_indices] = True

            # Apply two-gate mask
            valid_mask &= two_gate_mask

            # Add two-gate metrics
            for k, v in two_gate_metrics.items():
                metrics[f"filter/{k}"] = v

            logger.info(
                f"Applied two-gate filtering (oversampling detected, {len(two_gate_indices)}/{total_samples} samples passed)"  # noqa: E501
            )

            # Check if any groups now have too few samples after two-gate filtering
            two_gate_filtered_groups = set()
            for uid in unique_uids:
                group_indices = uid_to_indices[uid]
                group_valid_mask = valid_mask[group_indices]
                group_valid_count = group_valid_mask.sum().item()

                # If two-gate filtering reduced group below min_group_size, filter entire group
                if group_valid_count < self.config.min_group_size:
                    valid_mask[group_indices] = False
                    two_gate_filtered_groups.add(uid)

            if two_gate_filtered_groups:
                logger.info(
                    f"Filtered {len(two_gate_filtered_groups)} groups with insufficient samples after two-gate filtering"  # noqa: E501
                )
                metrics["rejection/two_gate_insufficient_groups"] = len(two_gate_filtered_groups)

        # ========== Stage 1: Group-Level Filtering ==========
        # Filter entire groups based on variance and length criteria
        # Note: Checks are performed on ORIGINAL groups (before any filtering)
        low_variance_groups = set()
        over_long_groups = set()
        solve_none = 0
        solve_all = 0
        number_larger_than_max_all = 0
        group_length_vars = []

        for uid in unique_uids:
            group_indices = uid_to_indices[uid]

            # Check reward variance on ORIGINAL group (before filtering)
            if self.config.reject_low_variance_groups and "rewards" in batch_data and len(group_indices) > 1:
                # Get rewards for ALL samples in this group (not filtered)
                group_rewards = batch_data["rewards"][group_indices]

                # Check variance across all samples in this group
                reward_std = group_rewards.std().item()
                if reward_std < 1e-3:
                    # Mark entire group as invalid
                    valid_mask[group_indices] = False
                    low_variance_groups.add(uid)

                # Track solve_none and solve_all
                if (group_rewards == 0).all():
                    solve_none += 1
                if (group_rewards == 1).all():
                    solve_all += 1

            if self.config.remove_clip:
                # Track response length statistics for this group (on original group)
                group_lengths = response_lengths[group_indices]

                # Compute standard deviation of response lengths
                # Handle single-element groups where std() would give a warning
                length_std = group_lengths.float().std().item() if len(group_lengths) > 1 else 0.0
                group_length_vars.append(length_std)

                # Group-level length filtering (NOT individual sample filtering)
                if self.config.max_response_length is not None:
                    # Count how many responses are >= max_response_length
                    num_larger_than_max = (group_lengths >= self.config.max_response_length).sum().item()
                    number_larger_than_max_all += num_larger_than_max

                    # Filter entire group if ALL responses are over-length
                    if num_larger_than_max == len(group_lengths):
                        valid_mask[group_indices] = False
                        over_long_groups.add(uid)

                    elif self.config.remove_clip and self.config.min_rollout_n is not None:
                        # With remove_clip: filter entire group if too few samples are under max_response_length
                        num_under_max = len(group_lengths) - num_larger_than_max
                        if num_under_max < self.config.min_rollout_n:
                            valid_mask[group_indices] = False
                            over_long_groups.add(uid)

        # Log metrics
        metrics["batch/low_variance_groups"] = len(low_variance_groups)
        metrics["batch/overlong_groups"] = len(over_long_groups)
        # Use union to avoid double-counting groups that are both low variance and over-long
        metrics["batch/filter_all_groups"] = len(low_variance_groups | over_long_groups)

        metrics["batch/solve_none_ratio"] = solve_none / len(unique_uids) if len(unique_uids) > 0 else 0
        metrics["batch/solve_all_ratio"] = solve_all / len(unique_uids) if len(unique_uids) > 0 else 0
        metrics["batch/response_clip_ratio"] = number_larger_than_max_all / total_samples if total_samples > 0 else 0
        metrics["batch/length_vars"] = np.mean(group_length_vars) if group_length_vars else 0.0

        if low_variance_groups:
            logger.info(f"Found {len(low_variance_groups)} groups with low reward variance")
        if over_long_groups:
            logger.info(f"Found {len(over_long_groups)} groups with all responses over max length")

        # ========== Stage 2: Individual Sample Filtering ==========
        # Apply reward threshold filtering to individual samples

        # Apply reward threshold filtering
        if self.config.reward_threshold is not None:
            reward_mask = batch_data["rewards"] >= self.config.reward_threshold
            valid_mask &= reward_mask
            reward_rejection_rate = 1 - reward_mask.float().mean().item()
            metrics["rejection/reward_rejection_rate"] = reward_rejection_rate
            if reward_rejection_rate > HIGH_REJECTION_RATE_THRESHOLD:
                self._log_high_reward_rejection_rate(reward_rejection_rate)

        # NOTE: NO individual sample-level length filtering is applied here.
        # Length considerations are handled in two ways:
        # 1. GROUP level: Entire groups filtered if all/most samples are over-length (Stage 1)
        # 2. SELECTION level: The 'efficiency' strategy naturally prefers shorter samples (Stage 3)
        # This approach reduces skip rates and avoids unnecessarily discarding valid data.

        # ========== Stage 3: Group-Level Processing and Sample Selection ==========
        # IMPORTANT: Both prompt (group) and sample selection must happen here because:
        # 1. We need to see sample quality to decide which prompts to keep
        # 2. Prompt selection depends on how many samples survive filtering
        # 3. Joint optimization: select best prompts that have best samples
        # 4. The sampler can't predict which prompts will have good samples

        # Track group statistics
        complete_groups_count = 0
        rejected_poor_groups_count = 0
        rejected_single_groups_count = 0
        selected_group_uids: set[str] = set()
        valid_group_uids: set[str] = set()

        # First pass: Collect all valid groups with their statistics
        valid_groups = []  # List of (uid, group_valid_indices, avg_reward, group_size)

        for uid in unique_uids:
            group_indices = uid_to_indices[uid]

            # Get valid samples in this group
            group_valid_mask = valid_mask[group_indices]
            group_valid_count = group_valid_mask.sum().item()

            # Skip groups that were marked as low variance or over-length in Stage 1
            if uid in low_variance_groups or uid in over_long_groups:
                rejected_poor_groups_count += 1
                continue

            # Skip groups with insufficient valid samples after filtering
            if group_valid_count < self.config.min_group_size:
                if group_valid_count == 1:
                    rejected_single_groups_count += 1
                else:
                    rejected_poor_groups_count += 1
                continue

            # Get valid indices for this group
            group_valid_indices = [group_indices[i] for i in range(len(group_indices)) if group_valid_mask[i]]

            # Calculate group statistics for selection
            group_rewards = batch_data["rewards"][group_valid_indices]
            avg_reward = group_rewards.mean().item()

            # Store valid group info
            valid_groups.append((uid, group_valid_indices, avg_reward, len(group_valid_indices)))
            valid_group_uids.add(uid)

        # If we have more valid groups than target, prioritize complete groups
        groups_to_process = valid_groups
        if self.config.target_num_groups is not None and len(valid_groups) > self.config.target_num_groups:
            # Separate groups into complete (target_group_size) and incomplete
            complete_groups = []
            incomplete_groups = []

            for group_info in valid_groups:
                uid, group_valid_indices, avg_reward, group_size = group_info
                if group_size == self.config.target_group_size:
                    complete_groups.append(group_info)
                else:
                    incomplete_groups.append(group_info)

            # First, prioritize complete groups
            if len(complete_groups) >= self.config.target_num_groups:
                # If we have enough complete groups, randomly select from them
                indices = torch.randperm(len(complete_groups), generator=self._generator)
                selected_indices = indices[: self.config.target_num_groups].tolist()
                groups_to_process = [complete_groups[i] for i in selected_indices]
            else:
                # Include all complete groups and fill the rest with incomplete groups
                groups_to_process = complete_groups.copy()

                # Randomly select from incomplete groups to fill remaining slots
                remaining_slots = self.config.target_num_groups - len(complete_groups)
                if remaining_slots > 0 and len(incomplete_groups) > 0:
                    indices = torch.randperm(len(incomplete_groups), generator=self._generator)
                    selected_indices = indices[: min(remaining_slots, len(incomplete_groups))].tolist()
                    groups_to_process.extend([incomplete_groups[i] for i in selected_indices])

            # Log group selection statistics
            n_complete_selected = sum(1 for _, _, _, size in groups_to_process if size == self.config.target_group_size)
            logger.info(
                f"Selected {len(groups_to_process)} groups out of {len(valid_groups)} valid groups "
                f"({n_complete_selected} complete, {len(groups_to_process) - n_complete_selected} incomplete)"
            )
            metrics["batch/groups_before_selection"] = len(valid_groups)
            metrics["batch/groups_after_selection"] = len(groups_to_process)
            metrics["batch/complete_groups_before_selection"] = len(complete_groups)
            metrics["batch/complete_groups_after_selection"] = n_complete_selected

        # Second pass: Process selected groups and get final indices
        selected_indices: list[int] = []

        for uid, group_valid_indices, avg_reward, _ in groups_to_process:
            # Record current length to calculate how many samples we add
            count_before = len(selected_indices)

            # Apply sample selection ONLY with oversampling and if we have more samples than needed
            if has_oversampling and len(group_valid_indices) > self.config.target_group_size:
                # Get data for selection
                group_rewards = batch_data["rewards"][group_valid_indices]
                group_lengths = response_lengths[group_valid_indices]

                # Apply selection strategy
                selected_local = self._select_samples(group_rewards, group_lengths, self.config.target_group_size)

                # Map back to global indices
                selected_indices.extend([group_valid_indices[idx] for idx in selected_local])
            else:
                # Use all valid samples (no oversampling or group size <= target)
                selected_indices.extend(group_valid_indices)

            # Calculate how many samples we added for this group
            group_selected_count = len(selected_indices) - count_before
            if group_selected_count == self.config.target_group_size:
                complete_groups_count += 1
            if group_selected_count > 0:
                selected_group_uids.add(uid)

        # Convert to tensor
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)

        # ========== Stage 4: Compile Metrics ==========

        # Check if batch size meets ray_trainer expectations
        selected_sample_count = selected_indices.numel()
        selected_group_count = len(selected_group_uids)

        # Calculate expected batch size (what ray_trainer expects)
        # expected_batch_size = rollout.n * train_batch_size
        if self.config.target_num_groups is not None:
            # target_num_groups is essentially train_batch_size
            expected_sample_count = self.config.target_group_size * self.config.target_num_groups
        else:
            # No target specified, can't determine expectation
            expected_sample_count = None

        if selected_sample_count == 0:
            logger.error("CRITICAL: No samples selected! Batch is empty. Check filtering criteria.")
            logger.error("This batch will be skipped by ray_trainer!")
            metrics["batch/critical_empty_batch"] = 1
        elif expected_sample_count is not None and selected_sample_count < expected_sample_count:
            # Calculate how many times samples would need to be repeated
            repeat_factor = (expected_sample_count + selected_sample_count - 1) // selected_sample_count

            if selected_sample_count < expected_sample_count * 0.5:  # Less than 50% of expected
                logger.warning(
                    f"SEVERE: Very low batch size: {selected_sample_count} samples selected "
                    f"(expected {expected_sample_count} = {self.config.target_group_size} * {self.config.target_num_groups}). "  # noqa: E501
                )
                logger.warning(
                    f"Ray_trainer will skip this batch (rejection_sample=True by default). "
                    f"Would need {repeat_factor}x repetition if rejection_sample=False."
                )
                metrics["batch/severe_low_batch_size"] = 1
            else:
                logger.warning(
                    f"Low batch size: {selected_sample_count} samples selected "
                    f"(expected {expected_sample_count}). "
                    f"Ray_trainer will skip this batch (rejection_sample=True by default)."
                )
                metrics["batch/low_batch_size_warning"] = 1

            metrics["batch/batch_size_ratio"] = selected_sample_count / expected_sample_count
            metrics["batch/expected_repeat_factor"] = repeat_factor

        metrics.update(
            {
                "batch/total_samples_generated": total_samples,
                "batch/total_samples_selected": selected_sample_count,
                "batch/expected_samples": expected_sample_count if expected_sample_count is not None else 0,
                "batch/selection_rate": selected_sample_count / total_samples if total_samples > 0 else 0,
                "batch/complete_groups_selected": complete_groups_count,
                "batch/poor_groups_rejected": rejected_poor_groups_count,
                "batch/single_sample_groups_rejected": rejected_single_groups_count,
                "batch/avg_group_size": (
                    selected_sample_count / selected_group_count if selected_group_count > 0 else 0
                ),
                "batch/selected_groups": selected_group_count,
            }
        )

        # ========== Stage 5: Track Per-Prompt Statistics for Dynamic Batch Sampler ==========
        if "prompt_index" in batch_data:
            prompt_indices = batch_data["prompt_index"]
            prompt_filter_stats = {}

            # Build a mapping from UID to prompt_index
            uid_to_prompt = {}
            for idx, uid in enumerate(uids):
                if uid not in uid_to_prompt:
                    # Convert to int to ensure consistent types
                    uid_to_prompt[uid] = int(prompt_indices[idx])

            # Count groups before and after filtering for each prompt
            # Convert np.unique result to Python int to avoid JSON serialization issues
            unique_prompt_indices = np.unique(prompt_indices).tolist()
            for prompt_idx in unique_prompt_indices:
                # Count UIDs (groups) that belong to this prompt
                prompt_uids_before = [uid for uid in unique_uids if uid_to_prompt.get(uid) == prompt_idx]

                # Count how many of these UIDs survived filtering (even if later dropped during selection)
                prompt_uids_valid = [uid for uid in valid_group_uids if uid_to_prompt.get(uid) == prompt_idx]

                # Track how many valid UIDs were ultimately selected for training
                prompt_uids_selected = [uid for uid in selected_group_uids if uid_to_prompt.get(uid) == prompt_idx]

                prompt_filter_stats[prompt_idx] = {
                    "before": len(prompt_uids_before),
                    "after": len(prompt_uids_valid),
                    "selected": len(prompt_uids_selected),
                }

            # Expose raw prompt-level stats for downstream consumers (dynamic batch sampler)
            metrics["prompt_filter_stats"] = prompt_filter_stats

        self.metrics = metrics

        # Return results
        if return_indices:
            return selected_indices, metrics
        else:
            # Return filtered batch data
            filtered_batch = {}
            for key, tensor in batch_data.items():
                if torch.is_tensor(tensor) and tensor.shape[0] == total_samples:
                    filtered_batch[key] = tensor[selected_indices]
                else:
                    filtered_batch[key] = tensor
            return filtered_batch, metrics

    def _select_samples(self, rewards: torch.Tensor, lengths: torch.Tensor, target_n: int) -> torch.Tensor:
        """
        Select samples within a group using the configured strategy.

        This method handles length preferences through selection strategies rather than
        hard filtering. Selection strategies:

        - 'uniform': Random selection without any bias, preserves natural distribution
        - 'efficiency' / 'efficiency_stochastic': Adaptive per-group strategy:
          * Each group (same prompt, multiple responses) is processed independently
          * Maintains proportional representation of positive/negative responses
          * Within each group:
            - Allocates slots proportionally to available positive/negative ratio
            - Positive responses (reward > 0): Select by reward/length ratio
            - Negative responses (reward ≤ 0): Select shortest responses
            - Edge case: Ensures at least 1 sample of minority type if available
          * Both types prefer shorter: positive via efficiency, negative via direct length
          * Preserves model's natural success/failure distribution per prompt
          * Stochastic version adds probabilistic sampling for exploration

        Args:
            rewards: Reward values for each sample
            lengths: Length values for each sample
            target_n: Number of samples to select

        Returns:
            Tensor of selected sample indices (local to the group)

        Raises:
            AssertionError: If inputs are inconsistent or invalid
        """
        n_samples = len(rewards)
        assert len(rewards) == len(lengths), (
            f"Rewards and lengths must have same length: {len(rewards)} vs {len(lengths)}"
        )
        assert target_n > 0, f"Target number of samples must be positive, got {target_n}"
        assert n_samples > target_n, (
            f"n_samples ({n_samples}) must be > target_n ({target_n}), caller should check this"
        )

        strategy = self.config.sample_selection_strategy

        if strategy == "uniform":
            # Random selection - no bias, preserves natural distribution
            perm = torch.randperm(n_samples, generator=self._generator)
            return perm[:target_n]

        elif strategy in ["efficiency", "efficiency_stochastic"]:
            # Efficiency-based selection
            epsilon = 1e-6
            efficiency = rewards / (lengths.float() + epsilon)

            # Separate positive and negative samples
            positive_mask = rewards > 0
            negative_mask = ~positive_mask

            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]

            selected = []
            deterministic = strategy == "efficiency"

            # Calculate proportional allocation based on available samples
            total_available = len(positive_indices) + len(negative_indices)

            if total_available == 0:
                return torch.tensor([], dtype=torch.long)

            # If total available matches target exactly, take all samples
            if total_available <= target_n:
                selected.extend(positive_indices.tolist())
                selected.extend(negative_indices.tolist())
                return torch.tensor(selected)

            # Proportional selection to maintain model's natural distribution
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Both types exist - maintain ratio
                # Use rounding to nearest integer for better allocation
                positive_ratio = len(positive_indices) / total_available
                n_positive = round(target_n * positive_ratio)

                # Ensure we don't exceed target_n and handle edge cases
                n_positive = max(1, min(n_positive, target_n - 1))  # At least 1, at most target_n - 1
                n_negative = target_n - n_positive

                # Ensure minority class gets at least 1 sample if available
                if n_negative == 0 and len(negative_indices) > 0:
                    n_negative = 1
                    n_positive = target_n - 1
            elif len(positive_indices) > 0:
                # Only positive samples
                n_positive = target_n
                n_negative = 0
            else:
                # Only negative samples
                n_positive = 0
                n_negative = target_n

            # Handle positive samples
            if n_positive > 0 and len(positive_indices) > 0:
                positive_efficiency = efficiency[positive_mask]
                n_positive = min(n_positive, len(positive_indices))

                if len(positive_indices) <= n_positive:
                    # Need all positive samples
                    selected.extend(positive_indices.tolist())
                else:
                    # Select best positive by efficiency
                    if deterministic:
                        sorted_idx = torch.argsort(positive_efficiency, descending=True)
                        selected.extend(positive_indices[sorted_idx[:n_positive]].tolist())
                    else:
                        # Stochastic selection
                        scores = positive_efficiency - positive_efficiency.min() + epsilon
                        selected.extend(self._stochastic_sample(scores, n_positive, positive_indices))

            # Handle negative samples
            if n_negative > 0 and len(negative_indices) > 0:
                negative_lengths = lengths[negative_mask]
                n_negative = min(n_negative, len(negative_indices))

                if len(negative_indices) <= n_negative:
                    # Need all negative samples
                    selected.extend(negative_indices.tolist())
                else:
                    # Select shortest negative samples
                    if deterministic:
                        sorted_idx = torch.argsort(negative_lengths)
                        selected.extend(negative_indices[sorted_idx[:n_negative]].tolist())
                    else:
                        # Stochastic selection by inverse length
                        inverse_lengths = 1.0 / (negative_lengths + epsilon)
                        weights = inverse_lengths / inverse_lengths.sum()
                        # Normalize for softmax (convert probabilities to logits)
                        log_weights = torch.log(weights + epsilon)
                        selected.extend(self._stochastic_sample(log_weights, n_negative, negative_indices))

            return torch.tensor(selected)

    def _stochastic_sample(self, weights: torch.Tensor, n_samples: int, indices: torch.Tensor) -> list[int]:
        """Helper method for stochastic sampling based on weights."""
        probs = F.softmax(weights, dim=0)
        sampled_idx = torch.multinomial(probs, n_samples, replacement=False, generator=self._generator)
        return indices[sampled_idx].tolist()

    def get_metrics(self) -> dict[str, float]:
        """Get the latest filtering metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all accumulated metrics."""
        self.metrics = {}
        if self._two_gate_filter:
            self._two_gate_filter.reset_stats()
