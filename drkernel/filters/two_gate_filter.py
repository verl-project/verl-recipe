# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""
Two-Gate Rejection Filter for Data Integrity.

Ported unchanged from DR.Kernel (https://github.com/hkust-nlp/KernelGYM) at
verl_patch/trainer/code/filters/two_gate_filter.py.

This module implements the principled rejection sampling approach described in
"Rejection Sampling is All You Need for Systematic Inference-Training Mismatch".

The filter addresses two sources of numerical corruption:

1. Gate 1: Systematic Bias Detection
   - Compares average log probabilities between FSDP (FP32) and vLLM (BFloat16)
   - Uses verl_F.masked_mean for efficient computation
   - Rejects sequences where |mean_diff| > bias_epsilon

2. Gate 2: Numerical Instability Detection
   - Identifies tokens sampled from precision-limited distributions
   - Checks if log_prob(sampled) - log_prob(max) < threshold
   - Default threshold: -15.0 (balanced for BFloat16 and vocab-aware filtering)
   - Rejects sequences containing any unstable tokens

By filtering corrupted samples, we ensure training on numerically sound data only.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)

# Constants for better maintainability
EXTREME_PPL_THRESHOLD = -13.8  # log_prob threshold for PPL > 1e6
MAX_BATCH_HISTORY = 100  # Maximum number of batches to keep in history


@dataclass
class FilterConfig:
    """Configuration for two-gate rejection filter."""

    # Gate 1: Systematic Bias Check
    enable_gate1: bool = True
    bias_epsilon: float = 0.01  # Max tolerable average log-prob difference per token (1% tolerance)

    # Gate 2: Logit Instability Check
    enable_gate2: bool = True
    instability_threshold: float = -15.0  # Default: max(hardware_limit, vocab_aware_threshold)

    # Hardware limits:
    # - FP16: -9.7 (ln(2^-14))
    # - BFloat16: -87.3 theoretical, -20 practical

    # Vocab-aware (k=100):
    # - 32k vocab: -15.0
    # - 155k vocab: -16.6

    # Choose max for safety

    # Logging and debugging
    log_rejected_samples: bool = False
    log_rejection_reasons: bool = True
    save_rejection_stats: bool = True

    # Batch processing
    process_in_chunks: bool = False
    chunk_size: int = 32


class RejectionStats:
    """Track rejection statistics across batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_samples = 0
        self.gate1_rejections = 0
        self.gate2_rejections = 0
        self.both_gates_rejections = 0

        # Detailed stats
        self.bias_values: list[float] = []
        self.instability_counts: list[int] = []
        self.rejection_reasons: list[str] = []

        # Gate 2 detailed metrics
        self.gate2_min_diffs: list[float] = []  # Minimum log-prob differences
        self.gate2_avg_diffs: list[float] = []  # Average log-prob differences per sequence

        # Temporal tracking
        self.batch_rejection_history: list[float] = []  # Recent batch rejection rates (last 100)

        # Length-based tracking
        self.rejections_by_length: dict[str, tuple[int, int]] = {}  # {length_bucket: (rejected, total)}

        # PPL tracking (separate for rollout vs FSDP)
        self.rollout_ppl_accepted: list[float] = []  # Rollout (vLLM) log-ppl of accepted
        self.rollout_ppl_rejected: list[float] = []  # Rollout (vLLM) log-ppl of rejected
        self.fsdp_ppl_accepted: list[float] = []  # FSDP log-ppl of accepted
        self.fsdp_ppl_rejected: list[float] = []  # FSDP log-ppl of rejected
        self.ppl_drift_accepted: list[float] = []  # log-PPL(rollout) - log-PPL(FSDP) for accepted
        self.ppl_drift_rejected: list[float] = []  # log-PPL(rollout) - log-PPL(FSDP) for rejected
        self.extreme_ppl_tokens: int = 0  # Count of tokens with ppl > 1e6
        self.total_tokens_processed: int = 0

    def update(self, batch_size: int, gate1_rejected: torch.Tensor, gate2_rejected: torch.Tensor) -> None:
        """Update statistics with batch results.

        Args:
            batch_size: Size of the processed batch
            gate1_rejected: Boolean tensor indicating Gate 1 rejections
            gate2_rejected: Boolean tensor indicating Gate 2 rejections
        """
        self.total_samples += batch_size

        gate1_count = gate1_rejected.sum().item()
        gate2_count = gate2_rejected.sum().item()
        both_count = (gate1_rejected & gate2_rejected).sum().item()

        self.gate1_rejections += gate1_count
        self.gate2_rejections += gate2_count
        self.both_gates_rejections += both_count

    def get_metrics(self) -> dict[str, float]:
        """Get current rejection metrics."""
        if self.total_samples == 0:
            return {
                "total_samples": 0,
                "acceptance_rate": 1.0,
                "gate1_rejection_rate": 0.0,
                "gate2_rejection_rate": 0.0,
                "both_gates_rejection_rate": 0.0,
            }

        total_rejections = self.gate1_rejections + self.gate2_rejections - self.both_gates_rejections

        metrics = {
            "total_samples": self.total_samples,
            "acceptance_rate": 1 - (total_rejections / self.total_samples),
            "gate1_rejection_rate": self.gate1_rejections / self.total_samples,
            "gate2_rejection_rate": self.gate2_rejections / self.total_samples,
            "both_gates_rejection_rate": self.both_gates_rejections / self.total_samples,
            "unique_rejections": total_rejections,
            "avg_bias": np.mean(self.bias_values) if self.bias_values else 0.0,
            "max_bias": np.max(self.bias_values) if self.bias_values else 0.0,
        }

        # Gate 2 detailed metrics
        if self.gate2_min_diffs:
            metrics["gate2_min_diff_avg"] = np.mean(self.gate2_min_diffs)
            metrics["gate2_min_diff_min"] = np.min(self.gate2_min_diffs)
            metrics["gate2_min_diff_max"] = np.max(self.gate2_min_diffs)

        # Temporal metrics (moving average of last 10 batches)
        if len(self.batch_rejection_history) > 0:
            recent_history = self.batch_rejection_history[-10:]
            metrics["rejection_rate_ma10"] = np.mean(recent_history)
            metrics["rejection_rate_trend"] = recent_history[-1] - recent_history[0] if len(recent_history) > 1 else 0.0

        # Length-based metrics
        if self.rejections_by_length:
            for bucket, (rejected, total) in self.rejections_by_length.items():
                if total > 0:
                    metrics[f"rejection_rate_{bucket}"] = rejected / total

        # PPL metrics (dual tracking for rollout vs FSDP)
        if self.rollout_ppl_accepted:
            metrics["rollout_ppl_accepted_mean"] = np.exp(np.mean(self.rollout_ppl_accepted))
            metrics["rollout_ppl_accepted_median"] = np.exp(np.median(self.rollout_ppl_accepted))

        if self.rollout_ppl_rejected:
            metrics["rollout_ppl_rejected_mean"] = np.exp(np.mean(self.rollout_ppl_rejected))
            metrics["rollout_ppl_rejected_median"] = np.exp(np.median(self.rollout_ppl_rejected))

        if self.fsdp_ppl_accepted:
            metrics["fsdp_ppl_accepted_mean"] = np.exp(np.mean(self.fsdp_ppl_accepted))
            metrics["fsdp_ppl_accepted_median"] = np.exp(np.median(self.fsdp_ppl_accepted))

        if self.fsdp_ppl_rejected:
            metrics["fsdp_ppl_rejected_mean"] = np.exp(np.mean(self.fsdp_ppl_rejected))
            metrics["fsdp_ppl_rejected_median"] = np.exp(np.median(self.fsdp_ppl_rejected))

        # PPL drift metrics (difference between rollout and FSDP in log space)
        if self.ppl_drift_accepted:
            metrics["ppl_drift_accepted_mean"] = np.mean(self.ppl_drift_accepted)
            metrics["ppl_drift_accepted_max"] = np.max(np.abs(self.ppl_drift_accepted))

        if self.ppl_drift_rejected:
            metrics["ppl_drift_rejected_mean"] = np.mean(self.ppl_drift_rejected)
            metrics["ppl_drift_rejected_max"] = np.max(np.abs(self.ppl_drift_rejected))

        # Overall quality improvement from filtering
        if self.rollout_ppl_accepted and self.rollout_ppl_rejected:
            all_rollout_ppl = self.rollout_ppl_accepted + self.rollout_ppl_rejected
            metrics["rollout_ppl_all_mean"] = np.exp(np.mean(all_rollout_ppl))
            metrics["rollout_ppl_improvement"] = metrics["rollout_ppl_all_mean"] - metrics["rollout_ppl_accepted_mean"]

        if self.total_tokens_processed > 0:
            metrics["extreme_ppl_rate"] = self.extreme_ppl_tokens / self.total_tokens_processed

        return metrics


class TwoGateRejectionFilter:
    """
    Two-gate rejection filter for ensuring data integrity in RL training.

    This filter implements the principled approach from the paper, targeting
    the root causes of numerical corruption rather than symptoms.
    """

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        """
        Initialize the two-gate filter.

        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or FilterConfig()
        self.stats = RejectionStats()

    def _get_length_bucket(self, length: int) -> str:
        """Categorize sequence length into buckets."""
        if length < 512:
            return "short_0_512"
        elif length < 2048:
            return "medium_512_2k"
        elif length < 8192:
            return "long_2k_8k"
        else:
            return "very_long_8k+"

    def gate1_systematic_bias_check(
        self, logprobs_fsdp: torch.Tensor, logprobs_vllm: torch.Tensor, response_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gate 1: Check for systematic bias in log probabilities.

        This gate detects sequences where accumulated precision errors
        exceed acceptable bounds, indicating systematic bias contamination.

        Args:
            logprobs_fsdp: High-precision (FP32) log probabilities [bs, seq_len]
            logprobs_vllm: Lower-precision (BFloat16) log probabilities [bs, seq_len]
            response_mask: Boolean mask for valid tokens [bs, seq_len]

        Returns:
            passes: Boolean tensor indicating which samples pass [bs]
            bias_per_seq: Average bias per sequence for diagnostics [bs]

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        # Validate input shapes
        if logprobs_fsdp.shape != logprobs_vllm.shape:
            raise ValueError(
                f"FSDP and vLLM logprobs must have same shape: {logprobs_fsdp.shape} vs {logprobs_vllm.shape}"
            )
        if logprobs_fsdp.shape != response_mask.shape:
            raise ValueError(f"Logprobs and mask must have same shape: {logprobs_fsdp.shape} vs {response_mask.shape}")
        if logprobs_fsdp.device != logprobs_vllm.device or logprobs_fsdp.device != response_mask.device:
            raise ValueError("All tensors must be on the same device")

        # Use verl_F.masked_mean to compute masked means efficiently
        # This computes mean over the sequence dimension (axis=1) for each batch element
        fsdp_mean_per_seq = verl_F.masked_mean(logprobs_fsdp, response_mask, axis=1)  # [bs]
        vllm_mean_per_seq = verl_F.masked_mean(logprobs_vllm, response_mask, axis=1)  # [bs]

        # Compute the difference of means
        bias_per_seq = fsdp_mean_per_seq - vllm_mean_per_seq  # [bs]

        # Check if within tolerance (use abs in-place for memory efficiency)
        passes = bias_per_seq.abs() < self.config.bias_epsilon

        # Store stats
        if self.config.save_rejection_stats:
            self.stats.bias_values.extend(bias_per_seq.abs().cpu().numpy().tolist())

            # Track length-based rejections
            seq_lengths = response_mask.sum(dim=1).cpu().numpy()
            for length, rejected in zip(seq_lengths, ~passes, strict=False):
                bucket = self._get_length_bucket(int(length))
                if bucket not in self.stats.rejections_by_length:
                    self.stats.rejections_by_length[bucket] = [0, 0]
                self.stats.rejections_by_length[bucket][1] += 1  # total
                if rejected:
                    self.stats.rejections_by_length[bucket][0] += 1  # rejected

        return passes, bias_per_seq

    def gate2_logit_instability_check(
        self, rollout_log_probs: torch.Tensor, top_log_probs: torch.Tensor, response_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gate 2: Check for numerical instability in token sampling.

        This gate detects tokens sampled from precision-limited distributions
        where reduced precision corrupts the probability computation.

        Theory:
        - BFloat16: 7-bit mantissa limits precision at extreme ratios
        - FP16: Underflows at 2^-14 ≈ 6.1×10^-5
        - Vocab consideration:
          * 32k vocab: uniform log(p) ≈ -10.4
          * 155k vocab (Qwen): uniform log(p) ≈ -11.95

        Threshold Selection (use max of):
        1. Hardware limits:
           - FP16: -9.7 (ln(2^-14), underflow at 6.1e-5)
           - BFloat16: -20.0 (practical mantissa limit)

        2. Vocab-aware thresholds (k=100, reject 100× below uniform):
           - 32k vocab: -15.0 (uniform at -10.4)
           - 155k vocab: -16.6 (uniform at -11.95)

        Default: -15.0 (balanced for most vocabs)

        Args:
            rollout_log_probs: Log probabilities of sampled tokens [bs, seq_len]
            top_log_probs: Maximum log probabilities from vLLM (logprobs=1) [bs, seq_len]
            response_mask: Boolean mask for valid tokens [bs, seq_len]

        Returns:
            passes: Boolean tensor indicating which samples pass [bs]
            min_diffs: Minimum log prob differences per sequence [bs]

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        # Validate input shapes
        if rollout_log_probs.shape != top_log_probs.shape:
            raise ValueError(
                f"Rollout and top logprobs must have same shape: {rollout_log_probs.shape} vs {top_log_probs.shape}"
            )
        if rollout_log_probs.shape != response_mask.shape:
            raise ValueError(
                f"Logprobs and mask must have same shape: {rollout_log_probs.shape} vs {response_mask.shape}"
            )
        if rollout_log_probs.device != top_log_probs.device or rollout_log_probs.device != response_mask.device:
            raise ValueError("All tensors must be on the same device")

        # Simple: compute difference between sampled and max log probs
        logprob_diffs = rollout_log_probs - top_log_probs  # [bs, seq_len]

        # Apply mask (set masked positions to 0, which is > threshold)
        masked_diffs = torch.where(response_mask.bool(), logprob_diffs, torch.zeros_like(logprob_diffs))

        # Check stability: find minimum difference among valid tokens
        min_diffs_per_seq = (
            torch.where(response_mask.bool(), masked_diffs, torch.full_like(masked_diffs, float("inf")))
            .min(dim=-1)
            .values
        )  # [bs]

        # A sequence passes if its minimum difference is above threshold
        passes = min_diffs_per_seq > self.config.instability_threshold

        # Count unstable tokens per sequence for stats
        if self.config.save_rejection_stats:
            unstable_counts = ((masked_diffs < self.config.instability_threshold) & response_mask.bool()).sum(dim=-1)
            self.stats.instability_counts.extend(unstable_counts.cpu().numpy().tolist())

            # Store Gate 2 detailed metrics
            self.stats.gate2_min_diffs.extend(min_diffs_per_seq.cpu().numpy().tolist())

            # Calculate average diffs per sequence
            avg_diffs = verl_F.masked_mean(masked_diffs, response_mask, axis=1)
            self.stats.gate2_avg_diffs.extend(avg_diffs.cpu().numpy().tolist())

        return passes, min_diffs_per_seq

    def filter_batch(
        self, batch_data: dict[str, torch.Tensor], return_indices: bool = False
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, float]]:
        """
        Filter a batch through both gates.

        Args:
            batch_data: Dictionary containing:
                - old_log_probs: FSDP log probabilities [bs, seq_len]
                - rollout_log_probs: vLLM log probabilities of sampled tokens [bs, seq_len]
                - response_mask: Valid token mask [bs, seq_len]
                - top_log_probs (optional): Maximum log probs for gate 2 [bs, seq_len]
                - Additional tensors to filter
            return_indices: If True, return accepted indices instead of filtered data

        Returns:
            filtered_batch: Batch with only accepted samples (or indices if return_indices=True)
            metrics: Rejection statistics and diagnostics
        """
        # Validate required inputs
        if not batch_data:
            raise ValueError("batch_data cannot be empty")
        if "response_mask" not in batch_data:
            raise ValueError("batch_data must contain 'response_mask' key")

        bs = batch_data["response_mask"].shape[0]
        device = batch_data["response_mask"].device

        if bs == 0:
            logger.warning("Empty batch received, returning empty results")
            empty_tensor = torch.tensor([], dtype=torch.long, device=device)
            empty_metrics = {
                "total_samples": 0,
                "acceptance_rate": 1.0,
                "gate1_rejection_rate": 0.0,
                "gate2_rejection_rate": 0.0,
            }
            return (empty_tensor, empty_metrics) if return_indices else ({}, empty_metrics)

        # Initialize acceptance mask (all True initially)
        accept_mask = torch.ones(bs, dtype=torch.bool, device=device)
        gate1_rejected = torch.zeros(bs, dtype=torch.bool, device=device)
        gate2_rejected = torch.zeros(bs, dtype=torch.bool, device=device)

        # Gate 1: Systematic Bias Check
        if self.config.enable_gate1 and "old_log_probs" in batch_data and "rollout_log_probs" in batch_data:
            gate1_passes, bias_values = self.gate1_systematic_bias_check(
                logprobs_fsdp=batch_data["old_log_probs"],
                logprobs_vllm=batch_data["rollout_log_probs"],
                response_mask=batch_data["response_mask"],
            )
            gate1_rejected = ~gate1_passes
            accept_mask &= gate1_passes

            if self.config.log_rejection_reasons:
                rejected_indices = torch.where(gate1_rejected)[0]
                for idx in rejected_indices:
                    self.stats.rejection_reasons.append(f"Gate1: Sample {idx} rejected - bias={bias_values[idx]:.3f}")

        # Gate 2: Instability Check
        if self.config.enable_gate2 and "rollout_log_probs" in batch_data and "top_log_probs" in batch_data:
            gate2_passes, min_diffs = self.gate2_logit_instability_check(
                rollout_log_probs=batch_data["rollout_log_probs"],
                top_log_probs=batch_data["top_log_probs"],
                response_mask=batch_data["response_mask"],
            )
            gate2_rejected = ~gate2_passes
            accept_mask &= gate2_passes

            if self.config.log_rejection_reasons:
                rejected_indices = torch.where(gate2_rejected)[0]
                for idx in rejected_indices:
                    self.stats.rejection_reasons.append(f"Gate2: Sample {idx} rejected - min_diff={min_diffs[idx]:.3f}")

        # Update statistics
        self.stats.update(bs, gate1_rejected, gate2_rejected)

        # Calculate PPL for each sequence (track both rollout and FSDP)
        if self.config.save_rejection_stats and "rollout_log_probs" in batch_data and "old_log_probs" in batch_data:
            rollout_log_probs = batch_data["rollout_log_probs"]  # vLLM (BFloat16)
            fsdp_log_probs = batch_data["old_log_probs"]  # FSDP (FP32)
            response_mask = batch_data["response_mask"]

            # Calculate mean log prob per sequence (negative of log PPL)
            rollout_mean_log_probs = verl_F.masked_mean(rollout_log_probs, response_mask, axis=1)
            fsdp_mean_log_probs = verl_F.masked_mean(fsdp_log_probs, response_mask, axis=1)

            # Convert to log PPL (negative of mean log prob)
            rollout_log_ppl = -rollout_mean_log_probs
            fsdp_log_ppl = -fsdp_mean_log_probs

            # Calculate PPL drift (difference in log space)
            ppl_drift = rollout_log_ppl - fsdp_log_ppl  # Positive means rollout has higher PPL

            # Track PPL for accepted and rejected sequences
            for accepted, roll_ppl, fsdp_ppl, drift in zip(
                accept_mask, rollout_log_ppl, fsdp_log_ppl, ppl_drift, strict=False
            ):
                if accepted:
                    self.stats.rollout_ppl_accepted.append(roll_ppl.item())
                    self.stats.fsdp_ppl_accepted.append(fsdp_ppl.item())
                    self.stats.ppl_drift_accepted.append(drift.item())
                else:
                    self.stats.rollout_ppl_rejected.append(roll_ppl.item())
                    self.stats.fsdp_ppl_rejected.append(fsdp_ppl.item())
                    self.stats.ppl_drift_rejected.append(drift.item())

            # Count extreme PPL tokens (PPL > 1e6)
            extreme_tokens = (rollout_log_probs < EXTREME_PPL_THRESHOLD) & response_mask.bool()
            self.stats.extreme_ppl_tokens += extreme_tokens.sum().item()
            self.stats.total_tokens_processed += response_mask.sum().item()

        # Track batch rejection rate in history
        batch_rejection_rate = 1 - accept_mask.float().mean().item()
        self.stats.batch_rejection_history.append(batch_rejection_rate)
        if len(self.stats.batch_rejection_history) > MAX_BATCH_HISTORY:
            self.stats.batch_rejection_history.pop(0)

        # Return indices if requested
        if return_indices:
            accepted_indices = torch.where(accept_mask)[0]
            metrics = self.stats.get_metrics()
            metrics["batch_acceptance_rate"] = accept_mask.float().mean().item()
            return accepted_indices, metrics

        # Filter the batch
        filtered_batch = {}
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == bs:
                filtered_batch[key] = tensor[accept_mask]
            else:
                # Keep non-batch tensors as-is
                filtered_batch[key] = tensor

        # Compute metrics
        metrics = self.stats.get_metrics()
        metrics["batch_acceptance_rate"] = accept_mask.float().mean().item()
        metrics["batch_size_original"] = bs
        metrics["batch_size_filtered"] = accept_mask.sum().item()

        return filtered_batch, metrics

    def reset_stats(self) -> None:
        """Reset rejection statistics."""
        self.stats.reset()

    def get_summary(self) -> str:
        """Get human-readable summary of rejection statistics."""
        metrics = self.stats.get_metrics()

        summary = [
            "=== Two-Gate Rejection Filter Summary ===",
            f"Total samples processed: {metrics['total_samples']}",
            f"Overall acceptance rate: {metrics['acceptance_rate']:.1%}",
            f"Gate 1 (bias) rejection rate: {metrics['gate1_rejection_rate']:.1%}",
            f"Gate 2 (instability) rejection rate: {metrics['gate2_rejection_rate']:.1%}",
            f"Both gates rejection rate: {metrics['both_gates_rejection_rate']:.1%}",
        ]

        if metrics["avg_bias"] > 0:
            summary.extend(
                [
                    f"Average bias magnitude: {metrics['avg_bias']:.4f}",
                    f"Maximum bias magnitude: {metrics['max_bias']:.4f}",
                ]
            )

        return "\n".join(summary)
