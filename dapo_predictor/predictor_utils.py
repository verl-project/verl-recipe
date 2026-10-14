"""Utilities for predictor-driven prompt reordering in DAPO."""

from __future__ import annotations


def snake_sort_indices(scores: list[float], n_samples_per_prompt: int, dp_world_size: int) -> list[int]:
    """Return sample indices after prompt-level score sort with serpentine DP packing.

    The input scores are expected to be repeated per prompt group. We first collapse them to
    prompt-level scores, sort prompts descending by score, assign prompt groups to DP ranks in
    snake/serpentine order, then expand back to sample indices.
    """
    if n_samples_per_prompt <= 0:
        raise ValueError("n_samples_per_prompt must be positive")
    if dp_world_size <= 0:
        raise ValueError("dp_world_size must be positive")
    if len(scores) % n_samples_per_prompt != 0:
        raise ValueError("scores length must be divisible by n_samples_per_prompt")

    num_prompts = len(scores) // n_samples_per_prompt
    prompt_scores = [scores[i * n_samples_per_prompt] for i in range(num_prompts)]
    sorted_prompt_indices = sorted(range(num_prompts), key=lambda idx: prompt_scores[idx], reverse=True)

    dp_buckets: list[list[int]] = [[] for _ in range(dp_world_size)]
    for sorted_pos, prompt_idx in enumerate(sorted_prompt_indices):
        block = sorted_pos // dp_world_size
        offset = sorted_pos % dp_world_size
        dp_rank = offset if block % 2 == 0 else dp_world_size - 1 - offset
        dp_buckets[dp_rank].append(prompt_idx)

    ordered_prompt_indices = [prompt_idx for bucket in dp_buckets for prompt_idx in bucket]
    sample_indices: list[int] = []
    for prompt_idx in ordered_prompt_indices:
        start = prompt_idx * n_samples_per_prompt
        sample_indices.extend(range(start, start + n_samples_per_prompt))
    return sample_indices
