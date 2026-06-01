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
Dataset utilities for DanceGRPO training.

Provides:
  - ``PickAPicDataset`` – loads prompt-only entries from JSON files.
  - ``collate_fn``       – batches samples into tensors / numpy arrays.
  - ``create_rl_dataset`` – factory that returns (dataset, collate_fn).
  - ``create_rl_sampler`` – factory that returns a (possibly shuffled) sampler.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler

__all__ = [
    "PickAPicDataset",
    "collate_fn",
    "create_rl_dataset",
    "create_rl_sampler",
]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a list of sample dicts into batched tensors / numpy arrays.

    Tensor values are stacked along dim 0.  Non-tensor values are packed into
    a 1-D ``np.ndarray`` of ``dtype=object``.

    Args:
        data_list: List of dicts produced by ``PickAPicDataset.__getitem__``.

    Returns:
        Batched dict ready for :class:`~verl.DataProto`.
    """
    tensors: dict[str, list] = defaultdict(list)
    non_tensors: dict[str, list] = defaultdict(list)

    for sample in data_list:
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    out: dict = {}
    for key, vals in tensors.items():
        try:
            out[key] = torch.stack(vals, dim=0)
        except Exception:
            # Fall back to object array if shapes differ.
            out[key] = np.fromiter(vals, dtype=object, count=len(vals))

    for key, vals in non_tensors.items():
        out[key] = np.fromiter(vals, dtype=object, count=len(vals))

    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PickAPicDataset(Dataset):
    """Minimal dataset that loads only the ``prompt`` field from JSON files.

    Each JSON file must be a list of objects containing a ``"prompt"`` key.

    Args:
        data_files: Path or list of paths to JSON data files.
        max_samples: Maximum number of samples to load (``-1`` for all).
    """

    def __init__(
        self,
        data_files: str | list[str],
        max_samples: int = -1,
    ) -> None:
        if isinstance(data_files, str):
            data_files = [data_files]

        self.data: list[dict] = []

        for data_file in data_files:
            if not os.path.exists(data_file):
                continue
            with open(data_file, encoding="utf-8") as fh:
                entries = json.load(fh)
            for entry in entries:
                if "prompt" in entry:
                    self.data.append({"prompt": entry["prompt"]})
                if max_samples > 0 and len(self.data) >= max_samples:
                    break
            if max_samples > 0 and len(self.data) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with a ``"prompts"`` key (numpy scalar string)."""
        return {"prompts": self.data[idx]["prompt"], "dummy": torch.ones(1)}


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_rl_dataset(
    data_paths: str | list[str],
    data_config: DictConfig,
    tokenizer=None,
    processor=None,
    is_train: bool = True,
    max_samples: int = -1,
) -> tuple[Dataset, callable]:
    """Create a :class:`PickAPicDataset` and its corresponding collate function.

    The ``tokenizer`` and ``processor`` arguments are accepted for API
    compatibility with verl's standard dataset factories but are unused,
    since this dataset contains only text prompts.

    Args:
        data_paths: Path(s) to JSON data file(s).
        data_config: Hydra/OmegaConf data config node.
        tokenizer: Unused (kept for API compatibility).
        processor: Unused (kept for API compatibility).
        is_train: Whether to create a training dataset (unused, for API compat).
        max_samples: Maximum number of samples to load (``-1`` for all).

    Returns:
        ``(dataset, collate_fn)`` tuple.
    """
    dataset = PickAPicDataset(data_files=data_paths, max_samples=max_samples)
    return dataset, collate_fn


def create_rl_sampler(data_config: DictConfig, dataset: Dataset) -> Sampler:
    """Create a (possibly shuffled) sampler for *dataset*.

    Args:
        data_config: Hydra/OmegaConf data config node. Must contain ``shuffle``
            (bool) and optionally ``seed`` (int).
        dataset: The dataset to sample from.

    Returns:
        A :class:`~torch.utils.data.Sampler` instance.
    """
    from torch.utils.data import SequentialSampler

    if data_config.shuffle:
        from torchdata.stateful_dataloader.sampler import RandomSampler

        generator = torch.Generator()
        seed = data_config.get("seed", None)
        if seed is not None:
            generator.manual_seed(seed)
        return RandomSampler(data_source=dataset, generator=generator)

    return SequentialSampler(data_source=dataset)
