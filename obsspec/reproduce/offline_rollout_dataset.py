#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

from verl import DataProto

_STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


def _join(root: str, filename: str) -> str:
    return f"{root.rstrip('/')}/{filename}"


def _publish(local_path: str, destination: str) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = f"{destination}.tmp"
    shutil.copyfile(local_path, temporary_destination)
    os.replace(temporary_destination, destination)


def convert_batch(batch: DataProto) -> dict[str, list[torch.Tensor]]:
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"].bool()
    world_loss_mask = batch.batch["world_loss_mask"].bool()
    prompt_width = batch.batch["prompts"].shape[1]

    input_ids_list = []
    loss_mask_list = []
    for index in range(len(batch)):
        prompt_length = int(attention_mask[index, :prompt_width].sum())
        response_length = int(attention_mask[index, prompt_width:].sum())
        ids = input_ids[index, attention_mask[index]].clone()
        loss_mask = torch.zeros(ids.shape, dtype=torch.bool)
        loss_mask[prompt_length : prompt_length + response_length] = world_loss_mask[index, :response_length]
        input_ids_list.append(ids)
        loss_mask_list.append(loss_mask)

    return {
        "input_ids": input_ids_list,
        "loss_mask": loss_mask_list,
    }


def convert_step(source_dir: str, output_dir: str, step: int, overwrite: bool = False) -> None:
    destination = _join(output_dir, f"step_{step:06d}.pt")
    if Path(destination).exists() and not overwrite:
        print(f"Step {step}: already exists")
        return

    with tempfile.TemporaryDirectory(prefix=f"convert-rollout-step-{step:06d}-") as temporary_dir:
        source = _join(source_dir, f"step_{step:06d}.pkl")
        local_source = os.path.join(temporary_dir, f"step_{step:06d}.pkl")
        local_output = os.path.join(temporary_dir, f"step_{step:06d}.pt")
        shutil.copyfile(source, local_source)
        torch.save(convert_batch(DataProto.load_from_disk(local_source)), local_output)
        _publish(local_output, destination)
    print(f"Step {step}: converted")


def _list_step_paths(dataset_dir: str) -> list[tuple[int, str]]:
    paths = [str(path) for path in Path(dataset_dir).glob("step_*.pt")]

    steps = []
    for path in paths:
        if match := _STEP_PATTERN.search(os.path.basename(path)):
            steps.append((int(match.group(1)), path))
    return sorted(steps)


def _validate_step_paths(step_paths: list[tuple[int, str]], dataset_dir: str) -> None:
    if not step_paths:
        raise FileNotFoundError(f"No step_XXXXXX.pt files found in {dataset_dir}")
    steps = [step for step, _ in step_paths]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError(f"Step files are not contiguous: {steps}")


def _load_processed_batch(path: str, mmap: bool = False) -> dict[str, list[torch.Tensor]]:
    batch = torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
    if set(batch) != {"input_ids", "loss_mask"}:
        raise ValueError(f"{path} contains unexpected keys: {sorted(batch)}")
    if len(batch["input_ids"]) != len(batch["loss_mask"]):
        raise ValueError(f"{path} input and mask counts differ")
    for index, (input_ids, loss_mask) in enumerate(zip(batch["input_ids"], batch["loss_mask"], strict=True)):
        if input_ids.ndim != 1 or loss_mask.ndim != 1 or input_ids.shape != loss_mask.shape:
            raise ValueError(f"{path} sample {index} has incompatible input and mask shapes")
    return batch


class OfflineRolloutDataset(Dataset):
    def __init__(
        self,
        parquet_files,
        tokenizer=None,
        config=None,
        processor=None,
        max_samples: int = -1,
    ):
        del tokenizer, processor
        if max_samples not in (-1, None):
            raise ValueError("Offline rollout data must be consumed as complete step batches")

        dataset_dirs = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        if len(dataset_dirs) != 1:
            raise ValueError(f"Expected one offline rollout directory, got {dataset_dirs}")
        self.dataset_dir = str(dataset_dirs[0])
        self.step_paths = _list_step_paths(self.dataset_dir)
        _validate_step_paths(self.step_paths, self.dataset_dir)

        expected_batch_size = config.get("train_batch_size", None) if config is not None else None
        if expected_batch_size is None:
            raise ValueError("data.train_batch_size is required for offline rollout data")
        self.batch_size = int(expected_batch_size)
        self._cached_step_index = None
        self._cached_batch = None

    def __len__(self) -> int:
        return len(self.step_paths) * self.batch_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        step_index, sample_index = divmod(index, self.batch_size)
        if step_index != self._cached_step_index:
            step, path = self.step_paths[step_index]
            batch = _load_processed_batch(path, mmap=True)
            if len(batch["input_ids"]) != self.batch_size:
                raise ValueError(f"Step {step} contains {len(batch['input_ids'])} samples, expected {self.batch_size}")
            self._cached_step_index = step_index
            self._cached_batch = batch

        input_ids = self._cached_batch["input_ids"][sample_index]
        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(input_ids.shape[0], dtype=torch.long),
            "loss_mask": self._cached_batch["loss_mask"][sample_index],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for step in range(args.start_step, args.end_step + 1):
        convert_step(args.source_dir, args.output_dir, step, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
