"""Preprocess Endless Terminals to parquet format."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import tarfile
from pathlib import Path

import datasets
from huggingface_hub import snapshot_download

from verl.utils.hdfs_io import copy, makedirs

DEFAULT_REPO_ID = "obiwan96/endless-terminals"
SYSTEM_PROMPT = (
    "You are a highly capable Linux terminal agent. "
    "Complete the user's task by running commands and verifying the result. "
    "When the task is complete, call done."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/echo",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_tasks(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/task.toml"))


def read_instruction(task_dir: Path) -> str:
    return (task_dir / "instruction.md").read_text(encoding="utf-8").strip()


def archive_task(task_dir: Path) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", dereference=True) as archive:

            def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                return info

            for path in sorted(task_dir.iterdir(), key=lambda item: item.name):
                archive.add(path, arcname=path.name, recursive=True, filter=normalize)
    return output.getvalue()


def build_row(task_dir: Path, root: Path, index: int, split: str) -> dict:
    task_id = task_dir.relative_to(root).as_posix()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": read_instruction(task_dir)},
        ],
        "data_source": DEFAULT_REPO_ID,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "id": task_id,
            "task_binary": archive_task(task_dir),
        },
    }


def main() -> None:
    args = parse_args()
    if args.local_dataset_path is not None:
        root = Path(args.local_dataset_path).expanduser()
    else:
        root = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        )
    tasks = discover_tasks(root)
    if len(tasks) <= args.val_size:
        raise ValueError(f"Found {len(tasks)} tasks, which is not enough for val_size={args.val_size}")

    val_indices = set(random.Random(args.seed).sample(range(len(tasks)), args.val_size))
    train_rows = []
    val_rows = []
    for index, task_dir in enumerate(tasks):
        split = "val" if index in val_indices else "train"
        row = build_row(task_dir, root, index, split)
        (val_rows if split == "val" else train_rows).append(row)

    local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "val.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_path)
    print(f"Wrote {len(train_rows)} training tasks to {train_path}")
    print(f"Wrote {len(val_rows)} validation tasks to {val_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
