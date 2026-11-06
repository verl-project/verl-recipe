"""Preprocess Endless Terminals to parquet format."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import re
import tarfile
from pathlib import Path

import datasets
import tomllib
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "obiwan96/endless-terminals"

# Exclude tasks that pass without agent actions or fail with the reference solution.
EXCLUDED_TASK_IDS = {
    "task_000000_00b7d96d",
    "task_000000_015c9641",
    "task_000000_02c27199",
    "task_000000_047019b5",
    "task_000000_089a919e",
    "task_000000_090301e3",
    "task_000000_0fffd67d",
    "task_000000_1241450d",
    "task_000000_13fbab8c",
    "task_000000_16508e14",
    "task_000000_16bae6c6",
    "task_000000_1710ec8a",
    "task_000000_1b07588c",
    "task_000000_1f806866",
    "task_000000_20bcf14e",
    "task_000000_23df8ab8",
    "task_000000_249b5651",
    "task_000000_270ab31c",
    "task_000000_2adee5b6",
    "task_000000_2b7a4e31",
    "task_000000_2fe597e0",
    "task_000000_30f9989a",
    "task_000000_31052eaa",
    "task_000000_3553ac20",
    "task_000000_36b30266",
    "task_000000_37a884bd",
    "task_000000_3af3408a",
    "task_000000_3bc4e68f",
    "task_000000_3d7a5b47",
    "task_000000_3e17389f",
    "task_000000_3e33ee25",
    "task_000000_420256fa",
    "task_000000_48d6fcd3",
    "task_000000_4ade8e9f",
    "task_000000_4c58315f",
    "task_000000_4cf3402e",
    "task_000000_4d781498",
    "task_000000_5521d3f2",
    "task_000000_56b13a16",
    "task_000000_59fdb1fb",
    "task_000000_5af0e862",
    "task_000000_5cae14d2",
    "task_000000_5d058fcf",
    "task_000000_61c0222b",
    "task_000000_630be072",
    "task_000000_636d236a",
    "task_000000_64be070f",
    "task_000000_65047b3d",
    "task_000000_6512ac67",
    "task_000000_654fa70c",
    "task_000000_6666507e",
    "task_000000_6723c42c",
    "task_000000_68703a5d",
    "task_000000_6a2cabd7",
    "task_000000_6b63e4c3",
    "task_000000_6c9cb030",
    "task_000000_6cc566b3",
    "task_000000_6d56eecc",
    "task_000000_6db57743",
    "task_000000_6e1be1cd",
    "task_000000_6e3c2b00",
    "task_000000_6fb185de",
    "task_000000_76d074ff",
    "task_000000_772affc8",
    "task_000000_7768b5ce",
    "task_000000_77ddada3",
    "task_000000_7cd2e4d9",
    "task_000000_7df1b438",
    "task_000000_7f5e2ef2",
    "task_000000_80f183a5",
    "task_000000_810c9b7d",
    "task_000000_8113449f",
    "task_000000_8626cdca",
    "task_000000_871bbccb",
    "task_000000_8867cc8b",
    "task_000000_8a392e48",
    "task_000000_8e1c176b",
    "task_000000_8e4a0ced",
    "task_000000_8fdd5a8f",
    "task_000000_945c928e",
    "task_000000_960fedcd",
    "task_000000_9719b9f3",
    "task_000000_97713302",
    "task_000000_9aa135c8",
    "task_000000_9dc27c3f",
    "task_000000_9e9b3164",
    "task_000000_9f6b6b6c",
    "task_000000_a0490b81",
    "task_000000_a4c7babc",
    "task_000000_a5de49c7",
    "task_000000_a7b59ea5",
    "task_000000_a8f61a57",
    "task_000000_ac644c33",
    "task_000000_ad881697",
    "task_000000_ad8e96da",
    "task_000000_ae953c04",
    "task_000000_b54561d8",
    "task_000000_b8d72e99",
    "task_000000_bb96f8da",
    "task_000000_bc9c5a3e",
    "task_000000_bd62e478",
    "task_000000_bf0a877f",
    "task_000000_c0e29c43",
    "task_000000_c66c3f8e",
    "task_000000_c78637fe",
    "task_000000_c7b6b77a",
    "task_000000_c9ba8288",
    "task_000000_cbb4abe0",
    "task_000000_ce45e6d1",
    "task_000000_ce51a6fd",
    "task_000000_cec40cc1",
    "task_000000_d06c6b98",
    "task_000000_d1c6c341",
    "task_000000_d4471c33",
    "task_000000_d76c7b86",
    "task_000000_d7badcc8",
    "task_000000_db0e409e",
    "task_000000_dcd52169",
    "task_000000_dcea1df0",
    "task_000000_dd913255",
    "task_000000_df3321eb",
    "task_000000_df919fac",
    "task_000000_e2e88388",
    "task_000000_e4ac86e9",
    "task_000000_e7dce417",
    "task_000000_e93c2b82",
    "task_000000_efb4f0f7",
    "task_000000_f09c372e",
    "task_000000_f0c18c44",
    "task_000000_f0f23518",
    "task_000000_f2747fb6",
    "task_000000_f34cabcd",
    "task_000000_f6a3b0a9",
    "task_000000_f94276c3",
    "task_000000_fffbc984",
}
SYSTEM_PROMPT = (
    "You are a highly capable Linux terminal agent. "
    "Complete the user's task by running commands and verifying the result. "
    "When the task is complete, call done."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/terminal_obsspec",
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
    # Build from the archived Dockerfile rather than a prebuilt image.
    config = (task_dir / "task.toml").read_text(encoding="utf-8")
    environment = re.search(r"(?ms)^\[environment\][ \t]*(?:#[^\n]*)?\n(.*?)(?=^\[|\Z)", config)
    if environment is not None:
        settings = re.sub(r"(?m)^[ \t]*docker_image[ \t]*=[^\n]*(?:\n|$)", "", environment[1])
        config = config[: environment.start(1)] + settings + config[environment.end(1) :]
    if "docker_image" in tomllib.loads(config).get("environment", {}):
        raise ValueError(f"Could not remove Docker image from {task_dir}")
    config_bytes = config.encode("utf-8")
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
                if path.name == "task.toml":
                    info = normalize(archive.gettarinfo(str(path), arcname=path.name))
                    info.size = len(config_bytes)
                    archive.addfile(info, io.BytesIO(config_bytes))
                else:
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


def prepare_data(args: argparse.Namespace) -> tuple[Path, Path]:
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
    # Split before filtering to keep validation membership stable when tasks are excluded.
    train_rows = []
    val_rows = []
    for index, task_dir in enumerate(tasks):
        split = "val" if index in val_indices else "train"
        if task_dir.relative_to(root).as_posix() in EXCLUDED_TASK_IDS:
            continue
        row = build_row(task_dir, root, index, split)
        (val_rows if split == "val" else train_rows).append(row)

    local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "endless-terminals-train.parquet")
    val_path = os.path.join(local_save_dir, "endless-terminals-val.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_path)
    print(f"Wrote {len(train_rows)} training tasks to {train_path}")
    print(f"Wrote {len(val_rows)} validation tasks to {val_path}")

    return Path(train_path), Path(val_path)


if __name__ == "__main__":
    prepare_data(parse_args())
