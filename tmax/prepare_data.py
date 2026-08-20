"""Download TMax Harbor tasks and convert them to verl RL parquet files."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import shutil
import subprocess
import tarfile
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs

DEFAULT_DATASET = "tmax/TMax-15K-Harbor@latest"
DATA_SOURCE = "tmax/TMax-15K-Harbor"
SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.

Your response must include a THOUGHT section before your action where you
explain your reasoning. After the THOUGHT, you must call the `bash` tool
with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`
count as a single action).

Failure to follow these rules — calling no tool, calling a tool other than
`bash`, or omitting the THOUGHT — will cause your response to be rejected.
"""
INSTANCE_TEMPLATE = """Please solve this task:

{{task}}

You can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,
etc.) to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase / environment by finding and reading relevant files.
2. If applicable, create a script to reproduce the issue or expected behaviour.
3. Implement the change(s) by editing the source code or environment state.
4. Verify your fix works by running your script (or relevant test) again.
5. Test edge cases to ensure your fix is robust.
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine it with any other command. After this command, you cannot
   continue working on this task.

## Important Rules

1. Every response must contain exactly one tool call to `bash`.
2. Directory and environment-variable changes ARE persistent across calls
   in this harness — you can `cd` and `export` and subsequent commands will
   see the change. (This differs from upstream mini-swe-agent's subshell
   model; treat the shell as a long-running login shell.)
3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.
4. Interactive commands are not possible. Use `yes`/`no` piping or
   non-interactive flags as appropriate.
5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,
   etc. to filter large outputs.

## Useful command examples

### Create a new file:
`cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF`

### Edit files with sed:
`sed -i 's/old_string/new_string/g' filename.py`        # all occurrences
`sed -i '1s/old_string/new_string/' filename.py`         # first on line 1
`sed -i '1,10s/old_string/new_string/g' filename.py`     # lines 1-10

### View file content:
`nl -ba filename.py | sed -n '10,20p'`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/tmax")
    parser.add_argument("--download_dir", default="~/data/tmax/harbor")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_tasks(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/task.toml"))


def dataset_directory_name(dataset: str) -> str:
    return dataset.split("@", 1)[0].rsplit("/", 1)[-1]


def download_dataset(dataset: str, download_dir: Path) -> Path:
    executable = shutil.which("harbor")
    if executable is None:
        raise FileNotFoundError("Harbor CLI not found")
    root = download_dir / dataset_directory_name(dataset)
    if discover_tasks(root):
        print(f"Using existing Harbor export at {root}")
        return root
    download_dir.mkdir(parents=True, exist_ok=True)
    command = [executable, "download", dataset, "--export", "--output-dir", str(download_dir)]
    subprocess.run(command, check=True)
    return root


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


def render_instance(instruction: str) -> str:
    return INSTANCE_TEMPLATE.replace("{{task}}", instruction)


def build_row(task_dir: Path, index: int, split: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_instance(read_instruction(task_dir))},
        ],
        "data_source": DATA_SOURCE,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "id": task_dir.name,
            "task_binary": archive_task(task_dir),
        },
    }


def main() -> None:
    args = parse_args()
    if args.val_size < 0:
        raise ValueError("--val_size must be non-negative")

    if args.local_dataset_path is not None:
        root = Path(args.local_dataset_path).expanduser()
    else:
        root = download_dataset(
            args.dataset,
            Path(args.download_dir).expanduser(),
        )

    tasks = discover_tasks(root)
    if not tasks:
        raise ValueError(f"No Harbor tasks found under {root}")
    if len(tasks) <= args.val_size:
        raise ValueError(f"Found {len(tasks)} tasks, which is not enough for val_size={args.val_size}")

    val_indices = set(random.Random(args.seed).sample(range(len(tasks)), args.val_size))
    train_rows = []
    val_rows = []
    for index, task_dir in enumerate(tasks):
        if index in val_indices:
            val_rows.append(build_row(task_dir, index, "val"))
        else:
            train_rows.append(build_row(task_dir, index, "train"))

    local_save_dir = os.path.expanduser(args.local_save_dir)
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
