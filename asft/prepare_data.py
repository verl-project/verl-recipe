#!/usr/bin/env python3
"""Prepare ASFT benchmark data (med + math) as parquet for fsdp_sft_trainer.

Downloads from HuggingFace chichi56/ASFT if not already present.
Outputs train + val parquet with columns: prompt, response.

Usage:
  python prepare_all_data.py --output_dir /path/to/output
  python prepare_all_data.py --dataset med --output_dir /path/to/output
  python prepare_all_data.py --dataset math --output_dir /path/to/output
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASETS = {
    "med": {
        "hf_file": "train_medmcqa_alpaca_10k.jsonl",
        "subdir": "med",
    },
    "math": {
        "hf_file": "numina_cot_10k.jsonl",
        "subdir": "math",
    },
}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_dataset(name: str, output_root: Path, max_samples: int = -1, val_size: int = 500):
    cfg = DATASETS[name]
    out_dir = output_root / cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    src_jsonl = out_dir / cfg["hf_file"]
    if not src_jsonl.exists():
        # Also check recipe data dir
        recipe_path = Path(__file__).parent / "data" / cfg["hf_file"]
        if recipe_path.exists():
            src_jsonl = recipe_path
        else:
            downloaded = hf_hub_download(
                repo_id="chichi56/ASFT",
                repo_type="dataset",
                filename=cfg["hf_file"],
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
            )
            src_jsonl = Path(downloaded).resolve()

    rows = load_jsonl(src_jsonl)
    if max_samples > 0:
        rows = rows[:max_samples]

    converted = []
    for item in rows:
        converted.append(
            {
                "prompt": item.get("instruction", ""),
                "response": item.get("response", ""),
            }
        )

    df = pd.DataFrame(converted)

    # Split into train and val
    if val_size > 0 and len(df) > val_size:
        val_df = df.tail(val_size)
        train_df = df.head(len(df) - val_size)
    else:
        train_df = df
        val_df = df.head(min(500, len(df)))

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"[{name}] source={src_jsonl} total={len(rows)}")
    print(f"[{name}] train={len(train_df)} -> {train_path}")
    print(f"[{name}] val={len(val_df)} -> {val_path}")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["med", "math", "all"], default="all")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=500)
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in datasets:
        prepare_dataset(name, output_root, args.max_samples, args.val_size)

    print("Done.")


if __name__ == "__main__":
    main()
