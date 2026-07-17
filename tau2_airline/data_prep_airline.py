"""Adapter E: tau2 airline tasks.json -> veRL parquet.

Each row is one airline task. The rollout conversation (system policy, greeting,
user turns) is generated dynamically inside ``tau2_agent_loop`` at rollout time,
so ``prompt`` here is only a short human-readable placeholder used by veRL for
batch shaping -- the loop rebuilds the real messages itself. The load-bearing
field is ``extra_info.task_id``, which the loop uses to select the task.

Routing to the custom loop is done via config
(``actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent``), so no
per-row ``agent_name`` column is required.

Usage:
    python data_prep_airline.py --out_dir ./data/tau2_airline --n_val 10
"""

import argparse
import os

import pandas as pd
from tau2.registry import registry


def task_placeholder_prompt(task) -> list[dict]:
    """A short, readable stand-in prompt. Not used for the actual rollout."""
    purpose = ""
    if getattr(task, "description", None) is not None:
        purpose = getattr(task.description, "purpose", "") or ""
    text = purpose.strip() or f"tau2 airline task {task.id}"
    return [{"role": "user", "content": f"[tau2 airline task; live conversation generated at rollout time] {text}"}]


def build_rows(domain: str, split: str | None):
    tasks = registry.get_tasks_loader(domain)(split)
    rows = []
    for i, t in enumerate(tasks):
        rows.append(
            {
                "data_source": f"tau2/{domain}",
                "prompt": task_placeholder_prompt(t),
                "ability": f"tau2-{domain}",
                "reward_model": {"style": "rule", "ground_truth": ""},  # unused: reward computed in-loop
                "extra_info": {"task_id": str(t.id), "index": i, "domain": domain},
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="airline")
    ap.add_argument("--split", default=None, help="tau2 task split name, or None for all")
    ap.add_argument("--out_dir", default="./data/tau2_airline")
    ap.add_argument("--n_val", type=int, default=10, help="hold out this many tasks for val")
    args = ap.parse_args()

    rows = build_rows(args.domain, args.split)
    print(f"loaded {len(rows)} {args.domain} tasks")

    os.makedirs(args.out_dir, exist_ok=True)
    # Deterministic split: first N as val, rest as train (small dataset).
    n_val = min(args.n_val, len(rows))
    val_rows = rows[:n_val] if n_val > 0 else rows
    train_rows = rows[n_val:] if n_val > 0 and n_val < len(rows) else rows

    for name, r in (("train", train_rows), ("test", val_rows)):
        df = pd.DataFrame(r)
        path = os.path.join(args.out_dir, f"{name}.parquet")
        df.to_parquet(path)
        print(f"wrote {path}: {len(df)} rows, cols={list(df.columns)}")


if __name__ == "__main__":
    main()
