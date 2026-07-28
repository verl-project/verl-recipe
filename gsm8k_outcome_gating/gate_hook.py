"""
gate_hook — the "correct test" for outcome-gating, wired into veRL's compute_advantage.

The scientific question (the user's own next-step, from the tau2 work):
    Under a SHAPED reward, an all-FAIL group can have std>0 (different partial credit per
    failed rollout). DAPO's std-based dynamic sampling KEEPS such a group (std>0) and GRPO then
    fabricates a "phantom advantage" inside it — rewarding the least-bad failure (e.g. the
    shortest give-up). A BINARY-OUTCOME gate drops the group (all same outcome) and kills the
    phantom. So binary-outcome gating catches exactly what DAPO's std filter misses.

GATE_MODE env selects the arm:
    none    — plain GRPO (control)
    std     — DAPO-style: zero advantage on groups whose reward std ≈ 0
    outcome — binary-outcome gate: zero advantage on groups whose OUTCOME is all-same

★ Design choice that matters: we ZERO THE ADVANTAGES of gated rows, we do NOT zero response_mask.
  The token-mean denominator (loss_mask.sum()) is therefore UNCHANGED across all three arms, so
  this comparison is FREE of the 1/live_frac learning-rate confound (which the GSM8K/MATH bake-off
  showed is what mask-dropping really does). The only thing that varies between arms is which
  groups' gradient is removed — exactly the intervention under test.

Installed via the same .pth + sitecustomize mechanism as the py3.10 compat shim, so it lands in
the Ray worker where compute_advantage runs. No-op unless GATE_MODE is set.
"""

from __future__ import annotations

import os

STD_EPS = 1e-6  # a group with reward spread below this is "no contrast" (DAPO would drop it)


def _enabled() -> bool:
    return os.environ.get("GATE_MODE") in ("none", "std", "outcome")


def partition(index, outcome, reward, mode):
    """Return (drop_row_ids, stats). index/outcome/reward are per-row lists (len = n_rows)."""
    groups = {}
    for i, uid in enumerate(index):
        groups.setdefault(uid, []).append(i)
    drop = []
    dead_groups = 0
    for uid, rows in groups.items():
        outs = [outcome[i] for i in rows]
        rews = [reward[i] for i in rows]
        if mode == "std":
            mean = sum(rews) / len(rews)
            var = sum((r - mean) ** 2 for r in rews) / len(rews)
            is_dead = var**0.5 <= STD_EPS  # no reward contrast at all
        elif mode == "outcome":
            is_dead = len(set(outs)) == 1  # all-same OUTCOME (all-fail or all-pass)
        else:  # none
            is_dead = False
        if is_dead:
            drop.extend(rows)
            dead_groups += 1
    stats = {
        "mode": mode,
        "total_groups": len(groups),
        "dead_groups": dead_groups,
        "total_rows": len(index),
        "dropped_rows": len(drop),
        "live_frac": round(1 - len(drop) / max(1, len(index)), 4),
    }
    return drop, stats


def zero_adv(adv, drop_rows):
    """Zero the advantage of dropped rows. adv: torch/np [n_rows, seq] or [n_rows]. Returns adv."""
    if not drop_rows:
        return adv
    import numpy as np

    idx = list(drop_rows)
    if hasattr(adv, "index_fill_"):  # torch
        import torch

        t = torch.tensor(idx, dtype=torch.long, device=adv.device)
        adv.index_fill_(0, t, 0.0)
        return adv
    adv[np.asarray(idx)] = 0.0  # numpy
    return adv


def _row_outcomes(data):
    """Per-row (reward_scalar, binary_outcome). outcome from the raw success signal, not shaped."""
    # outcome_binary: exposed by the shaped reward via extra_info if present; else threshold scores.
    ntb = data.non_tensor_batch
    scores = data.batch["token_level_scores"]
    row_reward = scores.sum(dim=-1) if scores.dim() > 1 else scores
    row_reward = row_reward.detach().cpu().tolist()
    if "outcome_binary" in ntb:
        outcome = [int(x) for x in ntb["outcome_binary"]]
    else:
        outcome = [1 if r > 0.5 else 0 for r in row_reward]  # fallback: shaped>0.5 ~ pass
    return row_reward, outcome


def install():
    if not _enabled():
        return
    mode = os.environ["GATE_MODE"]
    if mode == "none":
        print("[gate] GATE_MODE=none — plain GRPO, no hook installed", flush=True)
        return
    from verl.trainer.ppo import ray_trainer

    if getattr(ray_trainer, "_gate_installed", False):
        return
    import functools

    _orig = ray_trainer.compute_advantage

    @functools.wraps(_orig)
    def compute_advantage_gated(data, *args, **kwargs):
        data = _orig(data, *args, **kwargs)
        ntb = data.non_tensor_batch
        if "uid" not in ntb:
            raise RuntimeError("gate_hook: batch has no uid; cannot group.")
        index = list(ntb["uid"])
        row_reward, outcome = _row_outcomes(data)
        drop, st = partition(index, outcome, row_reward, mode)
        if drop:
            data.batch["advantages"] = zero_adv(data.batch["advantages"], drop)
        print(
            f"[gate] mode={mode} live_frac={st['live_frac']} "
            f"dropped_rows={st['dropped_rows']}/{st['total_rows']} "
            f"dead_groups={st['dead_groups']}/{st['total_groups']} (advantage-zeroed, denom unchanged)",
            flush=True,
        )
        return data

    ray_trainer.compute_advantage = compute_advantage_gated
    ray_trainer._gate_installed = True
    print(f"[gate] installed: mode={mode} (advantage-zeroing, confound-free)", flush=True)
