"""Offline (no-GPU, no-verl) tests for the gate hook and the shaped reward.

Run from this directory:  python test_gate_offline.py
Covers the pure-python pieces: group partitioning for both gate modes,
advantage zeroing, both phantom directions, and lambda scaling.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate_hook import partition, zero_adv  # noqa: E402


def reload_reward(**env):
    for k in ("GATE_LAMBDA", "GATE_PHANTOM"):
        os.environ.pop(k, None)
    os.environ.update(env)
    import gate_shaped_reward

    return importlib.reload(gate_shaped_reward)


def test_partition_std_keeps_the_phantom_group():
    # group "a": all-fail but with reward spread (shaped) -> DAPO's std gate keeps it
    # group "b": mixed outcomes -> everyone keeps it
    index = ["a"] * 4 + ["b"] * 4
    outcome = [0, 0, 0, 0, 1, 0, 1, 0]
    reward = [-0.1, -0.2, -0.05, -0.3, 1.0, -0.2, 0.9, -0.1]
    drop, st = partition(index, outcome, reward, "std")
    assert drop == [], f"std gate should keep std>0 groups, dropped {drop}"
    assert st["dead_groups"] == 0


def test_partition_outcome_drops_the_phantom_group():
    index = ["a"] * 4 + ["b"] * 4
    outcome = [0, 0, 0, 0, 1, 0, 1, 0]
    reward = [-0.1, -0.2, -0.05, -0.3, 1.0, -0.2, 0.9, -0.1]
    drop, st = partition(index, outcome, reward, "outcome")
    assert sorted(drop) == [0, 1, 2, 3], f"outcome gate must drop the all-fail group, got {drop}"
    assert st["dead_groups"] == 1 and st["live_frac"] == 0.5


def test_partition_std_drops_degenerate_groups():
    # identical rewards (post-collapse homogenization) -> std gate finally fires
    index = ["a"] * 4
    drop, st = partition(index, [0, 0, 0, 0], [-0.3, -0.3, -0.3, -0.3], "std")
    assert sorted(drop) == [0, 1, 2, 3] and st["dead_groups"] == 1


def test_partition_none_drops_nothing():
    index = ["a"] * 4
    drop, st = partition(index, [0, 0, 0, 0], [-0.3, -0.3, -0.3, -0.3], "none")
    assert drop == [] and st["live_frac"] == 1.0


def test_zero_adv_numpy():
    import numpy as np

    adv = np.ones((4, 3))
    out = zero_adv(adv, [1, 3])
    assert out[1].sum() == 0 and out[3].sum() == 0 and out[0].sum() == 3


def test_short_phantom_rewards_brevity_in_failures():
    g = reload_reward()  # defaults: lambda=0.30, phantom=short
    short_wrong = g.compute_score("openai/gsm8k", "5", "7")["score"]
    long_wrong = g.compute_score("openai/gsm8k", "x" * 512 + " 5", "7")["score"]
    assert short_wrong > long_wrong, "short phantom must favor the shortest failure"
    assert abs(long_wrong - (-0.30)) < 1e-9, "full-length wrong answer scores exactly -lambda"


def test_long_phantom_rewards_verbosity_in_failures():
    g = reload_reward(GATE_PHANTOM="long")
    short_wrong = g.compute_score("openai/gsm8k", "5", "7")["score"]
    long_wrong = g.compute_score("openai/gsm8k", "x" * 512 + " 5", "7")["score"]
    assert long_wrong > short_wrong, "long phantom must favor the longest failure"


def test_lambda_scaling():
    g = reload_reward(GATE_LAMBDA="0.50")
    assert abs(g.compute_score("openai/gsm8k", "x" * 512 + " 5", "7")["score"] - (-0.50)) < 1e-9


def test_outcome_rides_along_and_correctness():
    g = reload_reward()
    s = g.compute_score("openai/gsm8k", "so the total is #### 7", "7")
    assert s["outcome_binary"] == 1 and s["acc"] == 1.0 and s["score"] > 0.5
    s = g.compute_score("openai/gsm8k", "#### 5", "7")
    assert s["outcome_binary"] == 0 and s["score"] <= 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} offline tests passed")
