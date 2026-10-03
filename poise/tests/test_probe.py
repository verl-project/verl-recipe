# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest
from recipe.poise.probe import ProbeBank, ProbeConfig, ProbeRow, RidgeProbe, domain_for, sibling_indices


def batch(step=0):
    rng = np.random.default_rng(step)
    return dict(
        uids=[f"{step}_{i}" for i in [0, 1, 0, 1]],
        domains=["math"] * 4,
        rewards=np.array([1, 0, 0, 1], dtype=np.float32),
        prompt=rng.normal(size=(4, 6)),
        response=rng.normal(size=(4, 6)),
        scalars=rng.random((4, 3)),
    )


@pytest.fixture
def config():
    return ProbeConfig(prompt_pca_dim=2, response_pca_dim=3, buffer_rows={"math": 12})


def test_bootstrap_transition_and_sibling_baseline(config):
    bank = ProbeBank(config)
    for step in range(16):
        data = batch(step)
        advantages, metrics = bank.advantages(**data)
        np.testing.assert_array_equal(advantages, [1, -1, -1, 1])
        assert metrics["poise/bootstrap"] == 1
        assert not bank.probes
        bank.commit()
        assert len(bank.buffers["math"]) <= 6
    data = batch(16)
    predictions = bank.probes["math"].predict(data["prompt"], data["response"], data["scalars"])
    advantages, metrics = bank.advantages(**data)
    np.testing.assert_allclose(advantages, data["rewards"] - predictions[[2, 3, 0, 1]], atol=1e-6)
    assert metrics["poise/bootstrap"] == 0
    old_probe = bank.probes["math"]
    assert bank.completed_steps == 16
    bank.commit()
    assert bank.probes["math"] is not old_probe
    assert bank.completed_steps == 17


@pytest.mark.parametrize("checkpoint_step", [8, 15, 16, 17])
def test_resume_matches_uninterrupted(config, tmp_path, checkpoint_step):
    original = ProbeBank(config)
    for step in range(checkpoint_step):
        original.advantages(**batch(step))
        original.commit()
    original.save(tmp_path)
    restored = ProbeBank(config)
    restored.load(tmp_path, checkpoint_step)
    for step in range(checkpoint_step, 19):
        a, ma = original.advantages(**batch(step))
        b, mb = restored.advantages(**batch(step))
        np.testing.assert_array_equal(a, b)
        assert ma == mb
        assert original.commit() == restored.commit()
    with pytest.raises(ValueError, match="steps differ"):
        restored.load(tmp_path, checkpoint_step + 1)
    with pytest.raises(ValueError, match="configuration differs"):
        ProbeBank(replace(config, bootstrap_steps=8)).load(tmp_path, checkpoint_step)


def test_no_fit_or_checkpoint_before_actor_update(config, tmp_path):
    bank = ProbeBank(config)
    bank.advantages(**batch())
    assert bank.completed_steps == 0 and not bank.buffers["math"]
    with pytest.raises(RuntimeError, match="previous actor update"):
        bank.advantages(**batch())
    with pytest.raises(RuntimeError, match="uncommitted"):
        bank.save(tmp_path)


def test_pair_targets_and_independent_domain_fifo(config):
    bank = ProbeBank(replace(config, bootstrap_steps=2, buffer_rows={"math": 5, "code": 4}))
    data = batch()
    data["domains"] = ["math", "code", "math", "code"]
    bank.advantages(**data)
    bank.commit()
    math_targets = [row.target for pair in bank.buffers["math"] for row in pair]
    assert math_targets == [0, 1]
    for step in [1, 2, 3]:
        data = batch(step)
        bank.advantages(**data)
        bank.commit()
    assert len(bank.buffers["math"]) == 2
    assert len(bank.buffers["code"]) == 1
    assert bank.observed_steps == {"math": 4, "code": 2}


def test_reordering_and_own_rollout_independence(config):
    bank = ProbeBank(replace(config, bootstrap_steps=1))
    bank.advantages(**batch())
    bank.commit()
    bank2 = ProbeBank(replace(config, bootstrap_steps=1))
    bank2.probes = bank.probes
    bank2.completed_steps = 1
    data = batch(1)
    first, _ = bank.advantages(**data)
    data["response"][0] += 100
    second, _ = bank2.advantages(**data)
    assert first[0] == second[0]  # Rollout 0's own features cannot affect its baseline.
    np.testing.assert_array_equal(sibling_indices(["a", "b", "a", "b"], ["math"] * 4), [2, 3, 0, 1])


@pytest.mark.parametrize("uids,domains", [(["a"], ["math"]), (["a"] * 3, ["math"] * 3), (["a"] * 2, ["math", "code"])])
def test_invalid_pairs(uids, domains):
    with pytest.raises(ValueError):
        sibling_indices(uids, domains)


def test_unseen_domain_fails_at_bootstrap_transition(config):
    bank = ProbeBank(replace(config, bootstrap_steps=1, buffer_rows={"math": 12, "code": 12}))
    bank.advantages(**batch())
    with pytest.raises(ValueError, match="no examples"):
        bank.commit()


@pytest.mark.parametrize(
    "source,expected",
    [("math__aime", "math"), ("codegen__taco", "code"), ("stem_web", "other"), ("openai/gsm8k", "math")],
)
def test_domain_routing(source, expected):
    assert domain_for(source) == expected


def test_probe_learns_signal_and_refits():
    rng = np.random.default_rng(5)
    prompt, response = (rng.normal(size=(192, 2)).astype(np.float32) for _ in range(2))
    scalars = rng.random((192, 3), dtype=np.float32)
    targets = 0.2 + 0.6 * scalars[:, 0]
    rows = [ProbeRow(*row) for row in zip(prompt[:128], response[:128], scalars[:128], targets[:128], strict=True)]
    config = ProbeConfig(prompt_pca_dim=2, response_pca_dim=2, ridge_alpha=0.01)
    probe = RidgeProbe(config).fit(rows)
    predictions = probe.predict(prompt[128:], response[128:], scalars[128:])
    constant_mse = np.mean((targets[:128].mean() - targets[128:]) ** 2)
    assert np.mean((predictions - targets[128:]) ** 2) < 0.01 * constant_mse
    coefficients = probe.regressor[-1].coef_.copy()

    probe.fit([row._replace(target=1.0 - row.target) for row in rows])
    refitted = probe.predict(prompt[128:], response[128:], scalars[128:])
    assert np.mean((refitted - (1.0 - targets[128:])) ** 2) < 0.01 * constant_mse
    assert not np.allclose(coefficients, probe.regressor[-1].coef_)
