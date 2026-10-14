"""CPU regressions for explicit outcomes and the advantage-hook contract."""

import sys
from types import ModuleType, SimpleNamespace

import gate_hook
import gate_shaped_reward
import numpy as np
import pytest


class TensorStub:
    """Minimal tensor interface used by _row_outcomes; values backed by NumPy."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def dim(self):
        return self.values.ndim

    def sum(self, dim):
        return TensorStub(self.values.sum(axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def test_half_reward_keeps_correct_outcome(monkeypatch):
    monkeypatch.setattr(gate_shaped_reward, "LAMBDA", 0.5)
    monkeypatch.setattr(gate_shaped_reward, "PHANTOM", "short")
    score = gate_shaped_reward.compute_score("gsm8k", "x" * 512 + " #### 42", "42")
    assert score["score"] == 0.5
    data = SimpleNamespace(
        batch={"token_level_scores": TensorStub([score["score"]])},
        non_tensor_batch={"outcome_binary": [score["outcome_binary"]]},
    )
    assert gate_hook._row_outcomes(data) == ([0.5], [1])


def test_missing_outcome_is_error():
    data = SimpleNamespace(batch={"token_level_scores": TensorStub([[0, 0.5]])}, non_tensor_batch={})
    with pytest.raises(RuntimeError, match="explicit outcome_binary"):
        gate_hook._row_outcomes(data)


@pytest.mark.parametrize(("labels", "mode"), [([0.3], "outcome"), ([0], "outcomme")])
def test_invalid_inputs_rejected(labels, mode):
    with pytest.raises(ValueError):
        gate_hook.partition(["a"], labels, [0.0], mode)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="matching lengths"):
        gate_hook.partition(["a", "b"], [0], [0.0, 0.0], "outcome")


def test_installed_hook_preserves_mask_and_live_advantages(monkeypatch):
    trainer = ModuleType("verl.trainer.ppo.ray_trainer")
    trainer.compute_advantage = lambda data, *args, **kwargs: data
    ppo = ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = trainer
    monkeypatch.setitem(sys.modules, "verl", ModuleType("verl"))
    monkeypatch.setitem(sys.modules, "verl.trainer", ModuleType("verl.trainer"))
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", ppo)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", trainer)
    monkeypatch.setenv("GATE_MODE", "outcome")
    gate_hook.install()
    installed = trainer.compute_advantage
    gate_hook.install()
    assert trainer.compute_advantage is installed
    mask = np.array([[1, 1], [1, 0], [1, 1], [1, 0]])
    original_mask = mask.copy()
    adv = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    data = SimpleNamespace(
        batch={
            "token_level_scores": TensorStub([-0.1, -0.2, 0.8, -0.1]),
            "advantages": adv.copy(),
            "response_mask": mask,
        },
        non_tensor_batch={"uid": ["dead", "dead", "live", "live"], "outcome_binary": [0, 0, 1, 0]},
    )
    result = trainer.compute_advantage(data)
    np.testing.assert_array_equal(result.batch["advantages"][:2], 0)
    np.testing.assert_array_equal(result.batch["advantages"][2:], adv[2:])
    np.testing.assert_array_equal(result.batch["response_mask"], original_mask)


def test_unknown_install_mode_rejected(monkeypatch):
    monkeypatch.setenv("GATE_MODE", "outcomme")
    with pytest.raises(ValueError, match="Unknown GATE_MODE"):
        gate_hook.install()
