# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl_tinker.backends._loss import make_branching_loss
from verl_tinker.data.datum_processing import _datums_to_forward_td, _datums_to_update_actor_td
from verl_tinker.tinker_ops import normalize_tinker_loss_spec

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize(
    ("name", "wire_config", "expected"),
    [
        ("cross_entropy", None, {"name": "cross_entropy"}),
        ("importance_sampling", None, {"name": "importance_sampling"}),
        ("ppo", None, {"name": "ppo", "clip_ratio_low": 0.2, "clip_ratio_high": 0.2}),
        (
            "ppo",
            {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
            {"name": "ppo", "clip_ratio_low": 0.1, "clip_ratio_high": 0.1},
        ),
        ("cispo", None, {"name": "cispo", "clip_ratio_low": 1.0, "clip_ratio_high": 3.0}),
        ("dro", {"beta": 0.05}, {"name": "dro", "dro_beta": 0.05}),
        ("custom_from_config", {"ignored": float("nan")}, {"name": "custom_from_config"}),
    ],
)
def test_normalize_tinker_loss_spec(name, wire_config, expected):
    actual = normalize_tinker_loss_spec(name, wire_config)
    assert actual["name"] == expected["name"]
    for key, value in expected.items():
        if key != "name":
            assert actual[key] == pytest.approx(value)


@pytest.mark.parametrize(
    ("name", "config", "match"),
    [
        ("unknown", None, "Unsupported loss"),
        ("importance_sampling", {"beta": 0.1}, "does not accept"),
        ("ppo", {"clip_low_threshold": 1.1}, "requires 0 <="),
        ("cispo", {"clip_high_threshold": 0.9}, "requires 0 <="),
        ("dro", None, "requires a positive"),
        ("dro", {"beta": 0.0}, "requires a positive"),
    ],
)
def test_normalize_tinker_loss_spec_rejects_invalid_requests(name, config, match):
    with pytest.raises(ValueError, match=match):
        normalize_tinker_loss_spec(name, config)


def _branching_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "_target_": "verl.workers.config.actor.FSDPActorConfig",
                    "strategy": "fsdp",
                    "rollout_n": 1,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "loss_agg_mode": "token-mean",
                    "entropy_coeff": 0.25,
                    "use_kl_loss": True,
                    "policy_loss": {
                        "_target_": "verl.workers.config.actor.PolicyLossConfig",
                        "loss_mode": "vanilla",
                    },
                }
            }
        }
    )


def test_branching_loss_uses_isolated_request_configs(monkeypatch):
    captured = []

    def fake_ppo_loss(*, config, model_output, data, dp_group=None):
        captured.append(config)
        return torch.tensor(2.0), {}

    monkeypatch.setattr("verl_tinker.backends._loss.ppo_loss", fake_ppo_loss)
    startup = _branching_config()
    loss_fn = make_branching_loss(startup)

    for spec in (
        normalize_tinker_loss_spec("ppo", {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}),
        normalize_tinker_loss_spec("dro", {"beta": 0.05}),
        normalize_tinker_loss_spec("importance_sampling"),
    ):
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor_data(data, "__tinker_loss_spec__", spec)
        loss_fn(model_output={}, data=data)

    assert captured[0] is not captured[1]
    assert captured[0].policy_loss is not captured[1].policy_loss
    assert len({id(cfg.global_batch_info) for cfg in captured}) == len(captured)
    assert len({id(cfg.policy_loss) for cfg in captured}) == len(captured)
    assert len({id(cfg.policy_loss.rollout_correction) for cfg in captured}) == len(captured)
    assert captured[0].policy_loss.loss_mode == "vanilla"
    assert captured[0].clip_ratio_low == pytest.approx(0.1)
    assert captured[0].clip_ratio_high == pytest.approx(0.1)
    assert captured[0].clip_ratio_c == pytest.approx(1e9)
    assert captured[1].policy_loss.loss_mode == "dro"
    assert captured[1].policy_loss.dro_beta == pytest.approx(0.05)
    assert captured[2].policy_loss.loss_mode == "bypass_mode"
    rollout_correction = captured[2].policy_loss.rollout_correction
    assert rollout_correction.rollout_is == "token"
    assert rollout_correction.rollout_is_threshold == pytest.approx(2.0)
    assert rollout_correction.rollout_is_batch_normalize is False
    assert rollout_correction.rollout_rs is None
    assert rollout_correction.rollout_rs_threshold is None
    assert rollout_correction.bypass_mode is True
    assert rollout_correction.loss_type == "reinforce"
    assert all(cfg.loss_agg_mode == "token-sum" for cfg in captured)
    assert all(cfg.entropy_coeff == 0 and not cfg.use_kl_loss for cfg in captured)
    assert startup.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert startup.actor_rollout_ref.actor.entropy_coeff == pytest.approx(0.25)
    assert startup.actor_rollout_ref.actor.use_kl_loss is True
    assert startup.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


def test_custom_from_config_preserves_isolated_startup_actor_config(monkeypatch):
    startup = _branching_config()
    captured = []
    initial_global_batch_info = []

    def fake_ppo_loss(*, config, model_output, data, dp_group=None):
        captured.append(config)
        initial_global_batch_info.append(dict(config.global_batch_info))
        config.global_batch_info["mutated_by_verl"] = True
        return torch.tensor(2.0), {}

    monkeypatch.setattr("verl_tinker.backends._loss.ppo_loss", fake_ppo_loss)
    loss_fn = make_branching_loss(startup)
    for _ in range(2):
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor_data(data, "__tinker_loss_spec__", {"name": "custom_from_config"})
        loss_fn(model_output={}, data=data)

    assert captured[0] is not captured[1]
    assert captured[0].policy_loss is not captured[1].policy_loss
    assert captured[0].policy_loss.loss_mode == "vanilla"
    assert captured[0].loss_agg_mode == "token-mean"
    assert captured[0].entropy_coeff == pytest.approx(0.25)
    assert captured[0].use_kl_loss is True
    assert captured[1].global_batch_info == {"mutated_by_verl": True}
    assert initial_global_batch_info == [{}, {}]
    assert "global_batch_info" not in startup.actor_rollout_ref.actor


def _capture_request_configs(monkeypatch, *specs):
    captured = []

    def fake_ppo_loss(*, config, model_output, data, dp_group=None):
        captured.append(config)
        return torch.tensor(0.0), {}

    monkeypatch.setattr("verl_tinker.backends._loss.ppo_loss", fake_ppo_loss)
    loss_fn = make_branching_loss(_branching_config())
    for spec in specs:
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor_data(data, "__tinker_loss_spec__", spec)
        loss_fn(model_output={}, data=data)
    return captured


def test_ppo_dual_clip_sentinel_is_above_verl_stabilized_ratio(monkeypatch):
    (config,) = _capture_request_configs(monkeypatch, normalize_tinker_loss_spec("ppo"))
    policy_loss = get_policy_loss_fn("vanilla")
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.full((1, 1), 20.0)
    advantages = -torch.ones(1, 1)
    response_mask = torch.ones(1, 1, dtype=torch.bool)

    loss, _ = policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
    )

    max_stabilized_ratio = torch.exp(torch.tensor(20.0)).item()
    assert config.clip_ratio_c > max_stabilized_ratio
    assert loss.item() == pytest.approx(max_stabilized_ratio)


def test_native_dro_uses_beta_and_token_sum(monkeypatch):
    beta = 0.5
    (config,) = _capture_request_configs(monkeypatch, normalize_tinker_loss_spec("dro", {"beta": beta}))
    policy_loss = get_policy_loss_fn("dro")
    old_log_prob = torch.tensor([[-1.0, -2.0]])
    log_prob = torch.tensor([[-0.5, -1.5]])
    advantages = torch.tensor([[2.0, -1.0]])
    response_mask = torch.ones(1, 2, dtype=torch.bool)

    loss, _ = policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
    )

    expected_per_token = -(log_prob * advantages - 0.5 * beta * (log_prob - old_log_prob).square())
    assert loss.item() == pytest.approx(expected_per_token.sum().item())


def test_importance_sampling_uses_detached_token_tis_reinforce(monkeypatch):
    (config,) = _capture_request_configs(monkeypatch, normalize_tinker_loss_spec("importance_sampling"))
    policy_loss = get_policy_loss_fn("bypass_mode")
    rollout_log_prob = torch.tensor([[-1.0, -1.0]])
    log_prob = torch.tensor([[-1.0, math.log(3.0) - 1.0]], requires_grad=True)
    advantages = torch.full((1, 2), 2.0)
    response_mask = torch.ones(1, 2, dtype=torch.bool)

    loss, metrics = policy_loss(
        old_log_prob=rollout_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=config,
    )
    loss.backward()

    expected_weights = torch.tensor([[1.0, 2.0]])
    expected_loss = -(expected_weights * log_prob.detach() * advantages).sum()
    assert loss.item() == pytest.approx(expected_loss.item())
    torch.testing.assert_close(log_prob.grad, -expected_weights * advantages)
    assert metrics["rollout_corr/rollout_is_max"] == pytest.approx(2.0)


def test_cross_entropy_token_sum_preserves_signed_custom_loss_weights():
    loss_fn = make_branching_loss(_branching_config())
    data = TensorDict(
        {
            "loss_mask": torch.nested.as_nested_tensor([torch.tensor([0.0, 2.0, -3.0])], layout=torch.jagged),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "dp_size", 2)
    tu.assign_non_tensor_data(data, "batch_num_tokens", -1.0)
    tu.assign_non_tensor_data(data, "__loss_mode__", "sft")
    model_output = {"log_probs": torch.nested.as_nested_tensor([torch.tensor([-1.0, -2.0, -3.0])], layout=torch.jagged)}

    loss, metrics = loss_fn(model_output=model_output, data=data)

    # Rolled weights are [2, -3, 0], so local weighted NLL is
    # 1*2 + 2*(-3) = -4; dp_size compensates FSDP's mean reduction.
    assert loss.item() == pytest.approx(-8.0)
    assert metrics["loss"].aggregate() == pytest.approx(-8.0)


def _tensor_data(data, dtype="float32"):
    return SimpleNamespace(data=data, dtype=dtype, shape=[len(data)])


def test_rl_translation_preserves_wire_advantages():
    datum = SimpleNamespace(
        model_input=SimpleNamespace(to_ints=lambda: [10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": _tensor_data([11, 12, 13], dtype="int64"),
            "logprobs": _tensor_data([0.0, -0.2, -0.3]),
            "advantages": _tensor_data([0.0, 0.25, 0.25]),
            "response_mask": _tensor_data([0.0, 1.0, 1.0]),
        },
    )

    td = _datums_to_update_actor_td([datum], mini_batch_size=1)

    torch.testing.assert_close(td["advantages"], torch.tensor([[0.25, 0.25]]))


def test_forward_translation_rejects_multi_target_custom_loss_before_engine():
    datum = SimpleNamespace(
        model_input=SimpleNamespace(to_ints=lambda: [10, 11]),
        loss_fn_inputs={
            "target_tokens": SimpleNamespace(
                data=[11, 12, 21, 22],
                dtype="int64",
                shape=[2, 2],
            )
        },
    )

    with pytest.raises(HTTPException, match="only 1-D target_tokens") as exc_info:
        _datums_to_forward_td([datum])

    assert exc_info.value.status_code == 422
