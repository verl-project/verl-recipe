# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tinker import AdamParams, SampleResponse
from verl_tinker.tinker_ops import (
    forward_backward,
    load_state,
    optim_step,
    sample,
    save_state,
)


class _Prompt:
    def to_ints(self):
        return [101, 102, 103]


class _Engine:
    def __init__(self, stop_reason="stop", token_ids=None, log_probs=None):
        self.sampling_params = None
        self.stop_reason = stop_reason
        self.token_ids = token_ids if token_ids is not None else [201, 202]
        self.log_probs = log_probs if log_probs is not None else [-0.1, -0.2]

    async def generate(self, request_id, prompt_ids, sampling_params):
        assert prompt_ids == [101, 102, 103]
        self.sampling_params = sampling_params
        return SimpleNamespace(
            token_ids=self.token_ids,
            log_probs=self.log_probs,
            stop_reason=self.stop_reason,
            extra_fields={
                "prompt_ids": [[11, 12], [13, None], [0, 0]],
                "prompt_logprobs": [[-1.1, -2.2], [-3.3, None], [0.0, 0.0]],
            },
        )


class _OptimStepEngine:
    def __init__(self):
        self.optim_step_params = None

    def optim_step(self, optim_step_params=None):
        self.optim_step_params = optim_step_params
        return [{"grad_norm": 1.0}, {}, {"grad_norm": 3.0, "lr": 1e-5}]


class _LoadStateEngine:
    def __init__(self):
        self.load_calls = []

    def load_checkpoint(self, local_dir, zero_optimizer_grad=False):
        self.load_calls.append((local_dir, zero_optimizer_grad))


class _SaveStateEngine:
    def __init__(self):
        self.save_calls = []

    def save_checkpoint(self, local_dir, step):
        self.save_calls.append((local_dir, step))


class _ForwardBackwardEngine:
    world_size = 1

    def __init__(self, result_td):
        self.result_td = result_td

    def forward_backward(self, _data):
        return self.result_td


def _cross_entropy_datum(prefix, target_tokens):
    return SimpleNamespace(
        model_input=SimpleNamespace(to_ints=lambda: prefix),
        loss_fn_inputs={
            "target_tokens": SimpleNamespace(data=target_tokens),
        },
    )


async def _to_thread_inline(func, /, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_forward_backward_returns_verl_model_logprobs_for_cross_entropy(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    monkeypatch.setattr("verl_tinker.tinker_ops._datums_to_sft_td", lambda *args, **kwargs: object())
    log_probs = torch.nested.as_nested_tensor(
        [torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-4.0, -5.0, -6.0, -7.0])],
        layout=torch.jagged,
    )
    result_td = TensorDict({"log_probs": log_probs}, batch_size=[2])
    result_td.set_non_tensor("metrics", {"loss": 4.25})
    datums = [
        _cross_entropy_datum([10, 11], [11, 12]),
        _cross_entropy_datum([20, 21, 22], [21, 22, 23]),
    ]

    result = await forward_backward(_ForwardBackwardEngine(result_td), datums, "cross_entropy")

    assert result["loss_fn_outputs"] == [
        {"logprobs": {"data": [-1.0, -2.0], "dtype": "float32", "shape": [2]}},
        {"logprobs": {"data": [-4.0, -5.0, -6.0], "dtype": "float32", "shape": [3]}},
    ]
    assert result["metrics"]["loss:mean"] == 4.25


@pytest.mark.asyncio
async def test_forward_backward_zero_fills_missing_cross_entropy_model_logprobs(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    monkeypatch.setattr("verl_tinker.tinker_ops._datums_to_sft_td", lambda *args, **kwargs: object())
    result_td = TensorDict({}, batch_size=[])
    result_td.set_non_tensor("metrics", {"loss": 4.25})

    result = await forward_backward(
        _ForwardBackwardEngine(result_td),
        [_cross_entropy_datum([10, 11], [11, 12])],
        "cross_entropy",
    )

    assert result["loss_fn_outputs"] == [
        {"logprobs": {"data": [0.0, 0.0], "dtype": "float32", "shape": [2]}},
    ]
    assert result["metrics"]["loss:mean"] == 4.25


@pytest.mark.asyncio
async def test_sample_formats_topk_prompt_logprobs_for_current_tinker_schema():
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(max_tokens=4, temperature=0.7, top_p=0.9, top_k=20, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=2,
    )
    engine = _Engine()

    result = await sample(engine, req)

    expected = [[], [(11, -1.1), (12, -2.2)], [(13, -3.3)]]
    assert result["topk_prompt_logprobs"] == expected
    assert not isinstance(result["topk_prompt_logprobs"], dict)
    assert engine.sampling_params["prompt_logprobs"] == 2

    if hasattr(SampleResponse, "model_validate"):
        response = SampleResponse.model_validate(result)
        assert response.topk_prompt_logprobs == expected


@pytest.mark.asyncio
async def test_sample_fans_out_multiple_sequences_and_offsets_seed():
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(max_tokens=4, temperature=0.7, top_p=0.9, top_k=20, stop=None, seed=10),
        num_samples=3,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )

    class MultiEngine(_Engine):
        def __init__(self):
            super().__init__()
            self.seeds = []

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.seeds.append(sampling_params["seed"])
            seed = sampling_params["seed"]
            return SimpleNamespace(token_ids=[seed], log_probs=[-0.1], stop_reason="stop", extra_fields={})

    engine = MultiEngine()
    result = await sample(engine, req)

    assert engine.seeds == [10, 11, 12]
    assert [sequence["tokens"] for sequence in result["sequences"]] == [[10], [11], [12]]


@pytest.mark.asyncio
async def test_sample_scalar_prompt_logprobs_have_leading_unscored_token():
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(max_tokens=1, temperature=1, top_p=1, top_k=-1, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=True,
        topk_prompt_logprobs=0,
    )

    result = await sample(_Engine(), req)

    assert result["prompt_logprobs"] == [None, -1.1, -3.3]


@pytest.mark.asyncio
async def test_sample_forwards_token_stop_ids_and_seed():
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(
            max_tokens=4,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            stop=[151645],
            seed=123,
        ),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )
    engine = _Engine()

    await sample(engine, req)

    assert engine.sampling_params["stop_token_ids"] == [151645]
    assert engine.sampling_params["include_stop_str_in_output"] is True
    assert "stop" not in engine.sampling_params
    assert engine.sampling_params["seed"] == 123


@pytest.mark.asyncio
async def test_sample_forwards_string_stop():
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(
            max_tokens=4,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            stop=["</answer>", "<|im_end|>"],
            seed=None,
        ),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )
    engine = _Engine()

    await sample(engine, req)

    assert engine.sampling_params["stop"] == ["</answer>", "<|im_end|>"]
    assert engine.sampling_params["include_stop_str_in_output"] is True
    assert "stop_token_ids" not in engine.sampling_params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_stop_reason", "tinker_stop_reason"),
    [
        ("completed", "stop"),
        ("stop", "stop"),
        (None, "stop"),
        ("length", "length"),
        ("aborted", "length"),
    ],
)
async def test_sample_normalizes_stop_reason_for_tinker_schema(backend_stop_reason, tinker_stop_reason):
    req = SimpleNamespace(
        prompt=_Prompt(),
        sampling_params=SimpleNamespace(max_tokens=4, temperature=0.7, top_p=0.9, top_k=20, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
    )

    result = await sample(_Engine(stop_reason=backend_stop_reason), req)

    assert result["sequences"][0]["stop_reason"] == tinker_stop_reason
    if hasattr(SampleResponse, "model_validate"):
        SampleResponse.model_validate(result)


@pytest.mark.asyncio
async def test_optim_step_merges_one_to_all_worker_metrics(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _OptimStepEngine()
    result = await optim_step(engine)

    assert result == {
        "type": "optim_step",
        "metrics": {
            "grad_norm:mean": 2.0,
            "lr:mean": 1e-5,
        },
    }
    assert engine.optim_step_params is None


@pytest.mark.asyncio
async def test_optim_step_passes_adam_params_as_verl_optim_step_params(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _OptimStepEngine()

    await optim_step(
        engine,
        AdamParams(
            learning_rate=2e-5,
            beta1=0.8,
            beta2=0.9,
            eps=1e-7,
            weight_decay=0.01,
            grad_clip_norm=0.0,
        ),
    )

    assert engine.optim_step_params == {
        "lr": 2e-5,
        "betas": (0.8, 0.9),
        "eps": 1e-7,
        "weight_decay": 0.01,
    }


@pytest.mark.asyncio
async def test_optim_step_warns_for_ignored_grad_clip_override(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _OptimStepEngine()

    with pytest.warns(UserWarning, match="grad_clip_norm is accepted"):
        await optim_step(engine, AdamParams(learning_rate=2e-5, grad_clip_norm=1.0))

    assert engine.optim_step_params["lr"] == 2e-5


@pytest.mark.asyncio
async def test_load_state_zeroes_optimizer_grad_when_optimizer_false(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _LoadStateEngine()

    result = await load_state(
        engine,
        "/tmp/checkpoint-root",
        {"tinker://verl-remote-actor/state/test": "/tmp/local-checkpoint"},
        "tinker://verl-remote-actor/state/test",
        load_optimizer=False,
    )

    assert result == {"type": "load_weights", "path": "tinker://verl-remote-actor/state/test"}
    assert engine.load_calls == [("/tmp/local-checkpoint", True)]


@pytest.mark.asyncio
async def test_load_state_preserves_loaded_optimizer_by_default(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _LoadStateEngine()

    await load_state(
        engine,
        "/tmp/checkpoint-root",
        {"tinker://verl-remote-actor/state/test": "/tmp/local-checkpoint"},
        "tinker://verl-remote-actor/state/test",
    )

    assert engine.load_calls == [("/tmp/local-checkpoint", False)]


@pytest.mark.asyncio
async def test_save_state_creates_router_local_directory_for_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    engine = _SaveStateEngine()
    saved_state_paths = {}
    saved_state_metadata = {}
    checkpoint_root = tmp_path / "checkpoints"

    result = await save_state(
        engine,
        str(checkpoint_root),
        saved_state_paths,
        saved_state_metadata,
        {"epoch": 1},
        step=3,
        name="final",
    )

    uri = "tinker://verl-tinker/state/final"
    local_dir = checkpoint_root / "final"
    assert result == {"type": "save_weights", "path": uri}
    assert engine.save_calls == [(str(local_dir), 3)]
    assert saved_state_paths == {uri: str(local_dir)}
    assert saved_state_metadata == {uri: {"epoch": 1}}
    assert (local_dir / "metadata.json").read_text() == '{"epoch": 1}\n'
