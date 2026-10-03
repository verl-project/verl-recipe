# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from recipe.poise.features import capture_hidden, entropy_features, pool_hidden


def nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


@pytest.mark.parametrize("packed", [True, False])
def test_pooling_includes_think_end_and_excludes_masked_tokens(packed):
    ids = nested([[10, 11, 1, 2, 3, 99], [10, 4, 5]])
    responses = nested([[1, 2, 3, 99], [4, 5]])
    mask = nested([[1, 1, 1, 0], [1, 1]])
    hidden = torch.arange(12).float().reshape(2, 6, 1)
    if packed:
        hidden = torch.cat([hidden[0], hidden[1, :3]]).unsqueeze(0)
    result = pool_hidden(hidden, ids, responses, mask, packed=packed, pool_tokens=2, think_end_ids=[2])
    torch.testing.assert_close(result["poise_prompt"], torch.tensor([[0.5], [6.0]]))
    torch.testing.assert_close(result["poise_response"], torch.tensor([[2.5], [7.5]]))


def test_empty_response_and_last_multitoken_marker():
    ids = nested([[0, 1, 2, 1, 2, 3], [0, 1]])
    responses = nested([[1, 2, 1, 2, 3], [1]])
    mask = nested([[1, 1, 1, 1, 1], [0]])
    result = pool_hidden(
        torch.arange(8).float().reshape(1, 8, 1), ids, responses, mask, packed=True, pool_tokens=2, think_end_ids=[1, 2]
    )
    torch.testing.assert_close(result["poise_response"], torch.tensor([[3.5], [0.0]]))


def test_hook_cleanup_on_forward_failure():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    with (
        pytest.raises(RuntimeError, match="forward failed"),
        capture_hidden(
            model,
            layer=0,
            input_ids=nested([[0, 1]]),
            responses=nested([[1]]),
            response_mask=nested([[1]]),
            packed=True,
            pool_tokens=1,
            think_end_ids=[],
        ),
    ):
        raise RuntimeError("forward failed")
    assert not model.model.layers[0]._forward_hooks


class CharTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_entropy_spans_and_untagged_response():
    tokenizer = CharTokenizer()
    text = "<think>abc</think> xy"
    entropies = np.arange(len(text), dtype=np.float32)
    output = entropy_features(tokenizer, tokenizer.encode(text), entropies)
    np.testing.assert_allclose(output, [entropies.mean(), entropies[7:10].mean(), entropies[-2:].mean()])
    np.testing.assert_array_equal(entropy_features(tokenizer, tokenizer.encode("abc"), [1, 2, 3]), [2, 0, 0])
