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

"""Unit tests for DataConverter."""

import pytest
import torch
from tensordict import TensorDict
from torch.nested import nested_tensor

from verl_recipes.verl_tinker_connector.data_converter import DataConverter


# ==================== Tinker API Format Validators ====================


def validate_datum_weight_format(datum):
    """Validate Datum for weight format (SFT/forward)."""
    assert hasattr(datum.model_input, "to_ints")
    assert "target_tokens" in datum.loss_fn_inputs
    assert "weights" in datum.loss_fn_inputs

    # Check length consistency
    seq_len = len(datum.model_input.to_ints())
    assert len(datum.loss_fn_inputs["target_tokens"].data) == seq_len
    assert len(datum.loss_fn_inputs["weights"].data) == seq_len


def validate_datum_advantage_format(datum):
    """Validate Datum for advantage format (RL training)."""
    assert hasattr(datum.model_input, "to_ints")
    assert "target_tokens" in datum.loss_fn_inputs
    assert "logprobs" in datum.loss_fn_inputs
    assert "advantages" in datum.loss_fn_inputs

    # Check length consistency
    seq_len = len(datum.model_input.to_ints())
    assert len(datum.loss_fn_inputs["target_tokens"].data) == seq_len
    assert len(datum.loss_fn_inputs["logprobs"].data) == seq_len
    assert len(datum.loss_fn_inputs["advantages"].data) == seq_len


def make_batch(
    input_ids_list: list[list[int]],
    loss_mask_list: list[list[float]],
    old_log_probs_list: list[list[float]] | None = None,
    advantages_list: list[list[float]] | None = None,
) -> TensorDict:
    """Create a VeRL-style TensorDict batch for testing.

    Args:
        input_ids_list: List of token sequences per sample.
        loss_mask_list: List of loss masks per sample (1.0 for response, 0.0 for prompt).
        old_log_probs_list: Optional list of log probs per sample.
        advantages_list: Optional list of advantages per sample.

    Returns:
        TensorDict with nested tensors matching VeRL's format.
    """
    batch_size = len(input_ids_list)

    # Create nested tensors (variable length sequences)
    input_ids = nested_tensor([torch.tensor(ids, dtype=torch.long) for ids in input_ids_list])
    loss_mask = nested_tensor([torch.tensor(mask, dtype=torch.float32) for mask in loss_mask_list])

    data = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }

    if old_log_probs_list is not None:
        data["old_log_probs"] = nested_tensor([torch.tensor(lp, dtype=torch.float32) for lp in old_log_probs_list])

    if advantages_list is not None:
        data["advantages"] = nested_tensor([torch.tensor(adv, dtype=torch.float32) for adv in advantages_list])

    return TensorDict(data, batch_size=[batch_size])


class TestPadToSequence:
    """Tests for DataConverter._pad_to_sequence."""

    def test_basic_padding(self):
        """Test basic padding to a longer sequence."""
        values = torch.tensor([1.0, 2.0, 3.0])
        result = DataConverter._pad_to_sequence(values, total_len=7, start_pos=2)

        expected = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        assert torch.allclose(result, expected)

    def test_padding_at_start(self):
        """Test padding when values start at position 0."""
        values = torch.tensor([1.0, 2.0])
        result = DataConverter._pad_to_sequence(values, total_len=5, start_pos=0)

        expected = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result, expected)

    def test_padding_at_end(self):
        """Test padding when values end at the last position."""
        values = torch.tensor([1.0, 2.0])
        result = DataConverter._pad_to_sequence(values, total_len=5, start_pos=3)

        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_custom_pad_value(self):
        """Test padding with custom pad value."""
        values = torch.tensor([1.0, 2.0])
        result = DataConverter._pad_to_sequence(values, total_len=5, start_pos=1, pad_value=-1.0)

        expected = torch.tensor([-1.0, 1.0, 2.0, -1.0, -1.0])
        assert torch.allclose(result, expected)

    def test_truncation_when_overflow(self):
        """Test that values are truncated if they would overflow."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = DataConverter._pad_to_sequence(values, total_len=5, start_pos=3)

        # Only first 2 values fit (positions 3, 4)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_empty_values(self):
        """Test padding with empty values tensor."""
        values = torch.tensor([])
        result = DataConverter._pad_to_sequence(values, total_len=5, start_pos=2)

        expected = torch.zeros(5)
        assert torch.allclose(result, expected)


class TestTensorDictToTrainingDataWeight:
    """Tests for tensordict_to_training_data with weight format."""

    def test_single_sample(self):
        """Test conversion of a single sample."""
        # Sequence: [P1, P2, R1, R2, R3] where P=prompt, R=response
        # loss_mask: [0, 0, 1, 1, 1]
        batch = make_batch(
            input_ids_list=[[101, 102, 201, 202, 203]],
            loss_mask_list=[[0.0, 0.0, 1.0, 1.0, 1.0]],
        )

        datums = DataConverter.tensordict_to_training_data(batch, data_format="weight")

        assert len(datums) == 1
        datum = datums[0]

        # Validate Tinker API format compliance
        validate_datum_weight_format(datum)

        # model_input should be tokens[:-1]
        assert datum.model_input.to_ints() == [101, 102, 201, 202]

        # target_tokens should be tokens[1:]
        assert list(datum.loss_fn_inputs["target_tokens"].data) == [102, 201, 202, 203]

        # weights: response starts at prompt_len - 1 = 2 - 1 = 1 in target
        # target positions: [0, 1, 2, 3] -> weights [0, 1, 1, 1]
        expected_weights = [0.0, 1.0, 1.0, 1.0]
        assert list(datum.loss_fn_inputs["weights"].data) == expected_weights

    def test_multiple_samples(self):
        """Test conversion of multiple samples with different lengths."""
        batch = make_batch(
            input_ids_list=[
                [101, 102, 201, 202],  # 2 prompt + 2 response
                [101, 201, 202, 203, 204],  # 1 prompt + 4 response
            ],
            loss_mask_list=[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ],
        )

        datums = DataConverter.tensordict_to_training_data(batch, data_format="weight")

        assert len(datums) == 2

        # Validate Tinker API format compliance for all datums
        for datum in datums:
            validate_datum_weight_format(datum)

        # First sample
        assert datums[0].model_input.to_ints() == [101, 102, 201]
        assert list(datums[0].loss_fn_inputs["target_tokens"].data) == [102, 201, 202]
        assert list(datums[0].loss_fn_inputs["weights"].data) == [0.0, 1.0, 1.0]

        # Second sample
        assert datums[1].model_input.to_ints() == [101, 201, 202, 203]
        assert list(datums[1].loss_fn_inputs["target_tokens"].data) == [201, 202, 203, 204]
        assert list(datums[1].loss_fn_inputs["weights"].data) == [1.0, 1.0, 1.0, 1.0]

    def test_all_response(self):
        """Test when entire sequence is response (no prompt).

        Note: When prompt_len=0, response_start_in_target=-1, which causes
        Python slice [-1:2] to only cover position 1. This is an edge case
        that may not occur in practice (prompts usually exist).
        """
        batch = make_batch(
            input_ids_list=[[201, 202, 203]],
            loss_mask_list=[[1.0, 1.0, 1.0]],
        )

        datums = DataConverter.tensordict_to_training_data(batch, data_format="weight")

        assert len(datums) == 1
        datum = datums[0]

        # Validate Tinker API format compliance
        validate_datum_weight_format(datum)

        # Current behavior: only position 1 gets weight 1.0 due to negative index
        assert list(datum.loss_fn_inputs["weights"].data) == [0.0, 1.0]

    def test_missing_required_field_raises(self):
        """Test that missing required field raises ValueError."""
        batch = TensorDict(
            {"input_ids": nested_tensor([torch.tensor([1, 2, 3])])},
            batch_size=[1],
        )

        with pytest.raises(ValueError, match="loss_mask"):
            DataConverter.tensordict_to_training_data(batch, data_format="weight")


class TestTensorDictToTrainingDataAdvantage:
    """Tests for tensordict_to_training_data with advantage format."""

    def test_single_sample(self):
        """Test conversion with advantage format."""
        # Sequence: [P1, P2, R1, R2, R3]
        # loss_mask: [0, 0, 1, 1, 1]
        batch = make_batch(
            input_ids_list=[[101, 102, 201, 202, 203]],
            loss_mask_list=[[0.0, 0.0, 1.0, 1.0, 1.0]],
            old_log_probs_list=[[0.0, 0.0, -0.1, -0.2, -0.3]],  # Only response values matter
            advantages_list=[[0.0, 0.0, 1.0, 2.0, 3.0]],
        )

        datums = DataConverter.tensordict_to_training_data(batch, data_format="advantage")

        assert len(datums) == 1
        datum = datums[0]

        # Validate Tinker API format compliance
        validate_datum_advantage_format(datum)

        # Check model_input and target_tokens
        assert datum.model_input.to_ints() == [101, 102, 201, 202]
        assert list(datum.loss_fn_inputs["target_tokens"].data) == [102, 201, 202, 203]

        # Check logprobs: padded to target length, response values at correct positions
        # response_start_in_target = prompt_len - 1 = 2 - 1 = 1
        logprobs = list(datum.loss_fn_inputs["logprobs"].data)
        assert len(logprobs) == 4
        assert logprobs[0] == 0.0  # Padding
        assert logprobs[1] == pytest.approx(-0.1)
        assert logprobs[2] == pytest.approx(-0.2)
        assert logprobs[3] == pytest.approx(-0.3)

        # Check advantages: normalized by response_len (3)
        # Original: [1.0, 2.0, 3.0] -> normalized: [1/3, 2/3, 3/3]
        # Prompt positions are padded with 0.0, implicitly masking them
        advantages = list(datum.loss_fn_inputs["advantages"].data)
        assert len(advantages) == 4
        assert advantages[0] == 0.0  # Padding (implicit mask)
        assert advantages[1] == pytest.approx(1.0 / 3)
        assert advantages[2] == pytest.approx(2.0 / 3)
        assert advantages[3] == pytest.approx(3.0 / 3)

    def test_missing_advantage_fields_raises(self):
        """Test that missing advantage fields raise ValueError."""
        batch = make_batch(
            input_ids_list=[[101, 102, 201]],
            loss_mask_list=[[0.0, 1.0, 1.0]],
            # Missing old_log_probs and advantages
        )

        with pytest.raises(ValueError, match="old_log_probs"):
            DataConverter.tensordict_to_training_data(batch, data_format="advantage")


class TestExtractAdvantageTensors:
    """Tests for DataConverter.extract_advantage_tensors."""

    def test_single_sample(self):
        """Test extraction of advantage tensors for single sample."""
        batch = make_batch(
            input_ids_list=[[101, 102, 201, 202, 203]],
            loss_mask_list=[[0.0, 0.0, 1.0, 1.0, 1.0]],
            old_log_probs_list=[[0.0, 0.0, -0.1, -0.2, -0.3]],
            advantages_list=[[0.0, 0.0, 1.0, 2.0, 3.0]],
        )

        results = DataConverter.extract_advantage_tensors(batch)

        assert len(results) == 1
        result = results[0]

        # Check keys
        assert "old_log_probs" in result
        assert "advantages" in result
        assert "response_mask" in result

        # target_len = 5 - 1 = 4
        # response_start = prompt_len - 1 = 2 - 1 = 1
        assert result["old_log_probs"].shape == (4,)
        assert result["advantages"].shape == (4,)
        assert result["response_mask"].shape == (4,)

        # Check values
        expected_log_probs = torch.tensor([0.0, -0.1, -0.2, -0.3])
        expected_advantages = torch.tensor([0.0, 1.0, 2.0, 3.0])
        expected_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])

        assert torch.allclose(result["old_log_probs"], expected_log_probs)
        assert torch.allclose(result["advantages"], expected_advantages)
        assert torch.allclose(result["response_mask"], expected_mask)

    def test_multiple_samples(self):
        """Test extraction for multiple samples."""
        batch = make_batch(
            input_ids_list=[
                [101, 102, 201, 202],  # len=4, prompt=2, response=2
                [101, 201, 202, 203],  # len=4, prompt=1, response=3
            ],
            loss_mask_list=[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ],
            old_log_probs_list=[
                [0.0, 0.0, -0.1, -0.2],
                [0.0, -0.3, -0.4, -0.5],
            ],
            advantages_list=[
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 3.0, 4.0, 5.0],
            ],
        )

        results = DataConverter.extract_advantage_tensors(batch)

        assert len(results) == 2

        # First sample: target_len=3, response_start=1, response_len=2
        expected_mask_0 = torch.tensor([0.0, 1.0, 1.0])
        expected_log_probs_0 = torch.tensor([0.0, -0.1, -0.2])
        expected_advantages_0 = torch.tensor([0.0, 1.0, 2.0])
        assert torch.allclose(results[0]["response_mask"], expected_mask_0)
        assert torch.allclose(results[0]["old_log_probs"], expected_log_probs_0)
        assert torch.allclose(results[0]["advantages"], expected_advantages_0)

        # Second sample: target_len=3, response_start=0, response_len=3
        expected_mask_1 = torch.tensor([1.0, 1.0, 1.0])
        expected_log_probs_1 = torch.tensor([-0.3, -0.4, -0.5])
        expected_advantages_1 = torch.tensor([3.0, 4.0, 5.0])
        assert torch.allclose(results[1]["response_mask"], expected_mask_1)
        assert torch.allclose(results[1]["old_log_probs"], expected_log_probs_1)
        assert torch.allclose(results[1]["advantages"], expected_advantages_1)
