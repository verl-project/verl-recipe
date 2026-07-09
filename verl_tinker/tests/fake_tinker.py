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

"""Fake Tinker backend for unit testing.

This module provides mock implementations of Tinker APIs for testing
TinkerServer without requiring a real Tinker connection.
"""

from dataclasses import dataclass, field
from typing import Any


# ==================== Sampling Response Types ====================


@dataclass
class FakeSampledSequence:
    """Fake SampledSequence matching tinker.SampledSequence format."""

    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str = "completed"


@dataclass
class FakeSampleResponse:
    """Fake SampleResponse matching tinker.SampleResponse format."""

    sequences: list[FakeSampledSequence] = field(default_factory=list)
    type: str = "sample"
    prompt_logprobs: list[float | None] = field(default_factory=list)
    topk_prompt_logprobs: Any = None


# ==================== Fake Clients ====================


class FakeSamplingClient:
    """Fake SamplingClient for testing."""

    def __init__(self, model_path: str = None, base_model: str = None):
        self.model_path = model_path
        self.base_model = base_model

    async def sample_async(
        self,
        prompt: Any,
        num_samples: int = 1,
        sampling_params: Any = None,
        include_prompt_logprobs: bool = False,
    ) -> FakeSampleResponse:
        """Fake sampling - returns deterministic tokens."""
        # Get prompt length
        if hasattr(prompt, "to_ints"):
            prompt_len = len(prompt.to_ints())
        else:
            prompt_len = 5

        # Generate fake response tokens
        response_len = 10
        if sampling_params and hasattr(sampling_params, "max_tokens"):
            response_len = min(sampling_params.max_tokens, 20)

        sequences = []
        for _ in range(num_samples):
            seq = FakeSampledSequence(
                tokens=list(range(1000, 1000 + response_len)),
                logprobs=[-0.1 * (i + 1) for i in range(response_len)],
                stop_reason="length" if response_len >= 10 else "completed",
            )
            sequences.append(seq)

        # prompt_logprobs: [None, float, float, ...]
        prompt_logprobs = [None] + [-0.5 * (i + 1) for i in range(prompt_len - 1)]

        return FakeSampleResponse(
            sequences=sequences,
            type="sample",
            prompt_logprobs=prompt_logprobs if include_prompt_logprobs else [],
            topk_prompt_logprobs=None,
        )


class FakeServiceClient:
    """Fake ServiceClient factory for testing."""

    def __init__(self):
        self.sampling_clients: list[FakeSamplingClient] = []

    def create_sampling_client(
        self,
        model_path: str = None,
        base_model: str = None,
    ) -> FakeSamplingClient:
        client = FakeSamplingClient(model_path=model_path, base_model=base_model)
        self.sampling_clients.append(client)
        return client


# ==================== Fake Tinker Module Components ====================


class FakeModelInput:
    """Fake ModelInput matching tinker.ModelInput."""

    def __init__(self, token_ids: list[int]):
        self._token_ids = token_ids

    @classmethod
    def from_ints(cls, token_ids: list[int]) -> "FakeModelInput":
        return cls(list(token_ids))

    def to_ints(self) -> list[int]:
        return self._token_ids


@dataclass
class FakeSamplingParams:
    """Fake SamplingParams matching tinker.SamplingParams."""

    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = -1
