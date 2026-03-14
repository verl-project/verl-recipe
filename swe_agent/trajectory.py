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
"""Trajectory structures and strict replay reconstruction for SWE-Agent.

This module groups together:
- per-turn records captured online during SWE-Agent ↔ VERL interaction
- aligned trajectory results consumed by training
- the strict post-episode reconstructor that replays full prompts to verify
  token-level consistency before a rollout is accepted for learning

Design intent:
- keep trajectory data structures and their core operations in one place
- let ``swe_agent_loop.py`` focus on interaction orchestration only
- fail closed on alignment mismatches instead of silently training on
  ambiguous trajectories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class TurnRecord:
    """Single SWE-Agent model turn captured for offline trajectory reconstruction."""

    turn_index: int
    request_id: str
    messages: list[dict[str, str]]
    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    response_logprobs: list[float]
    finish_reason: str = "stop"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[dict[str, Any]] = None


@dataclass
class AlignmentFailure:
    """Strict replay alignment failure for one episode."""

    turn_index: int
    reason: str
    details: str = ""


@dataclass
class AlignedTrajectory:
    """Reconstructed training trajectory after strict replay validation."""

    ok: bool
    initial_prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    num_turns: int
    failure: Optional[AlignmentFailure] = None


@dataclass
class TrajectoryAccumulator:
    """Mutable accumulator used while rebuilding aligned trajectories."""

    initial_prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)


RenderChatIds = Callable[[list[dict[str, str]], bool], Awaitable[list[int]]]


class TrajectoryReconstructor:
    """Strictly rebuilds aligned trajectories from recorded turns."""

    def __init__(self, render_chat_ids: RenderChatIds):
        self._render_chat_ids = render_chat_ids

    async def reconstruct(self, turn_records: list[TurnRecord]) -> AlignedTrajectory:
        """Rebuild an aligned trajectory from recorded turns via full-prompt replay."""
        if not turn_records:
            return AlignedTrajectory(
                ok=True,
                initial_prompt_ids=[],
                response_ids=[],
                response_mask=[],
                response_logprobs=[],
                num_turns=0,
            )

        acc = TrajectoryAccumulator()
        expected_prefix_ids: Optional[list[int]] = None

        for record in turn_records:
            prompt_ids = await self._render_record_prompt(record)

            failure = self._validate_record_prompt(record, prompt_ids)
            if failure is not None:
                return self._failure_result(turn_records, acc, failure)

            if expected_prefix_ids is None:
                acc.initial_prompt_ids = list(prompt_ids)
                expected_prefix_ids = list(prompt_ids)
            else:
                failure = self._append_prompt_delta(acc, expected_prefix_ids, prompt_ids, record)
                if failure is not None:
                    return self._failure_result(turn_records, acc, failure)

            failure = self._validate_response_lengths(record)
            if failure is not None:
                return self._failure_result(turn_records, acc, failure)

            after_assistant_ids, assistant_span_ids, failure = await self._replay_assistant(record, prompt_ids)
            if failure is not None:
                return self._failure_result(turn_records, acc, failure)

            failure, trailing_template_ids = self._validate_assistant_span(record, assistant_span_ids)
            if failure is not None:
                return self._failure_result(turn_records, acc, failure)

            self._append_model_response(acc, record, trailing_template_ids)
            expected_prefix_ids = after_assistant_ids

        return AlignedTrajectory(
            ok=True,
            initial_prompt_ids=acc.initial_prompt_ids,
            response_ids=acc.response_ids,
            response_mask=acc.response_mask,
            response_logprobs=acc.response_logprobs,
            num_turns=len(turn_records),
        )

    async def _render_record_prompt(self, record: TurnRecord) -> list[int]:
        return await self._render_chat_ids(record.messages, add_generation_prompt=True)

    @staticmethod
    def _validate_record_prompt(record: TurnRecord, prompt_ids: list[int]) -> Optional[AlignmentFailure]:
        if prompt_ids == record.prompt_ids:
            return None
        return AlignmentFailure(
            turn_index=record.turn_index,
            reason="prompt_mismatch",
            details=f"rendered={len(prompt_ids)} recorded={len(record.prompt_ids)}",
        )

    @staticmethod
    def _append_prompt_delta(
        acc: TrajectoryAccumulator,
        expected_prefix_ids: list[int],
        prompt_ids: list[int],
        record: TurnRecord,
    ) -> Optional[AlignmentFailure]:
        if prompt_ids[: len(expected_prefix_ids)] != expected_prefix_ids:
            return AlignmentFailure(
                turn_index=record.turn_index,
                reason="prompt_prefix_mismatch",
                details=f"expected_prefix_len={len(expected_prefix_ids)} prompt_len={len(prompt_ids)}",
            )

        delta_ids = prompt_ids[len(expected_prefix_ids) :]
        acc.response_ids.extend(delta_ids)
        acc.response_mask.extend([0] * len(delta_ids))
        acc.response_logprobs.extend([0.0] * len(delta_ids))
        return None

    @staticmethod
    def _validate_response_lengths(record: TurnRecord) -> Optional[AlignmentFailure]:
        if len(record.response_ids) == len(record.response_logprobs):
            return None
        return AlignmentFailure(
            turn_index=record.turn_index,
            reason="logprob_length_mismatch",
            details=f"ids={len(record.response_ids)} logprobs={len(record.response_logprobs)}",
        )

    async def _replay_assistant(
        self,
        record: TurnRecord,
        prompt_ids: list[int],
    ) -> tuple[list[int], list[int], Optional[AlignmentFailure]]:
        assistant_messages = [
            *record.messages,
            {"role": "assistant", "content": record.response_text},
        ]
        after_assistant_ids = await self._render_chat_ids(assistant_messages, add_generation_prompt=False)

        if after_assistant_ids[: len(prompt_ids)] != prompt_ids:
            return (
                [],
                [],
                AlignmentFailure(
                    turn_index=record.turn_index,
                    reason="assistant_replay_prefix_mismatch",
                    details=f"prompt_len={len(prompt_ids)} replay_len={len(after_assistant_ids)}",
                ),
            )

        assistant_span_ids = after_assistant_ids[len(prompt_ids) :]
        return after_assistant_ids, assistant_span_ids, None

    @staticmethod
    def _validate_assistant_span(
        record: TurnRecord,
        assistant_span_ids: list[int],
    ) -> tuple[Optional[AlignmentFailure], list[int]]:
        """Validate and return (failure_or_none, trailing_template_tokens).

        Chat templates may append formatting tokens after the assistant
        content (e.g. ``\\n`` after ``<|im_end|>``) that vLLM never
        includes in ``response_ids``.  When ``response_ids`` is a strict
        prefix of the replayed span and the surplus is at most a few
        tokens, we accept the match and return the surplus so the caller
        can add it to the accumulator with mask=0.
        """
        if assistant_span_ids == record.response_ids:
            return None, []
        gen = record.response_ids
        max_trailing = 3
        if (
            len(assistant_span_ids) > len(gen)
            and (len(assistant_span_ids) - len(gen)) <= max_trailing
            and assistant_span_ids[: len(gen)] == gen
        ):
            trailing = assistant_span_ids[len(gen) :]
            return None, trailing
        return AlignmentFailure(
            turn_index=record.turn_index,
            reason="assistant_span_mismatch",
            details=f"replayed={len(assistant_span_ids)} generated={len(record.response_ids)}",
        ), []

    @staticmethod
    def _append_model_response(
        acc: TrajectoryAccumulator,
        record: TurnRecord,
        trailing_template_ids: list[int],
    ) -> None:
        acc.response_ids.extend(record.response_ids)
        acc.response_mask.extend([1] * len(record.response_ids))
        acc.response_logprobs.extend(record.response_logprobs)
        if trailing_template_ids:
            acc.response_ids.extend(trailing_template_ids)
            acc.response_mask.extend([0] * len(trailing_template_ids))
            acc.response_logprobs.extend([0.0] * len(trailing_template_ids))

    @staticmethod
    def _failure_result(
        turn_records: list[TurnRecord],
        acc: TrajectoryAccumulator,
        failure: AlignmentFailure,
    ) -> AlignedTrajectory:
        return AlignedTrajectory(
            ok=False,
            initial_prompt_ids=acc.initial_prompt_ids,
            response_ids=[],
            response_mask=[],
            response_logprobs=[],
            num_turns=len(turn_records),
            failure=failure,
        )
