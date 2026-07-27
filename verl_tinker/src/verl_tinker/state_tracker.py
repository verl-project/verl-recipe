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

"""In-memory actor, rollout, checkpoint, and sampler version tracking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class StateTrackerError(ValueError):
    """Base class for invalid or unavailable tracked state."""


class UnknownSamplerError(StateTrackerError):
    """Raised when a sampling session is not registered."""


class UnknownSamplerPathError(StateTrackerError):
    """Raised when a sampler checkpoint path is not registered."""


class StaleSamplerError(StateTrackerError):
    """Raised when no configured resource contains the requested weights."""


class SamplingResource(str, Enum):
    """Physical resource selected when a sampling request is executed."""

    TEACHER = "teacher"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    REFERENCE = "reference"


@dataclass(frozen=True)
class SamplerBinding:
    """The immutable weight identity selected when a session is created."""

    sampler_id: str | None
    base_model: str
    model_path: str | None = None
    teacher_model_path: str | None = None
    actor_id: int | None = None

    @property
    def is_teacher(self) -> bool:
        return self.teacher_model_path is not None


class ModelStateTracker:
    """Tracks which actor weights are resident in the single mutable rollout.

    IDs describe weight identity only. Optimizer and gradient state are deliberately
    outside this tracker; callers may therefore choose to skip a state load whenever
    the saved actor ID already matches the current actor ID.
    """

    def __init__(
        self,
        *,
        actor_model_identifiers: Iterable[str],
        teacher_models: Iterable[tuple[str, str]] = (),
    ):
        actor_identifiers = tuple(dict.fromkeys(str(value) for value in actor_model_identifiers if value))
        if not actor_identifiers:
            raise StateTrackerError("At least one actor model identifier is required")
        self.actor_base_model = actor_identifiers[0]
        self.actor_model_identifiers = frozenset(actor_identifiers)
        self._teacher_model_paths: dict[str, str] = {}
        self.configure_teacher_models(teacher_models)
        self.actor_id = 0
        self.rollout_id: int | None = 0
        self._next_actor_id = 1
        self._rollout_enabled = True
        self._reference_enabled = False

        self._state_path_to_actor_id: dict[str, int] = {}
        self._sampler_path_to_actor_id: dict[str, int] = {}
        self._samplers: dict[str, SamplerBinding] = {}

    def configure_resources(self, *, rollout_enabled: bool, reference_enabled: bool) -> None:
        """Record which late-bound inference resources exist on this server."""
        self._rollout_enabled = bool(rollout_enabled)
        self._reference_enabled = bool(reference_enabled)
        if not self._rollout_enabled:
            self.rollout_id = None

    def configure_teacher_models(self, teacher_models: Iterable[tuple[str, str]]) -> None:
        """Register public teacher identifiers against their loaded model paths."""
        for identifier, model_path in teacher_models:
            identifier = str(identifier)
            model_path = str(model_path)
            previous = self._teacher_model_paths.get(identifier)
            if previous is not None and previous != model_path:
                raise StateTrackerError(
                    f"Teacher identifier {identifier!r} is ambiguous between {previous!r} and {model_path!r}"
                )
            self._teacher_model_paths[identifier] = model_path

    @staticmethod
    def sampler_id(session_id: str, sampling_session_seq_id: int) -> str:
        """Build the retry-stable ID shape used by the Tinker SDK."""
        return f"{session_id}:sample:{int(sampling_session_seq_id)}"

    def _allocate_actor_id(self) -> int:
        actor_id = self._next_actor_id
        self._next_actor_id += 1
        return actor_id

    def actor_updated(self) -> int:
        """Record a successful or potentially partial actor-weight mutation."""
        self.actor_id = self._allocate_actor_id()
        return self.actor_id

    def rollout_synchronized(self) -> int:
        if not self._rollout_enabled:
            raise StateTrackerError("Cannot synchronize rollout weights when rollout is disabled")
        self.rollout_id = self.actor_id
        return self.rollout_id

    def rollout_synchronization_failed(self) -> None:
        self.rollout_id = None

    def state_saved(self, path: str) -> int:
        self._state_path_to_actor_id[path] = self.actor_id
        return self.actor_id

    def is_state_path(self, path: str) -> bool:
        return path in self._state_path_to_actor_id

    def should_skip_state_load(self, path: str) -> bool:
        saved_actor_id = self._state_path_to_actor_id.get(path)
        return saved_actor_id is not None and saved_actor_id == self.actor_id

    def state_loaded(self, path: str) -> int:
        saved_actor_id = self._state_path_to_actor_id.get(path)
        if saved_actor_id is None:
            saved_actor_id = self._allocate_actor_id()
            self._state_path_to_actor_id[path] = saved_actor_id
        self.actor_id = saved_actor_id
        return self.actor_id

    def state_load_failed(self) -> int:
        """Give potentially partially loaded actor weights an unmatchable identity."""
        return self.actor_updated()

    def sampler_path_saved(self, path: str) -> int:
        if self.rollout_id is None:
            raise StateTrackerError("Cannot register sampler path while rollout state is unknown")
        self._sampler_path_to_actor_id[path] = self.rollout_id
        return self.rollout_id

    def actor_id_for_sampler_path(self, path: str) -> int:
        try:
            return self._sampler_path_to_actor_id[path]
        except KeyError as exc:
            raise UnknownSamplerPathError(f"Unknown sampler checkpoint path: {path}") from exc

    def register_actor_sampler(
        self,
        sampler_id: str,
        *,
        base_model: str,
        actor_id: int,
        model_path: str | None = None,
    ) -> SamplerBinding:
        binding = SamplerBinding(
            sampler_id=sampler_id,
            base_model=base_model,
            model_path=model_path,
            actor_id=int(actor_id),
        )
        return self._register_sampler(binding)

    def register_teacher_sampler(
        self,
        sampler_id: str,
        *,
        teacher_model_path: str,
        base_model: str,
    ) -> SamplerBinding:
        binding = SamplerBinding(
            sampler_id=sampler_id,
            teacher_model_path=teacher_model_path,
            base_model=base_model,
            model_path=teacher_model_path,
        )
        return self._register_sampler(binding)

    def resolve_sampler_target(
        self,
        *,
        base_model: str | None,
        model_path: str | None,
        sampler_id: str | None = None,
    ) -> SamplerBinding:
        """Resolve teacher intent or bind a training session to an actor ID."""
        target = model_path if model_path is not None else base_model
        teacher_model_path = self._teacher_model_paths.get(target) if target is not None else None
        if teacher_model_path is not None:
            binding = SamplerBinding(
                sampler_id=sampler_id,
                teacher_model_path=teacher_model_path,
                base_model=base_model or target,
                model_path=teacher_model_path,
            )
        elif model_path is not None:
            if self.is_state_path(model_path):
                raise StateTrackerError(f"Training-state checkpoint cannot be sampled: {model_path}")
            if base_model is not None and base_model not in self.actor_model_identifiers:
                raise StateTrackerError(
                    f"Sampler checkpoint base_model must be one of "
                    f"{sorted(self.actor_model_identifiers)!r}, got {base_model!r}"
                )
            actor_id = 0 if model_path in self.actor_model_identifiers else self.actor_id_for_sampler_path(model_path)
            binding = SamplerBinding(
                sampler_id=sampler_id,
                base_model=base_model or self.actor_base_model,
                model_path=model_path,
                actor_id=actor_id,
            )
        elif base_model in self.actor_model_identifiers:
            binding = SamplerBinding(
                sampler_id=sampler_id,
                base_model=self.actor_base_model,
                actor_id=0,
            )
        else:
            raise StateTrackerError(f"Unknown sampling model: base_model={base_model!r}, model_path={model_path!r}")

        if sampler_id is not None:
            return self._register_sampler(binding)
        return binding

    def resolve_sampling_request(
        self,
        *,
        sampling_session_id: str | None,
        base_model: str | None,
        model_path: str | None,
    ) -> SamplerBinding:
        """Resolve an existing session or a direct raw sampling target."""
        if sampling_session_id is not None:
            return self.get_sampler(sampling_session_id)
        return self.resolve_sampler_target(base_model=base_model, model_path=model_path)

    def _register_sampler(self, binding: SamplerBinding) -> SamplerBinding:
        if binding.sampler_id is None:
            raise StateTrackerError("Cannot register a sampler without a sampler ID")
        previous = self._samplers.get(binding.sampler_id)
        if previous is not None and previous != binding:
            raise StateTrackerError(f"Sampler ID {binding.sampler_id!r} was reused for a different target")
        self._samplers[binding.sampler_id] = binding
        return binding

    def get_sampler(self, sampler_id: str) -> SamplerBinding:
        try:
            return self._samplers[sampler_id]
        except KeyError as exc:
            raise UnknownSamplerError(f"Unknown sampling session: {sampler_id}") from exc

    def resolve_resource(self, binding: SamplerBinding, *, scalar_prompt_logprobs: bool) -> SamplingResource:
        """Late-bind a session's weights to a resource that can fulfill the request."""
        if binding.is_teacher:
            return SamplingResource.TEACHER

        if binding.actor_id is None:
            raise StateTrackerError(f"Training sampling session {binding.sampler_id!r} has no actor ID")
        if self._rollout_enabled and self.rollout_id == binding.actor_id:
            return SamplingResource.ROLLOUT
        if scalar_prompt_logprobs and self.actor_id == binding.actor_id:
            return SamplingResource.ACTOR
        if scalar_prompt_logprobs and binding.actor_id == 0 and self._reference_enabled:
            return SamplingResource.REFERENCE

        operation = "scalar prompt logprobs" if scalar_prompt_logprobs else "autoregressive sampling"
        raise StaleSamplerError(
            f"Sampling session {binding.sampler_id!r} is bound to actor ID {binding.actor_id}, "
            f"but {operation} cannot be served by the configured resources "
            f"(actor_id={self.actor_id}, rollout_id={self.rollout_id}, "
            f"rollout_enabled={self._rollout_enabled}, reference_enabled={self._reference_enabled})"
        )

    def sampler_ids(self) -> list[str]:
        return list(self._samplers)

    def valid_sampler_ids(self) -> list[str]:
        return [
            sampler_id
            for sampler_id, binding in self._samplers.items()
            if binding.is_teacher
            or (
                binding.actor_id is not None
                and (
                    (self._rollout_enabled and self.rollout_id == binding.actor_id)
                    or self.actor_id == binding.actor_id
                    or (binding.actor_id == 0 and self._reference_enabled)
                )
            )
        ]
