"""Canonical trajectory exchange from source rollout to learner view."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from tensordict.tensorclass import NonTensorData


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("message role must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text in the v1 recipe")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Trajectory:
    """Tokenizer-neutral record exchanged between policy runtimes.

    Ordinary responses exclude terminal EOS from the probability contract.
    An EOS-only response retains that action and uses an empty response string.
    """

    trajectory_id: str
    prompt_id: str
    messages: tuple[Message, ...]
    source_agent_id: str
    source_policy_version: int
    response_text: str
    reward: float
    source_logp_sum: float
    source_token_count: int
    finish_reason: str = "unknown"
    terminal_only: bool = False
    source_prompt_ids: tuple[int, ...] | None = field(default=None, compare=False, hash=False, repr=False)
    source_response_ids: tuple[int, ...] | None = field(default=None, compare=False, hash=False, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("trajectory_id", "prompt_id", "source_agent_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if not self.messages:
            raise ValueError("messages must contain at least one prompt message")
        if self.source_policy_version < 0:
            raise ValueError("source_policy_version must be non-negative")
        if self.source_token_count <= 0:
            raise ValueError("source_token_count must be positive")
        if not math.isfinite(self.reward) or not math.isfinite(self.source_logp_sum):
            raise ValueError("reward and source_logp_sum must be finite")
        if self.terminal_only and self.response_text:
            raise ValueError("a terminal-only trajectory must have empty response text")
        if self.terminal_only and self.source_token_count != 1:
            raise ValueError("a terminal-only trajectory must contain exactly one source action")
        if (self.source_prompt_ids is None) != (self.source_response_ids is None):
            raise ValueError("source prompt and response ids must either both be present or both be absent")
        if self.source_prompt_ids is not None:
            if not self.source_prompt_ids or not self.source_response_ids:
                raise ValueError("stored source token ids must be non-empty")
            if len(self.source_response_ids) != self.source_token_count:
                raise ValueError("source_token_count must match stored source_response_ids")

    @property
    def source_mean_logp(self) -> float:
        return self.source_logp_sum / self.source_token_count


@dataclass(frozen=True)
class TrainingView:
    """One canonical trajectory encoded in a particular learner's token space."""

    trajectory_id: str
    target_agent_id: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    truncated: bool = False
    prompt_length_before_truncation: int | None = None
    response_length_before_truncation: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("a training view must have at least one prompt token")
        if not self.response_ids:
            raise ValueError("a training view must have at least one response token")
        lengths = (
            ("prompt", self.prompt_length_before_truncation, len(self.prompt_ids)),
            ("response", self.response_length_before_truncation, len(self.response_ids)),
        )
        for name, before_truncation, retained in lengths:
            if before_truncation is not None and before_truncation < retained:
                raise ValueError(
                    f"{name}_length_before_truncation={before_truncation} is smaller than retained length {retained}"
                )

    @property
    def prompt_was_truncated(self) -> bool:
        length = self.prompt_length_before_truncation
        return length is not None and length > len(self.prompt_ids)

    @property
    def response_was_truncated(self) -> bool:
        length = self.response_length_before_truncation
        return length is not None and length > len(self.response_ids)

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.response_ids

    @property
    def response_mask(self) -> tuple[bool, ...]:
        return (True,) * len(self.response_ids)


class TrajectoryTooLongError(ValueError):
    pass


def _extract_input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("tokenizer unexpectedly returned a batched result")
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _target_eos_token_id(tokenizer: Any) -> int:
    """Resolve one concrete EOS action in the target tokenizer namespace."""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, str | bytes):
        if not eos_token_id:
            raise ValueError("target tokenizer has an empty eos_token_id sequence")
        eos_token_id = eos_token_id[0]
    if eos_token_id is None:
        raise ValueError("target tokenizer has no eos_token_id for a terminal-only trajectory")
    return int(eos_token_id)


def build_training_view(
    trajectory: Trajectory,
    *,
    target_agent_id: str,
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
    overflow: Literal["error", "truncate"] = "error",
) -> TrainingView:
    """Retokenize a trajectory with the learner's tokenizer and chat template."""

    if max_prompt_length <= 0 or max_response_length <= 0:
        raise ValueError("maximum lengths must be positive")
    if overflow not in ("error", "truncate"):
        raise ValueError(f"unsupported overflow policy {overflow!r}")

    if target_agent_id == trajectory.source_agent_id and trajectory.source_prompt_ids is not None:
        # Keep rollout token ids for on-policy trajectories.
        prompt_ids = list(trajectory.source_prompt_ids)
        response_ids = list(trajectory.source_response_ids)
    else:
        messages: Sequence[dict[str, str]] = [message.as_dict() for message in trajectory.messages]
        prompt_ids = _extract_input_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **(apply_chat_template_kwargs or {}),
            )
        )
        if trajectory.terminal_only:
            response_ids = [_target_eos_token_id(tokenizer)]
        else:
            response_ids = _extract_input_ids(tokenizer(trajectory.response_text, add_special_tokens=False))

    prompt_length_before_truncation = len(prompt_ids)
    response_length_before_truncation = len(response_ids)
    prompt_overflow = max(0, len(prompt_ids) - max_prompt_length)
    response_overflow = max(0, len(response_ids) - max_response_length)
    truncated = prompt_overflow > 0 or response_overflow > 0
    if truncated and overflow == "error":
        raise TrajectoryTooLongError(
            f"trajectory {trajectory.trajectory_id!r} exceeds target limits: "
            f"prompt={len(prompt_ids)}/{max_prompt_length}, response={len(response_ids)}/{max_response_length}"
        )
    if prompt_overflow:
        prompt_ids = prompt_ids[prompt_overflow:]
    if response_overflow:
        response_ids = response_ids[:max_response_length]
    if not response_ids:
        raise ValueError(f"trajectory {trajectory.trajectory_id!r} encodes to an empty target response")

    return TrainingView(
        trajectory_id=trajectory.trajectory_id,
        target_agent_id=target_agent_id,
        prompt_ids=tuple(prompt_ids),
        response_ids=tuple(response_ids),
        truncated=truncated,
        prompt_length_before_truncation=prompt_length_before_truncation,
        response_length_before_truncation=response_length_before_truncation,
    )


@dataclass(frozen=True)
class RolloutRecord:
    """A source-tokenized rollout before actor log-probabilities are attached."""

    trajectory_id: str
    prompt_id: str
    messages: tuple[Message, ...]
    source_agent_id: str
    source_policy_version: int
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_text: str
    reward: float
    finish_reason: str = "unknown"
    terminal_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def training_view(self) -> TrainingView:
        return TrainingView(
            trajectory_id=self.trajectory_id,
            target_agent_id=self.source_agent_id,
            prompt_ids=self.prompt_ids,
            response_ids=self.response_ids,
        )

    def with_source_log_probs(self, log_probs: torch.Tensor) -> Trajectory:
        if log_probs.ndim != 1 or log_probs.numel() != len(self.response_ids):
            raise ValueError(
                f"trajectory {self.trajectory_id!r} expected {len(self.response_ids)} source log-probs, "
                f"got shape {tuple(log_probs.shape)}"
            )
        if not bool(torch.isfinite(log_probs).all()):
            raise ValueError(f"trajectory {self.trajectory_id!r} has non-finite source log-probabilities")
        return Trajectory(
            trajectory_id=self.trajectory_id,
            prompt_id=self.prompt_id,
            messages=self.messages,
            source_agent_id=self.source_agent_id,
            source_policy_version=self.source_policy_version,
            response_text=self.response_text,
            reward=self.reward,
            source_logp_sum=float(log_probs.float().sum().item()),
            source_token_count=len(self.response_ids),
            finish_reason=self.finish_reason,
            terminal_only=self.terminal_only,
            source_prompt_ids=self.prompt_ids,
            source_response_ids=self.response_ids,
            metadata=self.metadata,
        )


def _unwrap(value: Any) -> Any:
    return value.data if isinstance(value, NonTensorData) else value


def _rows(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return list(value.unbind())
    return [_unwrap(item) for item in value]


def _messages(raw_prompt: Any) -> tuple[Message, ...]:
    raw_prompt = _unwrap(raw_prompt)
    messages = []
    for raw_message in raw_prompt:
        raw_message = _unwrap(raw_message)
        if not isinstance(raw_message, Mapping):
            raise TypeError("each raw_prompt message must be a mapping")
        content = _unwrap(raw_message.get("content"))
        if not isinstance(content, str):
            raise TypeError("HACPO only supports text message content")
        messages.append(Message(role=str(_unwrap(raw_message.get("role", ""))), content=content))
    if not messages:
        raise ValueError("raw_prompt must contain at least one message")
    return tuple(messages)


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return {int(token_id) for token_id in value}
    return {int(value)}


def _terminal_eos_ids(tokenizer: Any, terminal_token_ids: Any = None) -> set[int]:
    return _token_id_set(getattr(tokenizer, "eos_token_id", None)) | _token_id_set(terminal_token_ids)


def materialize_rollout_records(
    *,
    data: Any,
    keys: Sequence[str],
    tokenizer: Any,
    source_agent_id: str,
    source_policy_version: int,
    terminal_token_ids: Any = None,
) -> tuple[RolloutRecord, ...]:
    """Convert TransferQueue rollout fields into canonical trajectories."""

    required = ("prompts", "responses", "response_mask", "rm_scores", "raw_prompt", "hacpo_prompt_id")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"rollout data is missing fields: {missing}")

    columns = {name: _rows(data[name]) for name in required}
    extra_fields = _rows(data["extra_fields"]) if "extra_fields" in data else [{} for _ in keys]
    for name, values in (*columns.items(), ("extra_fields", extra_fields)):
        if len(values) != len(keys):
            raise ValueError(f"rollout field {name!r} has {len(values)} rows, expected {len(keys)}")

    eos_ids = _terminal_eos_ids(tokenizer, terminal_token_ids)
    records = []
    for index, key in enumerate(keys):
        prompt_ids = tuple(int(token_id) for token_id in columns["prompts"][index].tolist())
        response_ids = [int(token_id) for token_id in columns["responses"][index].tolist()]
        response_mask = columns["response_mask"][index].bool()
        if response_mask.ndim != 1 or response_mask.numel() != len(response_ids):
            raise ValueError(f"trajectory {key!r} has an invalid response mask")
        if not bool(response_mask.all()):
            raise ValueError(f"trajectory {key!r} contains tool or observation tokens unsupported by single-turn HACPO")
        if not prompt_ids or not response_ids:
            raise ValueError(f"trajectory {key!r} has no prompt or generated action tokens")

        terminal_only = False
        if response_ids[-1] in eos_ids:
            if len(response_ids) == 1:
                terminal_only = True
            else:
                response_ids.pop()

        reward = float(columns["rm_scores"][index].float().sum().item())
        if not math.isfinite(reward):
            raise ValueError(f"trajectory {key!r} has a non-finite reward")
        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if terminal_only:
            response_text = ""
        metadata = _unwrap(extra_fields[index])
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        records.append(
            RolloutRecord(
                trajectory_id=str(key),
                prompt_id=str(_unwrap(columns["hacpo_prompt_id"][index])),
                messages=_messages(columns["raw_prompt"][index]),
                source_agent_id=source_agent_id,
                source_policy_version=source_policy_version,
                prompt_ids=prompt_ids,
                response_ids=tuple(response_ids),
                response_text=response_text,
                reward=reward,
                finish_reason=str(metadata.get("stop_reason", "unknown")),
                terminal_only=terminal_only,
                metadata=metadata,
            )
        )
    return tuple(records)


def attach_source_log_probs(
    records: Sequence[RolloutRecord],
    response_log_probs: torch.Tensor,
) -> tuple[Trajectory, ...]:
    """Attach source-actor scores returned by verl's no-padding engine."""

    rows = _rows(response_log_probs)
    if len(rows) != len(records):
        raise ValueError(f"received {len(rows)} source log-prob rows for {len(records)} rollouts")
    return tuple(record.with_source_log_probs(log_probs) for record, log_probs in zip(records, rows, strict=True))
