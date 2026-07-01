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

"""Pydantic request/response models for the legacy remote-worker JSON API.

Each schema class declares its fields (the contract), and inherits conversion
logic from _TensorDictSchema. To understand what an endpoint expects, just
read the field declarations — no need to look at conversion methods.

Data categories in each schema:
  - TensorData fields: tensors serialized as flat list + shape + dtype
  - list[Any] fields: per-sample non-tensor data (e.g. multi_modal_inputs)
  - scalar fields: meta/config values (e.g. calculate_entropy, mini_batch_size)
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
import torch
from pydantic import BaseModel
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.utils import tensordict_utils as tu


# ==================== TensorData ====================


class TensorData(BaseModel):
    """A tensor serialized as flat list + shape + dtype.

    For NestedTensors (jagged), set nested=True with values (flat) and offsets.
    """

    data: list
    shape: list[int]
    dtype: str  # "float32", "int64", etc.
    nested: bool = False
    offsets: Optional[list[int]] = None

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorData:
        if t.is_nested:
            values, offsets = t.values(), t.offsets()
            return cls(
                data=values.cpu().tolist(),
                shape=[len(offsets) - 1],
                dtype=str(values.dtype).replace("torch.", ""),
                nested=True,
                offsets=offsets.cpu().tolist(),
            )
        return cls(data=t.cpu().flatten().tolist(), shape=list(t.shape), dtype=str(t.dtype).replace("torch.", ""))

    def to_tensor(self) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        if self.nested:
            return torch.nested.nested_tensor_from_jagged(
                torch.tensor(self.data, dtype=dtype),
                offsets=torch.tensor(self.offsets, dtype=torch.int64),
            )
        return torch.tensor(self.data, dtype=dtype).reshape(self.shape)


# ==================== Base class for TensorDict conversion ====================


def _to_json_safe(v):
    """Convert numpy/torch values to JSON-serializable Python types."""
    if isinstance(v, torch.Tensor):
        return v.cpu().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, list):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_json_safe(val) for k, val in v.items()}
    return v


class _TensorDictSchema(BaseModel):
    """Base class that provides automatic TensorDict ↔ Pydantic conversion.

    Subclasses declare fields and class-level metadata tuples:
      _tensor_fields: required TensorData fields
      _optional_tensor_fields: optional TensorData fields (may be None)
      _non_tensor_batch_fields: per-sample list fields (NonTensorStack)
      _meta_fields: scalar config/meta fields (NonTensorData)

    The extra_tensors / extra_meta catch-alls preserve unknown fields for
    forward compatibility — new upstream fields won't be silently dropped.
    """

    _tensor_fields: ClassVar[tuple[str, ...]] = ()
    _optional_tensor_fields: ClassVar[tuple[str, ...]] = ()
    _non_tensor_batch_fields: ClassVar[tuple[str, ...]] = ()
    _meta_fields: ClassVar[tuple[str, ...]] = ()

    extra_tensors: dict[str, TensorData] = {}
    extra_meta: dict[str, Any] = {}

    def to_tensordict(self) -> TensorDict:
        tensors: dict[str, TensorData] = {}
        for name in self._tensor_fields:
            tensors[name] = getattr(self, name)
        for name in self._optional_tensor_fields:
            val = getattr(self, name)
            if val is not None:
                tensors[name] = val
        tensors.update(self.extra_tensors)

        non_tensor_batch: dict[str, list] = {}
        for name in self._non_tensor_batch_fields:
            val = getattr(self, name)
            if val is not None:
                non_tensor_batch[name] = val

        meta: dict[str, Any] = {}
        for name in self._meta_fields:
            val = getattr(self, name)
            if val is not None:
                meta[name] = val
        meta.update(self.extra_meta)

        # Build TensorDict
        batch_tensors = {k: v.to_tensor() for k, v in tensors.items()}
        batch_size = 0
        for v in tensors.values():
            batch_size = (len(v.offsets) - 1) if v.nested else v.shape[0]
            break
        td = TensorDict(batch_tensors, batch_size=[batch_size])
        for k, v in non_tensor_batch.items():
            td.set(k, NonTensorStack.from_list([NonTensorData(item) for item in v]))
        for k, v in meta.items():
            tu.assign_non_tensor_data(td, k, v)
        return td

    @classmethod
    def from_tensordict(cls, td) -> _TensorDictSchema:
        # Decompose TensorDict
        tensors, non_tensor_batch, meta_info = {}, {}, {}
        for k in td.keys():
            v = td.get(k)
            if isinstance(v, torch.Tensor):
                tensors[k] = TensorData.from_tensor(v)
            elif isinstance(v, NonTensorStack):
                non_tensor_batch[k] = _to_json_safe(v.tolist())
            else:
                raw = v.data if isinstance(v, NonTensorData) else v
                meta_info[k] = _to_json_safe(raw)

        # Route to known fields vs extras
        all_known_tensors = set(cls._tensor_fields + cls._optional_tensor_fields)
        kwargs: dict[str, Any] = {}
        extra_tensors = {}
        for k, v in tensors.items():
            if k in all_known_tensors:
                kwargs[k] = v
            else:
                extra_tensors[k] = v
        kwargs["extra_tensors"] = extra_tensors

        for k in cls._non_tensor_batch_fields:
            kwargs[k] = non_tensor_batch.pop(k, None)

        for k in cls._meta_fields:
            if k in meta_info:
                kwargs[k] = meta_info.pop(k)

        kwargs["extra_meta"] = meta_info
        return cls(**kwargs)


# ==================== /v1/connect ====================


class ServerCapabilities(BaseModel):
    """Server-side config flags returned in ConnectResponse for client validation."""

    use_kl_loss: bool = False
    use_kl_in_reward: bool = False
    use_critic: bool = False
    use_reference_policy: bool = False
    no_rollout_deployment: bool = False


class ConnectResponse(BaseModel):
    status: str
    world_size: Optional[int] = None
    session_id: Optional[str] = None
    no_rollout_deployment: Optional[bool] = None
    capabilities: Optional[ServerCapabilities] = None


# ==================== Simple schemas ====================


class StatusResponse(BaseModel):
    status: str


class SampleRequest(BaseModel):
    request_id: str
    prompt_ids: list[int]
    sampling_params: dict[str, Any] = {}
    image_data: Optional[list[Any]] = None
    video_data: Optional[list[Any]] = None


class SaveCheckpointRequest(BaseModel):
    path: str
    global_step: int = 0
    max_ckpt_to_keep: Optional[int] = None


class LoadCheckpointRequest(BaseModel):
    path: str


# ==================== /v1/forward & /v1/compute_ref_log_prob ====================
#
# Both endpoints share the same request/response schema.
# compute_ref_log_prob is just compute_log_prob on the reference model.


class ForwardRequest(_TensorDictSchema):
    """Input for /v1/forward and /v1/compute_ref_log_prob.

    Consumed by: dp_actor.compute_log_prob() → model forward pass.
    """

    _tensor_fields = ("input_ids", "attention_mask", "position_ids", "responses")
    _optional_tensor_fields = ("response_mask", "prompts")
    _non_tensor_batch_fields = ("multi_modal_inputs", "uid")
    _meta_fields = ("calculate_entropy", "compute_loss", "no_lora_adapter")

    # Required: core input tensors (may be NestedTensor / jagged)
    input_ids: TensorData
    attention_mask: TensorData
    position_ids: TensorData
    responses: TensorData
    # Optional: prefix grouping
    response_mask: Optional[TensorData] = None
    prompts: Optional[TensorData] = None
    # Optional: vision-language models
    multi_modal_inputs: Optional[list[Any]] = None
    uid: Optional[list[Any]] = None
    # Meta: controls what the forward pass computes
    calculate_entropy: bool = True
    compute_loss: bool = False
    no_lora_adapter: bool = False


class ForwardResponse(_TensorDictSchema):
    """Output of /v1/forward and /v1/compute_ref_log_prob."""

    _tensor_fields = ("log_probs",)
    _optional_tensor_fields = ("entropys", "sum_pi_squared")

    log_probs: TensorData
    entropys: Optional[TensorData] = None  # when calculate_entropy=True
    sum_pi_squared: Optional[TensorData] = None  # when calculate_sum_pi_squared=True


# ==================== /v1/forward_backward ====================


class UpdateActorRequest(_TensorDictSchema):
    """Input for /v1/forward_backward.

    Consumed by: dp_actor.update_policy() → forward + ppo_loss() + backward.
    """

    _tensor_fields = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "old_log_probs",
        "advantages",
    )
    _optional_tensor_fields = ("ref_log_prob", "rollout_is_weights", "rollout_log_probs", "prompts")
    _non_tensor_batch_fields = ("multi_modal_inputs", "uid")
    _meta_fields = (
        "mini_batch_size",
        "epochs",
        "seed",
        "temperature",
        "calculate_entropy",
        "global_batch_size",
        "multi_turn",
        "dataloader_kwargs",
    )

    # Required: core training tensors
    input_ids: TensorData  # may be NestedTensor (jagged)
    attention_mask: TensorData
    position_ids: TensorData
    responses: TensorData
    response_mask: TensorData
    old_log_probs: TensorData  # from compute_log_prob, used in ratio computation
    advantages: TensorData  # from advantage estimation (GAE / GRPO)
    # Conditional: depends on training config
    ref_log_prob: Optional[TensorData] = None  # required when use_kl_loss=True
    rollout_is_weights: Optional[TensorData] = None  # importance sampling correction
    rollout_log_probs: Optional[TensorData] = None  # bypass mode
    prompts: Optional[TensorData] = None  # prefix grouping
    # Optional: vision-language models
    multi_modal_inputs: Optional[list[Any]] = None
    uid: Optional[list[Any]] = None
    # Training hyperparams (set by ray_trainer)
    mini_batch_size: int
    epochs: int = 1
    seed: int = 42
    temperature: float = 1.0
    calculate_entropy: bool = False
    global_batch_size: Optional[int] = None
    multi_turn: bool = False
    dataloader_kwargs: dict[str, Any] = {"shuffle": True}


class UpdateActorResponse(_TensorDictSchema):
    """Output of /v1/forward_backward. Contains training metrics."""

    _meta_fields = ("metrics",)

    metrics: dict[str, Any] = {}


# ==================== /v1/compute_values (critic forward) ====================


class ComputeValuesRequest(_TensorDictSchema):
    """Input for /v1/compute_values.

    Consumed by: critic_wg.infer_batch() -> value model forward pass.
    Same core input shape as ForwardRequest (the critic sees the same
    sequence), but without actor-specific fields.

    Note: compute_loss is always set to False server-side. The field is
    included for schema completeness but ignored.
    """

    _tensor_fields = ("input_ids", "attention_mask", "position_ids", "responses")
    _optional_tensor_fields = ("response_mask", "prompts")
    _non_tensor_batch_fields = ("multi_modal_inputs", "uid")
    _meta_fields = ("compute_loss",)

    input_ids: TensorData
    attention_mask: TensorData
    position_ids: TensorData
    responses: TensorData
    response_mask: Optional[TensorData] = None
    prompts: Optional[TensorData] = None
    multi_modal_inputs: Optional[list[Any]] = None
    uid: Optional[list[Any]] = None
    compute_loss: bool = False


class ComputeValuesResponse(_TensorDictSchema):
    """Output of /v1/compute_values."""

    _tensor_fields = ("values",)

    values: TensorData


# ==================== /v1/update_critic ====================


class UpdateCriticRequest(_TensorDictSchema):
    """Input for /v1/update_critic.

    Consumed by: critic_wg.train_mini_batch() → forward + value_loss + backward.
    """

    _tensor_fields = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "values",
        "returns",
    )
    _optional_tensor_fields = ("prompts",)
    _non_tensor_batch_fields = ("multi_modal_inputs", "uid")
    _meta_fields = ("global_batch_size", "mini_batch_size", "epochs", "seed", "dataloader_kwargs")

    input_ids: TensorData
    attention_mask: TensorData
    position_ids: TensorData
    responses: TensorData
    response_mask: TensorData
    values: TensorData
    returns: TensorData
    prompts: Optional[TensorData] = None
    multi_modal_inputs: Optional[list[Any]] = None
    uid: Optional[list[Any]] = None
    global_batch_size: int = 0
    mini_batch_size: int = 1
    epochs: int = 1
    seed: int = 42
    dataloader_kwargs: dict[str, Any] = {"shuffle": True}


class UpdateCriticResponse(_TensorDictSchema):
    """Output of /v1/update_critic. Contains critic training metrics."""

    _meta_fields = ("metrics",)

    metrics: dict[str, Any] = {}


# ==================== /v1/compute_advantages (convenience) ====================


class ComputeAdvantagesRequest(_TensorDictSchema):
    """Input for /v1/compute_advantages.

    Server internally runs compute_values → GAE → returns
    (advantages, returns, values) in a single round-trip.
    """

    _tensor_fields = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "token_level_rewards",
    )
    _optional_tensor_fields = ("prompts",)
    _non_tensor_batch_fields = ("multi_modal_inputs", "uid")
    _meta_fields = ("gamma", "lam")

    input_ids: TensorData
    attention_mask: TensorData
    position_ids: TensorData
    responses: TensorData
    response_mask: TensorData
    token_level_rewards: TensorData
    prompts: Optional[TensorData] = None
    multi_modal_inputs: Optional[list[Any]] = None
    uid: Optional[list[Any]] = None
    gamma: float = 1.0
    lam: float = 1.0


class ComputeAdvantagesResponse(_TensorDictSchema):
    """Output of /v1/compute_advantages."""

    _tensor_fields = ("advantages", "returns", "values")

    advantages: TensorData
    returns: TensorData
    values: TensorData
