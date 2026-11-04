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

"""Megatron self-attention with an opt-in external-KV training path."""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention
from megatron.core.typed_torch import apply_module
from torch import Tensor

from .context import TPRAttentionContext, get_tpr_attention_context
from .rectangular_attention import rectangular_causal_attention


class TPRSelfAttention(SelfAttention):
    """Self-attention that consumes per-layer external KV in tree mode.

    With no active :class:`TPRAttentionContext`, this class delegates the
    complete call to Megatron's ``SelfAttention.forward``.  The external-KV
    branch is intentionally restricted to the first MVP configuration and
    bypasses only Megatron/MindSpeed's core-attention wrapper.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor | tuple[Tensor, Tensor]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor | None]:
        options = dict(
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )
        context = get_tpr_attention_context()
        if context is None:
            return super().forward(
                hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                **options,
            )

        self._validate_tree_forward(
            context=context,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **options,
        )
        return self._tree_forward(hidden_states, context)

    def _tree_forward(
        self,
        hidden_states: Tensor,
        context: TPRAttentionContext,
    ) -> tuple[Tensor, Tensor | None]:
        # Reuse Megatron's projection, GQA reshaping, and optional Q/K norms.
        output_gate = getattr(self.config, "attention_output_gate", False)
        qkv = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states=None,
            split_qkv=True,
            output_gate=output_gate,
        )
        if output_gate:
            query, new_key, new_value, gate = qkv
        else:
            query, new_key, new_value = qkv
            gate = None

        # Megatron ColumnParallelLinear owns SP all-gather on the QKV input.
        # The attention kernel and per-layer KVStack still see the *complete*
        # logical segment sequence, never an SP-local KV fragment.
        if (
            getattr(self.config, "sequence_parallel", False)
            and (getattr(self.config, "tensor_model_parallel_size", 1) > 1)
            and query.shape[0] != context.local_suffix_length
        ):
            raise ValueError(
                "Megatron SP QKV projection must restore the full TPR segment: "
                f"expected {context.local_suffix_length}, got {query.shape[0]}"
            )

        q_pos_emb, k_pos_emb = _as_rotary_pair(context.suffix_rotary_pos_emb)
        # A sharded backend receives rank-local RoPE that was already sliced
        # by SegmentExecutor; passing cp_group here would shard it a second time.
        cp_group = None if context.attention_backend is not None else getattr(self.pg_collection, "cp", None)
        rope_kwargs = {
            "config": self.config,
            "cu_seqlens": None,
            "mscale": getattr(self, "_yarn_concentration_factor", 1.0),
            "cp_group": cp_group,
        }
        query = apply_rotary_pos_emb(query, q_pos_emb, **rope_kwargs)
        new_key = apply_rotary_pos_emb(new_key, k_pos_emb, **rope_kwargs)

        if context.attention_backend is not None:
            core_attn_out = context.attention_backend.attention(
                self.layer_number,
                query,
                new_key,
                new_value,
                softmax_scale=_core_attention_softmax_scale(self.core_attention),
            )
        else:
            past_kv = context.get_past_kv(self.layer_number)
            if past_kv is None:
                key = new_key
                value = new_value
            else:
                past_key, past_value = past_kv
                _validate_past_compatibility(self.layer_number, past_key, past_value, new_key, new_value)
                key = _concat_sequence(past_key, new_key)
                value = _concat_sequence(past_value, new_value)

            core_attn_out = rectangular_causal_attention(
                query,
                key,
                value,
                softmax_scale=_core_attention_softmax_scale(self.core_attention),
                dropout_p=0.0,
            )

        # Pop retains post-RoPE K/raw V graphs. Push compacts graph-free views
        # now, so later layers do not keep this layer's full QKV storage alive.
        context.set_new_kv(self.layer_number, new_key, new_value)
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)
        output = apply_module(self.linear_proj)(core_attn_out)
        return output

    def _validate_tree_forward(
        self,
        *,
        context: TPRAttentionContext,
        hidden_states: Tensor,
        **unsupported: object,
    ) -> None:
        if not self.training:
            raise RuntimeError("TPR external-KV mode is restricted to training")
        tp_size = getattr(self.config, "tensor_model_parallel_size", 1)
        if tp_size < 1:
            raise ValueError(f"invalid tensor_model_parallel_size={tp_size}")
        sp_enabled = tp_size > 1 and getattr(self.config, "sequence_parallel", False)
        # Attention receives sequence-sharded hidden activations under native
        # Megatron SP. ColumnParallelLinear all-gathers them for full-segment
        # QKV inside get_query_key_value_tensors(); do not shard KV or RoPE.
        if sp_enabled and context.local_suffix_length % tp_size:
            raise ValueError(
                f"TPR segment length {context.local_suffix_length} must be divisible "
                f"by TP={tp_size} under native sequence parallelism (padding deferred)"
            )
        expected_input_len = context.local_suffix_length // tp_size if sp_enabled else context.local_suffix_length
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                f"TPR external-KV mode requires hidden_states [sequence, 1, hidden], got {hidden_states.shape}"
            )
        if hidden_states.shape[0] != expected_input_len:
            raise ValueError(
                f"hidden suffix length ({hidden_states.shape[0]}) does not match "
                f"expected {'SP-local' if sp_enabled else 'CP-local'} length "
                f"({expected_input_len}) for logical suffix {context.local_suffix_length}"
            )
        if context.suffix_rotary_pos_emb is None:
            raise ValueError("TPRAttentionContext.suffix_rotary_pos_emb is required in TPR mode")
        _validate_rotary_sequence_length(context.suffix_rotary_pos_emb, context.local_suffix_length)

        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise NotImplementedError(f"TPR external-KV MVP does not support: {', '.join(active)}")

        boolean_restrictions = {
            "activation checkpointing": getattr(self, "checkpoint_core_attention", False),
            "fused_single_qkv_rope": getattr(self.config, "fused_single_qkv_rope", False),
            "flash_decode": getattr(self.config, "flash_decode", False),
            "QKV offload": getattr(self, "offload_qkv_linear", False),
            "core-attention offload": getattr(self, "offload_core_attention", False),
            "attention-projection offload": getattr(self, "offload_attn_proj", False),
        }
        enabled = [name for name, value in boolean_restrictions.items() if value]
        if enabled:
            raise NotImplementedError(f"TPR external-KV MVP does not support: {', '.join(enabled)}")

        if getattr(self, "attention_type", "self") != "self":
            raise NotImplementedError("TPR external-KV mode requires self-attention")
        mask_type = getattr(self, "attn_mask_type", "causal")
        mask_name = getattr(mask_type, "name", mask_type)
        if mask_name not in ("causal", "causal_bottom_right"):
            raise NotImplementedError(f"TPR external-KV mode requires causal attention, got {mask_type!r}")
        if getattr(self.config, "window_size", None) is not None:
            raise NotImplementedError("TPR external-KV mode does not support window_size (sliding-window attention)")
        if getattr(self.config, "softmax_type", "vanilla") != "vanilla":
            raise NotImplementedError(
                "TPR external-KV mode requires softmax_type='vanilla'; sink softmax is unsupported"
            )
        core_attention = getattr(self.core_attention, "local_attn", self.core_attention)
        for name in ("softmax_offset", "pse"):
            if getattr(core_attention, name, None) is not None:
                raise NotImplementedError(f"TPR external-KV mode does not support core-attention {name}")

        if float(getattr(self.config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("TPR external-KV MVP requires attention_dropout=0")
        # Megatron's QKV and output projections already own TP collectives.
        # Their attention inputs/outputs are rank-local head shards, so TPR
        # must never all-gather KV along the TP dimension here.
        for name in ("pipeline_model_parallel_size", "expert_model_parallel_size"):
            value = getattr(self.config, name, 1)
            if value != 1:
                raise NotImplementedError(f"TPR external-KV MVP requires {name}=1, got {value}")
        context_parallel_size = getattr(self.config, "context_parallel_size", 1)
        expected_context_parallel_size = (
            1 if context.attention_backend is None else context.attention_backend.parallel_size
        )
        if context_parallel_size != expected_context_parallel_size:
            raise ValueError(
                "TPR attention backend/config CP size mismatch: "
                f"backend={expected_context_parallel_size}, config={context_parallel_size}"
            )


def _as_rotary_pair(rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
    if rotary_pos_emb is None:
        raise ValueError("suffix rotary position embedding is required")
    if isinstance(rotary_pos_emb, tuple):
        if len(rotary_pos_emb) != 2 or not all(isinstance(item, Tensor) for item in rotary_pos_emb):
            raise TypeError("suffix rotary position embedding tuple must contain two tensors")
        return rotary_pos_emb
    if not isinstance(rotary_pos_emb, Tensor):
        raise TypeError("suffix rotary position embedding must be a tensor or a pair of tensors")
    return rotary_pos_emb, rotary_pos_emb


def _validate_rotary_sequence_length(
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor],
    suffix_length: int,
) -> None:
    for name, tensor in zip(("query", "key"), _as_rotary_pair(rotary_pos_emb), strict=True):
        if tensor.ndim == 0 or tensor.shape[0] != suffix_length:
            raise ValueError(
                f"{name} suffix rotary embedding sequence length must be {suffix_length}, got {tensor.shape}"
            )


def _validate_past_compatibility(
    layer_number: int,
    past_key: Tensor,
    past_value: Tensor,
    new_key: Tensor,
    new_value: Tensor,
) -> None:
    for name, past, new in (("K", past_key, new_key), ("V", past_value, new_value)):
        if past.shape[1:] != new.shape[1:]:
            raise ValueError(
                f"layer {layer_number} past/new {name} shapes are incompatible: past={past.shape}, new={new.shape}"
            )
        if past.device != new.device or past.dtype != new.dtype:
            raise ValueError(
                f"layer {layer_number} past/new {name} device and dtype must match, "
                f"past=({past.device}, {past.dtype}), new=({new.device}, {new.dtype})"
            )


def _concat_sequence(prefix: Tensor, suffix: Tensor) -> Tensor:
    # Kept as a tiny seam so tests can assert that external KV remains part of
    # the autograd graph without introducing any storage transformation here.
    return torch.cat((prefix, suffix), dim=0)


def _core_attention_softmax_scale(core_attention) -> float | None:
    """Read the scale through MindSpeed's Ulysses wrapper when present."""

    scale = getattr(core_attention, "softmax_scale", None)
    if scale is not None:
        return scale
    local_attention = getattr(core_attention, "local_attn", None)
    return None if local_attention is None else getattr(local_attention, "softmax_scale", None)
