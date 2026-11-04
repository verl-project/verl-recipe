"""TPR-aware Megatron activation recomputation.

Do not patch global Megatron/MindSpeed code.  On a TPR model instance, replace
only TransformerBlock._checkpointed_forward.  Keep the *native* Megatron
tensor_parallel.checkpoint (including its TP RNG handling) but make each
layer's external K/V explicit checkpoint outputs.  Each replay gets a fresh
TPR collector; the prefix anchors and CP backend remain shared references.

Scope: dense FA, Megatron full/uniform/1 recomputation, PP=1.
"""

from __future__ import annotations

import inspect
import weakref
from types import MethodType
from typing import Any

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock

from verl_tpr.attention import TPRSelfAttention
from verl_tpr.context import get_tpr_attention_context, use_tpr_attention_context


def validate_tpr_recompute_config(config: Any) -> bool:
    """Return whether the explicitly supported TPR recompute mode is selected."""
    granularity = getattr(config, "recompute_granularity", None)
    if granularity is None:
        return False
    if (
        granularity != "full"
        or getattr(config, "recompute_method", None) != "uniform"
        or getattr(config, "recompute_num_layers", None) != 1
    ):
        raise NotImplementedError("TPR recompute supports only full/uniform/1")
    if getattr(config, "fp8", None) not in (False, None) or getattr(config, "fp4", None):
        raise NotImplementedError("TPR recompute does not support FP8/FP4")
    if getattr(config, "cpu_offloading", False) or getattr(config, "swap_attention", False):
        raise NotImplementedError("TPR recompute cannot be combined with activation offload")
    if getattr(config, "distribute_saved_activations", False):
        raise NotImplementedError("TPR recompute has not validated distributed saved activations")
    return True


def _checkpointed_tpr_forward(self: TransformerBlock, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Instance-only replacement for TransformerBlock._checkpointed_forward.

    Match the installed native method's signature and reject unknown *active*
    options. This deliberately supports one attention-only decoder block at a
    time; cross-attention and non-FA layers are not silently approximated.
    """
    original = self._tpr_native_checkpointed_forward
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    p = bound.arguments

    for name in (
        "context",
        "context_mask",
        "attention_bias",
        "packed_seq_params",
        "padding_mask",
        "rotary_pos_cos",
        "rotary_pos_sin",
        "rotary_pos_cos_sin",
        "sequence_len_offset",
    ):
        if p.get(name) is not None:
            raise NotImplementedError(f"TPR full recompute does not support {name}")
    if p.get("use_inner_quantization_context", False):
        raise NotImplementedError("TPR full recompute does not support inner quantization")

    hidden = p["hidden_states"]
    attention_mask = p.get("attention_mask")
    rotary_pos_emb = p.get("rotary_pos_emb")
    outer = get_tpr_attention_context()
    if outer is None:
        # Explicitly called on a TPR model without a tree: keep normal native
        # checkpoint behavior, including the ordinary attention implementation.
        return original(*args, **kwargs)

    # IMPORTANT: Megatron CheckpointFunction keeps run_function on its
    # autograd context. Capturing outer *strongly* creates a cross-language
    # cycle: checkpoint ctx -> run_layer -> outer -> returned KV -> grad_fn
    # -> checkpoint ctx. NPU tensors retained through this cycle can survive
    # Python gc and accumulate between actor updates / benchmark iterations.
    # SegmentExecutor holds outer strongly until backward completes, so
    # checkpoint replay can safely dereference it through a weak reference.
    outer_ref = weakref.ref(outer)

    # Models with multiple TransformerBlocks are supported independently.
    # The target shape is full/uniform/1, so each layer owns exactly one
    # checkpoint and one (K,V) side-output pair.
    for layer in self.layers:
        if not isinstance(getattr(layer, "self_attention", None), TPRSelfAttention):
            raise NotImplementedError("TPR recompute currently requires dense FA layers")

        def run_layer(x: torch.Tensor, *, _layer=layer):
            # Never capture 'outer' directly in this checkpoint closure.
            # In the executor, _backward_segment keeps the context alive
            # across the forward, replay, and completion of backward.
            current_outer = outer_ref()
            if current_outer is None:
                raise RuntimeError(
                    "TPR checkpoint replay outlived its attention context; "
                    "keep the SegmentExecutor context alive through backward"
                )
            replay = current_outer.new_collector()
            with use_tpr_attention_context(replay):
                layer_output, layer_context = _layer(
                    hidden_states=x,
                    attention_mask=attention_mask,
                    context=None,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    inference_context=None,
                    packed_seq_params=None,
                    padding_mask=None,
                )
            if layer_context is not None:
                raise NotImplementedError("TPR recompute does not support layer context outputs")
            new_key, new_value = replay.new_key_values[_layer.layer_number]
            return layer_output, new_key, new_value

        if torch.is_grad_enabled():
            if not hidden.requires_grad:
                raise RuntimeError("TPR full recompute needs a grad-carrying TransformerBlock input")
            hidden, key, value = tensor_parallel.checkpoint(
                run_layer,
                False,  # distribute_saved_activations: intentionally gated off
                hidden,
            )
            outer.set_new_kv(layer.layer_number, key, value)
        else:
            # Push: graph-free and never needs checkpoint replay. Capture KV
            # directly via the regular TPR attention collector.
            hidden, _key, _value = run_layer(hidden)
            outer.set_new_kv(layer.layer_number, _key, _value)

    return hidden


def install_tpr_full_recompute(model: torch.nn.Module, config: Any) -> int:
    """Install on real model instances; return number of patched blocks."""
    if not validate_tpr_recompute_config(config):
        return 0
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    if not blocks:
        raise RuntimeError("TPR full recompute found no Megatron TransformerBlock")
    # Validate every block before changing any method. Non-dense models
    # must fail at initialization, never halfway through a PPO optimizer step.
    for block in blocks:
        if not getattr(block, "layers", None):
            raise NotImplementedError("TPR recompute requires a nonempty dense TransformerBlock")
        if any(not isinstance(getattr(layer, "self_attention", None), TPRSelfAttention) for layer in block.layers):
            raise NotImplementedError("TPR recompute currently requires dense FA layers")

    patched = 0
    for block in blocks:
        if getattr(block, "_tpr_full_recompute_installed", False):
            continue
        native = getattr(block, "_checkpointed_forward", None)
        if not callable(native):
            raise RuntimeError("installed Megatron TransformerBlock lacks _checkpointed_forward")
        # Require the native entry fields used by the TPR adapter. Fail early
        # if the installed MindSpeed repatch exposes a different API.
        native_fields = set(inspect.signature(native).parameters)
        required = {"hidden_states", "attention_mask", "rotary_pos_emb"}
        if not required <= native_fields:
            raise NotImplementedError(f"unsupported Megatron checkpoint signature: missing {required - native_fields}")
        block._tpr_native_checkpointed_forward = native
        block._checkpointed_forward = MethodType(_checkpointed_tpr_forward, block)
        block._tpr_full_recompute_installed = True
        patched += 1
    return patched


def assert_tpr_full_recompute_installed(model: torch.nn.Module, config: Any) -> None:
    """Reject a config-only enablement that silently loses external KV grads."""
    if not validate_tpr_recompute_config(config):
        return
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    if not blocks or not all(getattr(m, "_tpr_full_recompute_installed", False) for m in blocks):
        raise RuntimeError("TPR full recompute enabled without TPR-aware block adaptation")
