# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""DR.Kernel actor-side runtime patches.

Two independent patches live here:

1. **Qwen3-MoE all-experts loop** (``apply_qwen3_moe_checkpoint_patch``).

2. **MRS per-token catastrophic-ratio veto** (``compute_policy_loss_bypass_mode_with_veto``).
   A tiny extension of verl's ``rollout_correction`` subsystem: any sequence
   containing a token whose ``pi_train / pi_rollout`` drops below
   ``rollout_token_veto_threshold`` (paper value: 1e-4) has its entire
   ``response_mask`` row zeroed for the gradient step. It composes with
   ``rollout_rs`` purely by mask multiplication, so applying it as a post-step of
   the upstream rejection mask is mathematically identical to baking it into the
   helper.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

import verl.trainer.ppo.core_algos as core_algos
import verl.trainer.ppo.rollout_corr_helper as rollout_corr_helper
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss


# ----------------------------------------------------------------------------
# Patch 1: Qwen3-MoE sparse block — loop over all experts (gradient-checkpoint safe)
# ----------------------------------------------------------------------------
def _qwen3_moe_sparse_block_forward_all_experts(self, hidden_states: torch.Tensor):
    """Drop-in replacement for ``Qwen3MoeSparseMoeBlock.forward`` (transformers
    4.57.x) that loops over all experts so the saved-tensor count is constant
    across the forward pass and the non-reentrant checkpoint recomputation."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # DR.Kernel NPU patch: iterate over ALL experts, not the data-dependent
    # "hit" subset (`expert_hit = ...nonzero()`), so non-reentrant gradient
    # checkpointing saves the same number of tensors in forward and recompute.
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def apply_qwen3_moe_checkpoint_patch() -> bool:
    """Patch ``Qwen3MoeSparseMoeBlock.forward`` in place. Idempotent; returns True
    if the patch is in effect (already-patched counts as success)."""
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe
    except Exception:
        # transformers build without qwen3_moe — nothing to patch.
        return False

    block = getattr(modeling_qwen3_moe, "Qwen3MoeSparseMoeBlock", None)
    if block is None:
        return False
    if getattr(block.forward, "_drkernel_all_experts_patch", False):
        return True  # already patched

    _qwen3_moe_sparse_block_forward_all_experts._drkernel_all_experts_patch = True
    block.forward = _qwen3_moe_sparse_block_forward_all_experts
    return True


# Apply on import so it runs wherever this external_lib is loaded (actor worker).
_QWEN3_MOE_PATCH_APPLIED = apply_qwen3_moe_checkpoint_patch()


# ----------------------------------------------------------------------------
# Patch 2: MRS per-token catastrophic-ratio veto
# ----------------------------------------------------------------------------
# Capture the upstream implementations *before* we override the registry, so the
# wrapper can delegate to the real bypass loss and restore the real helper.
_UPSTREAM_BYPASS_LOSS = core_algos.compute_policy_loss_bypass_mode
_UPSTREAM_CORR_AND_RS = rollout_corr_helper.compute_rollout_correction_and_rejection_mask

# Metric keys carry the same "rollout_corr/" prefix the upstream helper applies
# to every metric in its final pass, so the veto metrics sit alongside the rest.
_VETO_SEQ_FRACTION_KEY = "rollout_corr/rollout_veto_seq_fraction"
_VETO_TOKEN_FRACTION_KEY = "rollout_corr/rollout_veto_catastrophic_token_fraction"


def _apply_token_veto(
    modified_response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_token_veto_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Zero the full row of any sequence with a catastrophic-ratio token."""
    log_ratio = old_log_prob - rollout_log_prob
    log_veto_threshold = math.log(rollout_token_veto_threshold)
    catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask.bool()
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
    veto_mask = (~has_catastrophic).to(dtype=modified_response_mask.dtype)
    modified_response_mask = modified_response_mask * veto_mask
    metrics = {
        _VETO_SEQ_FRACTION_KEY: has_catastrophic.float().mean().item(),
        _VETO_TOKEN_FRACTION_KEY: verl_F.masked_mean(catastrophic_tokens.float(), response_mask).item(),
    }
    return modified_response_mask, metrics


def _read_veto_threshold(config) -> Optional[float]:
    """Pull ``rollout_token_veto_threshold`` out of the rollout_correction config."""
    if config is None or not hasattr(config, "policy_loss"):
        return None
    rollout_corr_config = config.policy_loss.get("rollout_correction", None)
    if rollout_corr_config is None:
        return None
    return rollout_corr_config.get("rollout_token_veto_threshold", None)


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode_with_veto(*args, config=None, **kwargs):
    """``bypass_mode`` policy loss + DR.Kernel per-token catastrophic-ratio veto.

    Overrides the upstream ``bypass_mode`` registration. When no veto threshold
    is configured this is a transparent passthrough to the upstream loss.
    """
    veto_threshold = _read_veto_threshold(config)
    if not veto_threshold:
        return _UPSTREAM_BYPASS_LOSS(*args, config=config, **kwargs)
    if veto_threshold <= 0:
        raise ValueError(f"rollout_token_veto_threshold must be positive, got {veto_threshold}.")

    def _helper_with_veto(*h_args, **h_kwargs):
        proto, mask, metrics = _UPSTREAM_CORR_AND_RS(*h_args, **h_kwargs)
        mask, veto_metrics = _apply_token_veto(
            modified_response_mask=mask,
            old_log_prob=h_kwargs["old_log_prob"],
            rollout_log_prob=h_kwargs["rollout_log_prob"],
            response_mask=h_kwargs["response_mask"],
            rollout_token_veto_threshold=veto_threshold,
        )
        metrics.update(veto_metrics)
        return proto, mask, metrics

    # The upstream bypass loss imports the helper by name from the module at call
    # time, so patching the module attribute before delegating is enough. The
    # actor computes the loss single-threaded per micro-batch, so the global
    # swap is safe; the finally clause restores it unconditionally.
    rollout_corr_helper.compute_rollout_correction_and_rejection_mask = _helper_with_veto
    try:
        return _UPSTREAM_BYPASS_LOSS(*args, config=config, **kwargs)
    finally:
        rollout_corr_helper.compute_rollout_correction_and_rejection_mask = _UPSTREAM_CORR_AND_RS
