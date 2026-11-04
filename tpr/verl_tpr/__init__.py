"""Model-side TPR Attention/Context/CANN and lazy backwards exports."""

from importlib import import_module

# Public TPR model/runtime exports; no optional/legacy symbols in __all__.
_PRODUCTION_EXPORTS = {
    "build_sharded_rotary_pos_emb": "verl_tpr.executor",
    "build_suffix_rotary_pos_emb": "verl_tpr.executor",
    "classify_ring_block": "verl_tpr.ring",
    "get_tpr_attention_context": "verl_tpr.context",
    "iter_valid_sequence_slices": "verl_tpr.ring",
    "KVPrefixAnchors": "verl_tpr.state",
    "KVPrefixState": "verl_tpr.state",
    "KVStack": "verl_tpr.state",
    "KVStackEntry": "verl_tpr.state",
    "make_ring_sequence_shard": "verl_tpr.ring",
    "make_tpr_module_spec_provider": "verl_tpr.adapter",
    "PastKVAnchors": "verl_tpr.state",
    "rectangular_causal_attention": "verl_tpr.rectangular_attention",
    "replace_self_attention_with_tpr": "verl_tpr.adapter",
    "ring_cp_attention": "verl_tpr.ring",
    "RingBlockKind": "verl_tpr.ring",
    "RingCPAttentionBackend": "verl_tpr.ring",
    "RingCPBackend": "verl_tpr.ring",
    "RingLocalKVBlock": "verl_tpr.ring",
    "round_up_sequence_length": "verl_tpr.ring",
    "SegmentExecutor": "verl_tpr.executor",
    "SequenceShard": "verl_tpr.ring",
    "TPRAttentionBackend": "verl_tpr.context",
    "TPRAttentionContext": "verl_tpr.context",
    "TPRSelfAttention": "verl_tpr.attention",
    "use_tpr_attention_context": "verl_tpr.context",
}
__all__ = sorted(_PRODUCTION_EXPORTS)


def __getattr__(name):
    if name not in _PRODUCTION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PRODUCTION_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
