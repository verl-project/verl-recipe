from typing import Optional

import torch
from fastapi import HTTPException
from tensordict import TensorDict


# ============================================================
# Datum → TensorDict translation (inverse of data_converter.py)
# ============================================================


def _tensor_data_to_torch(td) -> torch.Tensor:
    """Inverse of TensorData.from_torch — pydantic TensorData to a flat torch tensor."""
    dtype = torch.float32 if td.dtype == "float32" else torch.int64
    t = torch.tensor(td.data, dtype=dtype)
    if td.shape and len(td.shape) > 1:
        t = t.view(*td.shape)
    return t


def _coerce_one_metric_value(v) -> Optional[float]:
    """Best-effort convert a single verl metric value to a Python float.

    Returns None if the value can't be sensibly turned into a scalar.
    Handles:
      - ``Metric(value=…, aggregation=…)``  → unwrap ``.value``
      - ``NonTensorData`` / ``NonTensorStack`` → unwrap ``.data`` / iterate
      - ``torch.Tensor``                    → ``.item()`` if scalar else ``.mean().item()``
      - ``numpy`` scalars                   → ``float(np_value)``
      - Python int / float                  → ``float(v)``
      - ``None`` / non-numeric strings      → ``None``
    """
    # Unwrap Metric / NonTensorData wrappers (use ``.value`` / ``.data``).
    if hasattr(v, "value"):
        v = v.value
    if hasattr(v, "data") and not torch.is_tensor(v) and not isinstance(v, (list, tuple)):
        # NonTensorData has a .data attribute; numpy arrays have .data but it's a memoryview.
        inner = v.data
        # Heuristic: only unwrap if the .data attribute is itself a numeric type or wrapper.
        if isinstance(inner, (int, float)) or torch.is_tensor(inner) or hasattr(inner, "value"):
            v = inner

    if v is None:
        return None
    if isinstance(v, bool):
        # avoid bool-as-int sneaking through
        return float(v)
    if isinstance(v, (int, float)):
        return float(v) if v == v else None  # filter NaN
    if torch.is_tensor(v):
        v = v.detach().float()
        if v.numel() == 0:
            return None
        return v.mean().item() if v.numel() > 1 else v.item()
    if hasattr(v, "item") and callable(v.item):
        # numpy scalar / 0-D tensor / similar
        try:
            return float(v.item())
        except Exception:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_verl_metrics_to_floats(raw_metrics) -> dict[str, float]:
    """Flatten verl's per-microbatch metric dict into a Tinker-shaped
    ``dict[str, float]`` where every key carries a ``:<reduction>`` suffix.

    verl's ``train_mini_batch`` (engine_workers.py:298-321) calls
    ``py_functional.append_to_dict`` over each micro-batch's metrics, which
    **always wraps values into lists** (one entry per micro-batch ×
    DP rank). So the dict our shim sees is e.g.::

        {
          "loss":               [0.42, 0.43, ...],          # list of floats
          "actor/pg_loss":      [Metric(0.41), Metric(0.42)], # list of Metrics
          "actor/pg_clipfrac":  [0.0],                        # list of one float
          "lr":                 [1e-4],
          "grad_norm":          [0.78],
          "distillation/loss":  [Metric(2.3)],                # our top-K loss
        }

    A bare ``float(list_of_things)`` raises, so the original shim silently
    dropped everything except literal scalars. Two things this helper does:

    1. Take the mean of each list (or unwrap a singleton) so the cookbook
       sees real numbers.
    2. Append the Tinker SDK's required ``:<reduction>`` suffix
       (``chunked_fwdbwd_helpers.py:108`` does ``name, reduction =
       key.split(":")`` and raises on missing suffix). All verl metrics
       are already chunk-aggregated (mean of micro-batches × DP), so the
       cross-chunk reduction is also ``:mean`` for every key.
    """
    out: dict[str, float] = {}
    for k, v in (raw_metrics or {}).items():
        try:
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    continue
                vals = [fv for fv in (_coerce_one_metric_value(item) for item in v) if fv is not None]
                if not vals:
                    continue
                value: Optional[float] = sum(vals) / len(vals)
            else:
                value = _coerce_one_metric_value(v)
            if value is None:
                continue
            key = str(k)
            if ":" not in key:
                # Tinker SDK requires a ``:<reduction>`` suffix on every
                # metric key (chunked_fwdbwd_helpers.py:108). verl emits
                # raw names — chunk-aggregated metrics combine via mean,
                # which matches Tinker's :mean reducer semantics.
                key = f"{key}:mean"
            out[key] = value
        except Exception:
            # Last-resort guard — keep extraction loop forward-compatible
            # with new verl metric types. Anything we can't coerce is
            # silently dropped, same as the old behavior.
            continue
    return out


def _datums_to_update_actor_td(
    datums, mini_batch_size: int, temperature: float = 1.0, pad_to_multiple: int = 1
) -> TensorDict:
    """Translate a Tinker forward_backward payload (list[Datum]) into the
    TensorDict shape expected by verl's actor_rollout_wg.forward_backward.

    The target shape mirrors what verl's ``left_right_2_no_padding`` produces
    on the Tinker-server client side, because the downstream actor code path is
    the same:

      - ``input_ids``    : jagged NestedTensor (B, j1)   full sequence per sample
      - ``position_ids`` : jagged NestedTensor (B, j1)   arange per sample
      - ``prompts``      : jagged NestedTensor (B, p_i)  per-sample prompts
      - ``responses``    : jagged NestedTensor (B, r_i)  per-sample responses
      - ``attention_mask``: rectangular (B, max_seq)     defensive; not consumed
                            in rmpad path but the schema marks it required
      - ``response_mask`` : rectangular (B, max_resp)    1.0 on response positions
      - ``loss_mask``     : == response_mask (used by forward_backward_batch
                            to compute ``batch_num_tokens``)
      - ``old_log_probs`` : rectangular (B, max_resp)
      - ``advantages``    : rectangular (B, max_resp) — denormalised from the
                            wire (Tinker divides by response_len for SUM
                            aggregation; verl wants the raw per-sample value).

    Reconstruction rules:
      - full_tokens = ModelInput.to_ints() + [target_tokens[-1]]   (= tokens[:-1] + tokens[-1])
      - The response window is read directly from
        loss_fn_inputs["response_mask"] (target-indexed). The data_converter
        packs it explicitly so we don't need to infer the response window
        from non-zero advantages (which is unreliable in PPO, where genuine
        zero advantages occur when V-hat predicts the return exactly).
      - response_start_full = response_start_in_target + 1   (target = tokens[1:])
    """
    if not datums:
        raise HTTPException(
            status_code=422,
            detail="Critic request carried no datums; forward_input.data must be non-empty.",
        )

    per_sample_fulls = []
    per_sample_positions = []
    per_sample_prompts = []
    per_sample_responses_tok = []
    per_sample_resp_mask = []
    per_sample_old_logp = []
    per_sample_advantages = []
    per_sample_rollout_isw: list[Optional[torch.Tensor]] = []

    for d in datums:
        prefix = list(d.model_input.to_ints())  # tokens[:-1]
        loss_inputs = d.loss_fn_inputs
        target_tokens = _tensor_data_to_torch(loss_inputs["target_tokens"]).long()
        padded_logp = _tensor_data_to_torch(loss_inputs["logprobs"]).float()
        padded_adv = _tensor_data_to_torch(loss_inputs["advantages"]).float()

        # Find the response window. Sources in priority order:
        #   1. ``response_mask``  — modelchef's verl_tinker_connector packs
        #      this explicitly (handles PPO zero-advantage cleanly).
        #   2. ``mask``           — tinker_cookbook's rl/data_processing.py
        #      uses this name with the same semantics. Usually stripped before
        #      send via _remove_mask in rl/train.py, but cheap to check.
        #   3. ``weights``        — tinker_cookbook's SL data_processing pack
        #      uses ``weights`` (1.0 on response, 0.0 on prompt).
        #   4. Non-zero positions of ``advantages`` ∪ ``logprobs``.
        #      The cookbook's RL builder writes logprobs=0 on prompt tokens
        #      and the actual sampled logprob on response tokens
        #      (data_processing.py:180). Actual logprobs are essentially
        #      never exactly 0 under stochastic sampling, so this is a
        #      reliable fallback even when the whole sample's advantage is
        #      zero (a trajectory whose reward equals the group mean).
        if "response_mask" in loss_inputs:
            padded_mask = _tensor_data_to_torch(loss_inputs["response_mask"]).float()
        elif "mask" in loss_inputs:
            padded_mask = _tensor_data_to_torch(loss_inputs["mask"]).float()
        elif "weights" in loss_inputs:
            padded_mask = _tensor_data_to_torch(loss_inputs["weights"]).float()
        else:
            padded_mask = ((padded_adv != 0) | (padded_logp != 0)).float()

        full_tokens = torch.tensor(prefix + [int(target_tokens[-1].item())], dtype=torch.long)
        valid_len = full_tokens.numel()

        # Read response window from the explicit mask (target-indexed).
        mask_positions = (padded_mask > 0).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Datum has no inferrable response window: "
                    "response_mask / mask / weights absent and both advantages "
                    "and logprobs are all zero. Caller must pack one of these "
                    "fields or send non-degenerate data."
                ),
            )
        response_start_in_target = int(mask_positions[0].item())
        # Extend the response window to the end of the target sequence so the
        # invariant ``sum(prompt_lens + response_lens) == total_input_ids_nnz``
        # (which verl's ``no_padding_2_padding`` asserts at padding.py:131)
        # always holds. The cookbook's RL builder can leave trailing zeros in
        # the mask (post-response observation / EOS delimiter) and interior
        # zeros for multi-turn observations — those positions sit inside the
        # response window with response_mask=0, which ppo_loss masks out of
        # the loss. The denormalised advantages/logprobs are taken from the
        # same extended slice, preserving the on-wire values verbatim.
        target_len = padded_mask.numel()
        response_len_in_mask = int(mask_positions.numel())
        response_len = target_len - response_start_in_target  # extend to end of target
        response_start_full = response_start_in_target + 1
        prompt_len = response_start_full

        prompt_tokens = full_tokens[:prompt_len]
        response_tokens = full_tokens[response_start_full : response_start_full + response_len]
        # Slice + denormalise (wire-side advantages are divided by response_len
        # for SUM aggregation in Tinker; verl wants the raw value).
        old_log_probs = padded_logp[response_start_in_target : response_start_in_target + response_len]
        advantages = (
            padded_adv[response_start_in_target : response_start_in_target + response_len] * response_len_in_mask
        )
        resp_mask = padded_mask[response_start_in_target : response_start_in_target + response_len]

        # Optional client-supplied importance-sampling correction. verl's
        # ``compute_policy_loss_*`` family reads ``data["rollout_is_weights"]``
        # and folds it into the IS ratio (off-policy correction, behavior-
        # policy reweighting, custom clipping). Forwarding it lets users
        # configure the correction from the SDK side with no redeploy. Same
        # response-window slicing as old_log_probs / advantages.
        if "rollout_is_weights" in loss_inputs:
            padded_isw = _tensor_data_to_torch(loss_inputs["rollout_is_weights"]).float()
            rollout_isw = padded_isw[response_start_in_target : response_start_in_target + response_len]
        else:
            rollout_isw = None

        per_sample_fulls.append(full_tokens)
        per_sample_positions.append(torch.arange(valid_len, dtype=torch.long))
        per_sample_prompts.append(prompt_tokens)
        per_sample_responses_tok.append(response_tokens)
        per_sample_resp_mask.append(resp_mask)
        per_sample_old_logp.append(old_log_probs)
        per_sample_advantages.append(advantages)
        per_sample_rollout_isw.append(rollout_isw)

    # Pad the batch up to a multiple of the trainer's data-parallel world size.
    # verl splits each forward_backward batch across the FSDP ranks
    # (``chunk_tensordict``, tensordict_utils.py:341) and asserts the length is
    # evenly divisible by the chunk count. The RL sampler can drop a degenerate
    # trajectory (e.g. an empty response), leaving 31 sequences for an 8-way
    # split -> ``AssertionError('... got 31 and 8')``. Append zero-loss filler
    # rows — a clone of the first real sample with response_mask / advantages /
    # old_log_probs zeroed — so they carry valid tokens for the rmpad and
    # token-accounting invariants (``no_padding_2_padding`` at padding.py:131)
    # but contribute no gradient: ppo_loss masks them out, and batch_num_tokens
    # is ``loss_mask.sum()`` which the zeroed rows leave unchanged. The filler
    # rows are never read back — the response loop iterates the original
    # ``datums`` and the metrics are batch-mean aggregates.
    pad_n = (-len(datums)) % max(1, pad_to_multiple)
    if pad_n:
        real_has_isw = any(w is not None for w in per_sample_rollout_isw)
        for _ in range(pad_n):
            per_sample_fulls.append(per_sample_fulls[0].clone())
            per_sample_positions.append(per_sample_positions[0].clone())
            per_sample_prompts.append(per_sample_prompts[0].clone())
            per_sample_responses_tok.append(per_sample_responses_tok[0].clone())
            per_sample_resp_mask.append(torch.zeros_like(per_sample_resp_mask[0]))
            per_sample_old_logp.append(torch.zeros_like(per_sample_old_logp[0]))
            per_sample_advantages.append(torch.zeros_like(per_sample_advantages[0]))
            per_sample_rollout_isw.append(torch.zeros_like(per_sample_old_logp[0]) if real_has_isw else None)

    B = len(per_sample_fulls)
    max_seq = max(t.numel() for t in per_sample_fulls)
    max_resp = max(t.numel() for t in per_sample_responses_tok)

    # Nested (jagged) tensors: the rmpad path keys off input_ids.is_nested and
    # uses input_ids.offsets() to drive flash_attn_varlen.
    input_ids_nt = torch.nested.as_nested_tensor(per_sample_fulls, layout=torch.jagged)
    position_ids_nt = torch.nested.as_nested_tensor(per_sample_positions, layout=torch.jagged)
    prompts_nt = torch.nested.as_nested_tensor(per_sample_prompts, layout=torch.jagged)
    responses_nt = torch.nested.as_nested_tensor(per_sample_responses_tok, layout=torch.jagged)

    # Rectangular padded tensors. response_mask/old_log_probs/advantages are
    # response-only; ppo_loss runs ``data.select(...).to_padded_tensor()`` on
    # them, which is a no-op if they're already rectangular.
    attention_mask = torch.zeros(B, max_seq, dtype=torch.long)
    response_mask = torch.zeros(B, max_resp, dtype=torch.float32)
    old_log_probs = torch.zeros(B, max_resp, dtype=torch.float32)
    advantages = torch.zeros(B, max_resp, dtype=torch.float32)
    for i in range(B):
        seq = per_sample_fulls[i].numel()
        resp = per_sample_responses_tok[i].numel()
        attention_mask[i, :seq] = 1
        response_mask[i, :resp] = per_sample_resp_mask[i]
        old_log_probs[i, :resp] = per_sample_old_logp[i]
        advantages[i, :resp] = per_sample_advantages[i]

    # Optional rollout_is_weights: all-or-nothing per batch. Mixed presence
    # is rejected because a silent broadcast (e.g. defaulting missing samples
    # to 1.0) would mask client bugs where some Datums dropped the field.
    n_with_isw = sum(1 for w in per_sample_rollout_isw if w is not None)
    if 0 < n_with_isw < B:
        raise HTTPException(
            status_code=422,
            detail=(
                f"rollout_is_weights present on {n_with_isw}/{B} datums; mixed presence is "
                "ambiguous — pack the field on every datum or none."
            ),
        )
    td_fields = {
        "input_ids": input_ids_nt,
        "position_ids": position_ids_nt,
        "prompts": prompts_nt,
        "responses": responses_nt,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        # forward_backward_batch reads data["loss_mask"].sum() to compute
        # batch_num_tokens; matches left_right_2_no_padding behavior.
        "loss_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if n_with_isw == B:
        rollout_isw = torch.zeros(B, max_resp, dtype=torch.float32)
        for i in range(B):
            resp = per_sample_responses_tok[i].numel()
            rollout_isw[i, :resp] = per_sample_rollout_isw[i]
        td_fields["rollout_is_weights"] = rollout_isw

    td = TensorDict(
        td_fields,
        batch_size=[B],
    )
    # Verl meta — set what the trainer would normally set in _update_actor.
    from verl.utils import tensordict_utils as tu

    # Use the padded batch size B (== mini_batch_size when no padding was
    # needed) so the whole padded batch is treated as one mini-/global batch.
    tu.assign_non_tensor_data(td, "mini_batch_size", B)
    tu.assign_non_tensor_data(td, "global_batch_size", B)
    tu.assign_non_tensor_data(td, "epochs", 1)
    tu.assign_non_tensor_data(td, "seed", 42)
    tu.assign_non_tensor_data(td, "temperature", float(temperature))
    tu.assign_non_tensor_data(td, "calculate_entropy", False)
    tu.assign_non_tensor_data(td, "distillation_use_topk", False)
    tu.assign_non_tensor_data(td, "compute_loss", True)
    tu.assign_non_tensor_data(td, "dataloader_kwargs", {"shuffle": False})
    # Pin max_response_len so ``no_padding_2_padding`` pads the model log_probs
    # to the same width as our rectangular response-aligned tensors
    # (response_mask / old_log_probs / advantages). Without this it defaults to
    # ``response_lens.max()`` of the *current micro-batch*, which can be smaller
    # than the whole-batch max and yields a (B, micro_max) vs (B, global_max)
    # shape mismatch in ``ppo_loss``.
    tu.assign_non_tensor_data(td, "max_response_len", max_resp)
    return td


def _datums_to_sft_td(datums, mini_batch_size: int, temperature: float = 1.0, pad_to_multiple: int = 1) -> TensorDict:
    """SFT-shape TensorDict for verl's ``sft_loss`` (cross_entropy on the wire).

    Dispatches on ``target_tokens`` shape:
      - 1-D ``(valid_len-1,)`` → plain SFT: build a TD that verl's
        ``sft_loss`` consumes (loss_mask + log_probs).
      - 2-D ``(valid_len-1, K)`` → top-K teacher distillation (SDFT,
        off_policy_reasoning, prompt_distillation): build a TD with
        ``teacher_ids`` + ``teacher_logprobs`` and
        ``distillation_use_topk=True`` so verl's engine calls our
        branching loss as an in-forward logit processor.

    Cookbook SFT Datum carries:
      - ``model_input.to_ints()`` → tokens[:-1]                (length valid_len-1)
      - ``loss_fn_inputs["target_tokens"]``                    1-D length valid_len-1
                                                               OR 2-D (valid_len-1, K) for distillation
      - ``loss_fn_inputs["weights"]``                          same rank as target_tokens; for 1-D
        case 1.0 on response positions, for 2-D case renormalised teacher
        probability over the top-K slots (0 for invalid/masked slots).

    verl's ``sft_loss`` (workers/utils/losses.py:28) in NO_PADDING mode:
      log_prob_flatten = log_prob.values()
      loss_mask_flatten = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
      loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size

    So ``data["loss_mask"]`` is a nested tensor of shape (B, j1=valid_len) per
    sample, and verl rolls it left by one so that ``log_prob[i] =
    logP(token[i+1]|...)`` aligns with the mask. The roll-by-(-1) means
    ``loss_mask_post[i] = loss_mask[(i+1) % n]``. We want
    ``loss_mask_post[i] = weights[i]`` for i in ``[0, valid_len-1)``, which gives:

      loss_mask[i+1] = weights[i]   for i in [0, valid_len-1)
      loss_mask[0]   = 0.0          (wraps to the unused last position)
    """
    if not datums:
        raise HTTPException(
            status_code=422,
            detail="Critic request carried no datums; forward_input.data must be non-empty.",
        )

    # Probe the first datum to decide which translator to use. All datums in a
    # batch must agree on shape (forward_backward sends a homogeneous list).
    first_target = _tensor_data_to_torch(datums[0].loss_fn_inputs["target_tokens"])
    if first_target.dim() == 2:
        return _datums_to_topk_distill_td(
            datums,
            mini_batch_size=mini_batch_size,
            temperature=temperature,
            pad_to_multiple=pad_to_multiple,
        )

    per_sample_fulls = []
    per_sample_positions = []
    per_sample_loss_masks = []

    for d in datums:
        prefix = list(d.model_input.to_ints())
        target = _tensor_data_to_torch(d.loss_fn_inputs["target_tokens"]).long()
        if target.dim() != 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"_datums_to_sft_td (1-D branch) received target_tokens with shape {tuple(target.shape)}; "
                    "all datums in a batch must agree on dimensionality."
                ),
            )
        if "weights" in d.loss_fn_inputs:
            weights = _tensor_data_to_torch(d.loss_fn_inputs["weights"]).float()
        elif "mask" in d.loss_fn_inputs:
            # tinker_cookbook RL data_processing uses ``mask`` with the same
            # semantics (1.0 on response). Either field works.
            weights = _tensor_data_to_torch(d.loss_fn_inputs["mask"]).float()
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "SFT Datum missing loss_fn_inputs['weights'] (or 'mask'). "
                    "cross_entropy datums must carry one of these to mark the "
                    "response window."
                ),
            )
        # Tinker wire contract: target_tokens = tokens[1:] and weights aligns
        # one-to-one with target_tokens, so both must be length len(prefix) =
        # valid_len - 1. Catch the mismatch loudly — otherwise the silent
        # ``loss_mask[1:] = weights[: valid_len - 1]`` truncation below would
        # quietly corrupt the loss.
        prefix_len = len(prefix)
        if target.numel() != prefix_len or weights.numel() != prefix_len or target.numel() == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"SFT Datum shape mismatch: prefix len={prefix_len}, "
                    f"target_tokens len={target.numel()}, weights len={weights.numel()}. "
                    "All three must equal valid_len - 1."
                ),
            )

        full = torch.tensor(prefix + [int(target[-1].item())], dtype=torch.long)
        valid_len = full.numel()
        loss_mask = torch.zeros(valid_len, dtype=torch.float32)
        loss_mask[1:] = weights

        per_sample_fulls.append(full)
        per_sample_positions.append(torch.arange(valid_len, dtype=torch.long))
        per_sample_loss_masks.append(loss_mask)

    # World-size divisibility padding (see _datums_to_update_actor_td). Filler
    # rows clone the first sample with an all-zero loss_mask, so sft_loss masks
    # them out and batch_num_tokens (loss_mask.sum()) is unchanged.
    pad_n = (-len(datums)) % max(1, pad_to_multiple)
    for _ in range(pad_n):
        per_sample_fulls.append(per_sample_fulls[0].clone())
        per_sample_positions.append(per_sample_positions[0].clone())
        per_sample_loss_masks.append(torch.zeros_like(per_sample_loss_masks[0]))

    B = len(per_sample_fulls)
    td = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(per_sample_fulls, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(per_sample_positions, layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(per_sample_loss_masks, layout=torch.jagged),
        },
        batch_size=[B],
    )
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor_data(td, "mini_batch_size", B)
    tu.assign_non_tensor_data(td, "global_batch_size", B)
    tu.assign_non_tensor_data(td, "epochs", 1)
    tu.assign_non_tensor_data(td, "seed", 42)
    tu.assign_non_tensor_data(td, "temperature", float(temperature))
    tu.assign_non_tensor_data(td, "calculate_entropy", False)
    tu.assign_non_tensor_data(td, "compute_loss", True)
    tu.assign_non_tensor_data(td, "dataloader_kwargs", {"shuffle": False})
    # Sentinel read by engine._make_branching_loss to dispatch to sft_loss
    # instead of ppo_loss. RL TDs (built by _datums_to_update_actor_td)
    # leave this key absent and fall through to the ppo branch.
    tu.assign_non_tensor_data(td, "__loss_mode__", "sft")
    return td


def _datums_to_topk_distill_td(
    datums, mini_batch_size: int, temperature: float = 1.0, pad_to_multiple: int = 1
) -> TensorDict:
    """Top-K teacher-distillation TD for cookbook SDFT / off_policy_reasoning
    / prompt_distillation flows.

    Cookbook builds Datums with:
      - ``loss_fn_inputs["target_tokens"]`` shape ``(valid_len-1, K)`` —
        teacher's top-K token IDs at each completion position.
      - ``loss_fn_inputs["weights"]``       shape ``(valid_len-1, K)`` —
        renormalised teacher probability over those K tokens (the K weights
        sum to 1 at response positions, all zeros at prompt / masked /
        invalid-slot positions).

    The resulting per-position loss in cookbook semantics:
        L_t = -Σ_k weights[t,k] · log_student(target_tokens[t,k])

    verl's FSDP engine exposes a per-TD ``distillation_use_topk`` non-tensor
    flag (``transformer_impl.py:1057, 1105``) that triggers an in-forward
    logit-processor call. The processor receives the student logits while
    they're still in scope, returns a dict of ``(1, total_nnz)``-shaped
    tensors which the engine stashes into ``model_output``. The final
    loss reads back from there.

    We ride this hook without enabling verl's full distillation framework
    (which would require teacher_models config). The engine's
    branching loss_fn (engine._make_branching_loss) implements the
    in-forward processor and the final-loss aggregation. This translator
    just packs the cookbook's tensors into the shape verl expects:

      - ``teacher_ids``        nested (B, j1, K)  long
      - ``teacher_logprobs``  nested (B, j1, K)  float — log of
        weights, with invalid slots (weights==0) clamped to a large
        negative value so exp(.) reads 0 (DistillationLossConfig default
        is -10.0 elsewhere in verl).
      - ``loss_mask``               nested (B, j1)     float — 1.0 at any
        position with at least one nonzero teacher weight, 0.0 elsewhere.
      - Plus the same shape contract as the 1-D SFT path:
        ``input_ids``, ``position_ids``, ``loss_mask``.

    The j1 axis aligns with verl's ``input_ids.offsets()`` over the full
    valid_len, same convention used by verl's ``log_probs`` jagged tensor
    in ``prepare_model_outputs``. To match this we pad both teacher
    tensors with one leading row of zeros (matching ``loss_mask[0]=0`` so
    the roll-by-(-1) discards the wrap position).
    """
    if not datums:
        raise HTTPException(
            status_code=422,
            detail="Critic request carried no datums; forward_input.data must be non-empty.",
        )

    per_sample_fulls = []
    per_sample_positions = []
    per_sample_loss_masks = []
    per_sample_topk_ids = []
    per_sample_topk_log_probs = []
    LOG_PROB_MIN_CLAMP = -10.0  # matches DistillationLossConfig default

    for d in datums:
        prefix = list(d.model_input.to_ints())
        target = _tensor_data_to_torch(d.loss_fn_inputs["target_tokens"]).long()
        if target.dim() != 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    "_datums_to_topk_distill_td expects 2-D target_tokens "
                    f"shape (N, K); got shape {tuple(target.shape)}."
                ),
            )
        if "weights" not in d.loss_fn_inputs:
            raise HTTPException(
                status_code=422,
                detail="top-K distillation Datum requires loss_fn_inputs['weights'].",
            )
        weights = _tensor_data_to_torch(d.loss_fn_inputs["weights"]).float()
        if weights.shape != target.shape:
            raise HTTPException(
                status_code=422,
                detail=f"weights shape {tuple(weights.shape)} must match target_tokens shape {tuple(target.shape)}.",
            )
        # Tinker contract: target/weights are length valid_len-1, indexed in
        # target-space (= input-space shifted by 1).
        prefix_len = len(prefix)
        target_len, K = target.shape
        if target_len != prefix_len:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"top-K Datum shape mismatch: prefix len={prefix_len}, "
                    f"target_tokens shape={tuple(target.shape)}. "
                    "Target_len must equal valid_len - 1."
                ),
            )
        if target_len == 0:
            raise HTTPException(
                status_code=422,
                detail="target cannot be empty",
            )

        # Build the full input_ids: prefix + [target[-1, 0]] (use the top-1
        # teacher token at the last position as the "final" token; only the
        # prefix matters for model prediction at preceding positions, and
        # the last position is unused — loss_mask[0]=0 wraps to it).
        final_tok = int(target[target_len - 1, 0].item())
        full = torch.tensor(prefix + [final_tok], dtype=torch.long)
        valid_len = full.numel()

        # loss_mask: 1.0 at every position with any nonzero teacher weight,
        # shifted to align with verl's roll-by-(-1) convention. Same as 1-D
        # path: loss_mask[i+1] = (weights[i, :].sum() > 0) for i in [0, prefix_len),
        # loss_mask[0] = 0.
        per_pos_active = (weights.sum(dim=-1) > 0).float()  # (target_len,)
        loss_mask = torch.zeros(valid_len, dtype=torch.float32)
        loss_mask[1:] = per_pos_active

        # teacher_ids / teacher_logprobs: pad with a zero row at
        # position 0 (wrap slot) so the j1 axis matches input_ids' valid_len.
        # verl rolls these by -1 alongside the loss_mask to align with the
        # student logits at predict-next-token positions.
        ids_padded = torch.zeros(valid_len, K, dtype=torch.long)
        ids_padded[1:] = target
        lp_padded = torch.full((valid_len, K), LOG_PROB_MIN_CLAMP, dtype=torch.float32)
        # log(weights) with -10.0 clamp on zeros so exp() yields ~0.
        log_weights = torch.where(
            weights > 0, weights.clamp_min(1e-12).log(), torch.full_like(weights, LOG_PROB_MIN_CLAMP)
        )
        lp_padded[1:] = log_weights

        per_sample_fulls.append(full)
        per_sample_positions.append(torch.arange(valid_len, dtype=torch.long))
        per_sample_loss_masks.append(loss_mask)
        per_sample_topk_ids.append(ids_padded)
        per_sample_topk_log_probs.append(lp_padded)

    # World-size divisibility padding (see _datums_to_update_actor_td). Filler
    # rows clone the first sample with an all-zero loss_mask; the topk_ids /
    # topk_log_probs values are irrelevant because the masked-out positions
    # contribute no loss and don't change batch_num_tokens (loss_mask.sum()).
    pad_n = (-len(datums)) % max(1, pad_to_multiple)
    for _ in range(pad_n):
        per_sample_fulls.append(per_sample_fulls[0].clone())
        per_sample_positions.append(per_sample_positions[0].clone())
        per_sample_loss_masks.append(torch.zeros_like(per_sample_loss_masks[0]))
        per_sample_topk_ids.append(per_sample_topk_ids[0].clone())
        per_sample_topk_log_probs.append(per_sample_topk_log_probs[0].clone())

    B = len(per_sample_fulls)
    td = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(per_sample_fulls, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(per_sample_positions, layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(per_sample_loss_masks, layout=torch.jagged),
            "teacher_ids": torch.nested.as_nested_tensor(per_sample_topk_ids, layout=torch.jagged),
            "teacher_logprobs": torch.nested.as_nested_tensor(per_sample_topk_log_probs, layout=torch.jagged),
        },
        batch_size=[B],
    )
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor_data(td, "mini_batch_size", B)
    tu.assign_non_tensor_data(td, "global_batch_size", B)
    tu.assign_non_tensor_data(td, "epochs", 1)
    tu.assign_non_tensor_data(td, "seed", 42)
    tu.assign_non_tensor_data(td, "temperature", float(temperature))
    tu.assign_non_tensor_data(td, "calculate_entropy", False)
    tu.assign_non_tensor_data(td, "compute_loss", True)
    tu.assign_non_tensor_data(td, "dataloader_kwargs", {"shuffle": False})
    # Enables top-K distillation loss computation. Two engine paths:
    # - Fused (use_fused_kernels=True, e.g. VeOmni): ForCausalLMLoss routes
    #   to chunk_topk_distill_function, computing forward-KL without
    #   materializing full [B, L, V] logits.
    # - Eager (use_fused_kernels=False, e.g. FSDP): the branching loss's
    #   in-forward logit processor receives materialized student_logits and
    #   computes top-K weighted CE inline.
    tu.assign_non_tensor_data(td, "distillation_use_topk", True)
    # Sentinel read by branching_loss to route the final-loss call (and, in
    # the eager path, the logit-processor call) to the topk_distill arm.
    tu.assign_non_tensor_data(td, "__loss_mode__", "topk_distill")
    return td


def _datums_to_forward_td(datums, pad_to_multiple: int = 1) -> TensorDict:
    """Subset of _datums_to_update_actor_td for /api/v1/forward (compute_log_prob).

    The forward request only carries ``target_tokens`` — no response/advantage
    info — so we treat the full sequence as eligible (loss_mask = ones).
    Still needs the rmpad-friendly nested shape, since the underlying
    ``forward_backward_batch`` keys off ``input_ids.offsets()``.
    """
    if not datums:
        raise HTTPException(
            status_code=422,
            detail="Critic request carried no datums; forward_input.data must be non-empty.",
        )

    per_sample_fulls = []
    per_sample_positions = []
    per_sample_losses = []

    for d in datums:
        prefix = list(d.model_input.to_ints())
        target_tokens = _tensor_data_to_torch(d.loss_fn_inputs["target_tokens"]).long()
        if target_tokens.numel() == 0:
            raise HTTPException(
                status_code=422,
                detail="Received empty target",
            )
        full_tokens = torch.tensor(prefix + [int(target_tokens[-1].item())], dtype=torch.long)
        valid_len = full_tokens.numel()
        per_sample_fulls.append(full_tokens)
        per_sample_positions.append(torch.arange(valid_len, dtype=torch.long))
        per_sample_losses.append(torch.ones(valid_len, dtype=torch.float32))

    # World-size divisibility padding, same rationale as
    # _datums_to_update_actor_td: this compute_log_prob forward is also split
    # across the FSDP ranks. Filler rows clone the first sample; their logprobs
    # are never read (the response loop iterates the original datums and guards
    # ``i < len(train_logprobs_per_sample)``).
    pad_n = (-len(datums)) % max(1, pad_to_multiple)
    for _ in range(pad_n):
        per_sample_fulls.append(per_sample_fulls[0].clone())
        per_sample_positions.append(per_sample_positions[0].clone())
        per_sample_losses.append(per_sample_losses[0].clone())

    B = len(per_sample_fulls)
    max_seq = max(t.numel() for t in per_sample_fulls)

    input_ids_nt = torch.nested.as_nested_tensor(per_sample_fulls, layout=torch.jagged)
    position_ids_nt = torch.nested.as_nested_tensor(per_sample_positions, layout=torch.jagged)
    loss_mask_nt = torch.nested.as_nested_tensor(per_sample_losses, layout=torch.jagged)

    attention_mask = torch.zeros(B, max_seq, dtype=torch.long)
    for i in range(B):
        attention_mask[i, : per_sample_fulls[i].numel()] = 1

    td = TensorDict(
        {
            "input_ids": input_ids_nt,
            "position_ids": position_ids_nt,
            "loss_mask": loss_mask_nt,
            "attention_mask": attention_mask,
        },
        batch_size=[B],
    )
    # Temperature is read as ``micro_batch["temperature"]`` in prepare_model_inputs
    # and expanded via ``expand_as_nested(temperature, input_ids)`` — must be a
    # scalar/1-D meta value, not a tensor inside the batch.
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor_data(td, "temperature", 1.0)
    # compute_loss=False makes infer_batch pass loss_function=None so we don't
    # require response_mask / old_log_probs / advantages for /api/v1/forward.
    tu.assign_non_tensor_data(td, "compute_loss", False)
    tu.assign_non_tensor_data(td, "calculate_entropy", False)
    return td
