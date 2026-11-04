# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""Pooled decoder activations and the three entropy features used in the paper."""

import re
from contextlib import contextmanager

import numpy as np
import torch


def _last_marker_end(ids, marker):
    for start in range(len(ids) - len(marker), -1, -1):
        if marker and ids[start : start + len(marker)] == marker:
            return start + len(marker)
    return len(ids)


def pool_hidden(hidden, input_ids, responses, response_mask, *, packed, pool_tokens, think_end_ids):
    """Pool actual token states (no log-probability shift), including the final </think>."""
    prompt_rows, response_rows = [], []
    offset = 0
    for i, (ids, response, mask) in enumerate(
        zip(input_ids.unbind(), responses.unbind(), response_mask.unbind(), strict=True)
    ):
        length = len(ids)
        states = hidden[0, offset : offset + length] if packed else hidden[i, :length]
        offset += length
        prompt_length = length - len(response)
        if prompt_length <= 0 or len(mask) != len(response):
            raise ValueError("POISE requires a nonempty prompt and aligned response mask")
        prompt_rows.append(states[:prompt_length][-pool_tokens:].mean(0).float())
        valid = mask.to(device=states.device, dtype=torch.bool)
        response_states = states[prompt_length:][valid]
        response_ids = response.to(device=states.device)[valid].tolist()
        end = _last_marker_end(response_ids, think_end_ids)
        selected = response_states[:end][-pool_tokens:]
        response_rows.append(selected.mean(0).float() if len(selected) else states.new_zeros(states.shape[-1]).float())
    return {"poise_prompt": torch.stack(prompt_rows), "poise_response": torch.stack(response_rows)}


@contextmanager
def capture_hidden(module, *, layer, input_ids, responses, response_mask, packed, pool_tokens, think_end_ids):
    """Keep only pooled vectors, and remove the hook even if forward fails."""
    while getattr(module, "module", None) is not None:
        module = module.module
    layers = getattr(getattr(module, "model", None), "layers", None)
    if layers is None or not 0 <= layer < len(layers):
        raise ValueError(f"POISE needs model.layers and a valid layer index, got {layer}")
    result = {}

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        result.update(
            pool_hidden(
                hidden.detach(),
                input_ids,
                responses,
                response_mask,
                packed=packed,
                pool_tokens=pool_tokens,
                think_end_ids=think_end_ids,
            )
        )

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield result
        if not result:
            raise RuntimeError("The POISE decoder hook was not called")
    finally:
        handle.remove()


def entropy_features(tokenizer, response_ids, entropies):
    """Match the reference implementation's decoded-text spans and offset fallback."""
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    think_span = think.span(1) if think else None
    answer_span = answer.span(1) if answer else None
    if answer is None and "</think>" in text:
        start = text.index("</think>") + len("</think>")
        start += len(text[start:]) - len(text[start:].lstrip())
        answer_span = (start, len(text)) if start < len(text) else None
    indices = []
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) != len(response_ids):
            raise ValueError("Decoded tokens do not align")
        for span in (think_span, answer_span):
            indices.append(
                [
                    i
                    for i, (start, end) in enumerate(encoded["offset_mapping"])
                    if span is not None and end > start and end > span[0] and start < span[1]
                ]
            )
    except (ValueError, TypeError, KeyError, NotImplementedError):
        counts = [
            min(len(response_ids), len(tokenizer.encode(text[s[0] : s[1]], add_special_tokens=False))) if s else 0
            for s in (think_span, answer_span)
        ]
        indices = [list(range(counts[0])), list(range(len(response_ids) - counts[1], len(response_ids)))]
    entropies = np.asarray(entropies, dtype=np.float32)
    return np.asarray(
        [
            float(entropies.mean()) if len(entropies) else 0.0,
            *(float(entropies[i].mean()) if i else 0.0 for i in indices),
        ],
        dtype=np.float32,
    )
