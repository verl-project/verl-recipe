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

"""Colocated engine helpers for the Tinker HTTP surface."""

import asyncio
import json
import logging
import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verl.workers.engine_workers_tinker import OptimStepParams

from .data.datum_processing import (
    _coerce_verl_metrics_to_floats,
    _datums_to_forward_td,
    _datums_to_sft_td,
    _datums_to_update_actor_td,
)


logger = logging.getLogger(__name__)

GLOBAL_SESSION_ID = "verl-remote-actor"
GLOBAL_MODEL_ID = "verl-remote-actor-model"
GLOBAL_SAMPLING_SESSION_ID = "verl-remote-actor-sampler"
STATE_METADATA_FILE = "metadata.json"


def get_configured_model_name(engine) -> str:
    """Return the client-visible model name configured for this server."""
    return str(engine.config.server.get("model_name") or engine.config.actor_rollout_ref.model.path)


def get_supported_models(engine) -> dict:
    """Return Tinker server capability metadata."""
    try:
        model_name = get_configured_model_name(engine)
    except Exception:
        model_name = None
    try:
        max_ctx = int(engine.config.actor_rollout_ref.rollout.max_response_length)
    except Exception:
        max_ctx = 4096
    return {"supported_models": [{"model_name": model_name, "max_context_length": max_ctx}]}


def get_base_model_name(engine, model_to_base_model: dict[str, str]) -> str:
    """Return the client-visible base model/tokenizer id."""
    return model_to_base_model.get(GLOBAL_MODEL_ID, get_configured_model_name(engine))


def _format_topk_prompt_logprobs(
    prompt_ids: Any,
    prompt_logprobs: Any,
) -> list[list[tuple[int, float]]] | None:
    """Convert rollout prompt-logprob arrays to tinker.SampleResponse shape.

    vLLM/SGLang replicas expose prompt ids and logprobs as parallel arrays:
    ``[prompt_position][topk_rank]``. The current tinker SDK expects each
    position to carry ``(token_id, logprob)`` pairs directly.
    """
    if prompt_ids is None or prompt_logprobs is None:
        return None

    out: list[list[tuple[int, float]]] = []
    for ids_row, logprobs_row in zip(prompt_ids, prompt_logprobs):
        if ids_row is None or logprobs_row is None:
            out.append([])
            continue

        row: list[tuple[int, float]] = []
        for token_id, logprob in zip(ids_row, logprobs_row):
            if token_id is None or logprob is None:
                continue
            row.append((int(token_id), float(logprob)))
        out.append(row)

    return out


def _normalize_tinker_stop_reason(stop_reason: Any) -> str:
    """Map rollout backend stop reasons to Tinker's public StopReason literals."""
    if stop_reason in (None, "stop", "completed"):
        return "stop"
    return "length"


def _merge_worker_metric_dicts(raw_metrics):
    """Merge metric dictionaries returned by Ray workers.

    During an optimization step, VeRL collects a metric dictionary from
    every Ray worker. Only the worker that actually processed the batch
    returns non-empty metrics; the others return empty dictionaries.

    This function filters out empty results and aggregates values from
    non-empty dictionaries into a single mapping of
    metric_name -> list[metric_value].
    """
    if not isinstance(raw_metrics, (list, tuple)):
        return raw_metrics

    merged = {}
    for worker_metrics in raw_metrics:
        if not worker_metrics:
            continue
        for key, value in worker_metrics.items():
            merged.setdefault(key, []).append(value)
    return merged


def _adam_params_to_optim_step_params(adam_params) -> OptimStepParams | None:
    if adam_params is None:
        return None

    grad_clip_norm = getattr(adam_params, "grad_clip_norm", None)
    if grad_clip_norm is not None:
        warnings.warn(
            "grad_clip_norm is accepted for Tinker API compatibility but is not used by verl-recipes.",
            UserWarning,
            stacklevel=2,
        )

    return OptimStepParams(
        lr=adam_params.learning_rate,
        eps=adam_params.eps,
        betas=(adam_params.beta1, adam_params.beta2),
        weight_decay=adam_params.weight_decay,
    )


async def forward(engine, datums) -> dict:
    td = _datums_to_forward_td(datums, pad_to_multiple=engine.world_size)
    result_td = await asyncio.to_thread(engine.compute_log_prob, td)

    outputs = []
    log_probs = result_td.get("log_probs")
    if log_probs.is_nested:
        per_sample = list(log_probs.unbind())
    else:
        per_sample = [log_probs[i] for i in range(log_probs.shape[0])]
    per_sample = per_sample[: len(datums)]
    if len(per_sample) != len(datums):
        raise RuntimeError(f"compute_log_prob returned {len(per_sample)} samples but request carried {len(datums)}.")

    for lp_t, datum in zip(per_sample, datums):
        lp = lp_t.detach().float().cpu()
        expected_valid_len = len(datum.model_input.to_ints()) + 1
        if lp.numel() != expected_valid_len:
            raise RuntimeError(
                f"compute_log_prob returned {lp.numel()} log-probs for valid_len={expected_valid_len}; "
                "expected full input length so trailing wrap-around drop matches Tinker's target_len contract."
            )
        lp_list = lp[:-1].tolist()
        outputs.append({"logprobs": {"data": lp_list, "dtype": "float32", "shape": [len(lp_list)]}})

    return {
        "type": "forward_backward",
        "loss_fn_output_type": "log_probs",
        "loss_fn_outputs": outputs,
        "metrics": {},
    }


async def forward_backward(engine, datums, loss_fn_name: str) -> dict:
    if loss_fn_name == "cross_entropy":
        td = _datums_to_sft_td(datums, mini_batch_size=len(datums), pad_to_multiple=engine.world_size)
    else:
        td = _datums_to_update_actor_td(datums, mini_batch_size=len(datums), pad_to_multiple=engine.world_size)

    result_td = await asyncio.to_thread(engine.forward_backward, td)
    from verl.utils import tensordict_utils as tu

    raw_metrics = tu.get(result_td, "metrics", {}) if result_td is not None else {}
    metrics = _coerce_verl_metrics_to_floats(raw_metrics or {})
    if not any(k.startswith("loss:") for k in metrics):
        metrics["loss:mean"] = 0.0

    logprobs_per_sample = None
    if result_td is not None:
        log_probs = result_td.get("log_probs", None)
        if log_probs is not None:
            if log_probs.is_nested:
                logprobs_per_sample = list(log_probs.unbind())
            else:
                logprobs_per_sample = [log_probs[i] for i in range(log_probs.shape[0])]

    outputs = []
    for i, datum in enumerate(datums):
        if logprobs_per_sample is not None and i < len(logprobs_per_sample):
            lp_t = logprobs_per_sample[i].detach().float().cpu()
            expected_valid_len = len(datum.model_input.to_ints()) + 1
            target_len = len(datum.loss_fn_inputs["target_tokens"].data)
            if lp_t.numel() == expected_valid_len:
                lp_list = lp_t[:-1].tolist()
            elif lp_t.numel() == target_len:
                lp_list = lp_t.tolist()
            else:
                lp_list = [0.0] * target_len
        elif "logprobs" in datum.loss_fn_inputs:
            lp_list = list(datum.loss_fn_inputs["logprobs"].data)
        else:
            target_len = len(datum.loss_fn_inputs["target_tokens"].data)
            lp_list = [0.0] * target_len
        outputs.append({"logprobs": {"data": lp_list, "dtype": "float32", "shape": [len(lp_list)]}})

    return {
        "type": "forward_backward",
        "loss_fn_output_type": loss_fn_name,
        "loss_fn_outputs": outputs,
        "metrics": metrics,
    }


async def optim_step(engine, adam_params=None) -> dict:
    optim_step_params = _adam_params_to_optim_step_params(adam_params)
    metrics = await asyncio.to_thread(engine.optim_step, optim_step_params)
    return {
        "type": "optim_step",
        "metrics": _coerce_verl_metrics_to_floats(_merge_worker_metric_dicts(metrics) or {}),
    }


async def save_weights_for_sampler(engine, named: bool) -> dict:
    await asyncio.to_thread(engine.update_weights)
    if named:
        path = f"tinker://verl-tinker/weights/step-{uuid.uuid4().hex[:8]}"
        return {"type": "save_weights_for_sampler", "path": path, "sampling_session_id": None}
    return {
        "type": "save_weights_for_sampler",
        "path": None,
        "sampling_session_id": GLOBAL_SAMPLING_SESSION_ID,
    }


def state_path_to_local(checkpoint_root: str, uri: str) -> str:
    tag = uri.rsplit("/", 1)[-1] if uri else uuid.uuid4().hex[:12]
    tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    return f"{checkpoint_root.rstrip('/')}/{tag}"


def state_metadata_path(local_dir: str) -> Path:
    return Path(local_dir) / STATE_METADATA_FILE


def load_state_metadata(
    checkpoint_root: str,
    saved_state_metadata: dict[str, dict[str, Any]],
    uri: str,
) -> dict[str, Any] | None:
    metadata = saved_state_metadata.get(uri)
    if metadata is not None:
        return dict(metadata)

    path = state_metadata_path(state_path_to_local(checkpoint_root, uri))
    if not path.exists():
        return None

    with path.open() as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid checkpoint metadata at {path}: expected object")
    saved_state_metadata[uri] = dict(metadata)
    return metadata


async def save_state(
    engine,
    checkpoint_root: str,
    saved_state_paths: dict[str, str],
    saved_state_metadata: dict[str, dict[str, Any]],
    state_metadata: dict[str, Any] | None,
    step: int,
    name: str | None,
) -> dict:
    tag = name or uuid.uuid4().hex[:12]
    uri = f"tinker://verl-tinker/state/{tag}"
    local_dir = state_path_to_local(checkpoint_root, uri)
    await asyncio.to_thread(engine.save_checkpoint, local_dir, step)
    saved_state_paths[uri] = local_dir
    if state_metadata is not None:
        saved_state_metadata[uri] = dict(state_metadata)
        state_metadata_path(local_dir).write_text(json.dumps(state_metadata) + "\n")
    logger.info(f"[tinker_router] save_weights uri={uri} -> {local_dir}")
    return {"type": "save_weights", "path": uri}


async def load_state(
    engine,
    checkpoint_root: str,
    saved_state_paths: dict[str, str],
    uri: str,
    load_optimizer: bool = True,
) -> dict:
    local_dir = saved_state_paths.get(uri) or state_path_to_local(checkpoint_root, uri)
    await asyncio.to_thread(engine.load_checkpoint, local_dir, zero_optimizer_grad=not load_optimizer)
    logger.info(f"[tinker_router] load_weights uri={uri} <- {local_dir}")
    return {"type": "load_weights", "path": uri}


def _apply_sampling_stop(sampling_params: dict[str, Any], stop: Any) -> None:
    """Translate Tinker SamplingParams.stop to rollout backend sampling params."""
    if stop is None:
        return
    if isinstance(stop, str):
        sampling_params["stop"] = stop
        sampling_params["include_stop_str_in_output"] = True
        return
    if not isinstance(stop, Sequence):
        raise TypeError(f"Unsupported sampling stop type: {type(stop).__name__}")

    stop_values = list(stop)
    if not stop_values:
        return
    if all(isinstance(value, str) for value in stop_values):
        sampling_params["stop"] = stop_values
        sampling_params["include_stop_str_in_output"] = True
        return
    if all(isinstance(value, int) for value in stop_values):
        sampling_params["stop_token_ids"] = stop_values
        sampling_params["include_stop_str_in_output"] = True
        return
    raise TypeError("Sampling stop sequences must contain only strings or only token ids")


async def sample(engine, req) -> dict:
    prompt_ids = list(req.prompt.to_ints())
    sp = req.sampling_params
    sampling_params = {
        "max_tokens": sp.max_tokens or 256,
        "temperature": float(sp.temperature),
        "top_p": float(sp.top_p),
        "top_k": int(sp.top_k),
        "n": int(req.num_samples),
        "logprobs": True,
    }
    _apply_sampling_stop(sampling_params, getattr(sp, "stop", None))
    if getattr(sp, "seed", None) is not None:
        sampling_params["seed"] = int(sp.seed)
    if req.prompt_logprobs:
        sampling_params["prompt_logprobs"] = 0
    if req.topk_prompt_logprobs:
        sampling_params["prompt_logprobs"] = int(req.topk_prompt_logprobs)

    result = await engine.generate(
        request_id=uuid.uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
    )
    token_ids = list(result.token_ids or [])
    log_probs = list(result.log_probs) if result.log_probs is not None else None
    tinker_stop_reason = _normalize_tinker_stop_reason(result.stop_reason)
    sequences = [
        {
            "stop_reason": tinker_stop_reason,
            "tokens": token_ids,
            "logprobs": log_probs,
        }
    ]

    extras = getattr(result, "extra_fields", None) or {}
    prompt_ids_raw = extras.get("prompt_ids")
    prompt_lp_raw = extras.get("prompt_logprobs")
    prompt_logprobs_out = None
    topk_prompt_logprobs_out = None
    if prompt_lp_raw is not None:
        if req.topk_prompt_logprobs:
            topk_prompt_logprobs_out = _format_topk_prompt_logprobs(prompt_ids_raw, prompt_lp_raw)
        else:
            prompt_logprobs_out = [(pos[0] if isinstance(pos, list) and pos else None) for pos in prompt_lp_raw]

    return {
        "type": "sample",
        "sequences": sequences,
        "prompt_logprobs": prompt_logprobs_out,
        "topk_prompt_logprobs": topk_prompt_logprobs_out,
    }
