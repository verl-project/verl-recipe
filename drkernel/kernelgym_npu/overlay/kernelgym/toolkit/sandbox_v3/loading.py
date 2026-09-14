# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""KernelBench model loading helpers (toolkit layer)."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import torch
import torch.nn as nn


def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point: str = "Model"
) -> tuple[nn.Module, callable, callable]:
    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None
    try:
        exec(model_original_src, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None
    get_init_inputs_fn = context.get("get_init_inputs")

    # sandbox_v3 mirrors verify_br_430.resolve_input_provider: prefer the
    # multi-shape get_input_groups() provider, but fall back to
    # wrapping a single get_inputs() as one group.
    raw_get_input_groups = context.get("get_input_groups")
    raw_get_inputs = context.get("get_inputs")

    def get_inputs_fn():
        if callable(raw_get_input_groups):
            return raw_get_input_groups()
        if callable(raw_get_inputs):
            return [raw_get_inputs()]
        raise AttributeError("Reference module must define get_inputs() or get_input_groups()")

    Model = context.get(entry_point)

    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src: str, entry_point: str = "ModelNew"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(model_custom_src)
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    return ModelNew, temp_file


def load_custom_model(model_custom_src: str, context: dict, build_directory: str = None) -> nn.Module:
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n") + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    del curr_context
    with torch.npu.device(device):
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats(device=device)
        torch.npu.synchronize(device=device)
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)
