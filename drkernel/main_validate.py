# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Stand-alone validation entry point for the DR.Kernel recipe.

Reuses the same `FullyAsyncRollouter.do_validate()` path that runs at
`test_freq` during training, but skips the trainer / message queue /
parameter-sync machinery entirely. The rollout workers load weights
themselves from `actor_rollout_ref.model.path` — point it at a merged
HF checkpoint (output of `scripts/merge_to_hf.sh`).

Run:
    python -m recipe.drkernel.main_validate \\
        --config-name=drkernel_kernel_trainer_native \\
        actor_rollout_ref.model.path=<merged_hf_dir> \\
        data.val_files=<val.parquet> \\
        ...

Outputs:
  - `val_metrics_<global_step>_<timestamp>.json` under
    `trainer.default_local_dir` with the same `val-core/*` / `val-aux/*`
    keys produced during training.
  - TensorBoard event file (if `trainer.logger` contains `"tensorboard"`).
"""

import json
import os
import re
import socket
import threading
from datetime import datetime
from pprint import pprint
from time import time

import hydra
import ray
from omegaconf import OmegaConf

# Use the DR.Kernel rollouter subclass so `val-core/<src>/reward/mean@N` is
# rewritten to the per-(traj, turn) mean (mirroring the training-side
# `critic/rewards/mean` overwrite). See
# `recipe/drkernel/trainer/drkernel_async_rollouter.py::_val_metrics_update`.
from recipe.drkernel import algos  # noqa: F401  # registers the "trloo" advantage estimator
from recipe.drkernel.trainer.drkernel_async_rollouter import DrKernelFullyAsyncRollouter

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking


def _extract_global_step(model_path: str) -> int:
    """Best-effort: pull `global_step_<N>` out of the model path so the
    metrics row in TB is plotted at the correct training step. Falls back
    to 0 when the path does not encode a step."""
    m = re.search(r"global_step_(\d+)", model_path or "")
    return int(m.group(1)) if m else 0


@ray.remote(num_cpus=1)
class DrKernelValidateTaskRunner:
    """Ray remote runner that spins up only the rollouter and runs one
    validation pass. No trainer, no message queue, no param sync."""

    def __init__(self):
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[VALIDATE] Starting DR.Kernel validation...")
        self._initialize_components(config)
        return self._run_validation(config)

    def _initialize_components(self, config) -> None:
        print(f"[VALIDATE] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[VALIDATE] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor

        print("[VALIDATE] Creating DrKernelFullyAsyncRollouter (rollout pool only)...")
        rollouter = DrKernelFullyAsyncRollouter.remote(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        self.components["rollouter"] = rollouter
        print("[VALIDATE] Rollouter created and initialized successfully")

    def _run_validation(self, config) -> dict:
        rollouter = self.components["rollouter"]

        print("[VALIDATE] Running do_validate()...")
        val_metrics = ray.get(rollouter.do_validate.remote())
        metrics = dict(val_metrics.metrics or {})
        timing = dict(val_metrics.timing_raw or {})

        # Per-prompt rollout table (only populated when
        # `actor_rollout_ref.rollout.val_kwargs.n > 1`; empty list otherwise).
        # Built by `DrKernelFullyAsyncRollouterImpl._compute_pass_at_n_and_per_prompt`
        # alongside the Pass@N metrics already merged into `metrics`.
        per_prompt: list = []
        try:
            per_prompt = ray.get(rollouter.get_per_prompt_table.remote()) or []
        except Exception as e:  # noqa: BLE001 — defensive: never block JSON dump on this
            print(f"[VALIDATE] get_per_prompt_table() raised {type(e).__name__}: {e}; writing aggregate metrics only.")

        print("[VALIDATE] Metrics:")
        pprint(metrics)
        print("[VALIDATE] Timing:")
        pprint(timing)
        if per_prompt:
            n_per_first = per_prompt[0].get("n_rollouts", "?") if per_prompt else "?"
            print(f"[VALIDATE] Per-prompt table: {len(per_prompt)} prompts x ~{n_per_first} rollouts each")

        # --- JSON dump ------------------------------------------------------
        out_dir = config.trainer.default_local_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.getcwd(), out_dir)
        os.makedirs(out_dir, exist_ok=True)

        global_step = _extract_global_step(config.actor_rollout_ref.model.path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"val_metrics_step{global_step}_{ts}.json")
        # `data.val_files` can come in as a plain str (single path) or a
        # ListConfig (list-form override). Coerce both to a JSON-safe value.
        val_files = config.data.val_files
        if OmegaConf.is_config(val_files):
            val_files = OmegaConf.to_container(val_files, resolve=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "model_path": config.actor_rollout_ref.model.path,
                    "val_files": val_files,
                    "metrics": metrics,
                    "timing": timing,
                    "per_prompt": per_prompt,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[VALIDATE] Wrote {out_path}")

        # --- Tracking (TensorBoard / console) -------------------------------
        # Build the same Tracking handle used by the trainer so the run shows
        # up at the same project/experiment in TB.
        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=list(config.trainer.logger),
            config=OmegaConf.to_container(config, resolve=True),
        )
        if metrics:
            logger.log(data=metrics, step=global_step)
        if timing:
            logger.log(data=timing, step=global_step)

        return {"metrics": metrics, "timing": timing, "out_path": out_path}


@hydra.main(config_path="config", config_name="drkernel_kernel_trainer_native", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    start_time = time()
    auto_set_device(config)

    # Mirror main.py: keep the rollout block mirrored into actor_rollout_ref
    # so the rollouter sees the right nnodes / gpus_per_node, then resolve
    # interpolations before `migrate_legacy_reward_impl` mutates the tree
    # (it deletes `reward_model` after forwarding `reward_kwargs`, which
    # would otherwise raise InterpolationKeyError when the kernel reward
    # manager later dereferences ${reward_model.*}).
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    OmegaConf.resolve(config)
    config = migrate_legacy_reward_impl(config)

    run_ppo(config, task_runner_class=DrKernelValidateTaskRunner)
    print(f"[VALIDATE] total time: {time() - start_time:.2f} seconds")

    # Explicit teardown — the validation path calls `do_validate()` once
    # and returns, so the rollout workers (vLLM replicas + reward-loop
    # actors spun up by `init_workers`) never see a natural shutdown the
    # way they would in training when `fit()` runs out of steps. Without
    # this, the driver hangs on Python interpreter shutdown waiting for
    # those non-daemon Ray actors. `ray.shutdown()` cancels them; the
    # subsequent `os._exit` is a hammer to bypass any lingering atexit
    # handlers from vLLM/Ascend that have also been observed to wedge.
    try:
        ray.shutdown()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup
        print(f"[VALIDATE] ray.shutdown() raised {type(e).__name__}: {e}")
    os._exit(0)


if __name__ == "__main__":
    main()
