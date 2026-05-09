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
"""v1 generation-only smoke test for the recipe/dynamo backend.

Spins up a DynamoReplica via ``init_standalone`` on one node, waits for the
frontend to advertise the model, sends a chat completion, prints the
response, then tears everything down. No verl trainer involved.

Pass criterion: an HTTP 200 chat completion with a non-empty assistant
message.

Usage (inside the verl container, after ``ray start --head``):
    python recipe/dynamo/scripts/smoke_dynamo_v1.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --tp 1 \
        --gpus-per-node 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Optional

import aiohttp
import ray
from omegaconf import OmegaConf

from recipe.dynamo.dynamo_async_server import DynamoReplica
from verl.workers.config import HFModelConfig

logger = logging.getLogger("smoke_dynamo_v1")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)


def _build_rollout_config(
    *,
    tp: int,
    gpus_per_node: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
) -> Any:
    cfg = OmegaConf.create({
        "_target_": "verl.workers.config.RolloutConfig",
        "name": "dynamo",
        "mode": "async",
        "nnodes": 1,
        "n_gpus_per_node": gpus_per_node,
        "tensor_model_parallel_size": tp,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max(8192, max_model_len),
        "max_num_seqs": 256,
        "dtype": "bfloat16",
        "enforce_eager": enforce_eager,
        "enable_sleep_mode": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "free_cache_engine": True,
        "load_format": "auto",
        "logprobs_mode": "processed_logprobs",
        "scheduling_policy": "fcfs",
        "disable_log_stats": True,
        "engine_kwargs": {
            "dynamo": {
                "namespace": "verl_smoke",
                "router_mode": "round-robin",
            }
        },
    })
    return cfg


def _build_model_config(model_path: str) -> Any:
    return OmegaConf.create({
        "path": model_path,
        "tokenizer_path": model_path,
        "trust_remote_code": True,
        "external_lib": None,
        "lora_rank": 0,
        "lora": {"rank": 0, "merge": False},
    })


async def _wait_frontend_serves_model(
    server_address: str, timeout: float = 600.0
) -> None:
    """Poll /v1/models on the frontend until at least one model is listed."""
    url = f"http://{server_address}/v1/models"
    deadline = time.monotonic() + timeout
    last_err: Optional[str] = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ids = {m.get("id") for m in data.get("data", [])}
                        if ids:
                            logger.info("frontend /v1/models lists: %s", sorted(ids))
                            return
                        last_err = "no models listed"
                    else:
                        last_err = f"HTTP {resp.status}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(2.0)
    raise RuntimeError(
        f"frontend did not advertise any model within {timeout}s "
        f"(last error: {last_err})"
    )


async def _chat_completion(server_address: str, model_id: str, prompt: str) -> str:
    url = f"http://{server_address}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"chat/completions HTTP {resp.status}: {body[:500]}")
            data = await resp.json()
    logger.info("chat/completions response: %s", json.dumps(data, indent=2)[:1000])
    return data["choices"][0]["message"]["content"]


async def amain(args: argparse.Namespace) -> int:
    rollout_cfg = _build_rollout_config(
        tp=args.tp,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    model_cfg_dict = _build_model_config(args.model)
    # HFModelConfig has a __post_init__ that does heavy work (download tokenizer
    # etc.). Construct via omega_conf_to_dataclass so post-init runs once here
    # in the driver process.
    from verl.utils.config import omega_conf_to_dataclass

    model_cfg = omega_conf_to_dataclass(model_cfg_dict, dataclass_type=HFModelConfig)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    replica = DynamoReplica(
        replica_rank=0,
        config=rollout_cfg,
        model_config=model_cfg,
        gpus_per_node=args.gpus_per_node,
    )

    logger.info("init_standalone() — creating trainer placeholder workers + servers")
    t0 = time.monotonic()
    await replica.init_standalone()
    logger.info("init_standalone done in %.1fs", time.monotonic() - t0)

    server_address = replica.server_address
    logger.info("frontend at %s", server_address)

    served_model_name = (
        rollout_cfg.engine_kwargs.dynamo.get("served_model_name", None)
        or args.model
    )

    rc = 0
    try:
        await _wait_frontend_serves_model(server_address)
        text = await _chat_completion(
            server_address,
            served_model_name,
            "Reply in one short sentence: what is 2+2?",
        )
        logger.info("ASSISTANT: %s", text.strip())
        if not text.strip():
            logger.error("FAIL: assistant message is empty")
            rc = 2
        else:
            logger.info("PASS: chat completion returned non-empty content")
    finally:
        logger.info("shutting down replica")
        try:
            await replica._server_handle.shutdown.remote()  # type: ignore[attr-defined]
        except Exception:
            logger.exception("replica shutdown raised; ignoring")

    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--enforce-eager", action="store_true", default=False)
    args = p.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
