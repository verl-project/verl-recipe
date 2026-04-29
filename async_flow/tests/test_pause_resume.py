#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
"""
Test vLLM pause/resume at DP (replica) granularity

This test verifies:
1) inflight requests submitted before pause can finish
2) requests submitted after pause are blocked (ray.get timeout)
3) resume releases blocked requests
"""

import asyncio
import os
import time
from uuid import uuid4

import ray


def _make_prompt_ids(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    if hasattr(out, "input_ids"):
        out = out.input_ids
    if hasattr(out, "tolist"):
        out = out.tolist()
    if isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    if not isinstance(out, list) or (out and not isinstance(out[0], int)):
        raise TypeError(
            f"prompt_ids must be list[int], got: {type(out)} / sample={out[:5] if isinstance(out, list) else out}"
        )
    return out


async def main():
    # ==================== CONFIGURATION ====================
    MODEL_PATH = "YOUR_MODEL_PATH"  # change your model path
    GPUS_PER_NODE = int(os.environ.get("TEST_GPUS_PER_NODE", "2"))
    TP_SIZE = int(os.environ.get("TEST_TP_SIZE", "1"))
    ROLLOUT_NAME = os.environ.get("TEST_ROLLOUT_NAME", "vllm")

    NUM_INFLIGHT = int(os.environ.get("TEST_NUM_INFLIGHT", "4"))
    NUM_AFTER_PAUSE = int(os.environ.get("TEST_NUM_AFTER_PAUSE", "8"))

    PRE_PAUSE_RUN_SECS = float(os.environ.get("TEST_PRE_PAUSE_RUN_SECS", "1.0"))
    BLOCK_PROBE_TIMEOUT = float(os.environ.get("TEST_BLOCK_PROBE_TIMEOUT", "5.0"))

    PROMPT_LEN = int(os.environ.get("TEST_PROMPT_LEN", "256"))
    RESP_LEN = int(os.environ.get("TEST_RESP_LEN", "1024"))

    MAX_NUM_SEQS = int(os.environ.get("TEST_MAX_NUM_SEQS", "4"))
    MAX_BATCHED_TOKENS = int(os.environ.get("TEST_MAX_BATCHED_TOKENS", "2048"))

    print("=" * 80)
    print("vLLM Pause Test (wait_for_inflight_requests=true) - REAL rollout path (pause via Replica) [Ray-only]")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {GPUS_PER_NODE}, TP Size: {TP_SIZE}")
    print(f"NUM_INFLIGHT={NUM_INFLIGHT}, NUM_AFTER_PAUSE={NUM_AFTER_PAUSE}")
    print(f"PRE_PAUSE_RUN_SECS={PRE_PAUSE_RUN_SECS}, BLOCK_PROBE_TIMEOUT={BLOCK_PROBE_TIMEOUT}")
    print(f"PROMPT_LEN={PROMPT_LEN}, RESP_LEN={RESP_LEN}")
    print(f"MAX_NUM_SEQS={MAX_NUM_SEQS}, MAX_BATCHED_TOKENS={MAX_BATCHED_TOKENS}")
    print("=" * 80)

    # ==================== Initialize Ray ====================
    print("\n[1] Initializing Ray...")
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_VERSION": "0.13.0",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        # ==================== Create Config ====================
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = GPUS_PER_NODE
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = MODEL_PATH
        config.actor_rollout_ref.rollout.name = ROLLOUT_NAME
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE

        config.actor_rollout_ref.rollout.prompt_length = PROMPT_LEN
        config.actor_rollout_ref.rollout.response_length = RESP_LEN
        config.actor_rollout_ref.rollout.max_num_seqs = MAX_NUM_SEQS
        config.actor_rollout_ref.rollout.max_num_batched_tokens = MAX_BATCHED_TOKENS

        # ==================== Create Rollout Replica ====================
        print("\n[3] Creating rollout replica...")

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        from recipe.async_flow.vllm_rollout.vllm_async_server import AsyncFlowReplica

        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

        def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
            worker_dict_cls = RayClassWithInitArgs(
                cls=ray.remote(vLLMAsyncRollout),
                config=self.config,
                model_config=self.model_config,
                device_mesh=None,
            )

            return worker_dict_cls

        def register_version_reporter(self, server):
            pass

        vLLMAsyncRollout.register_version_reporter = register_version_reporter

        AsyncFlowReplica.get_ray_class_with_init_args = get_ray_class_with_init_args
        rollout_server_class = AsyncFlowReplica

        replica = rollout_server_class(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=GPUS_PER_NODE,
        )

        await replica.init_standalone()
        server_handle = replica._server_handle
        server_address = replica._server_address
        print(f"Server address: {server_address}")

        # ==================== Load Tokenizer ====================
        print("\n[4] Loading tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        # ==================== Prepare Prompts ====================
        print("\n[5] Preparing prompts...")
        base_prompts = [
            "Write a very long story about a brave knight and dragon.",
            "Explain the history of the Roman Empire in great detail.",
            "Describe quantum computing and its applications thoroughly.",
            "Write an essay about climate change and its global effects.",
        ]

        inflight_prompt_ids = [
            _make_prompt_ids(tokenizer, base_prompts[i % len(base_prompts)]) for i in range(NUM_INFLIGHT)
        ]
        after_pause_prompt_ids = [
            _make_prompt_ids(tokenizer, base_prompts[i % len(base_prompts)]) for i in range(NUM_AFTER_PAUSE)
        ]

        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": False,
        }

        # ==================== Start inflight requests ====================
        print("\n[6] Submitting inflight requests (BEFORE pause)...")
        inflight_refs = []
        for i, prompt_ids in enumerate(inflight_prompt_ids):
            request_id = f"inflight_{i}_{uuid4().hex[:8]}"
            ref = server_handle.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
                image_data=None,
            )
            inflight_refs.append((i, request_id, ref))
            print(f"    inflight[{i}] submitted: {request_id}")

        print(f"\n[7] Letting inflight run for {PRE_PAUSE_RUN_SECS}s ...")
        await asyncio.sleep(PRE_PAUSE_RUN_SECS)

        # ==================== Pause via Replica ====================
        print("\n[8] Pausing via Replica.pause(wait_for_inflight_requests=true) ...")
        pause_start = time.perf_counter()

        pause_task = asyncio.create_task(replica.pause(wait_for_inflight_requests=True))
        await asyncio.sleep(1)
        pause_result = await pause_task

        pause_latency = time.perf_counter() - pause_start
        print(f"    pause_result={pause_result}")
        print(f"    pause latency={pause_latency * 1000:.2f}ms")

        # verify paused via Ray API
        print("\n[9] Verifying replica.is_paused() ...")
        st = await replica.is_paused()
        print(f"    is_paused={st}")
        assert st.get("is_paused") is True

        # ==================== Submit requests AFTER pause ====================
        print("\n[10] Submitting requests AFTER pause (should be BLOCKED while paused)...")
        after_pause_refs = []
        for i, prompt_ids in enumerate(after_pause_prompt_ids):
            request_id = f"afterpause_{i}_{uuid4().hex[:8]}"
            ref = server_handle.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
                image_data=None,
            )
            after_pause_refs.append((i, request_id, ref))
            print(f"    after_pause[{i}] submitted: {request_id}")

        # ==================== Collect inflight results (should finish) ====================
        print("\n[11] Collecting inflight results (expected: completed, NOT blocked)...")
        inflight_outs = []
        for i, request_id, ref in inflight_refs:
            try:
                out = ray.get(ref, timeout=180.0)
                inflight_outs.append((i, request_id, out, "got"))
            except Exception as e:
                inflight_outs.append((i, request_id, None, f"error: {e}"))

        # ==================== Probe after-pause requests: should TIMEOUT while paused ====================
        print(f"\n[12] Probing after-pause requests with timeout={BLOCK_PROBE_TIMEOUT}s (expected: TIMEOUT)...")
        blocked = 0
        got_early = 0
        probe_results = []
        for i, request_id, ref in after_pause_refs:
            try:
                out = ray.get(ref, timeout=BLOCK_PROBE_TIMEOUT)
                probe_results.append((i, request_id, out, "got_early"))
                got_early += 1
            except Exception as e:
                probe_results.append((i, request_id, None, f"timeout_or_error: {e}"))
                blocked += 1

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY (while paused)")
        print("=" * 80)
        print(f"Inflight count: {len(inflight_refs)} (expected: all finished)")
        print(f"After-pause count: {len(after_pause_refs)} (expected: most blocked/timeouts)")
        print(f"After-pause blocked(timeouts): {blocked}, got_early: {got_early}")

        assert blocked >= 1, (
            "Did not observe blocked requests while paused. "
            "This indicates pause gating is not applied on the rollout path. "
            "Fix by adding pause gate in vLLMHttpServerBase.generate."
        )

        inflight_errors = [x for x in inflight_outs if x[3] != "got"]
        assert not inflight_errors, f"Inflight requests errored (unexpected): {inflight_errors[:1]}"

        # ==================== Resume via Replica ====================
        print("\n[13] Resuming via Replica.resume() ...")
        resume_result = await replica.resume()
        print(f"    resume_result={resume_result}")

        st = await replica.is_paused()
        print(f"    is_paused after resume: {st.get('is_paused')}")
        assert st.get("is_paused") is False

        # ==================== Collect all after-pause results ====================
        print("\n[14] Collecting ALL after-pause results (should proceed after resume)...")
        after_pause_outs = []
        for i, request_id, ref in after_pause_refs:
            try:
                out = ray.get(ref, timeout=300.0)
                after_pause_outs.append((i, request_id, out, "got"))
            except Exception as e:
                after_pause_outs.append((i, request_id, None, f"error: {e}"))

        errors = [x for x in after_pause_outs if x[3] != "got"]
        assert not errors, "Some after-pause requests errored after resume (unexpected)."
        print("\n✅ Test passed: pause via replica blocks new submits, inflight completes, resume releases.")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
