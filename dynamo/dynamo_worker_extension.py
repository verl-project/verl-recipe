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
"""vLLM worker_extension_cls for the dynamo backend.

The base ``vLLMColocateWorkerExtension._get_zmq_handle`` (verl/workers/rollout
/vllm_rollout/utils.py) uses ``self.local_rank``, which is the rank of
the worker *within its TP group*. In the dynamo Route-B topology each DP shard
is a separate ``dynamo.vllm`` subprocess, so two DP shards' TP rank 0 would
both compute ``self.local_rank == 0`` and connect to the same IPC socket file.

Fix: read ``VERL_DYNAMO_RANK_OFFSET`` from env (set by DynamoHttpServer when
spawning each DP shard) and add it to ``self.local_rank`` so the IPC handle
encodes a node-global rank that matches what the trainer side computes
(``rollout_rank % local_world_size`` in vllm_rollout.py). Keep the same Ray
job id prefix as verl's native vLLM path so sender and receiver build identical
socket paths.
"""

from __future__ import annotations

import os

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

_RANK_OFFSET_ENV = "VERL_DYNAMO_RANK_OFFSET"


class vLLMDynamoColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker mixin for verl × dynamo.

    Override ``_get_zmq_handle`` to use a node-global rank (rather than the
    per-shard TP-local rank), so trainer-side BucketedWeightSender and
    engine-side BucketedWeightReceiver agree on the IPC socket path.
    """

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        offset = int(os.environ.get(_RANK_OFFSET_ENV, "0"))
        global_rank = self.local_rank + offset
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{global_rank}.sock"

    def update_weights_from_ipc(self, *args, **kwargs):
        """Run verl's weight reload inside vLLM's config context.

        vLLM 0.20's FlashInfer MoE post-load path calls
        get_current_vllm_config(). Native vLLM sets that context around engine
        internals, but verl invokes this worker extension through
        collective_rpc, so set it explicitly in the TP worker process.
        """
        vllm_config = getattr(getattr(self, "model_runner", None), "vllm_config", None)
        if vllm_config is None:
            return super().update_weights_from_ipc(*args, **kwargs)

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            return super().update_weights_from_ipc(*args, **kwargs)
