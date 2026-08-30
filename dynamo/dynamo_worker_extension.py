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

Each DP shard in the dynamo topology is a separate ``dynamo.vllm`` subprocess,
so two shards' TP rank 0 would both compute ``self.local_rank == 0`` and
connect to the same IPC socket file. DynamoHttpServer disambiguates by
injecting ``VERL_ZMQ_BASE_TRAINER_RANK=<shard rank offset>`` per subprocess —
verl's base ``vLLMColocateWorkerExtension._get_zmq_handle`` consumes it
natively (``int(base) + dp-resolved local rank``; dynamo shards run dp=1 so
the resolver is the identity), producing the node-global socket rank the
trainer/CE sender side computes. No ``_get_zmq_handle`` override needed.
"""

from __future__ import annotations

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class vLLMDynamoColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker mixin for verl × dynamo."""

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
