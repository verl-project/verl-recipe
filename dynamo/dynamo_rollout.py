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
"""ServerAdapter for the dynamo backend.

Inherits the vLLM ServerAdapter (HTTP path is identical: trainer rank reads
``replica.server_address`` and POSTs chat completions to it) and only
overrides the Ray actor name prefix used for sleep/wake/update_weights RPC,
so it lands on ``dynamo_server_*`` (created by DynamoReplica.launch_servers)
rather than ``vllm_server_*``.
"""

from verl.workers.rollout.vllm_rollout.vllm_rollout import (
    ServerAdapter as _VllmServerAdapter,
)


class ServerAdapter(_VllmServerAdapter):
    """Per-rank dynamo client.

    All HTTP-based generation goes through the frontend URL stored in
    ``RolloutReplica.server_address``; weight-update / wake-up / sleep
    requests go to the per-replica Ray actor named ``dynamo_server_{r}_{n}``.
    """

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"


__all__ = ["ServerAdapter"]
