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
"""DynamoHttpServer + DynamoReplica (recipe-side, m2).

m2 inherits the entire vLLM async server stack and only renames the Ray
actor prefix (``dynamo_`` instead of ``vllm_``). Functionally this is a
verl-vllm copy living under recipe/, used as the baseline against which
m3 will measure dynamo runtime injection.

m3 will start overriding ``DynamoHttpServer.launch_server`` to additionally
spin up a Dynamo Frontend subprocess and register the embedded ``AsyncLLM``
into a same-process ``DistributedRuntime`` (file discovery). m4 wires
``runtime_env`` so Ray actor subprocesses also import ``recipe.dynamo``.
"""

from __future__ import annotations

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer as _VllmHttpServer,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMReplica as _VllmReplica,
)


class DynamoHttpServer(_VllmHttpServer):
    """Per-node Ray actor. m2: identical to vLLMHttpServer. Future overrides
    (m3+) inject Dynamo Frontend + runtime registration into ``launch_server``.
    """

    pass


class DynamoReplica(_VllmReplica):
    """Replica for the dynamo backend. Reuses the entire vLLMReplica launch
    pipeline but binds the Ray actor name to ``dynamo_*`` so ServerAdapter
    (whose own ``_get_server_name_prefix`` reads ``config.name``) lands on
    these actors and not on someone else's vllm replicas in the same job.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebind to the recipe-side server class so subclass hooks (added in
        # m3+) take effect. vLLMReplica.__init__ already set this to
        # ``ray.remote(vLLMHttpServer)``; we replace it.
        self.server_class = ray.remote(DynamoHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"


__all__ = ["DynamoHttpServer", "DynamoReplica"]
