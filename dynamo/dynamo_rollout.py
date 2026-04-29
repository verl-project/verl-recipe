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
"""Dynamo rollout ServerAdapter (recipe-side, m2).

m2 is the "vLLM-equivalent" milestone: the trainer side talks to a per-node
:class:`recipe.dynamo.dynamo_async_server.DynamoHttpServer` Ray actor that
embeds a vLLM ``AsyncLLM`` directly (no Dynamo Frontend / runtime yet).
The actor name prefix is read from ``config.name`` by the vLLM ServerAdapter,
which is already ``"dynamo_"`` for this backend, so no behavior diverges
from vLLM in m2.

Subclassing keeps us automatically aligned with any upstream verl fixes to
the vLLM ServerAdapter. Dynamo-specific deltas (Frontend subprocess, dynamo
runtime injection) start landing in m3+ on the *server* side, not here.
"""

from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter as _VllmServerAdapter


class ServerAdapter(_VllmServerAdapter):
    """Per-rank dynamo client. Identical behavior to the vLLM ServerAdapter
    in m2; ``_get_server_name_prefix`` already returns ``"dynamo_"`` because
    it reads ``self.config.name``.
    """

    pass
