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
"""Recipe-side package for the Dynamo rollout backend.

Registration of the dynamo backend lives in verl proper:

  * ``verl.workers.rollout.base._ROLLOUT_REGISTRY[("dynamo", "async")]``
    -> :class:`recipe.dynamo.dynamo_rollout.ServerAdapter`
  * ``verl.workers.rollout.replica.RolloutReplicaRegistry["dynamo"]``
    -> :class:`recipe.dynamo.dynamo_async_server.DynamoReplica`

Both registry entries are populated at verl import time; this package
intentionally has no module-level side effects so that early importers
(e.g. Ray ``worker_process_setup_hook``) cannot pull
``verl.workers.rollout`` -> ``verl.utils.device`` and prematurely call
``torch.cuda.is_available()`` before Ray finalizes the actor's
``CUDA_VISIBLE_DEVICES``. (That ordering bug poisons the torch CUDA cache
and breaks ``transformer_engine`` import in CPU-only actors.)
"""
