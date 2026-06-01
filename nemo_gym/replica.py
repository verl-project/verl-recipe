# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class NeMoGymvLLMHttpServer(vLLMHttpServer):
    def apply_nemo_gym_server_patch(self):
        from recipe.nemo_gym.server_patch import (
            patch_hermes_tool_parser_thread_safety,
            patch_serving_chat_for_nemo_gym,
        )

        patch_serving_chat_for_nemo_gym()
        patch_hermes_tool_parser_thread_safety()


class NeMoGymvLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(NeMoGymvLLMHttpServer)
