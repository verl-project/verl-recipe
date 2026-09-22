# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Dataset adapter: NeMo Gym rollout rows to verl `tools_kwargs`.

`NeMoGymJSONLDataset` already parses NeMo Gym rollout inputs into `raw_prompt`
plus `extra_env_info`. A step-wise agent loop needs the per-task fields on
`tools_kwargs` instead, which is the only thing this subclass adds.
"""

from __future__ import annotations

from recipe.nemo_gym.dataset import NeMoGymJSONLDataset

_TASK_KEYS = ("initial_url", "verifier_metadata", "question", "task_id")


class BrowserJSONLDataset(NeMoGymJSONLDataset):
    def __getitem__(self, idx: int) -> dict:
        row = super().__getitem__(idx)
        env_info = row.get("extra_env_info") or {}

        question = env_info.get("question")
        if not question:
            # Fall back to the last user message, which is what the task text is.
            user_turns = [m.get("content", "") for m in row["raw_prompt"] if m.get("role") == "user"]
            question = user_turns[-1] if user_turns else ""

        create_kwargs = {key: env_info[key] for key in _TASK_KEYS if key in env_info}
        create_kwargs["question"] = question
        row["tools_kwargs"] = {"browser": {"create_kwargs": create_kwargs}}
        return row
