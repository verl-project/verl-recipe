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

answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


def map_fn(row: dict, *, data_source: str = None):
    if data_source == "Maxwell-Jia/AIME_2024":
        problem, answer = row["Problem"], row["Answer"]
    elif data_source == "yentinglin/aime_2025":
        problem, answer = row["problem"], row["answer"]
    prompt = problem + answer_format
    data = {
        "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer)},
        "agent_name": "tool_agent",
    }
    return data


def map_fn2(row: dict):
    content = row["prompt"][0]["content"]
    row["prompt"][0]["content"] = content + answer_format
    row["agent_name"] = "tool_agent"
    return row
