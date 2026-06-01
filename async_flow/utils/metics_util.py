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
def aggregate_metrics_before_reduce(nested_data) -> dict:
    """
    功能：聚合各个train的metric，数据格式为嵌套字典数据 list[list[dict]]。
    :param nested_data: 嵌套的字典列表，例如 [[{...}, {...}], [{...}]]
    :return: 整合后的字典 {version: {key: [values]}}
    """
    # 1. 初始化返回结果
    output = {}

    # 2. 统一输入格式：确保是 list[list[dict]]
    # 如果传入的是单个 list[dict]，包装成 list[list[dict]] 统一处理
    if isinstance(nested_data, list) and len(nested_data) > 0:
        if isinstance(nested_data[0], dict):
            nested_data = [nested_data]
    elif not nested_data:
        return output

    # 3. 处理嵌套列表
    for sub_list in nested_data:
        for data in sub_list:
            if not isinstance(data, dict):
                continue
            item = data.copy()
            version = item.pop("version", "unknown_version")
            if version not in output:
                output[version] = {}

            for key, value in item.items():
                if key not in output[version]:
                    output[version][key] = []
                output[version][key].append(value)
    return output


def reduce_timing_metrics(timing_metrics):
    consumer_name = timing_metrics["consumer_name"][0]
    experience_step = timing_metrics["experience_step"][0]
    e2e_time = f"e2e_max_{consumer_name}"
    wait_time = f"wait_max_{consumer_name}"
    compute_time = f"compute_max_{consumer_name}"
    # 取rank0估算指标
    timing_metrics["total_num_tokens"] = sum(timing_metrics["total_num_tokens"])
    timing_metrics[e2e_time] = sum(timing_metrics[e2e_time][:experience_step])
    timing_metrics[wait_time] = sum(timing_metrics[wait_time][:experience_step])
    timing_metrics[compute_time] = sum(timing_metrics[compute_time][:experience_step])
    e2e_tps = timing_metrics["total_num_tokens"] / timing_metrics[e2e_time]
    compute_tps = timing_metrics["total_num_tokens"] / timing_metrics[compute_time]
    timing_metrics.update({"perf/throughtput": e2e_tps})
    timing_metrics.update({"compute/throughtput": compute_tps})
    return timing_metrics
