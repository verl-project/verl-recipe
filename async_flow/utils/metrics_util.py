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
    Aggregates training metrics across different ranks or workers.
    :param nested_data: A nested list of dictionaries containing metric data,
                        e.g., [{key_rank1: [values]}, {key_rank2: [values]}, ...]
    :return: A consolidated dictionary grouped by version,
             e.g., {version: {key: [values]}}
    """
    output = {}

    for data in nested_data:
        if not isinstance(data, dict):
            continue
        item = data.copy()
        raw_versions = item.pop("version", ["unknown_version"])
        unique_versions = set(raw_versions)
        if len(unique_versions) > 1:
            raise ValueError(f"Inconsistent model versions found in nested_data: {unique_versions}.")
        version = list(unique_versions)[0]

        if version not in output:
            output[version] = {}
        for key, value in item.items():
            if key not in output[version]:
                output[version][key] = []
            if isinstance(value, list):
                output[version][key].extend(value)
            else:
                output[version][key].append(value)
    return output


def reduce_timing_metrics(timing_metrics):
    consumer_name = timing_metrics["consumer_name"][0]
    experience_step = timing_metrics["experience_step"][0]
    e2e_time = f"e2e_max_{consumer_name}"
    wait_time = f"wait_max_{consumer_name}"
    compute_time = f"compute_max_{consumer_name}"

    timing_metrics["total_num_tokens"] = sum(timing_metrics["total_num_tokens"])
    timing_metrics[e2e_time] = sum(timing_metrics[e2e_time][:experience_step])
    timing_metrics[wait_time] = sum(timing_metrics[wait_time][:experience_step])
    timing_metrics[compute_time] = sum(timing_metrics[compute_time][:experience_step])
    e2e_tps = timing_metrics["total_num_tokens"] / timing_metrics[e2e_time]
    compute_tps = timing_metrics["total_num_tokens"] / timing_metrics[compute_time]
    timing_metrics.update({"perf/throughtput": e2e_tps})
    timing_metrics.update({"compute/throughtput": compute_tps})
    return timing_metrics
