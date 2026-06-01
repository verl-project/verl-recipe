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
"""TransferQueue 全局配置常量"""

DEFAULT_TOPIC = "experience"
GROUP_SHARED_COLUMNS = {"prompt", "prompt_length", "prompt_uuid"}  # 硬编码：组内共享列名
NUM_SAMPLE_PER_SEGMENT = 1024
MAX_CONCURRENT_GETS = 10
ZMQ_HWM = 2000
PADDED_COLUMNS = {"prompt", "response"}
