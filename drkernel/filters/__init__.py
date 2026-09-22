# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""
DR.Kernel multi-turn rejection sampling (MRS) filters.

Port of `verl_patch/trainer/code/filters/` from
https://github.com/hkust-nlp/KernelGYM. The filter logic is unchanged from
upstream — only a dead import was dropped. A `dataproto_adapter` module is
added on top to translate between verl's `DataProto` and the dict-of-tensors
shape `PPOBatchFilter.filter_batch` expects.

Public API:
- PPOBatchFilter, PPOFilterConfig: the filter pipeline + its config
- filter_dataproto: convenience wrapper that takes a `DataProto` and returns
  a filtered `DataProto`
"""

from .dataproto_adapter import filter_dataproto
from .unified_filter import PPOBatchFilter, PPOFilterConfig

__all__ = ["PPOBatchFilter", "PPOFilterConfig", "filter_dataproto"]
