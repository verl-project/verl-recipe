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
"""Cluster performance tracer for verl / Asyncflow.

Quickstart
----------
Enable by setting the ``VERL_CLUSTER_TRACE`` environment variable to ``1``
and disable log deduplication before launching training::

    export VERL_CLUSTER_TRACE=1
    export RAY_DEDUP_LOG=false
    python train.py ...

After training, parse the Ray logs and generate a Chrome Trace JSON::

    python -m recipe.async_flow.utils.cluster_trace.log_parser /tmp/ray/session_latest/logs/ -o trace.json

Open ``trace.json`` in ``chrome://tracing`` or https://ui.perfetto.dev.
"""


# Lazy imports to avoid early dependency on verl
def __getattr__(name: str):
    if name in ("_ROLE_MAP", "_get_role", "install", "get_role", "get_rank"):
        from . import trace_logger

        return getattr(trace_logger, name)
    if name in ("parse_logs", "merge"):
        from . import log_parser

        if name == "parse_logs":
            return log_parser.parse_trace_logs
        if name == "merge":
            return log_parser.merge_events
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
