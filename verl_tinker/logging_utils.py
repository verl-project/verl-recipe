# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Logging setup shared by Tinker server driver and Ray Serve replicas."""

from __future__ import annotations

import logging
import os


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_LOG_LEVEL = logging.INFO
_CONFIGURED_PID: int | None = None


def configure_tinker_server_logging(level: int = _DEFAULT_LOG_LEVEL) -> None:
    """Configure process-wide logging for Tinker server processes.

    VeRL/Ray imports can attach a root handler while the root logger is still at
    WARNING. ``force=True`` makes this server's entrypoints restore INFO logs
    for this dedicated process.
    """
    global _CONFIGURED_PID

    current_pid = os.getpid()
    if _CONFIGURED_PID == current_pid:
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)
    _CONFIGURED_PID = current_pid
