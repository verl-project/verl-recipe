# Copyright 2026 HUMANLM team and/or its affiliates
# Copyright 2026 Bytedance Ltd. and/or its affiliates

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return False


def shorten(s: Any, max_chars: int = 240) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def debug_enabled(extra_info: dict[str, Any] | None) -> bool:
    if not extra_info:
        return False
    return as_bool(extra_info.get("_usim_debug_logs", False))


def debug_print(extra_info: dict[str, Any] | None, msg: str) -> None:
    """Debug print with a stable prefix. No-op unless debug is enabled via extra_info."""
    if not debug_enabled(extra_info):
        return
    key = extra_info.get("_usim_key") or extra_info.get("index") or "?"
    hname = extra_info.get("state_name") or "?"
    print(f"[USIM_DEBUG][key={key}][field={hname}] {msg}", flush=True)


def debug_print_kh(enabled: bool, key: str, state: str, msg: str) -> None:
    """Debug print when you already have the key + state/field name."""
    if not enabled:
        return
    print(f"[USIM_DEBUG][key={key}][state={state}] {msg}", flush=True)


@dataclass
class HumanLMDebugConfig:
    enabled: bool = False
    max_examples_per_batch: int = 1
    print_text: bool = False
    text_max_chars: int = 1200