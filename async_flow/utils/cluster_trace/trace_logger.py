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
"""
Auto-installed via VERL_CLUSTER_TRACE env var.
Patches performance._timer and performance.simple_timer to log trace events to stdout.
"""

import json
import time
from contextlib import contextmanager
from typing import Optional

# Lazy import to avoid early dependency on verl
_perf_mod = None
_original_timer = None
_original_simple_timer = None

# ── role map ───────────────────────────────────────────────────────────────
_ROLE_MAP: dict[str, str] = {
    # ── VeRL original workers ─────────────────────────────────────────
    "ActorRolloutRefWorker": "actor_rollout",
    "AsyncActorRolloutRefWorker": "actor_rollout",
    "CriticWorker": "critic",
    "TrainingWorker": "train",
    # ── Asyncflow workers ───────────────────────────────────────────────
    "AsyncFlowAgentLoopWorker": "rollout",
    "ActorForwardWorker": "actor_fwd",
    "ReferenceForwardWorker": "ref",
    "RefForwardWorker": "ref",  # Alias
    "RewardAdvWorker": "reward",
    "RewardWorker": "reward",  # Alias
    "ActorTrainWorker": "actor_train",
}


def _get_role(cls_name: str) -> str:
    """Get role label from class name."""
    return _ROLE_MAP.get(cls_name, "unknown")


# ── process-local state ────────────────────────────────────────────────────
_role: str = "unknown"
_rank: int = 0
_patched: bool = False


# ── public ─────────────────────────────────────────────────────────────────


def install(role: str, rank: int = 0) -> None:
    """Patch timers and log trace events to stdout. Idempotent."""
    global _role, _rank, _patched, _perf_mod, _original_timer, _original_simple_timer
    if _patched:
        return
    _role = role
    _rank = rank
    # Lazy import - only import when install is called
    import verl.utils.profiler.performance as perf

    _perf_mod = perf
    _original_timer = perf._timer
    _perf_mod._timer = _patched_timer

    # Also patch simple_timer for direct calls
    _original_simple_timer = perf.simple_timer
    _perf_mod.simple_timer = _patched_simple_timer

    # Patch at module level if this is NPU and mstx_profile is already imported
    try:
        # Reload mstx to get fresh import of _timer
        import importlib

        import verl.utils.profiler.mstx_profile as mstx

        importlib.reload(mstx)
    except ImportError:
        pass

    _patched = True


def get_role() -> str:
    """Get current role."""
    return _role


def get_rank() -> int:
    """Get current rank."""
    return _rank


def is_installed() -> bool:
    """Check if tracer is installed."""
    return _patched


# ── internal ────────────────────────────────────────────────────────────────


class _TraceContext:
    """Context manager that logs trace events to stdout on exit."""

    def __init__(self, name: str):
        self._name = name
        self._start: Optional[int] = None

    def __enter__(self):
        self._start = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is None:
            return
        end = time.time_ns()
        dur_us = (end - self._start) // 1_000
        ts_us = self._start // 1_000

        event = {
            "worker": _role,
            "rank": _rank,
            "ph": "X",  # Complete event
            "name": self._name,
            "ts": ts_us,
            "dur": dur_us,
        }
        print(f"[ClusterProfiler] {json.dumps(event)}", flush=True)


def _patched_timer(name: str, timing_raw: dict, *args, **kwargs):
    """Wrapper around original timer that logs trace events.

    Accepts *args and **kwargs to maintain compatibility with calls that
    pass extra parameters (e.g., color, domain, category) which are
    ignored by the actual _timer implementation.

    Note: This is a generator function to match the original _timer signature.
    Using yield from ensures timing is measured around the actual work, not
    just around generator creation.
    """
    with _TraceContext(name):
        yield from _original_timer(name, timing_raw)


@contextmanager
def _patched_simple_timer(name: str, timing_raw: dict):
    """Wrapper around original simple_timer that logs trace events."""
    with _TraceContext(name):
        with _original_simple_timer(name, timing_raw):
            yield
