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
Stdlib-only MBPP exec + timeout for multiprocessing ``spawn``.

Imported as a separate module so worker processes do not load ``verl``/torch.
"""

from __future__ import annotations

import multiprocessing as mp
import queue

MBPP_EXEC_TIMEOUT_SEC = 10.0


def _mbpp_exec_globals() -> dict:
    def _stdin_unavailable(*_args, **_kwargs):
        raise EOFError("input() is not available during MBPP reward evaluation")

    return {"input": _stdin_unavailable}


def _mbpp_evaluate_unsafe(code: str, test_setup: str, test_list: list) -> float:
    exec_globals = _mbpp_exec_globals()
    if test_setup:
        exec(test_setup, exec_globals)
    exec(code, exec_globals)

    total = len(test_list)
    if total == 0:
        return 0.0

    passed = 0
    for test in test_list:
        try:
            exec(test, exec_globals)
            passed += 1
        except Exception:
            pass

    return passed / total


def _mbpp_eval_worker(code: str, test_setup: str, test_list: list, out_q: mp.Queue) -> None:
    try:
        out_q.put(_mbpp_evaluate_unsafe(code, test_setup, test_list))
    except Exception:
        out_q.put(0.0)


def run_mbpp_eval_with_timeout(
    code: str,
    test_setup: str,
    test_list: list,
    timeout_sec: float = MBPP_EXEC_TIMEOUT_SEC,
) -> float:
    """Run MBPP evaluation in a spawned subprocess; return ``0.0`` on timeout or failure."""
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_mbpp_eval_worker,
        args=(code, test_setup, test_list, out_q),
    )
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        return 0.0

    try:
        return float(out_q.get(timeout=1.0))
    except queue.Empty:
        return 0.0
