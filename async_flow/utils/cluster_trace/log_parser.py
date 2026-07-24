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
Parse Ray logs and extract trace events for Chrome Trace Event Format.
Usage:
    python -m recipe.async_flow.utils.cluster_trace.log_parser logs/exp.log -o trace.json --only-compute
"""

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict


def parse_trace_logs(log_path: str) -> list[dict]:
    """解析日志文件，处理 Ray 的颜色转义字符并提取 JSON 负载。"""
    log_file = pathlib.Path(log_path)
    events: list[dict] = []

    if log_file.is_dir():
        log_files = sorted(log_file.glob("worker-*.log")) or list(log_file.glob("*"))
    else:
        log_files = [log_file]

    # 正则用于剔除类似 [36m 的颜色代码
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    for log_file_path in log_files:
        if not log_file_path.is_file():
            continue
        try:
            with open(log_file_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "[ClusterProfiler]" not in line:
                        continue
                    clean_line = ansi_escape.sub("", line)
                    start_idx = clean_line.find("{")
                    if start_idx == -1:
                        continue
                    try:
                        events.append(json.loads(clean_line[start_idx:]))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            print(f"Warning: Could not read {log_file_path}: {e}", file=sys.stderr)
    return events


def merge_events(events: list[dict], only_compute: bool = False) -> tuple[dict, int, int]:
    """合并事件。如果 only_compute 为 True，则剔除 e2e 容器块和非计算事件。"""
    all_names = {e.get("name", "") for e in events}

    # 动态识别 e2e 周期标志
    e2e_markers = {n for n in all_names if n.startswith("e2e_")}

    # 核心计算事件定义
    legacy_compute = {
        "rollout",
        "generate_sequences",
        "actor_fwd",
        "reward",
        "actor_train",
        # ── weight-update / vLLM replica profiling ──────────────────────
        "vllm_generate",
        "vllm_pause",
        "vllm_resume",
        "recv_weights",
        "load_weights",
    }
    # sync_* 编排子阶段 (driver) 也视为关注事件
    compute_timers = {n for n in all_names if "compute" in n or n.startswith("sync_") or n in legacy_compute}

    by_worker_rank: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        key = (event.get("worker", "unknown"), event.get("rank", 0))
        by_worker_rank[key].append(event)

    filtered_events = []
    for (worker, rank), worker_events in by_worker_rank.items():
        worker_events.sort(key=lambda e: e.get("ts", 0))

        # 提取当前 worker 所有的 e2e 标志位
        worker_e2e_markers = [e for e in worker_events if e.get("name") in e2e_markers]

        if not worker_e2e_markers:
            # 如果没找到 e2e 周期，直接按 compute 属性过滤
            for e in worker_events:
                if not only_compute or e.get("name") in compute_timers:
                    filtered_events.append(e)
            continue

        # 基于 e2e 周期构建容器
        iterations = []
        for e2e in worker_e2e_markers:
            start, end = e2e.get("ts", 0), e2e.get("ts", 0) + e2e.get("dur", 0)

            # 如果 only_compute=True，不把 e2e 自身加入 events 列表，防止长条连接
            initial_events = [] if only_compute else [e2e]

            iterations.append({"start": start, "end": end, "events": initial_events, "has_compute": False})

        # O(N) 同步扫描将事件归位
        iter_ptr, num_iters = 0, len(iterations)
        for event in worker_events:
            e_name = event.get("name", "")
            if e_name in e2e_markers:
                continue

            ts = event.get("ts", 0)
            while iter_ptr < num_iters and ts > iterations[iter_ptr]["end"]:
                iter_ptr += 1
            if iter_ptr >= num_iters:
                break

            curr_it = iterations[iter_ptr]
            if curr_it["start"] <= ts <= curr_it["end"]:
                # 开启 only_compute 时，只添加计算相关的子事件
                is_compute = e_name in compute_timers
                if not only_compute or is_compute:
                    curr_it["events"].append(event)

                if is_compute:
                    curr_it["has_compute"] = True

        # 最终汇总
        for it in iterations:
            if it["has_compute"]:
                filtered_events.extend(it["events"])

    # 转化为标准的 Chrome Trace 格式 (JSON)
    chrome_events = []
    role_pid = {}
    for event in filtered_events:
        worker = event.get("worker", "unknown")
        if worker not in role_pid:
            role_pid[worker] = len(role_pid)
        pid, rank = role_pid[worker], event.get("rank", 0)

        chrome_events.append(
            {
                "ph": "X",
                "name": event.get("name", "unknown"),
                "pid": pid,
                "tid": rank,
                "ts": event.get("ts", 0),
                "dur": event.get("dur", 0),
                "args": {k: v for k, v in event.items() if k not in ["ph", "name", "pid", "tid", "ts", "dur"]},
            }
        )

    # 生成进程/线程元数据
    meta_events = []
    for role, pid in role_pid.items():
        meta_events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": role}})
        tids = {e["tid"] for e in chrome_events if e["pid"] == pid}
        for tid in sorted(tids):
            meta_events.append(
                {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": f"{role}[rank={tid}]"}}
            )

    return {"traceEvents": meta_events + chrome_events}, len(events), len(chrome_events)


def main():
    parser = argparse.ArgumentParser(description="Generate clean Chrome Trace JSON from Ray logs")
    parser.add_argument("log_path", help="Log file path or directory")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument(
        "--only-compute", action="store_true", help="Hide e2e markers and non-compute events (wait/put/etc.)"
    )
    parser.add_argument("--compact", action="store_true", help="Minimal JSON size (no indent)")
    args = parser.parse_args()

    # 1. 解析
    events = parse_trace_logs(args.log_path)
    if not events:
        print(f"Error: No [ClusterProfiler] lines found in {args.log_path}", file=sys.stderr)
        sys.exit(1)

    # 2. 过滤与合并
    chrome_trace, total, kept = merge_events(events, only_compute=args.only_compute)

    # 3. 统计并写入
    print(f"Success: Parsed {total} events. Retained {kept} events.", file=sys.stderr)
    if args.only_compute:
        print("Mode: --only-compute (Pure computation blocks only)", file=sys.stderr)

    indent = None if args.compact else 2
    with open(args.output, "w") as f:
        json.dump(chrome_trace, f, indent=indent)


if __name__ == "__main__":
    main()
