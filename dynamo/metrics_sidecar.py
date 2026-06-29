#!/usr/bin/env python3
"""Dependency-free monitoring sidecar for the KV-cache experiment.

WHY: the previous run lost its KV/prefix-cache time-series because the one-shot
report scraped the metric endpoints AFTER the jobs died (all "Connection
refused"). This sidecar runs for the whole job lifetime and appends RAW
per-target metric snapshots to a JSONL file, so the data survives even after the
servers go away. The report builder then ingests the JSONL instead of (failing
to) scrape dead endpoints.

It scrapes two kinds of endpoints, re-discovered every cycle (verl writes the
target file only after the servers come up):
  1. Prometheus target files written by verl (rollout job targets).
     - vLLM arm: each target is a vLLM worker → exposes vllm:prefix_cache_*.
     - Dynamo arm: the single target is the FRONTEND → exposes dynamo_frontend_*.
  2. Dynamo worker /metrics endpoints recorded by dynamo_async_server when
     enable_worker_system_metrics=true (one file per replica/node, *.endpoints).
     These expose the engine-level vllm:prefix_cache_* that makes the Dynamo arm
     directly comparable to the vLLM arm.

Stdlib only: urllib, json, re, glob, time, datetime, argparse, pathlib.

JSONL line schema:
  {"ts": ISO8601, "cycle": int, "label": str, "target": "host:port",
   "source": "prom_target"|"worker_endpoint", "metrics": {name: float},
   "error": null|str}
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import time
import urllib.request
from pathlib import Path

PROM_TARGET_RE = re.compile(r"^\s*-\s+([A-Za-z0-9_.\-\[\]:]+:\d+)\s*$", re.MULTILINE)
METRIC_LINE_RE = re.compile(r"([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)")

# Metric families worth persisting. Engine-level vllm:prefix_cache_* is the
# apples-to-apples cross-backend KV hit-rate signal (present on vLLM workers AND
# on Dynamo workers via the system port).
METRIC_NAMES = {
    # vLLM engine (vLLM arm workers + Dynamo arm workers)
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    # Dynamo worker portable engine gauges (from worker system /metrics)
    "dynamo_component_kv_cache_hit_rate",
    "dynamo_component_gpu_cache_usage_percent",
    "dynamo_component_total_blocks",
    # Dynamo frontend
    "dynamo_frontend_cached_tokens_count",
    "dynamo_frontend_cached_tokens_sum",
    "dynamo_frontend_input_sequence_tokens_sum",
    "dynamo_frontend_input_sequence_tokens_count",
    "dynamo_frontend_request_duration_seconds_count",
    "dynamo_frontend_request_duration_seconds_sum",
    "dynamo_component_router_kv_hit_rate_sum",
    "dynamo_component_router_kv_hit_rate_count",
    "dynamo_frontend_active_requests",
    "dynamo_frontend_inflight_requests",
}


def discover_targets(target_globs: list[str], endpoint_globs: list[str]) -> list[tuple[str, str]]:
    """Return list of (host:port, source). Re-read each cycle so targets that
    appear only after servers start are picked up."""
    found: dict[str, str] = {}
    for pat in target_globs:
        for path in glob.glob(pat):
            try:
                text = Path(path).read_text(errors="replace")
            except OSError:
                continue
            for tgt in PROM_TARGET_RE.findall(text):
                found.setdefault(tgt, "prom_target")
    for pat in endpoint_globs:
        for path in glob.glob(pat):
            try:
                lines = Path(path).read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                tgt = line.strip()
                if tgt:
                    found.setdefault(tgt, "worker_endpoint")
    return sorted(found.items())


def scrape_one(target: str, timeout: float) -> tuple[dict[str, float], str | None]:
    url = f"http://{target}/metrics"
    try:
        data = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - record and continue
        return {}, str(exc)
    values: dict[str, float] = {}
    for line in data.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        if name not in METRIC_NAMES:
            continue
        try:
            values[name] = values.get(name, 0.0) + float(raw)
        except ValueError:
            continue
    return values, None


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cycle = 0
    print(
        f"[sidecar] start label={args.label} interval={args.interval}s output={out} "
        f"targets_glob={args.targets_glob} endpoints_glob={args.endpoints_glob}",
        flush=True,
    )
    while args.max_iters <= 0 or cycle < args.max_iters:
        cycle += 1
        ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        targets = discover_targets(args.targets_glob, args.endpoints_glob)
        n_ok = 0
        with out.open("a") as f:
            for target, source in targets:
                metrics, error = scrape_one(target, args.timeout)
                if error is None and metrics:
                    n_ok += 1
                f.write(
                    json.dumps(
                        {
                            "ts": ts,
                            "cycle": cycle,
                            "label": args.label,
                            "target": target,
                            "source": source,
                            "metrics": metrics,
                            "error": error,
                        }
                    )
                    + "\n"
                )
            f.flush()
        print(f"[sidecar] cycle={cycle} targets={len(targets)} scraped_ok={n_ok}", flush=True)
        if args.max_iters > 0 and cycle >= args.max_iters:
            break
        time.sleep(args.interval)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-glob", action="append", default=[], help="glob(s) for verl prometheus_*.yml target files")
    p.add_argument("--endpoints-glob", action="append", default=[], help="glob(s) for *.endpoints worker-metrics files")
    p.add_argument("--output", required=True, help="JSONL output path")
    p.add_argument("--label", default="", help="arm label written into each line, e.g. vllm / dynamo_kv / dynamo_rr")
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--timeout", type=float, default=4.0)
    p.add_argument("--max-iters", type=int, default=0, help="0 = run until killed (bound to job lifetime)")
    args = p.parse_args()
    if not args.targets_glob and not args.endpoints_glob:
        p.error("at least one of --targets-glob / --endpoints-glob is required")
    run(args)


if __name__ == "__main__":
    main()
