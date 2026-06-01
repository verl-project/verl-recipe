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

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional

from prometheus_client import Counter, Gauge, Histogram

import verl.utils.profiler

_AF_PROMETHEUS_METRICS_ENABLED = True
_original_marked_timer = verl.utils.profiler.marked_timer


logger = logging.getLogger(__name__)

# Global metrics cache
_PROMETHEUS_METRICS: dict[str, Any] = {}


def set_prometheus_metrics_enabled(enabled: bool):
    """Set global Prometheus enable status from configuration"""
    global _AF_PROMETHEUS_METRICS_ENABLED
    _AF_PROMETHEUS_METRICS_ENABLED = enabled


def init_prometheus_metrics(prefix=""):
    """Initialize Prometheus metrics, create rl_phase_active gauge"""
    global _PROMETHEUS_METRICS
    global _AF_PROMETHEUS_METRICS_ENABLED

    if not _AF_PROMETHEUS_METRICS_ENABLED:
        logger.warning("Prometheus client not enable. Set AF_PROMETHEUS_METRICS_ENABLE as true to enable.")
        return None

    # Initialize rl_phase_active gauge
    _PROMETHEUS_METRICS = {
        f"{prefix}_train_steps_total": Counter(
            f"{prefix}_train_steps_total", "Total number of completed training steps."
        ),
        f"{prefix}_processed_data_samples_total": Counter(
            f"{prefix}_processed_data_samples_total", "Total number of rollout data samples processed during training."
        ),
        f"{prefix}_phase_active": Gauge(
            f"{prefix}_phase_active",
            "Indicator gauge for current active training phase, see RLPhase class definition for details.",
            labelnames=["phase"],
        ),
        f"{prefix}_phase_duration_seconds": Histogram(
            f"{prefix}_phase_duration_seconds",
            "Histogram of processing duration in seconds for each RL training phase.",
            labelnames=["phase"],
            buckets=(1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000, 2000, 4000, 8000),
        ),
        f"{prefix}_rollout_duration_seconds": Histogram(
            f"{prefix}_rollout_duration_seconds",
            "Histogram of processing duration in seconds for rollout phase.",
            labelnames=["step"],
            buckets=(1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000, 2000, 4000, 8000),
        ),
        f"{prefix}_rollout_sequence_length": Histogram(
            f"{prefix}_rollout_sequence_length",
            "Histogram of token sequence lengths generated during rollout phase(include prompt and response).",
            labelnames=["step"],
            buckets=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
        ),
    }

    return _PROMETHEUS_METRICS


def update_metric(key: str, value: int | float, cumulate: bool = False, labels: dict[str, Any] = None):
    """Update Prometheus metric

    Args:
        key: metric name
        value: metric value
        cumulate: whether to cumulate (True for cumulate, False for set value)
        label: label dictionary
    """
    global _PROMETHEUS_METRICS
    global _AF_PROMETHEUS_METRICS_ENABLED

    if not _AF_PROMETHEUS_METRICS_ENABLED:
        return None

    if not _PROMETHEUS_METRICS:
        init_prometheus_metrics("af")

    metric = _PROMETHEUS_METRICS.get(key)

    if metric is None:
        logger.warning(f"The specified key {key} is not defined in prometheus metrics.")
        return None

    if isinstance(metric, Histogram):
        if labels is not None and isinstance(labels, dict):
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    else:
        if cumulate:
            if labels is not None and isinstance(labels, dict):
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        else:
            if labels is not None and isinstance(labels, dict):
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    return metric


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    prometheus_only: bool = False,
    labels: dict[str, Any] = None,
):
    labels = {"phase": name, **(labels or {})}
    update_metric("af_phase_active", value=1, labels=labels)
    start_time = time.perf_counter()

    if prometheus_only:
        yield
    else:
        # Use the existing marked_timer functionality
        with _original_marked_timer(name, timing_raw) as timer:
            yield timer

    elapsed_time = time.perf_counter() - start_time
    update_metric("af_phase_duration_seconds", elapsed_time, False, labels=labels)
    update_metric("af_phase_active", value=0, labels=labels)
