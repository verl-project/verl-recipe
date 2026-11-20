# Copyright 2026 Google LLC
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
"""Decide which sandbox images to keep warm, one training step ahead.

The sandbox provider only offers ``prewarm`` and ``unwarm``. When to call them is this recipe's decision, made here:

* :func:`sandbox_requests` reads which image each sample of a batch or dataset will run in.
* :class:`WarmPools` turns the batches that are currently live into ``prewarm``/``unwarm`` calls.
* :class:`LookaheadBatches` hands out training batches while warming the next one in the background.

Nothing here imports ``verl``, so the module can be tested without a trainer.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from uni_agent.sandbox.base import SandboxConfig
from uni_agent.tasks import TaskConfigResolver

logger = logging.getLogger(__name__)

GKE_PROVIDER = "gke_agent_sandbox"

# One warm pool: the image plus the provider options (JSON) its sandboxes are created with.
PoolKey = tuple[str, str]


def _task_of(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tools_kwargs = row.get("tools_kwargs") or (row.get("extra_info") or {}).get("tools_kwargs") or {}
    task = tools_kwargs.get("task")
    return task if isinstance(task, Mapping) else None


def _sample_tasks(samples: Any) -> list[Mapping[str, Any]]:
    """Return the serialized task of every sample in a dataset, a batch, or a list of rows."""
    frame = getattr(samples, "dataframe", samples)
    if hasattr(frame, "iloc"):  # pandas.DataFrame
        return [task for row in frame.to_dict(orient="records") if (task := _task_of(row))]
    if not hasattr(frame, "keys"):  # a datasets.Dataset or a plain list of rows
        return [task for row in frame if (task := _task_of(row))]
    # A batch (TensorDict or dict) holds one column per field.
    for column in ("tools_kwargs", "extra_info"):
        try:
            values = frame[column]
        except KeyError:
            continue
        values = values.tolist() if hasattr(values, "tolist") else values
        values = [values] if isinstance(values, Mapping) else values
        tasks = [task for value in values if isinstance(value, Mapping) and (task := _task_of({column: value}))]
        if tasks:
            return tasks
    return []


def sandbox_requests(samples: Any, resolver: TaskConfigResolver, per_sample: int = 1) -> dict[PoolKey, int]:
    """Count the GKE sandboxes ``samples`` will ask for, ``per_sample`` for each sample.

    Each sample is resolved against the Task Config exactly as the task runner does, so the image (after
    ``image_map``) and the provider options are the ones its sandbox will really use.
    """
    counts: collections.Counter[PoolKey] = collections.Counter()
    for task in _sample_tasks(samples):
        sandbox = resolver.resolve(task).get("sandbox") or {}
        if sandbox.get("provider") != GKE_PROVIDER:
            continue
        config = SandboxConfig.model_validate(sandbox)
        if config.image:
            counts[(config.image, json.dumps(config.sandbox_kwargs, sort_keys=True))] += per_sample
    return dict(counts)


class WarmPools:
    """Keep warm what the live batches need, and nothing else.

    A batch is live from :meth:`start` to :meth:`finish`. A pool is sized to the sum of what the live batches ask
    for, and deleted once no live batch uses its image. ``prewarm`` and ``unwarm`` are the provider's calls; they
    block, so they run one at a time on a worker thread. A failed call is logged and training carries on, because a
    sandbox without a warm pool still starts, only slower.
    """

    def __init__(self, prewarm: Callable[..., Any], unwarm: Callable[..., Any], timeout: float | None = None) -> None:
        self._prewarm = prewarm
        self._unwarm = unwarm
        self._timeout = timeout
        self._lock = threading.Lock()
        self._live: dict[str, dict[PoolKey, int]] = {}
        self._pending: dict[str, Future[None]] = {}
        self._warmed: dict[PoolKey, int] = {}  # only touched on the worker thread
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sandbox-warm")

    def start(self, batch_id: str, requests: Mapping[PoolKey, int]) -> None:
        """Start warming the pools for ``batch_id`` in the background."""
        with self._lock:
            self._live[batch_id] = dict(requests)
            self._pending[batch_id] = self._worker.submit(self._reconcile)

    def wait(self, batch_id: str) -> None:
        """Block until the warming started for ``batch_id`` has finished."""
        with self._lock:
            pending = self._pending.get(batch_id)
        if pending is not None:
            pending.result()

    def finish(self, batch_id: str) -> None:
        """Mark ``batch_id`` as done; pools no other live batch needs are deleted in the background."""
        with self._lock:
            self._pending.pop(batch_id, None)
            if self._live.pop(batch_id, None) is not None:
                self._worker.submit(self._reconcile)

    def close(self) -> None:
        """Delete every pool and stop the worker thread."""
        with self._lock:
            self._live.clear()
            self._pending.clear()
            done = self._worker.submit(self._reconcile)
        done.result()
        self._worker.shutdown()

    def _reconcile(self) -> None:
        with self._lock:
            wanted: collections.Counter[PoolKey] = collections.Counter()
            for requests in self._live.values():
                wanted.update(requests)
        grow: dict[str, dict[str, int]] = {}
        for (image, kwargs), replicas in wanted.items():
            if replicas > self._warmed.get((image, kwargs), 0):  # prewarm never shrinks a pool
                grow.setdefault(kwargs, {})[image] = replicas
        drop: dict[str, list[str]] = {}
        for image, kwargs in self._warmed:
            if (image, kwargs) not in wanted:
                drop.setdefault(kwargs, []).append(image)
        for kwargs, images in drop.items():
            try:
                self._unwarm(images, **json.loads(kwargs))
            except Exception:
                logger.warning("Could not delete %d sandbox warm pools", len(images), exc_info=True)
            for image in images:
                del self._warmed[(image, kwargs)]
        for kwargs, images in grow.items():
            try:
                ready = self._prewarm(images, wait=True, timeout=self._timeout, **json.loads(kwargs))
            except Exception:
                logger.warning("Could not warm %d sandbox images; they will cold-start", len(images), exc_info=True)
                continue
            for image, replicas in images.items():
                self._warmed[(image, kwargs)] = replicas
            late = [image for image, ok in (ready or {}).items() if not ok]
            if late:
                logger.warning("%d of %d sandbox warm pools were not ready in time", len(late), len(images))


class LookaheadBatches:
    """Hand out training batches whose sandboxes are warm, while the next batch's warm in the background.

    ``fetch`` returns the next batch and ``requests_of`` the pools it needs. Call :meth:`finish_oldest` once the
    rollouts of the oldest batch handed out are complete.
    """

    def __init__(
        self, fetch: Callable[[], Any], requests_of: Callable[[Any], Mapping[PoolKey, int]], pools: WarmPools
    ) -> None:
        self._fetch = fetch
        self._requests_of = requests_of
        self._pools = pools
        self._count = 0
        self._ahead: tuple[str, Any] | None = None
        self._handed_out: collections.deque[str] = collections.deque()

    def _fetch_ahead(self) -> None:
        self._count += 1
        batch_id = f"train_{self._count}"
        batch = self._fetch()
        self._pools.start(batch_id, self._requests_of(batch))
        self._ahead = (batch_id, batch)

    def prime(self) -> None:
        """Fetch the first batch and wait until its sandboxes are warm."""
        if self._ahead is None:
            self._fetch_ahead()
        self._pools.wait(self._ahead[0])

    def next_batch(self) -> Any:
        """Return the batch that was warmed ahead and start warming the one after it."""
        self.prime()
        batch_id, batch = self._ahead
        self._handed_out.append(batch_id)
        self._fetch_ahead()
        return batch

    def finish_oldest(self) -> None:
        """Release the pools of the oldest batch handed out."""
        if self._handed_out:
            self._pools.finish(self._handed_out.popleft())
