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
import os
from enum import Enum

import ray
from prometheus_client import CollectorRegistry, multiprocess, start_http_server

logger = logging.getLogger(__name__)


class MetricsServerStateEnum(str, Enum):
    """Metrics server enum"""

    INITIALIZED = "initialized"
    RUNNING = "running"
    FAILED = "failed"


_DEFAULT_PROMETHEUS_PORT = 9400


def _resolve_bind_addr() -> str:
    """Resolve metrics server bind address.

    Defaults to loopback (127.0.0.1). Only binds to 0.0.0.0 when the operator
    explicitly opts in via AF_PROMETHEUS_METRICS_BIND_ALL=true, or provides a
    concrete address via AF_PROMETHEUS_METRICS_BIND.
    """
    explicit = os.environ.get("AF_PROMETHEUS_METRICS_BIND")
    if explicit:
        return explicit
    if os.environ.get("AF_PROMETHEUS_METRICS_BIND_ALL", "").lower() == "true":
        return "0.0.0.0"
    return "127.0.0.1"


@ray.remote
class PrometheusMetricsServer:
    def __init__(self, port: int = None, addr: str = None):
        self.port = int(port) if port is not None else _DEFAULT_PROMETHEUS_PORT
        self.addr = addr if addr is not None else _resolve_bind_addr()
        self.server = None
        self._status = MetricsServerStateEnum.INITIALIZED
        logger.info(f"PrometheusMetricsServer initialized on {self.addr}:{self.port}")

    def start(self):
        try:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            self.server = start_http_server(self.port, addr=self.addr, registry=registry)
            logger.info(f"Prometheus metrics server started on {self.addr}:{self.port}")
            self._status = MetricsServerStateEnum.RUNNING
        except Exception as e:
            self._status = MetricsServerStateEnum.FAILED
            logger.error(f"Failed to start prometheus metrics server: {e}")

    def get_status(self):
        return self._status


def start_prometheus_servers(port: int = _DEFAULT_PROMETHEUS_PORT, addr: str = None):
    """Start prometheus metrics servers on all Ray nodes.

    Gated by the AF_PROMETHEUS_METRICS_ENABLE environment variable; metrics
    endpoints are only launched when it is set to 'true'.
    """
    if os.environ.get("AF_PROMETHEUS_METRICS_ENABLE", "").lower() != "true":
        logger.info("AF_PROMETHEUS_METRICS_ENABLE is not 'true', skipping prometheus metrics server start")
        return None

    bind_addr = addr if addr is not None else _resolve_bind_addr()

    # connect to the existing Ray cluster
    if not ray.is_initialized():
        ray.init(address="auto")

    logger.info("Connected to Ray cluster")

    # get all active nodes
    nodes = ray.nodes()
    alive_nodes = [node for node in nodes if node["Alive"]]

    logger.info(f"Found {len(alive_nodes)} alive nodes")

    servers = []

    # start prometheus server on each node
    for node in alive_nodes:
        node_id = node["NodeID"]
        node_ip = node["NodeManagerAddress"]

        logger.info(f"Starting server on node {node_id} ({node_ip})")

        # Use NodeAffinitySchedulingStrategy to run server on the specided node
        server = PrometheusMetricsServer.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            name=f"prometheus_server_{node_id[:8]}",
            lifetime="detached",
        ).remote(port=port, addr=bind_addr)

        servers.append(server)

    # Start servers
    logger.info("Starting all MetricsServers...")
    ray.get([server.start.remote() for server in servers])
    logger.info(f"Started {len(servers)} prometheus metrics servers")

    return servers


if __name__ == "__main__":
    servers = start_prometheus_servers()
    logger.info("\nPrometheus servers started successfully!")
    logger.info(f"You can access metrics at: http://<node_ip>:{_DEFAULT_PROMETHEUS_PORT}/metrics")
    logger.info("Servers will continue running in detached mode.")
