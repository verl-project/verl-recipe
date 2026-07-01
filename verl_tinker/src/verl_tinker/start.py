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

"""Server entry point for the VeRL-backed Tinker server."""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf
from ray import serve
from ray.serve.handle import DeploymentHandle

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

from .config_utils import process_config
from .tinker_router import ServerStatus, TinkerServer


logger = logging.getLogger(__name__)

USAGE = "Usage: python verl-recipes/src/verl_recipes/verl_tinker_server/start.py --config path_to_yaml"
SUPERVISOR_POLL_INTERVAL_S = 5.0
HEALTHZ_UNAVAILABLE_TIMEOUT_S = 30 * 60
HEALTHZ_REQUEST_TIMEOUT_S = 5.0


def init_ray(ray_address: str):
    """Initialize Ray connection."""
    if ray.is_initialized():
        return

    kwargs = {"runtime_env": get_ppo_ray_runtime_env()}
    if ray_address == "local":
        ray.init(**kwargs)
    else:
        ray.init(address=ray_address, **kwargs)
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")


def _poll_healthz(handle: DeploymentHandle) -> dict:
    payload = handle.healthz.remote().result(timeout_s=HEALTHZ_REQUEST_TIMEOUT_S)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected healthz dict response, got {type(payload).__name__}")
    return payload


def run_server(config) -> int:
    server_config = config["server"]
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    ray_address = server_config.get("ray_address", "local")
    max_runtime = server_config.get("server_max_runtime", None)

    logger.info(
        f"Starting Tinker server: host={host}, port={port}, max runtime:{max_runtime}, ray address: {ray_address}"
    )

    init_ray(ray_address)

    # SIGTERM -> KeyboardInterrupt so blocking=True handles both signals.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    serve.start(http_options={"host": host, "port": port})

    logger.info("Tinker server deployed (initializing from config)")
    exit_code = 0
    try:
        handle = serve.run(TinkerServer.bind(config), blocking=False)
        start_time = time.monotonic()
        healthz_unavailable_since: float | None = None
        while True:
            try:
                healthz_payload = _poll_healthz(handle)
            except Exception as e:
                now = time.monotonic()
                if healthz_unavailable_since is None:
                    healthz_unavailable_since = now
                unavailable_for = now - healthz_unavailable_since
                if unavailable_for >= HEALTHZ_UNAVAILABLE_TIMEOUT_S:
                    logger.error(
                        "Tinker server healthz has been unreachable through the Ray Serve deployment handle "
                        "for %.1f seconds. The router is likely dead or unreachable; stopping Serve and "
                        "exiting with failure. "
                        "Last healthz error: %r",
                        unavailable_for,
                        e,
                    )
                    exit_code = 1
                    break
                logger.warning(
                    "Tinker server healthz unavailable through the Ray Serve deployment handle "
                    "for %.1f seconds; continuing to poll. Last error: %r",
                    unavailable_for,
                    e,
                )
            else:
                healthz_unavailable_since = None
                status = healthz_payload.get("status")
                if status == ServerStatus.SHUTDOWN_COMPLETE.value:
                    logger.info("Tinker server reported shutdown_complete via healthz. Stopping Serve and exiting.")
                    break
                if status == ServerStatus.ERROR.value:
                    logger.error(
                        "Tinker server reported ERROR via healthz; stopping Serve and exiting with failure. "
                        "Healthz payload: %s",
                        healthz_payload,
                    )
                    exit_code = 1
                    break

            if max_runtime is not None:
                elapsed_time = time.monotonic() - start_time
                remaining_time = max_runtime - elapsed_time
                if remaining_time <= 0:
                    logger.info("Max runtime reached. Stopping Serve and exiting driver.")
                    break
            time.sleep(SUPERVISOR_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        logger.info("Shutting down Tinker server...")

    serve.shutdown()
    return exit_code


def log_config(title: str, config: DictConfig) -> None:
    line = "=" * 80
    logger.info("\n%s\n%s\n%s\n%s", line, title, line, OmegaConf.to_yaml(config, resolve=True))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(USAGE)

    if sys.argv[1] != "--config":
        raise SystemExit(USAGE)

    config_path = Path(sys.argv[2]).expanduser()

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"--config must point to a YAML file, got: {config_path}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)

    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected top-level YAML mapping in {config_path}")

    # Allow downstream code to add/modify fields.
    OmegaConf.set_struct(config, False)

    # Resolve ${oc.env:VAR,default} eagerly so config/env issues fail before server startup.
    OmegaConf.resolve(config)

    log_config("Received config from user", config)

    # do our best to check and format the config before launching server,
    # so any issue is caught early, note that erroreneous config can still
    # go down though
    try:
        config = process_config(config)
    except Exception:
        logger.exception("Tinker server config validation failed")
        raise

    log_config("Processed config", config)

    raise SystemExit(run_server(config))


if __name__ == "__main__":
    main()
