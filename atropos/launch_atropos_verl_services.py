#!/usr/bin/env python3
"""
Launch script for Atropos-VeRL integrated services.

This script orchestrates the startup of all required services:
1. Atropos environment server
2. VeRL inference engines (vLLM)
3. VeRL training process

Usage:
    python launch_atropos_verl_services.py --config recipe/atropos/config/gsm8k_grpo_example.yaml
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceLauncher:
    """Manages the lifecycle of Atropos and VeRL services."""

    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.processes: list[subprocess.Popen] = []
        self.config_path = config_path

        # Register signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, cleaning up services...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Terminate all launched processes."""
        for proc in self.processes:
            if proc.poll() is None:
                logger.info(f"Terminating process {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {proc.pid}")
                    proc.kill()

    def wait_for_service(
        self,
        url: str,
        service_name: str,
        timeout: int = 60,
        health_path: str = "/health",
    ) -> bool:
        """Wait for a service to become available."""
        logger.info(f"Waiting for {service_name} at {url}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}{health_path}", timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logger.error(f"{service_name} failed to start within {timeout} seconds")
        return False

    def _parse_host_port(self, url: str, default_port: int) -> tuple[str, int]:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        return host, port

    def launch_atropos_api_server(self) -> bool:
        """Launch the Atropos API server (trajectory handler)."""
        atropos_config = self.config.get("trainer", {}).get("atropos", {})
        api_url = atropos_config.get("api_url", "http://localhost:9001")
        api_host, api_port = self._parse_host_port(api_url, 9001)

        try:
            response = requests.get(f"{api_url}/status", timeout=2)
            if response.status_code == 200:
                logger.info("Atropos API server already running")
                return True
        except requests.exceptions.RequestException:
            pass

        atropos_path = os.environ.get("ATROPOS_PATH")
        if not atropos_path:
            possible_paths = [
                Path.home() / "atropos",
                Path.home() / "projects" / "atropos",
                Path("/opt/atropos"),
            ]
            for path in possible_paths:
                if (path / "atroposlib" / "cli" / "run_api.py").exists():
                    atropos_path = str(path)
                    break

        if not atropos_path:
            logger.error("Could not find Atropos installation. Set ATROPOS_PATH environment variable.")
            return False

        api_script = Path(atropos_path) / "atroposlib" / "cli" / "run_api.py"
        if not api_script.exists():
            logger.error(f"Atropos API script not found: {api_script}")
            return False

        cmd = [
            sys.executable,
            str(api_script),
            "--host",
            api_host,
            "--port",
            str(api_port),
        ]

        logger.info(f"Launching Atropos API server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=atropos_path, text=True)
        self.processes.append(proc)

        return self.wait_for_service(api_url, "Atropos API server", timeout=60, health_path="/status")

    def launch_atropos_server(self, vllm_url: str | None = None) -> bool:
        """Launch the Atropos environment server."""
        atropos_config = self.config.get("trainer", {}).get("atropos", {})
        api_url = atropos_config.get("api_url", "http://localhost:9001")
        environment = atropos_config.get("environment", "gsm8k")

        # Find Atropos installation
        atropos_path = os.environ.get("ATROPOS_PATH")
        if not atropos_path:
            # Try to find Atropos in common locations
            possible_paths = [
                Path.home() / "atropos",
                Path.home() / "projects" / "atropos",
                Path("/opt/atropos"),
            ]
            for path in possible_paths:
                if path.exists() and (path / "environments" / f"{environment}_server.py").exists():
                    atropos_path = str(path)
                    break

        if not atropos_path:
            logger.error("Could not find Atropos installation. Set ATROPOS_PATH environment variable.")
            return False

        # Launch Atropos server
        server_script = Path(atropos_path) / "environments" / f"{environment}_server.py"
        if not server_script.exists():
            logger.error(f"Atropos server script not found: {server_script}")
            return False

        cmd = [sys.executable, str(server_script), "serve", "--slurm", "false"]

        rollout_server_url = atropos_config.get("rollout_server_url", api_url)
        cmd.extend(["--env.rollout_server_url", rollout_server_url])

        if vllm_url:
            cmd.extend(["--openai.base_url", f"{vllm_url}/v1"])
            cmd.extend(["--openai.server_type", "openai"])
            cmd.extend(["--openai.api_key", "x"])

        logger.info(f"Launching Atropos server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=atropos_path, text=True)
        self.processes.append(proc)

        # Env server connects back to the API, so no direct health check here
        time.sleep(5)
        return True

    def launch_vllm_workers(self) -> bool:
        """Launch vLLM inference workers if configured."""
        inference_config = self.config.get("inference", {})
        if not inference_config and self.config.get("rollout", {}).get("name") == "vllm":
            inference_config = {"type": "vllm"}

        if not inference_config or inference_config.get("type") != "vllm":
            logger.info("vLLM not configured, skipping")
            return True

        # Check if vLLM is already running
        vllm_port = inference_config.get("port", 8000)
        vllm_url = f"http://localhost:{vllm_port}"

        try:
            response = requests.get(f"{vllm_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("vLLM server already running")
                return True
        except requests.exceptions.RequestException:
            pass

        # Launch vLLM server
        model_path = self.config.get("model", {}).get("path", "")
        if not model_path:
            logger.error("Model path not specified in config")
            return False

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--port",
            str(vllm_port),
            "--tensor-parallel-size",
            str(inference_config.get("tensor_parallel_size", 1)),
        ]

        if inference_config.get("dtype"):
            cmd.extend(["--dtype", inference_config["dtype"]])

        logger.info(f"Launching vLLM server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, text=True)
        self.processes.append(proc)

        # Wait for vLLM to start
        if not self.wait_for_service(vllm_url, "vLLM server", timeout=120, health_path="/health"):
            return False
            
        # Register inference endpoint with Atropos if supported
        try:
            atropos_config = self.config.get("trainer", {}).get("atropos", {})
            api_url = atropos_config.get("api_url", "http://localhost:9001")
            
            # Try to register the endpoint (optional, for future Atropos versions)
            response = requests.post(
                f"{api_url}/register_inference_endpoint",
                json={"url": vllm_url},
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Successfully registered vLLM endpoint {vllm_url} with Atropos")
            else:
                logger.debug(f"Endpoint registration not supported by Atropos (status {response.status_code})")
        except requests.exceptions.RequestException:
            # Endpoint registration is optional - don't fail if not supported
            logger.debug("Endpoint registration not supported by this Atropos version")
            
        return True

    def launch_training(self) -> subprocess.Popen:
        """Launch the VeRL training process."""
        # Use main_ppo by default; enable example script via VERL_ATROPOS_USE_EXAMPLE=1
        example_script = Path(__file__).parent / "example_gsm8k_grpo.py"
        use_example = os.getenv("VERL_ATROPOS_USE_EXAMPLE", "0") == "1"

        if use_example and example_script.exists():
            cmd = [
                sys.executable,
                str(example_script),
                "--config-path",
                str(Path(self.config_path).parent),
                "--config-name",
                Path(self.config_path).stem,
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "verl.trainer.main_ppo",
                "--config-path",
                str(Path(self.config_path).parent),
                "--config-name",
                Path(self.config_path).stem,
                "trainer_cls=recipe.atropos.grpo_atropos_trainer.RayGRPOAtroposTrainer",
            ]

        logger.info(f"Launching training: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, text=True)
        self.processes.append(proc)
        return proc

    def run(self):
        """Launch all services and monitor training."""
        logger.info("Starting Atropos-VeRL integrated training")

        # Launch services in order
        if not self.launch_atropos_api_server():
            logger.error("Failed to launch Atropos API server")
            self.cleanup()
            return 1

        if not self.launch_vllm_workers():
            logger.error("Failed to launch vLLM workers")
            self.cleanup()
            return 1

        inference_config = self.config.get("inference", {})
        if not inference_config and self.config.get("rollout", {}).get("name") == "vllm":
            inference_config = {"type": "vllm"}
        vllm_port = inference_config.get("port", 8000) if inference_config else 8000
        vllm_url = f"http://localhost:{vllm_port}" if inference_config else None

        if not self.launch_atropos_server(vllm_url=vllm_url):
            logger.error("Failed to launch Atropos environment server")
            self.cleanup()
            return 1

        # Small delay to ensure services are fully ready
        time.sleep(5)

        # Launch training
        training_proc = self.launch_training()

        # Monitor training process
        try:
            return_code = training_proc.wait()
            logger.info(f"Training completed with return code: {return_code}")
            return return_code
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return 1
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Launch Atropos-VeRL integrated services")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--atropos-path", type=str, help="Path to Atropos installation (overrides ATROPOS_PATH env var)"
    )
    parser.add_argument(
        "--skip-atropos-api",
        action="store_true",
        help="Skip launching Atropos API server (assume it's already running)",
    )
    parser.add_argument(
        "--skip-atropos",
        action="store_true",
        help="Skip launching Atropos environment server (assume it's already running)",
    )
    parser.add_argument(
        "--skip-vllm", action="store_true", help="Skip launching vLLM server (assume it's already running)"
    )

    args = parser.parse_args()

    if args.atropos_path:
        os.environ["ATROPOS_PATH"] = args.atropos_path

    launcher = ServiceLauncher(args.config)

    # Override launch methods if skip flags are set
    if args.skip_atropos_api:
        launcher.launch_atropos_api_server = lambda: True
    if args.skip_atropos:
        launcher.launch_atropos_server = lambda: True
    if args.skip_vllm:
        launcher.launch_vllm_workers = lambda: True

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
