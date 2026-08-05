import re
import time
from urllib.parse import urljoin

import requests
from tinker_cookbook import model_info

_LOCAL_RENDERERS = {
    # verl-tinker can serve models that are not in the hosted Tinker model
    # catalog. Cookbook 0.5.2 removed this small Qwen3 variant from its
    # metadata, but it uses the standard Qwen3 renderer.
    "Qwen/Qwen3-1.7B": "qwen3",
}


def recommended_renderer_name(model_name: str) -> str:
    """Resolve a Cookbook renderer, including verl-tinker-only models."""

    try:
        return model_info.get_recommended_renderer_name(model_name)
    except KeyError:
        if renderer_name := _LOCAL_RENDERERS.get(model_name):
            return renderer_name
        raise


def model_name_slug(model_name: str) -> str:
    """Return a W&B-safe run-name component for a model identifier or path."""

    normalized = model_name.rstrip("/").rsplit("/", 1)[-1].lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", normalized).strip("-._")
    return slug or "model"


def wait_for_healthz_ready(url: str, max_wait_time: int = 7200):
    healthz_url = urljoin(url.rstrip("/") + "/", "api/v1/healthz")
    print(f"waiting for server health at: {healthz_url}")

    ct = 0
    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait_time:
            raise TimeoutError(f"Timed out waiting for healthz ready after {max_wait_time} seconds at {healthz_url}")

        try:
            response = requests.get(healthz_url, timeout=10)
            response.raise_for_status()
            payload = response.json()

            status = payload.get("status")
            print(f"healthz waited time = {int(elapsed)} seconds: status={status}")

            if status == "ready":
                return payload
            if status == "error":
                raise Exception(f"Received error status from server: {payload}")

        except requests.RequestException as e:
            print(f"healthz ct={ct}: request failed: {e}")
        except ValueError as e:
            print(f"healthz ct={ct}: invalid JSON: {e}")

        ct += 1
        time.sleep(30)


def shutdown_server(url: str):
    shutdown_url = urljoin(url.rstrip("/") + "/", "api/v1/shutdown")
    print(f"shutting down server at: {shutdown_url}")
    try:
        response = requests.post(shutdown_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"failed to shut down server: {e}")
