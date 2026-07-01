import time
from urllib.parse import urljoin

import requests
from ray.serve import get_serve_http_client


def wait_for_url(psm: str, max_wait_time: int = 28800):
    # ex: url = "http://[****:***:cda2:1202:f11b:3ff4:****:****]:11503/"

    if psm is None or len(psm) == 0:
        # no psm given, so we just assume it's local development
        return "http://127.0.0.1:8000/"

    print(f"waiting for url from psm: {psm}")
    url = ""
    ct = 0
    start_time = time.monotonic()

    while url == "":
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait_time:
            raise TimeoutError(f"Timed out waiting for server URL after {max_wait_time} seconds for psm={psm}")

        url = get_serve_http_client(
            psm=psm,
        ).get_one_request_url()

        if url == "":
            ct += 1
            if ct % 12 == 0:
                print(f"Server not ready yet, waited: {ct // 12} minutes")
            time.sleep(5)

    return url


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
        time.sleep(10)


def shutdown_server(url: str):
    shutdown_url = urljoin(url.rstrip("/") + "/", "api/v1/shutdown")
    print(f"shutting down server at: {shutdown_url}")
    try:
        response = requests.post(shutdown_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"failed to shut down server: {e}")
