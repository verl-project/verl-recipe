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
"""Smoke-test the environment this recipe trains against. No GPU, no policy.

Walks one episode over the same HTTP contract the agent loop uses — seed, act,
observe, verify — so a reviewer can tell an environment problem from a training
problem before spending a node on it.

    gym env start --resources-server interactive_browser --no-agent --no-model
    python smoke_environment.py --url http://127.0.0.1:8000

With the default local-Chromium backend this needs no account and no
credentials; `--start-url` defaults to a page the environment ships itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp


async def run(base_url: str, start_url: str, goal_url: str, timeout_s: float) -> int:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    # One cookie jar == one environment session, exactly as the agent loop does it.
    async with aiohttp.ClientSession(timeout=timeout) as client:

        async def call(path: str, payload: dict) -> dict:
            async with client.post(f"{base_url}{path}", json=payload) as response:
                body = await response.json()
                print(f"{path} -> HTTP {response.status}")
                if response.status >= 400:
                    raise SystemExit(f"{path} failed: {json.dumps(body)[:400]}")
                return body

        await call("/seed_session", {"initial_url": start_url, "verifier_metadata": {"url_contains": goal_url}})

        first = await call("/browser_observe", {})
        observation = str(first.get("observation") or "")
        print(f"  observation: {observation.splitlines()[:3]}")
        if not observation:
            raise SystemExit("environment returned an empty observation")

        await call("/browser_navigate", {"url": start_url})
        await call("/browser_finish", {"answer": "smoke"})

        verdict = await call("/verify", {"verifier_metadata": {"url_contains": goal_url}})
        reward = float(verdict.get("reward", 0.0))
        print(f"  reward: {reward}")
    print("environment smoke OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="interactive_browser resources server")
    parser.add_argument("--start-url", default="site/index.html", help="task page; relative paths resolve in the env")
    parser.add_argument("--goal-url", default="index.html", help="substring the verifier looks for")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    return asyncio.run(run(args.url.rstrip("/"), args.start_url, args.goal_url, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
