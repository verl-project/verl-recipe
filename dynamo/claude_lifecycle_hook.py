"""Sandbox-side observer: forward Claude lifecycle identity, never prompt text."""

import json
import os
import sys
import time
import urllib.request
from uuid import uuid4


def main():
    source = json.load(sys.stdin)
    payload = {key: source[key] for key in ("hook_event_name", "session_id", "agent_id", "agent_type") if key in source}
    payload["event_id"] = uuid4().hex
    # Uni-Agent already supplies the rollout-scoped URL to Claude Code. The
    # hook inherits it, including any sandbox tunnel rewrite by the runner.
    endpoint = os.environ.get("DYNAMO_AGENT_EVENT_URL")
    if not endpoint:
        endpoint = os.environ["ANTHROPIC_BASE_URL"].rstrip("/").removesuffix("/v1") + "/agent_events"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 200:
                    raise RuntimeError(f"Lifecycle endpoint returned {response.status}")
                response.read()
            return 0
        except Exception as error:
            print(f"Dynamo lifecycle delivery failed: {type(error).__name__}: {error}", file=sys.stderr)
            if attempt < 2:
                time.sleep(0.2 * 2**attempt)
    # Exit 2 would instruct Claude to continue generating at SubagentStop.
    # Delivery failure must be observable without changing the agent's answer.
    return 1


if __name__ == "__main__":
    sys.exit(main())
