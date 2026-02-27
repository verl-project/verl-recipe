# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""
Docker container cleanup for SWE-Agent instances.

Provides last-resort cleanup of Docker containers that outlive their
SWE-Agent process (e.g. after force-kill).  Uses the ``verl.instance_id``
label injected by ``yaml_builder.py`` to precisely target only one
instance's containers.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def cleanup_instance_containers(instance_id: str) -> None:
    """Stop Docker containers belonging to a specific instance.

    Uses the ``verl.instance_id`` label to precisely target only this
    instance's containers.  The ``--rm`` flag on the container ensures it
    is auto-removed once stopped, so we only need ``docker stop``.

    This is idempotent — if no containers exist, it is a no-op.

    Args:
        instance_id: The unique instance identifier whose containers to stop.
    """
    try:
        find_proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-q",
            "--filter",
            f"label=verl.instance_id={instance_id}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await find_proc.communicate()
        container_ids = stdout.decode().strip().split()

        if not container_ids or container_ids == [""]:
            logger.debug(f"[{instance_id}] No residual containers found")
            return

        logger.info(f"[{instance_id}] Stopping {len(container_ids)} residual container(s): {', '.join(container_ids)}")

        stop_proc = await asyncio.create_subprocess_exec(
            "docker",
            "stop",
            "-t",
            "10",
            *container_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(stop_proc.communicate(), timeout=30.0)
        logger.info(f"[{instance_id}] Residual containers stopped successfully")

    except asyncio.TimeoutError:
        logger.warning(f"[{instance_id}] Timeout stopping residual containers")
    except Exception as e:
        logger.warning(f"[{instance_id}] Failed to cleanup containers: {e}")
