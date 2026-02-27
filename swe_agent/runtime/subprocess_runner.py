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
SWE-Agent subprocess runner.

Manages the ``sweagent run`` CLI subprocess lifecycle:
- Process creation and environment setup
- Timeout handling with graceful SIGTERM → SIGKILL escalation
- stdout/stderr log capture
- Patch extraction via ``PatchExtractor``
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from recipe.swe_agent.utils.patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)


async def execute_swe_agent(
    *,
    config_path: str,
    problem_statement: str,
    instance_id: str,
    output_dir: str,
    repo_path: str,
    exec_dir: str,
    swe_agent_timeout: int = 1800,
    proxy_port: int = 8080,
    problem_statement_id: Optional[str] = None,
) -> Optional[str]:
    """Execute SWE-Agent CLI and return the generated patch.

    Args:
        config_path: Path to SWE-Agent YAML config file.
        problem_statement: The problem statement text.
        instance_id: Unique instance identifier.
        output_dir: Directory for SWE-Agent output / logs.
        repo_path: Path to the repository (for git diff fallback).
        exec_dir: Working directory for the subprocess
                  (avoids YAML parsing issues with ``docker`` subdir).
        swe_agent_timeout: Overall timeout in seconds.
        proxy_port: ModelProxy port (for logging only).
        problem_statement_id: Optional task id passed to SWE-Agent. Defaults to instance_id.

    Returns:
        Generated patch string, or ``None`` on failure.
    """
    effective_problem_id = problem_statement_id or instance_id

    cmd = [
        "sweagent",
        "run",
        "--config",
        config_path,
        "--problem_statement.text",
        problem_statement,
        "--problem_statement.id",
        effective_problem_id,
    ]

    logger.info(f"[{instance_id}] Executing SWE-Agent (proxy port={proxy_port})...")

    env = os.environ.copy()
    process = None

    try:
        subprocess_start = time.time()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=exec_dir,
        )
        logger.info(f"[{instance_id}] Subprocess created (pid={process.pid}), waiting for completion...")

        # Wait with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=swe_agent_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - subprocess_start
            logger.error(f"[{instance_id}] SWE-Agent timed out after {elapsed:.1f}s (limit={swe_agent_timeout}s)")
            # Graceful: SIGTERM first, escalate to SIGKILL after 15 s
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=15.0)
                logger.info(f"[{instance_id}] SWE-Agent exited gracefully after SIGTERM")
            except asyncio.TimeoutError:
                logger.warning(f"[{instance_id}] SWE-Agent did not exit after SIGTERM, sending SIGKILL")
                process.kill()
                await process.wait()
            return None

        subprocess_elapsed = time.time() - subprocess_start
        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")

        # Persist logs
        _save_logs(output_dir, instance_id, stdout_text, stderr_text)

        if process.returncode != 0:
            logger.error(f"[{instance_id}] SWE-Agent failed (rc={process.returncode}) after {subprocess_elapsed:.1f}s")
            logger.error(f"[{instance_id}] stderr (last 2000): {stderr_text[-2000:]}")
            logger.error(f"[{instance_id}] stdout (last 1000): {stdout_text[-1000:]}")
        else:
            logger.info(f"[{instance_id}] SWE-Agent subprocess completed successfully in {subprocess_elapsed:.1f}s")

        # Extract patch
        extract_start = time.time()
        patch = await _extract_patch(output_dir, instance_id, repo_path)
        logger.info(f"[{instance_id}] Patch extraction took {time.time() - extract_start:.1f}s")

        if patch:
            logger.info(f"[{instance_id}] Successfully extracted patch ({len(patch)} chars)")
        else:
            logger.warning(f"[{instance_id}] No patch found in SWE-Agent output or git diff")

        return patch

    except asyncio.CancelledError:
        logger.warning(f"[{instance_id}] SWE-Agent task cancelled, terminating subprocess...")
        if process is not None and process.returncode is None:
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            except Exception:
                pass
        raise
    except FileNotFoundError:
        logger.error("SWE-Agent not found. Please install it with: pip install sweagent")
        return None
    except Exception as e:
        logger.exception(f"[{instance_id}] Error running SWE-Agent: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_logs(output_dir: str, instance_id: str, stdout_text: str, stderr_text: str) -> None:
    """Persist subprocess stdout/stderr to files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        stdout_path = os.path.join(output_dir, f"{instance_id}.stdout.log")
        stderr_path = os.path.join(output_dir, f"{instance_id}.stderr.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr_text)
        logger.info(f"[{instance_id}] Saved SWE-Agent subprocess logs: stdout={stdout_path}, stderr={stderr_path}")
    except Exception as e:
        logger.warning(f"[{instance_id}] Failed to save subprocess logs: {e}")


async def _extract_patch(output_dir: str, instance_id: str, repo_path: str) -> Optional[str]:
    """Extract patch via PatchExtractor (file → git diff fallback)."""
    extractor = PatchExtractor(
        output_dir=output_dir,
        instance_id=instance_id,
        repo_path=repo_path,
    )
    return await extractor.extract()
