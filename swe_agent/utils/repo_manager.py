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
Temporary repository management for SWE-Agent.

Creates ephemeral git repos from ``repo_content`` dicts (used by the simple
test-data path) and cleans them up after the agent run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


async def create_temp_repo(repo_content: dict[str, Optional[str]]) -> str:
    """Create a temporary git repository from a file-content mapping.

    Args:
        repo_content: Mapping of ``{relative_path: file_content}``.
                      If *content* is ``None``, a stub header is written.

    Returns:
        Absolute path to the new temporary repo directory.

    Raises:
        Exception: If git init / commit fails.
    """
    temp_dir = tempfile.mkdtemp(prefix="swe_repo_")

    try:
        abs_temp_dir = os.path.abspath(temp_dir)
        for file_path, content in repo_content.items():
            full_path = os.path.abspath(os.path.join(abs_temp_dir, file_path))
            # Path traversal protection
            if not full_path.startswith(abs_temp_dir + os.sep) and full_path != abs_temp_dir:
                logger.error(f"Skipping invalid file path (path traversal attempt): {file_path}")
                continue

            parent_dir = os.path.dirname(full_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            if content is None:
                content = f"# {file_path}\n"

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Initialize git repository
        cmds = [
            f"cd {temp_dir} && git init",
            f"cd {temp_dir} && git config user.email 'verl@swe-agent.local'",
            f"cd {temp_dir} && git config user.name 'VERL SWE-Agent'",
            f"cd {temp_dir} && git add -A",
            f"cd {temp_dir} && git commit --allow-empty -m 'Initial commit'",
        ]

        for cmd in cmds:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace")
                stdout_text = stdout.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Command failed (rc={proc.returncode}): {cmd}\n"
                    f"stderr={stderr_text[-500:]}\n"
                    f"stdout={stdout_text[-500:]}"
                )

        logger.debug(f"Created temp repo with {len(repo_content)} files at {temp_dir}")
        return temp_dir

    except Exception as e:
        logger.exception(f"Failed to create temp repo: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def cleanup_temp_repo(repo_path: Optional[str]) -> None:
    """Remove a temporary repository directory (no-op if *repo_path* is ``None``)."""
    if repo_path is None:
        return
    try:
        shutil.rmtree(repo_path, ignore_errors=True)
        logger.debug(f"Cleaned up temporary repo: {repo_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp repo: {e}")
