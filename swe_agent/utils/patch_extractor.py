"""
Unified Patch Extractor

Provides a single, clean interface for extracting patches from SWE-Agent runs.
Tries multiple strategies in order:
1. SWE-Agent output .patch file
2. git diff HEAD (staged + unstaged)
3. git diff (unstaged only)
"""

import asyncio
import glob
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class PatchExtractor:
    """Unified patch extraction utility.

    Simplifies patch extraction by trying multiple strategies
    in a clean, testable way.
    """

    def __init__(
        self,
        output_dir: str,
        instance_id: str,
        repo_path: Optional[str] = None,
    ):
        """Initialize patch extractor.

        Args:
            output_dir: SWE-Agent output directory.
            instance_id: Instance identifier.
            repo_path: Optional repository path for git diff fallback.
        """
        self.output_dir = output_dir
        self.instance_id = instance_id
        self.repo_path = repo_path

    async def extract(self) -> Optional[str]:
        """Extract patch using multiple strategies.

        Returns:
            Patch content string or None if no patch found.
        """
        # Strategy 1: Try to read .patch file
        patch = await self._try_patch_file()
        if patch:
            logger.info(f"Extracted patch from file ({len(patch)} chars)")
            return patch

        # Strategy 2: Fallback to git diff
        if self.repo_path:
            patch = await self._try_git_diff()
            if patch:
                logger.info(f"Extracted patch from git diff ({len(patch)} chars)")
                return patch

        logger.warning("No patch found via any strategy")
        return None

    async def _try_patch_file(self) -> Optional[str]:
        """Try to read patch from SWE-Agent output file."""
        patch_patterns = [
            os.path.join(self.output_dir, self.instance_id, f"{self.instance_id}.patch"),
            os.path.join(self.output_dir, f"{self.instance_id}.patch"),
            os.path.join(self.output_dir, "*.patch"),
        ]

        for pattern in patch_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    return self._read_patch_file(matches[0])
            elif os.path.exists(pattern):
                return self._read_patch_file(pattern)

        return None

    def _read_patch_file(self, path: str) -> Optional[str]:
        """Read and validate patch file."""
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logger.debug(f"Read patch from {path}")
                    return content
        except Exception as e:
            logger.error(f"Failed to read patch file {path}: {e}")
        return None

    async def _try_git_diff(self) -> Optional[str]:
        """Try to extract patch using git diff."""
        if not self.repo_path or not os.path.isdir(self.repo_path):
            return None

        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            logger.debug(f"Not a git repository: {self.repo_path}")
            return None

        # Try git diff HEAD first (includes staged + unstaged)
        patch = await self._run_git_diff("HEAD")
        if patch:
            return patch

        # Try git diff (unstaged only)
        patch = await self._run_git_diff()
        return patch

    async def _run_git_diff(self, ref: Optional[str] = None) -> Optional[str]:
        """Run git diff command."""
        cmd = ["git", "diff"]
        if ref:
            cmd.append(ref)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            if process.returncode == 0 and stdout:
                patch = stdout.decode("utf-8", errors="replace").strip()
                if patch:
                    ref_str = f" {ref}" if ref else ""
                    logger.debug(f"git diff{ref_str} returned {len(patch)} chars")
                    return patch
        except asyncio.TimeoutError:
            logger.error("git diff timed out")
        except Exception as e:
            logger.error(f"git diff failed: {e}")

        return None
