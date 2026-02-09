# recipe/swe_agent/tests/test_patch_extractor.py
import pytest
import tempfile
import os
from pathlib import Path
from recipe.swe_agent.utils.patch_extractor import PatchExtractor


@pytest.mark.asyncio
async def test_extract_from_patch_file():
    """Test extracting patch from .patch file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a patch file
        instance_id = "test-123"
        patch_dir = Path(tmpdir) / instance_id
        patch_dir.mkdir()
        patch_file = patch_dir / f"{instance_id}.patch"

        expected_patch = "diff --git a/test.py\n--- a/test.py\n+++ b/test.py"
        patch_file.write_text(expected_patch)

        extractor = PatchExtractor(output_dir=tmpdir, instance_id=instance_id)
        patch = await extractor.extract()

        assert patch == expected_patch


@pytest.mark.asyncio
async def test_extract_from_git_diff():
    """Test fallback to git diff when no patch file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        repo_path.mkdir()

        # Initialize git repo
        os.system(f"cd {repo_path} && git init && git config user.email 'test@test.com' && git config user.name 'Test'")

        # Create and commit a file
        test_file = repo_path / "test.py"
        test_file.write_text("original content")
        os.system(f"cd {repo_path} && git add test.py && git commit -m 'init'")

        # Modify the file
        test_file.write_text("modified content")

        extractor = PatchExtractor(
            output_dir=str(tmpdir),
            instance_id="test",
            repo_path=str(repo_path),
        )
        patch = await extractor.extract()

        assert patch is not None
        assert "test.py" in patch
        assert "modified content" in patch or "-original content" in patch


@pytest.mark.asyncio
async def test_extract_returns_none_when_no_changes():
    """Test that extractor returns None when no changes exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = PatchExtractor(
            output_dir=tmpdir,
            instance_id="test",
            repo_path="/nonexistent",
        )
        patch = await extractor.extract()

        assert patch is None
