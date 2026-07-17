"""Client-side HDFS support for ``AutoTokenizer.from_pretrained``."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_HDFS_PREFIX = "hdfs://"
_VERL_DIRECTORY_MARKER = ".directory_record.txt"
_CLIENT_DIRECTORY_MARKER = ".verl_tinker_hdfs_complete"
_MODEL_WEIGHT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)


def _cached_hdfs_path(hdfs_path: str) -> Path:
    """Return the same deterministic local cache path used by VERL."""

    normalized = hdfs_path.rstrip("/")
    cache_root = Path(os.environ.get("TINKER_HDFS_CACHE_DIR", tempfile.gettempdir()))
    cache_key = hashlib.md5(normalized.encode()).hexdigest()
    return cache_root / cache_key / normalized.rsplit("/", 1)[-1]


def _is_complete(local_path: Path) -> bool:
    if local_path.is_file():
        return True
    if not local_path.is_dir():
        return False
    return (local_path / _VERL_DIRECTORY_MARKER).exists() or (local_path / _CLIENT_DIRECTORY_MARKER).exists()


def _list_hdfs_tokenizer_files(hdfs_bin: str, hdfs_path: str) -> list[str]:
    result = subprocess.run(
        [hdfs_bin, "dfs", "-ls", hdfs_path],
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("-"):
            continue
        remote_path = fields[-1]
        if not remote_path.lower().endswith(_MODEL_WEIGHT_SUFFIXES):
            files.append(remote_path)
    if not files:
        raise RuntimeError(f"No tokenizer/config files found in {hdfs_path!r}")
    return files


def _resolve_hdfs_path(hdfs_path: str) -> str:
    normalized = hdfs_path.rstrip("/")
    local_path = _cached_hdfs_path(normalized)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # VERL uses the same MD5 lock name while copying models from HDFS. Using
    # flock here lets the client wait for and then reuse the server-side copy.
    lock_path = local_path.parent.parent / f"{local_path.parent.name}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _is_complete(local_path):
            print(f"Using cached HDFS tokenizer at {local_path}")
            return str(local_path)

        hdfs_bin = os.environ.get("HDFS_BIN") or shutil.which("hdfs")
        if not hdfs_bin:
            raise RuntimeError(
                f"Cannot load tokenizer from {hdfs_path!r}: no completed VERL cache exists at "
                f"{local_path}, and the hdfs executable is not available."
            )

        staging_path = local_path.with_name(f".{local_path.name}.partial-{os.getpid()}")
        if staging_path.exists():
            if staging_path.is_dir():
                shutil.rmtree(staging_path)
            else:
                staging_path.unlink()

        remote_files = _list_hdfs_tokenizer_files(hdfs_bin, normalized)
        print(f"Copying {len(remote_files)} tokenizer/config files from {normalized} to {local_path}")
        try:
            staging_path.mkdir()
            for remote_file in remote_files:
                subprocess.run([hdfs_bin, "dfs", "-get", remote_file, str(staging_path)], check=True)
            (staging_path / _CLIENT_DIRECTORY_MARKER).touch()
            if local_path.is_dir():
                shutil.rmtree(local_path)
            elif local_path.exists():
                local_path.unlink()
            os.replace(staging_path, local_path)
        except BaseException:
            if staging_path.is_dir():
                shutil.rmtree(staging_path, ignore_errors=True)
            elif staging_path.exists():
                staging_path.unlink()
            raise

    return str(local_path)


def install_hdfs_tokenizer_patch() -> None:
    """Teach Tinker and Transformers to resolve HDFS tokenizer paths locally."""

    from transformers import AutoTokenizer

    if not getattr(AutoTokenizer, "_verl_tinker_hdfs_patch_installed", False):
        original_from_pretrained = AutoTokenizer.from_pretrained

        def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
            if isinstance(pretrained_model_name_or_path, str) and pretrained_model_name_or_path.startswith(_HDFS_PREFIX):
                pretrained_model_name_or_path = _resolve_hdfs_path(pretrained_model_name_or_path)
            return original_from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        AutoTokenizer.from_pretrained = classmethod(from_pretrained)
        AutoTokenizer._verl_tinker_hdfs_patch_installed = True

    # Patch the exact Tinker SDK helper used by TrainingClient.get_tokenizer.
    # It strips ``model_name`` at the first colon, which turns an HDFS URI into
    # the invalid Hugging Face model ID ``hdfs``. Resolve before it can do so.
    from tinker.lib.public_interfaces import sampling_client, training_client

    if not getattr(sampling_client, "_verl_tinker_hdfs_patch_installed", False):
        original_load_tokenizer = sampling_client._load_tokenizer_from_model_info

        def load_tokenizer_from_model_info(model_name: str, tokenizer_id: str | None = None):
            hdfs_path = next(
                (
                    value
                    for value in (tokenizer_id, model_name)
                    if isinstance(value, str) and value.startswith(_HDFS_PREFIX)
                ),
                None,
            )
            if hdfs_path is not None:
                local_path = _resolve_hdfs_path(hdfs_path)
                return original_load_tokenizer(local_path, local_path)
            return original_load_tokenizer(model_name, tokenizer_id)

        sampling_client._load_tokenizer_from_model_info = load_tokenizer_from_model_info
        # training_client imported the helper by name, so replace that alias too.
        training_client._load_tokenizer_from_model_info = load_tokenizer_from_model_info
        sampling_client._verl_tinker_hdfs_patch_installed = True

    # Tinker Cookbook strips checkpoint suffixes with ``split(":")[0]``
    # before calling AutoTokenizer. For ``hdfs://...`` that produces the
    # invalid model ID ``hdfs``, so intercept its HF helper before the split.
    from tinker_cookbook import tokenizer_utils

    if not getattr(tokenizer_utils, "_verl_tinker_hdfs_patch_installed", False):
        original_get_hf_tokenizer = tokenizer_utils._get_hf_tokenizer

        def get_hf_tokenizer(model_name: str):
            if model_name.startswith(_HDFS_PREFIX):
                model_name = _resolve_hdfs_path(model_name)
            return original_get_hf_tokenizer(model_name)

        tokenizer_utils._get_hf_tokenizer = get_hf_tokenizer
        tokenizer_utils._verl_tinker_hdfs_patch_installed = True
