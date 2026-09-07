"""Build verl-compatible parquet datasets from local Harbor task directories.

A *Harbor task* is a directory containing at least::

    task.toml
    instruction.md

The directory may also contain ``environment/``, ``tests/``, ``solution/`` etc.,
but only ``task.toml`` and ``instruction.md`` are needed for dataset rows.

This module scans a root directory, picks every immediate subdirectory that
looks like a Harbor task, and writes a parquet file matching verl's RL row
schema::

    {
        "data_source": str,
        "prompt": [{"role": "user", "content": "..."}],
        "ability": str,
        "reward_model": {"style": "rule", "ground_truth": "..."},
        "extra_info": {"split": str, "index": int, "task_id": str, ...},
        "instance_id": str,         # equal to task_id; agent loop uses it
        "local_task_path": str,     # absolute path of the task directory
    }

Inspired by ``harbor_rl_sdk.dataset.verl`` but the OSS code path is dropped on
purpose — we only read from the local filesystem here.
"""

from __future__ import annotations

import hashlib
import logging
import tomllib
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# A directory must contain *all* of these files to be treated as a Harbor task.
_TASK_MARKER_FILES: tuple[str, ...] = ("task.toml", "instruction.md")


# ---------------------------------------------------------------------------
# Local scanning
# ---------------------------------------------------------------------------


def _looks_like_task_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / marker).is_file() for marker in _TASK_MARKER_FILES)


def iter_local_task_dirs(root: str | Path) -> Iterator[Path]:
    """Yield every immediate subdirectory of ``root`` that is a Harbor task.

    The order is deterministic (sorted by directory name) so that re-runs
    produce identical parquet files (and therefore identical hashes).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"harbor task root does not exist: {root_path}")

    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if _looks_like_task_dir(child):
            yield child


def parse_local_task_dir(local_dir: Path) -> dict[str, Any]:
    """Read ``task.toml`` and ``instruction.md`` from a single task dir."""
    task_toml = local_dir / "task.toml"
    instruction_md = local_dir / "instruction.md"
    task_config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    instruction = instruction_md.read_text(encoding="utf-8")
    return {
        "task_id": local_dir.name,
        "local_path": local_dir,
        "instruction": instruction,
        "task_config": task_config,
    }


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def task_to_verl_row(
    task: dict[str, Any],
    index: int,
    split: str,
    *,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Map a parsed local task to one verl RL row."""
    metadata = (
        task["task_config"].get("metadata", {})
        if isinstance(task["task_config"], dict)
        else {}
    )
    info: dict[str, Any] = {
        "split": split,
        "index": index,
        "task_id": task["task_id"],
        "local_path": str(task["local_path"]),
        "task_metadata": metadata,
    }
    if extra_info:
        info.update(extra_info)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": task["instruction"]}],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": task["task_id"],
        },
        "extra_info": info,
        # Top-level fields, transparently passed to AgentLoop.run via kwargs.
        # ``instance_id`` lets RemoteAgentLoop locate the task directory under
        # data.train_harbor_dir / data.val_harbor_dir without any template.
        "instance_id": task["task_id"],
        "local_task_path": str(task["local_path"]),
    }


def build_rows_from_local(
    root: str | Path,
    *,
    split: str,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Scan ``root`` and return verl rows for every Harbor task found."""
    rows: list[dict[str, Any]] = []
    for idx, task_dir in enumerate(iter_local_task_dirs(root)):
        if limit is not None and idx >= limit:
            break
        task = parse_local_task_dir(task_dir)
        rows.append(
            task_to_verl_row(
                task,
                index=idx,
                split=split,
                data_source=data_source,
                ability=ability,
                extra_info=extra_info,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------


def _signature_for(root: Path, split: str, limit: Optional[int]) -> str:
    """Stable short hash describing the inputs that produced a parquet file."""
    raw = f"{root.resolve()}|split={split}|limit={limit}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def build_verl_parquet_from_local(
    root: str | Path,
    *,
    split: str,
    cache_dir: str | Path,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Produce a parquet file under ``cache_dir`` (or ``output_path``).

    The parquet path is deterministic: ``<cache_dir>/<split>-<sig>.parquet``
    where ``sig`` is a hash of (root, split, limit), so calling this function
    twice with the same arguments and ``overwrite=False`` is a no-op.

    Returns the absolute path of the parquet file.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Building verl parquet from a Harbor directory requires the "
            "'datasets' package. Install via: pip install datasets"
        ) from exc

    root_path = Path(root)
    if output_path is not None:
        out = Path(output_path)
    else:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        out = cache_root / f"{split}-{_signature_for(root_path, split, limit)}.parquet"

    if out.exists() and not overwrite:
        logger.info("Reusing cached harbor parquet: %s", out)
        return out

    rows = build_rows_from_local(
        root_path,
        split=split,
        data_source=data_source,
        ability=ability,
        extra_info=extra_info,
        limit=limit,
    )
    if not rows:
        raise ValueError(
            f"no harbor tasks found under {root_path}; "
            "expected subdirectories containing task.toml and instruction.md"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(out))
    logger.info("Wrote %d harbor task rows to %s", len(rows), out)
    return out


def build_verl_parquets(
    train_root: Optional[str | Path],
    val_root: Optional[str | Path],
    *,
    cache_dir: str | Path,
    data_source: str = "harbor",
    ability: str = "agent",
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    overwrite: bool = False,
) -> tuple[Optional[Path], Optional[Path]]:
    """Convenience helper that builds train/val parquet files in one call.

    Either root may be ``None`` — in that case the corresponding parquet path
    is returned as ``None`` and the caller should keep its existing dataset.
    """
    train_path = (
        build_verl_parquet_from_local(
            train_root,
            split="train",
            cache_dir=cache_dir,
            data_source=data_source,
            ability=ability,
            limit=train_limit,
            overwrite=overwrite,
        )
        if train_root
        else None
    )
    val_path = (
        build_verl_parquet_from_local(
            val_root,
            split="val",
            cache_dir=cache_dir,
            data_source=data_source,
            ability=ability,
            limit=val_limit,
            overwrite=overwrite,
        )
        if val_root
        else None
    )
    return train_path, val_path


__all__ = [
    "iter_local_task_dirs",
    "parse_local_task_dir",
    "task_to_verl_row",
    "build_rows_from_local",
    "build_verl_parquet_from_local",
    "build_verl_parquets",
]
