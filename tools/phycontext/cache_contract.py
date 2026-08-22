#!/usr/bin/env python3
"""Shared integrity checks for cached Wan training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CURRENT_CACHE_SCHEMA = "phycontext.wan_ti2v_cache.v4"
FULL_RATE_DAS_CACHE_SCHEMAS = frozenset(
    {
        CURRENT_CACHE_SCHEMA,
    }
)
GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA = {
    CURRENT_CACHE_SCHEMA: (
        "float64_after_source_decode_until_integer_rasterization"
    ),
}
FLOAT64_GEOMETRY_CACHE_SCHEMAS = frozenset(GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA)
SUPPORTED_CACHE_SCHEMAS = frozenset(
    {
        "phycontext.wan_ti2v_cache.v3",
        CURRENT_CACHE_SCHEMA,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_cache_dataset_root(project_root: Path, cache: dict) -> Path:
    value = cache.get("dataset_root")
    if not value:
        raise ValueError("Wan cache is missing its external dataset root")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_cache_artifact(
    project_root: Path,
    descriptor: dict,
    label: str,
) -> Path:
    """Resolve and hash-check one project-local cache artifact."""
    if not isinstance(descriptor, dict):
        raise ValueError(f"Wan cache is missing its {label} descriptor")
    relative_value = descriptor.get("path")
    expected_hash = descriptor.get("sha256")
    if not relative_value or not expected_hash:
        raise ValueError(f"Wan cache {label} descriptor is incomplete")
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Wan cache {label} path must be project-root-relative")
    root = project_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Wan cache {label} artifact is missing: {path}")
    if _sha256(path) != expected_hash:
        raise ValueError(f"Wan cache {label} artifact hash mismatch: {path}")
    return path


def validate_cache_source_manifest(project_root: Path, cache: dict) -> Path:
    """Reject a cache when its source manifest has changed since preprocessing."""
    if cache.get("schema") not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError("Wan cache uses an unsupported schema")
    dataset_root = resolve_cache_dataset_root(project_root, cache)
    source_value = cache.get("source_manifest")
    expected_hash = cache.get("source_manifest_sha256")
    if not source_value or not expected_hash:
        raise ValueError("Wan cache is missing its source manifest contract")

    source_path = Path(source_value).expanduser()
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError("source manifest path must be dataset-root-relative")
    source_path = (dataset_root / source_path).resolve()
    if not source_path.is_relative_to(dataset_root) or not source_path.is_file():
        raise FileNotFoundError(f"Wan cache source manifest is missing: {source_path}")

    actual_hash = _sha256(source_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "stale Wan cache: source manifest changed after preprocessing; "
            "rebuild the cache before training "
            f"(cached={expected_hash}, current={actual_hash})"
        )
    summary_value = cache.get("source_dataset_summary")
    summary_hash = cache.get("source_dataset_summary_sha256")
    if not summary_value or not summary_hash:
        raise ValueError("Wan cache is missing its PhysSweep dataset summary contract")
    summary_path = Path(summary_value).expanduser()
    if summary_path.is_absolute() or ".." in summary_path.parts:
        raise ValueError("PhysSweep dataset summary path must be dataset-root-relative")
    summary_path = (dataset_root / summary_path).resolve()
    if not summary_path.is_relative_to(dataset_root) or not summary_path.is_file():
        raise FileNotFoundError(f"Wan cache dataset summary is missing: {summary_path}")
    if _sha256(summary_path) != summary_hash:
        raise ValueError("stale Wan cache: PhysSweep dataset summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("path_base") != "physweep_project_root":
        raise ValueError("PhysSweep dataset summary has an unsupported path base")
    if summary.get("manifest") != source_path.relative_to(dataset_root).as_posix():
        raise ValueError("PhysSweep dataset summary references a different manifest")
    if summary.get("manifest_sha256") != expected_hash:
        raise ValueError("PhysSweep dataset summary manifest hash mismatch")
    point_value = cache.get("source_point_trajectory_manifest")
    point_hash = cache.get("source_point_trajectory_manifest_sha256")
    if not point_value or not point_hash:
        raise ValueError("Wan cache is missing its point-trajectory contract")
    point_path = Path(point_value).expanduser()
    if point_path.is_absolute() or ".." in point_path.parts:
        raise ValueError("point-trajectory manifest path must be dataset-root-relative")
    point_path = (dataset_root / point_path).resolve()
    if not point_path.is_relative_to(dataset_root) or not point_path.is_file():
        raise FileNotFoundError(
            f"Wan cache point-trajectory manifest is missing: {point_path}"
        )
    if _sha256(point_path) != point_hash:
        raise ValueError("stale Wan cache: point-trajectory manifest changed")
    return source_path
