#!/usr/bin/env python3
"""Shared integrity checks for cached Wan training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CURRENT_CACHE_SCHEMA = "phycontext.wan_ti2v_cache.v7"
CANONICAL_CONDITION_FRAME_PROTOCOL = (
    "shared_first_frame_overrides_target_video_frame_zero_before_vae_encode"
)
FULL_RATE_DAS_CACHE_SCHEMAS = frozenset(
    {
        "phycontext.wan_ti2v_cache.v4",
        "phycontext.wan_ti2v_cache.v5",
        "phycontext.wan_ti2v_cache.v6",
        CURRENT_CACHE_SCHEMA,
    }
)
TRACK_CORRESPONDENCE_CACHE_SCHEMAS = frozenset(
    {"phycontext.wan_ti2v_cache.v6", CURRENT_CACHE_SCHEMA}
)
RASTER_STABLE_TRACK_CORRESPONDENCE_CACHE_SCHEMAS = frozenset(
    {CURRENT_CACHE_SCHEMA}
)
GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA = {
    "phycontext.wan_ti2v_cache.v4": (
        "float64_after_source_decode_until_integer_rasterization"
    ),
    "phycontext.wan_ti2v_cache.v5": (
        "float64_after_source_decode_until_integer_rasterization"
    ),
    "phycontext.wan_ti2v_cache.v6": (
        "float64_after_source_decode_until_integer_rasterization"
    ),
    CURRENT_CACHE_SCHEMA: (
        "float64_after_source_decode_until_integer_rasterization"
    ),
}
FLOAT64_GEOMETRY_CACHE_SCHEMAS = frozenset(GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA)
SOURCE_FILE_HASH_CACHE_SCHEMAS = frozenset(
    {
        "phycontext.wan_ti2v_cache.v4",
        "phycontext.wan_ti2v_cache.v5",
        "phycontext.wan_ti2v_cache.v6",
        CURRENT_CACHE_SCHEMA,
    }
)
SUPPORTED_CACHE_SCHEMAS = frozenset(
    {
        "phycontext.wan_ti2v_cache.v3",
        *FULL_RATE_DAS_CACHE_SCHEMAS,
    }
)


def select_cache_source_records(
    source_records: list[dict], selection: dict | None
) -> list[dict]:
    """Reproduce the source-record selection recorded by cache generation."""
    records = list(source_records)
    source_ids = [record.get("sample_id") for record in records]
    if any(not sample_id for sample_id in source_ids):
        raise ValueError("source manifest contains an empty sample_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source manifest contains duplicate sample_ids")
    if selection is None:
        return records
    if not isinstance(selection, dict):
        raise ValueError("Wan cache selection must be an object")
    if selection.get("mode") == "merged_shards":
        shard_selections = selection.get("shard_selections")
        if not isinstance(shard_selections, list) or not shard_selections:
            # Older merged manifests did not preserve the individual shard
            # selections. They were intended to cover the full source manifest.
            return records
        expected_ids = {
            record["sample_id"]
            for shard_selection in shard_selections
            for record in select_cache_source_records(records, shard_selection)
        }
        return [record for record in records if record["sample_id"] in expected_ids]

    selected = records
    split = selection.get("split")
    if split:
        selected = [record for record in selected if record["split"] == split]
    requested_scene_ids = set(selection.get("base_scene_ids") or [])
    if requested_scene_ids:
        known_scene_ids = {record["base_scene_id"] for record in selected}
        missing_scene_ids = sorted(requested_scene_ids - known_scene_ids)
        if missing_scene_ids:
            raise ValueError(
                f"cache selection contains unknown base scenes: {missing_scene_ids[:5]}"
            )
        selected = [
            record
            for record in selected
            if record["base_scene_id"] in requested_scene_ids
        ]
    sweep_axis = selection.get("sweep_axis")
    if sweep_axis:
        selected = [
            record
            for record in selected
            if record["sweep"]["mode"] == "base"
            or record["sweep"]["axis"] == sweep_axis
        ]
    limit_base_scenes = selection.get("limit_base_scenes")
    if limit_base_scenes is not None:
        limit_base_scenes = int(limit_base_scenes)
        if limit_base_scenes <= 0:
            raise ValueError("cache selection limit_base_scenes must be positive")
        scene_ids = list(
            dict.fromkeys(record["base_scene_id"] for record in selected)
        )[:limit_base_scenes]
        allowed_scene_ids = set(scene_ids)
        selected = [
            record
            for record in selected
            if record["base_scene_id"] in allowed_scene_ids
        ]

    shard_count = int(selection.get("shard_count", 1))
    shard_index = int(selection.get("shard_index", 0))
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("cache selection has an invalid shard index/count")
    scene_ids = list(dict.fromkeys(record["base_scene_id"] for record in selected))
    shard_scene_ids = {
        scene_id
        for index, scene_id in enumerate(scene_ids)
        if index % shard_count == shard_index
    }
    selected = [
        record for record in selected if record["base_scene_id"] in shard_scene_ids
    ]
    limit = selection.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("cache selection limit must be positive")
        selected = selected[:limit]
    return selected


def validate_cache_record_coverage(
    cache: dict,
    source_records: list[dict],
    *,
    label: str = "Wan cache",
) -> None:
    """Require exactly one cached record for every selected source record."""
    cache_records = cache.get("records")
    if not isinstance(cache_records, list):
        raise ValueError(f"{label} records must be a list")
    actual_ids = [item.get("sample_id") for item in cache_records]
    if any(not sample_id for sample_id in actual_ids):
        raise ValueError(f"{label} contains an empty sample_id")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"{label} contains duplicate sample_ids")
    expected_records = select_cache_source_records(
        source_records, cache.get("selection")
    )
    expected_ids = [record["sample_id"] for record in expected_records]
    missing = sorted(set(expected_ids) - set(actual_ids))
    unexpected = sorted(set(actual_ids) - set(expected_ids))
    if missing or unexpected:
        raise ValueError(
            f"{label} record coverage mismatch: expected={len(expected_ids)}, "
            f"actual={len(actual_ids)}, missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    expected_by_id = {
        record["sample_id"]: record for record in expected_records
    }
    changed = [
        item["sample_id"]
        for item in cache_records
        if item.get("record") != expected_by_id[item["sample_id"]]
    ]
    if changed:
        raise ValueError(f"{label} embeds changed source records: {changed[:5]}")


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


def resolve_cache_artifact_root(project_root: Path, cache: dict) -> Path:
    """Resolve the base directory used by cache artifact descriptors.

    Manifests written before external cache roots were supported omit both
    fields and remain project-root-relative.  New manifests bind descriptors
    to an explicit cache root so large artifacts can live on data storage
    without weakening path traversal checks.
    """
    path_base = cache.get("artifact_path_base", "project_root")
    value = cache.get("artifact_root")
    if path_base == "project_root":
        if value is not None:
            raise ValueError(
                "Wan cache artifact_root requires artifact_path_base=cache_root"
            )
        artifact_root = project_root.resolve()
    else:
        if path_base != "cache_root":
            raise ValueError(f"unsupported Wan cache artifact path base: {path_base}")
        if not value:
            raise ValueError("Wan cache is missing its cache artifact root")
        path = Path(value).expanduser()
        artifact_root = (
            path.resolve() if path.is_absolute() else (project_root / path).resolve()
        )
    dataset_root = resolve_cache_dataset_root(project_root, cache)
    if artifact_root.is_relative_to(dataset_root):
        raise ValueError("Wan cache artifacts must remain outside the published dataset")
    return artifact_root


def validate_cache_artifact(
    artifact_root: Path,
    descriptor: dict,
    label: str,
    *,
    verify_hash: bool = True,
) -> Path:
    """Resolve one artifact below its declared cache root.

    ``verify_hash=False`` is intended for training startup after an offline cache
    audit has already completed.  Descriptor completeness, path confinement and
    file existence remain mandatory; inference and audit callers retain the
    strict SHA-256 check by default.
    """
    if not isinstance(descriptor, dict):
        raise ValueError(f"Wan cache is missing its {label} descriptor")
    relative_value = descriptor.get("path")
    expected_hash = descriptor.get("sha256")
    if not relative_value or not expected_hash:
        raise ValueError(f"Wan cache {label} descriptor is incomplete")
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Wan cache {label} path must be cache-root-relative")
    root = artifact_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Wan cache {label} artifact is missing: {path}")
    if verify_hash and _sha256(path) != expected_hash:
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
