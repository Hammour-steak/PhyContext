#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from schema import iter_jsonl, validate_training_record
from project_defaults import CACHE_MANIFEST
from cache_contract import resolve_cache_dataset_root, validate_cache_source_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = CACHE_MANIFEST


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached Wan TI2V training inputs")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--skip-video-hashes", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    cache_path = (root / args.cache_manifest).resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema") != "phycontext.wan_ti2v_cache.v3":
        raise ValueError("cache must use the dense-point-track v3 schema")
    dataset_root = resolve_cache_dataset_root(root, cache)
    errors = []
    try:
        source_manifest = validate_cache_source_manifest(root, cache)
    except (ValueError, FileNotFoundError) as error:
        raise ValueError(f"invalid Wan cache source contract: {error}") from error
    point_records = {}
    source_point_manifest = dataset_root / cache["source_point_trajectory_manifest"]
    if (
        not source_point_manifest.is_file()
        or sha256(source_point_manifest)
        != cache["source_point_trajectory_manifest_sha256"]
    ):
        errors.append("source point trajectory manifest hash mismatch")
    else:
        point_manifest = json.loads(source_point_manifest.read_text(encoding="utf-8"))
        if point_manifest.get("schema") != "physweep.point_trajectory_manifest.v1":
            errors.append("source point trajectory manifest schema mismatch")
        point_records = {
            item["sample_id"]: item for item in point_manifest.get("records", [])
        }
    if "reused_cache_manifest" in cache:
        reused_cache = root / cache["reused_cache_manifest"]
        if (
            not reused_cache.is_file()
            or sha256(reused_cache) != cache["reused_cache_manifest_sha256"]
        ):
            errors.append("reused cache manifest hash mismatch")
    source_records = {
        record["sample_id"]: record for record in iter_jsonl(source_manifest)
    }
    expected_shape = (
        48,
        (int(cache["preprocess"]["frames"]) - 1) // 4 + 1,
        int(cache["preprocess"]["height"]) // 16,
        int(cache["preprocess"]["width"]) // 16,
    )
    expected_point_preprocess = {
        "source_schema": "physweep.point_trajectories.v1",
        "channels_per_object": 6,
        "max_objects": 3,
        "frame_selection": "evenly_spaced_from_first_to_last",
        "spatial_grid": "Wan_latent_grid",
        "spatial_transform": "cover_then_center_crop_to_preprocess_size",
        "preprocess_size_px": [
            int(cache["preprocess"]["width"]),
            int(cache["preprocess"]["height"]),
        ],
    }
    if cache.get("point_track_preprocess") != expected_point_preprocess:
        errors.append("point-track preprocessing contract mismatch; cache must be rebuilt")
    sample_ids = set()
    latent_paths = set()
    text_paths = set()
    base_ids = set()
    point_track_shapes = set()
    for item in cache["records"]:
        sample_id = item["sample_id"]
        if sample_id in sample_ids:
            errors.append(f"duplicate cache sample: {sample_id}")
            continue
        sample_ids.add(sample_id)
        base_ids.add(item["base_scene_id"])
        record = item["record"]
        if source_records.get(sample_id) != record:
            errors.append(f"cached source record changed: {sample_id}")
        try:
            validate_training_record(record, dataset_root, check_files=True)
        except ValueError as error:
            errors.append(f"invalid training record {sample_id}: {error}")

        latent_path = root / item["latent"]["path"]
        text_path = root / item["text_context"]["path"]
        if item["latent"]["path"] in latent_paths:
            errors.append(f"latent path is reused: {item['latent']['path']}")
        latent_paths.add(item["latent"]["path"])
        text_paths.add(item["text_context"]["path"])
        for path, expected_hash, label in (
            (latent_path, item["latent"]["sha256"], "latent"),
            (text_path, item["text_context"]["sha256"], "text"),
        ):
            if not path.is_file() or sha256(path) != expected_hash:
                errors.append(f"{label} file/hash mismatch: {sample_id}")
        if latent_path.is_file():
            latent = load_file(str(latent_path), device="cpu")["latent"]
            if tuple(latent.shape) != expected_shape:
                errors.append(
                    f"latent shape mismatch: {sample_id} {tuple(latent.shape)}"
                )
            if latent.dtype != torch.bfloat16 or not torch.isfinite(latent.float()).all():
                errors.append(f"invalid latent values: {sample_id}")
        if text_path.is_file():
            context = load_file(str(text_path), device="cpu")["context"]
            if context.ndim != 2 or context.shape[1] != 4096 or context.shape[0] > 512:
                errors.append(f"text context shape mismatch: {sample_id}")
            if context.dtype != torch.bfloat16 or not torch.isfinite(context.float()).all():
                errors.append(f"invalid text context values: {sample_id}")
        point_record = item.get("point_track")
        expected_point_record = point_records.get(sample_id)
        if not isinstance(point_record, dict) or expected_point_record is None:
            errors.append(f"missing point-track record: {sample_id}")
        else:
            expected_object_ids = [
                record["conditioning"]["physics"]["object"]["object_id"]
            ]
            if point_record.get("object_ids") != expected_object_ids:
                errors.append(f"point-track object binding mismatch: {sample_id}")
            if expected_point_record.get("object_ids") != expected_object_ids:
                errors.append(f"point trajectory object binding mismatch: {sample_id}")
            if point_record.get("source_point_trajectory") != expected_point_record.get("path"):
                errors.append(f"point trajectory binding mismatch: {sample_id}")
            if point_record.get("source_point_trajectory_sha256") != expected_point_record.get("sha256"):
                errors.append(f"point trajectory hash binding mismatch: {sample_id}")
            point_path = root / point_record["path"]
            if not point_path.is_file() or sha256(point_path) != point_record["sha256"]:
                errors.append(f"point-track file/hash mismatch: {sample_id}")
            else:
                point_track = load_file(str(point_path), device="cpu").get(
                    "point_track_map"
                )
                point_track_shapes.add(tuple(point_track.shape) if point_track is not None else ())
                expected_point_shape = (18, *expected_shape[1:])
                if point_track is None or tuple(point_track.shape) != expected_point_shape:
                    errors.append(f"point-track shape mismatch: {sample_id}")
                elif point_track.dtype != torch.float32 or not torch.isfinite(point_track).all():
                    errors.append(f"invalid point-track values: {sample_id}")
        if not args.skip_video_hashes:
            video_path = dataset_root / record["target"]["video"]
            if sha256(video_path) != item["source_video_sha256"]:
                errors.append(f"source video hash mismatch: {sample_id}")

    report = {
        "passed": not errors,
        "cache_manifest": str(cache_path),
        "sample_count": len(sample_ids),
        "base_scene_count": len(base_ids),
        "latent_count": len(latent_paths),
        "unique_text_context_count": len(text_paths),
        "expected_latent_shape": list(expected_shape),
        "point_track_shapes": [list(shape) for shape in sorted(point_track_shapes)],
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
