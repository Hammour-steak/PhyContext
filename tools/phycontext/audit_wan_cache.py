#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from schema import iter_jsonl, validate_training_record
from project_defaults import CACHE_MANIFEST
from cache_contract import (
    CANONICAL_CONDITION_FRAME_PROTOCOL,
    CURRENT_CACHE_SCHEMA,
    FLOAT64_GEOMETRY_CACHE_SCHEMAS,
    FULL_RATE_DAS_CACHE_SCHEMAS,
    GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA,
    SOURCE_FILE_HASH_CACHE_SCHEMAS,
    SUPPORTED_CACHE_SCHEMAS,
    resolve_cache_artifact_root,
    resolve_cache_dataset_root,
    validate_cache_source_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = CACHE_MANIFEST


def record_dynamic_object_ids(record: dict) -> list[str]:
    physics = record["conditioning"]["physics"]
    objects = physics.get("objects")
    if objects is None:
        objects = [physics["object"]]
    elif isinstance(objects, dict):
        objects = [objects[key] for key in sorted(objects)]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 3:
        raise ValueError("record must contain one to three dynamic objects")
    return [str(item["object_id"]) for item in objects]


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
    cache_schema = cache.get("schema")
    if cache_schema not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError("cache uses an unsupported trajectory schema")
    dataset_root = resolve_cache_dataset_root(root, cache)
    artifact_root = resolve_cache_artifact_root(root, cache)
    errors = []
    if (
        cache_schema == CURRENT_CACHE_SCHEMA
        and cache.get("preprocess", {}).get("condition_frame")
        != CANONICAL_CONDITION_FRAME_PROTOCOL
    ):
        errors.append("cache does not use the canonical TI2V condition-frame protocol")
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
        raw_point_records = point_manifest.get("records", [])
        if not isinstance(raw_point_records, list):
            errors.append("source point trajectory manifest records are not a list")
            raw_point_records = []
        point_sample_ids = [item.get("sample_id") for item in raw_point_records]
        if any(not sample_id for sample_id in point_sample_ids):
            errors.append("source point trajectory manifest has an empty sample id")
        if len(set(point_sample_ids)) != len(point_sample_ids):
            errors.append("source point trajectory manifest has duplicate samples")
        if (
            "record_count" in point_manifest
            and int(point_manifest["record_count"]) != len(raw_point_records)
        ):
            errors.append("source point trajectory manifest record count mismatch")
        point_records = dict(zip(point_sample_ids, raw_point_records))
    if "reused_cache_manifest" in cache:
        reused_value = Path(cache["reused_cache_manifest"])
        reused_cache = (
            reused_value if reused_value.is_absolute() else root / reused_value
        ).resolve()
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
    point_preprocess = cache.get("point_track_preprocess", {})
    trajectory_representation = point_preprocess.get(
        "representation", "dense_point_tracks"
    )
    legacy_point_preprocess = {
        "source_schema": "physweep.point_trajectories.v1",
        "max_objects": 3,
        "frame_selection": "evenly_spaced_from_first_to_last",
        "spatial_grid": "Wan_latent_grid",
        "spatial_transform": "cover_then_center_crop_to_preprocess_size",
        "preprocess_size_px": [
            int(cache["preprocess"]["width"]),
            int(cache["preprocess"]["height"]),
        ],
    }
    if cache_schema in FULL_RATE_DAS_CACHE_SCHEMAS:
        legacy_point_preprocess["pixel_center_convention"] = "opencv_half_pixel"
    if cache_schema in FLOAT64_GEOMETRY_CACHE_SCHEMAS:
        legacy_point_preprocess["geometry_compute_dtype"] = (
            GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA[cache_schema]
        )
    if trajectory_representation == "dense_point_tracks":
        expected_point_channels = 18
        expected_point_preprocess = {
            **legacy_point_preprocess,
            "channels_per_object": 6,
        }
        expected_point_frames = expected_shape[1]
    elif trajectory_representation == "das_3d_tracks":
        expected_point_channels = 12
        if cache_schema not in FULL_RATE_DAS_CACHE_SCHEMAS:
            errors.append(
                "das_3d_tracks requires the current full-rate DaS cache schema"
            )
        expected_point_preprocess = {
            "source_schema": "physweep.point_trajectories.v1",
            "representation": "das_3d_tracks",
            "channels_per_object": 4,
            "channels": [
                "identity_r",
                "identity_g",
                "identity_b",
                "visibility",
            ],
            "max_objects": 3,
            "frame_selection": "same_evenly_spaced_indices_as_preprocessed_video",
            "source_frame_alignment": "trajectory_count_equals_source_video_count",
            "spatial_grid": "Wan_VAE_grid_after_full_resolution_visibility",
            "spatial_transform": "cover_then_center_crop_to_preprocess_size",
            "pixel_center_convention": "opencv_half_pixel",
            "geometry_compute_dtype": GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA[
                CURRENT_CACHE_SCHEMA
            ],
            "preprocess_size_px": [
                int(cache["preprocess"]["width"]),
                int(cache["preprocess"]["height"]),
            ],
            "point_identity": "first_frame_camera_xyz_with_reciprocal_z_rgb",
            "color_normalization": "shared_xy_minmax_inverse_z_2_98_percentile",
            "visibility": "full_resolution_dynamic_and_static_nearest_depth_z_buffer",
            "visibility_depth_range": "source_camera_clip_start_end",
            "spatial_reduction": "visible_point_identity_mean_and_binary_occupancy",
            "temporal_encoding": "conditioner_first_frame_plus_learned_stride4_windows",
            "point_splat_radius_px": 1,
        }
        expected_point_frames = int(cache["preprocess"]["frames"])
    else:
        expected_point_channels = -1
        expected_point_preprocess = {}
        errors.append(
            f"unsupported point-track representation: {trajectory_representation}"
        )
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

        if cache_schema in SOURCE_FILE_HASH_CACHE_SCHEMAS:
            for record_key, hash_key, label in (
                ("first_frame", "source_first_frame_sha256", "first frame"),
                ("scene", "source_scene_sha256", "scene condition"),
            ):
                source_path = dataset_root / record["conditioning"][record_key]
                expected_hash = item.get(hash_key)
                if (
                    not expected_hash
                    or not source_path.is_file()
                    or sha256(source_path) != expected_hash
                ):
                    errors.append(f"{label} file/hash mismatch: {sample_id}")

        latent_path = artifact_root / item["latent"]["path"]
        text_path = artifact_root / item["text_context"]["path"]
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
        if cache_schema == CURRENT_CACHE_SCHEMA:
            latent_descriptor = item.get("latent", {})
            if (
                latent_descriptor.get("condition_frame_protocol")
                != CANONICAL_CONDITION_FRAME_PROTOCOL
                or latent_descriptor.get("condition_frame_sha256")
                != item.get("source_first_frame_sha256")
            ):
                errors.append(f"latent condition-frame binding mismatch: {sample_id}")
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
            expected_object_ids = record_dynamic_object_ids(record)
            if point_record.get("object_ids") != expected_object_ids:
                errors.append(f"point-track object binding mismatch: {sample_id}")
            if int(point_record.get("object_count", -1)) != len(expected_object_ids):
                errors.append(f"point-track object count mismatch: {sample_id}")
            if expected_point_record.get("object_ids") != expected_object_ids:
                errors.append(f"point trajectory object binding mismatch: {sample_id}")
            if int(expected_point_record.get("object_count", -1)) != len(
                expected_object_ids
            ):
                errors.append(f"point trajectory object count mismatch: {sample_id}")
            if point_record.get("source_point_trajectory") != expected_point_record.get("path"):
                errors.append(f"point trajectory binding mismatch: {sample_id}")
            if point_record.get("source_point_trajectory_sha256") != expected_point_record.get("sha256"):
                errors.append(f"point trajectory hash binding mismatch: {sample_id}")
            trajectory_relative = expected_point_record.get("path")
            trajectory_hash = expected_point_record.get("sha256")
            trajectory_path = (
                (dataset_root / trajectory_relative).resolve()
                if trajectory_relative
                else dataset_root
            )
            if (
                not trajectory_relative
                or not trajectory_path.is_relative_to(dataset_root)
                or not trajectory_path.is_file()
                or sha256(trajectory_path) != trajectory_hash
            ):
                errors.append(f"source point trajectory file/hash mismatch: {sample_id}")
            point_path = artifact_root / point_record["path"]
            if not point_path.is_file() or sha256(point_path) != point_record["sha256"]:
                errors.append(f"point-track file/hash mismatch: {sample_id}")
            else:
                point_track = load_file(str(point_path), device="cpu").get(
                    "point_track_map"
                )
                point_track_shapes.add(tuple(point_track.shape) if point_track is not None else ())
                expected_point_shape = (
                    expected_point_channels,
                    expected_point_frames,
                    *expected_shape[2:],
                )
                if point_track is None or tuple(point_track.shape) != expected_point_shape:
                    errors.append(f"point-track shape mismatch: {sample_id}")
                elif point_track.dtype != torch.float32 or not torch.isfinite(point_track).all():
                    errors.append(f"invalid point-track values: {sample_id}")
                elif trajectory_representation == "das_3d_tracks":
                    slots = point_track.reshape(3, 4, *point_track.shape[1:])
                    rgb = slots[:, :3]
                    visibility = slots[:, 3:4]
                    if bool((point_track < -1.0e-6).any()) or bool(
                        (point_track > 1.0 + 1.0e-6).any()
                    ):
                        errors.append(f"DaS point-track range mismatch: {sample_id}")
                    if not bool(
                        torch.logical_or(
                            visibility == 0, visibility == 1
                        ).all()
                    ):
                        errors.append(
                            f"DaS point-track visibility is nonbinary: {sample_id}"
                        )
                    background_rgb = rgb.masked_select(
                        visibility.expand_as(rgb) == 0
                    )
                    if len(background_rgb) and bool(
                        (background_rgb.abs() > 1.0e-6).any()
                    ):
                        errors.append(
                            f"DaS point-track background RGB is nonzero: {sample_id}"
                        )
                    object_count = int(point_record.get("object_count", -1))
                    if 1 <= object_count <= 3 and bool(
                        (
                            slots[object_count:].abs() > 1.0e-6
                        ).any()
                    ):
                        errors.append(
                            f"unused DaS point-track slots are nonzero: {sample_id}"
                        )
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
        "trajectory_representation": trajectory_representation,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
