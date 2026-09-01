#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from safetensors.torch import save_file

from cache_contract import (
    CANONICAL_CONDITION_FRAME_PROTOCOL,
    CENTER_PIXEL_TRACK_CORRESPONDENCE_CACHE_SCHEMAS,
    CURRENT_CACHE_SCHEMA,
    GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA,
    SUPPORTED_CACHE_SCHEMAS,
    resolve_cache_artifact_root,
)
from point_trajectory import (
    DAS_TRACK_CHANNELS_PER_OBJECT,
    POINT_TRAJECTORY_SCHEMA,
    TRACK_CHANNELS_PER_OBJECT,
    build_das_track_correspondence,
    rasterize_das_3d_tracks,
    rasterize_projected_tracks,
)
from project_defaults import (
    CACHE_ROOT,
    DATASET_MANIFEST,
    DATASET_ROOT,
    POINT_TRAJECTORY_MANIFEST,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from schema import SWEEP_AXES, iter_jsonl
from track_correspondence import validate_track_correspondence
from video_preprocess import cover_center_crop_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA = CURRENT_CACHE_SCHEMA
REUSABLE_CACHE_SCHEMAS = set(SUPPORTED_CACHE_SCHEMAS)
DEFAULT_DATASET = DATASET_ROOT
DEFAULT_CACHE = CACHE_ROOT
DEFAULT_WAN_REPO = Path(os.environ.get("PHYCONTEXT_WAN_REPO", "external/Wan2.2"))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("PHYCONTEXT_WAN_CHECKPOINT", "checkpoints/Wan2.2-TI2V-5B")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Wan2.2 TI2V video latents and T5 contexts"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET,
        required=DEFAULT_DATASET is None,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_MANIFEST,
        help="Source manifest relative to the PhysSweep project root",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--point-trajectory-manifest",
        type=Path,
        default=POINT_TRAJECTORY_MANIFEST,
        help="Manifest produced by export_point_trajectories.py",
    )
    parser.add_argument("--reuse-cache-manifest", type=Path)
    parser.add_argument("--wan-repo", type=Path, default=DEFAULT_WAN_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--base-scene-id", action="append", default=[])
    parser.add_argument("--sweep-axis", choices=SWEEP_AXES)
    parser.add_argument("--limit-base-scenes", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument("--frames", type=int, default=VIDEO_FRAMES)
    parser.add_argument(
        "--trajectory-representation",
        choices=("das_3d_tracks", "dense_point_tracks"),
        default="das_3d_tracks",
        help="point-track rasterization stored in the Wan cache",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evenly_spaced_frame_indices(
    source_frames: int, output_frames: int
) -> list[int]:
    """Select frames without silently duplicating a shorter source sequence."""
    if source_frames <= 0 or output_frames <= 0:
        raise ValueError("source and output frame counts must be positive")
    if source_frames < output_frames:
        raise ValueError(
            "source sequence has fewer frames than the requested Wan input"
        )
    return np.rint(
        np.linspace(0, source_frames - 1, output_frames)
    ).astype(np.int64).tolist()


def trajectory_frame_indices(
    representation: str,
    source_frames: int,
    video_frames: int,
    latent_frames: int,
) -> list[int]:
    if representation == "das_3d_tracks":
        return evenly_spaced_frame_indices(source_frames, video_frames)
    if representation == "dense_point_tracks":
        return evenly_spaced_frame_indices(source_frames, latent_frames)
    raise ValueError(f"unsupported trajectory representation: {representation}")


def expected_point_track_shape(
    representation: str,
    video_frames: int,
    latent_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if representation == "das_3d_tracks":
        channels = 3 * DAS_TRACK_CHANNELS_PER_OBJECT
        frames = video_frames
    elif representation == "dense_point_tracks":
        channels = 3 * TRACK_CHANNELS_PER_OBJECT
        frames = latent_shape[1]
    else:
        raise ValueError(f"unsupported trajectory representation: {representation}")
    return channels, frames, latent_shape[2], latent_shape[3]


def record_dynamic_object_ids(record: dict) -> list[str]:
    physics = record["conditioning"]["physics"]
    objects = physics.get("objects")
    if objects is None:
        objects = [physics["object"]]
    elif isinstance(objects, dict):
        objects = [objects[key] for key in sorted(objects)]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 3:
        raise ValueError("training record must contain one to three dynamic objects")
    object_ids = [str(item["object_id"]) for item in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("training record dynamic object ids must be unique")
    return object_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def portable_path(path: Path, root: Path) -> str:
    """Prefer a project-relative path, retaining absolute external paths."""
    try:
        return relative(path, root)
    except ValueError:
        return path.resolve().as_posix()


def descriptor_matches_file(
    descriptor: dict | None,
    path: Path,
    root: Path,
) -> bool:
    """Trust a cached artifact only when its manifest binding still verifies."""
    if not isinstance(descriptor, dict) or not path.is_file():
        return False
    return (
        descriptor.get("path") == relative(path, root)
        and descriptor.get("sha256") == sha256(path)
    )


def materialize_reusable_artifact(source: Path, target: Path) -> None:
    """Atomically place a verified reusable artifact below the new cache root."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_text_artifact(
    prompt_hash: str,
    text_root: Path,
    artifact_root: Path,
    reuse_artifact_root: Path,
    cached_text_by_hash: dict[str, dict],
    reused_text_by_hash: dict[str, dict],
    overwrite: bool,
) -> bool:
    """Materialize a trusted reusable context or request a fresh encoding."""
    target = text_root / f"{prompt_hash}.safetensors"
    return prepare_reusable_artifact(
        target,
        artifact_root,
        reuse_artifact_root,
        cached_text_by_hash.get(prompt_hash),
        reused_text_by_hash.get(prompt_hash),
        overwrite,
    )


def prepare_reusable_artifact(
    target: Path,
    artifact_root: Path,
    reuse_artifact_root: Path,
    current: dict | None,
    reusable: dict | None,
    overwrite: bool,
) -> bool:
    """Materialize one hash-bound artifact or request a fresh build."""
    current_matches = descriptor_matches_file(
        current, target, artifact_root
    )
    reusable_path = (
        reuse_artifact_root / reusable["path"]
        if isinstance(reusable, dict) and reusable.get("path")
        else None
    )
    reusable_matches = bool(
        reusable_path is not None
        and descriptor_matches_file(
            reusable, reusable_path, reuse_artifact_root
        )
    )
    if (
        reusable_matches
        and not overwrite
        and not current_matches
        and not target.exists()
    ):
        materialize_reusable_artifact(reusable_path, target)
        current_matches = descriptor_matches_file(
            {
                "path": relative(target, artifact_root),
                "sha256": reusable["sha256"],
            },
            target,
            artifact_root,
        )
    return should_build_local_artifact(
        target,
        current_matches=current_matches,
        reusable_matches=reusable_matches,
        overwrite=overwrite,
    )


def should_build_local_artifact(
    path: Path,
    *,
    current_matches: bool,
    reusable_matches: bool = False,
    overwrite: bool = False,
) -> bool:
    """Never let an untrusted local file shadow a verified reusable artifact."""
    return bool(
        overwrite
        or (
            not current_matches
            and (path.exists() or not reusable_matches)
        )
    )


def reusable_latent_protocol_matches(
    cache_schema: str,
    reused_preprocess: dict,
    current_preprocess: dict,
) -> bool:
    """Allow v4 migration while forbidding its noncanonical video latents."""
    if reused_preprocess == current_preprocess:
        return True
    legacy_preprocess = {
        key: value
        for key, value in current_preprocess.items()
        if key != "condition_frame"
    }
    if (
        cache_schema == "phycontext.wan_ti2v_cache.v4"
        and reused_preprocess == legacy_preprocess
    ):
        return False
    raise ValueError("reused cache uses different video preprocessing")


def index_reusable_text_contexts(records: dict[str, dict]) -> dict[str, dict]:
    """Index only descriptors present in a complete or interrupted cache."""
    result = {}
    for item in records.values():
        descriptor = item.get("text_context")
        if isinstance(descriptor, dict) and descriptor.get("prompt_sha256"):
            result[descriptor["prompt_sha256"]] = descriptor
    return result


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    selected = records
    if args.split:
        selected = [record for record in selected if record["split"] == args.split]
    if args.base_scene_id:
        requested = set(args.base_scene_id)
        known = {record["base_scene_id"] for record in selected}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"unknown base scene ids: {missing}")
        selected = [record for record in selected if record["base_scene_id"] in requested]
    if args.sweep_axis:
        selected = [
            record
            for record in selected
            if record["sweep"]["mode"] == "base"
            or record["sweep"]["axis"] == args.sweep_axis
        ]
    if args.limit_base_scenes is not None:
        if args.limit_base_scenes <= 0:
            raise ValueError("limit-base-scenes must be positive")
        scene_ids = []
        for record in selected:
            if record["base_scene_id"] not in scene_ids:
                scene_ids.append(record["base_scene_id"])
            if len(scene_ids) == args.limit_base_scenes:
                break
        allowed = set(scene_ids)
        selected = [record for record in selected if record["base_scene_id"] in allowed]
    base_scene_ids = list(dict.fromkeys(record["base_scene_id"] for record in selected))
    shard_scene_ids = {
        scene_id
        for index, scene_id in enumerate(base_scene_ids)
        if index % args.shard_count == args.shard_index
    }
    selected = [
        record for record in selected if record["base_scene_id"] in shard_scene_ids
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("record selection is empty")
    return selected


def load_video(path: Path, width: int, height: int, frame_count: int) -> torch.Tensor:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=4)
    if len(reader) == 0:
        raise ValueError(f"video has no frames: {path}")
    indices = np.asarray(
        evenly_spaced_frame_indices(len(reader), frame_count), dtype=np.int64
    )
    frames = reader.get_batch(indices).asnumpy()
    cropped = cover_center_crop_frames(frames, width, height)
    return torch.from_numpy(cropped).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def load_canonical_first_frame(path: Path, width: int, height: int) -> torch.Tensor:
    """Load the published group-shared TI2V condition frame."""
    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"))[None]
    cropped = cover_center_crop_frames(frame, width, height)
    return (
        torch.from_numpy(cropped)
        .permute(3, 0, 1, 2)
        .float()
        .div_(127.5)
        .sub_(1.0)
    )


def bind_canonical_condition_frame(
    video: torch.Tensor,
    canonical_first_frame: torch.Tensor,
) -> torch.Tensor:
    """Replace frame zero in-place before the full causal VAE encode."""
    if video.ndim != 4 or video.shape[1] < 1:
        raise ValueError("video must have shape [C, T, H, W] with T >= 1")
    if canonical_first_frame.shape != video[:, :1].shape:
        raise ValueError(
            "canonical first frame and target video preprocessing differ"
        )
    if canonical_first_frame.device != video.device:
        raise ValueError("canonical first frame and target video must share a device")
    video[:, :1].copy_(canonical_first_frame)
    return video


def atomic_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        save_file(tensors, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if (
        args.trajectory_representation == "dense_point_tracks"
        and args.cache_root == DEFAULT_CACHE
    ):
        raise ValueError(
            "dense_point_tracks requires an explicit dedicated --cache-root; "
            "the default root is reserved for full-rate das_3d_tracks"
        )
    if args.point_trajectory_manifest is None:
        raise ValueError("formal PhyContext caching requires dense point trajectories")
    if args.width % 32 or args.height % 32:
        raise ValueError("TI2V cache width and height must be divisible by 32")
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError("TI2V frame count must have the form 4n+1")
    root = args.project_root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    cache_root = (root / args.cache_root).resolve()
    if cache_root.is_relative_to(dataset_root):
        raise ValueError("Wan cache root must remain outside the published dataset")
    manifest_path = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else (dataset_root / args.manifest).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    manifest_hash = sha256(manifest_path)
    dataset_summary_path = manifest_path.parent / "summary.json"
    if not dataset_summary_path.is_file():
        raise FileNotFoundError(
            f"published dataset summary does not exist: {dataset_summary_path}"
        )
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    if dataset_summary.get("path_base") != "physweep_project_root":
        raise ValueError("PhysSweep summary has an unsupported path base")
    if dataset_summary.get("manifest") != relative(manifest_path, dataset_root):
        raise ValueError("PhysSweep summary points to a different training manifest")
    if dataset_summary.get("manifest_sha256") != manifest_hash:
        raise ValueError("PhysSweep summary manifest hash mismatch")
    dataset_summary_hash = sha256(dataset_summary_path)
    records = list(iter_jsonl(manifest_path))
    selected = select_records(records, args)
    order = {record["sample_id"]: index for index, record in enumerate(records)}
    point_manifest_path = None
    point_manifest_hash = None
    point_records = {}
    if args.point_trajectory_manifest is not None:
        point_manifest_path = (dataset_root / args.point_trajectory_manifest).resolve()
        point_manifest_hash = sha256(point_manifest_path)
        point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
        if point_manifest.get("schema") != "physweep.point_trajectory_manifest.v1":
            raise ValueError("point trajectory manifest uses an unsupported schema")
        if point_manifest["source_manifest_sha256"] != manifest_hash:
            raise ValueError("point trajectories belong to a different training manifest")
        if int(point_manifest.get("point_count", 0)) != 2048:
            raise ValueError("point trajectory manifest must contain 2048 points per object")
        raw_point_records = point_manifest.get("records")
        if not isinstance(raw_point_records, list):
            raise ValueError("point trajectory manifest records must be a list")
        point_sample_ids = [item.get("sample_id") for item in raw_point_records]
        if any(not sample_id for sample_id in point_sample_ids):
            raise ValueError("point trajectory manifest contains an empty sample id")
        if len(set(point_sample_ids)) != len(point_sample_ids):
            raise ValueError("point trajectory manifest contains duplicate samples")
        if (
            "record_count" in point_manifest
            and int(point_manifest["record_count"]) != len(raw_point_records)
        ):
            raise ValueError("point trajectory manifest record count is inconsistent")
        point_records = dict(zip(point_sample_ids, raw_point_records))
        missing_points = sorted(
            record["sample_id"]
            for record in selected
            if record["sample_id"] not in point_records
        )
        if missing_points:
            raise ValueError(f"point trajectories are missing samples: {missing_points}")
        mismatched_points = sorted(
            record["sample_id"]
            for record in selected
            if point_records[record["sample_id"]].get("object_ids")
            != record_dynamic_object_ids(record)
        )
        if mismatched_points:
            raise ValueError(
                f"point trajectories have mismatched object ids: {mismatched_points}"
            )

    checkpoint = args.checkpoint.resolve()
    preprocess = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "resize": "cover_then_center_crop",
        "condition_frame": CANONICAL_CONDITION_FRAME_PROTOCOL,
    }
    expected_latent_shape = (
        48,
        (args.frames - 1) // 4 + 1,
        args.height // 16,
        args.width // 16,
    )
    if args.trajectory_representation == "dense_point_tracks":
        point_track_preprocess = {
            "source_schema": POINT_TRAJECTORY_SCHEMA,
            "channels_per_object": TRACK_CHANNELS_PER_OBJECT,
            "max_objects": 3,
            "frame_selection": "evenly_spaced_from_first_to_last",
            "spatial_grid": "Wan_latent_grid",
            "spatial_transform": "cover_then_center_crop_to_preprocess_size",
            "pixel_center_convention": "opencv_half_pixel",
            "geometry_compute_dtype": GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA[
                CACHE_SCHEMA
            ],
            "preprocess_size_px": [args.width, args.height],
        }
    else:
        point_track_preprocess = {
            "source_schema": POINT_TRAJECTORY_SCHEMA,
            "representation": "das_3d_tracks",
            "channels_per_object": DAS_TRACK_CHANNELS_PER_OBJECT,
            "channels": ["identity_r", "identity_g", "identity_b", "visibility"],
            "max_objects": 3,
            "frame_selection": "same_evenly_spaced_indices_as_preprocessed_video",
            "source_frame_alignment": "trajectory_count_equals_source_video_count",
            "spatial_grid": "Wan_VAE_grid_after_full_resolution_visibility",
            "spatial_transform": "cover_then_center_crop_to_preprocess_size",
            "pixel_center_convention": "opencv_half_pixel",
            "geometry_compute_dtype": GEOMETRY_COMPUTE_DTYPE_BY_SCHEMA[
                CACHE_SCHEMA
            ],
            "preprocess_size_px": [args.width, args.height],
            "point_identity": "first_frame_camera_xyz_with_reciprocal_z_rgb",
            "color_normalization": "shared_xy_minmax_inverse_z_2_98_percentile",
            "visibility": "full_resolution_dynamic_and_static_nearest_depth_z_buffer",
            "visibility_depth_range": "source_camera_clip_start_end",
            "spatial_reduction": "visible_point_identity_mean_and_binary_occupancy",
            "temporal_encoding": "conditioner_first_frame_plus_learned_stride4_windows",
            "point_splat_radius_px": 1,
        }
    point_correspondence_preprocess = {
        "source_schema": POINT_TRAJECTORY_SCHEMA,
        "representation": "exact_material_point_correspondence",
        "frame_selection": "same_evenly_spaced_indices_as_preprocessed_video",
        "spatial_transform": "cover_then_center_crop_to_preprocess_size",
        "preprocess_size_px": [args.width, args.height],
        "point_axis": "fixed_object_slot_and_material_point_index",
        "visibility": "projected_center_pixel_winner_in_full_resolution_dynamic_and_static_point_center_z_buffer",
        "coordinate_serialization": "float32_with_float64_visible_raster_pixel_preserved",
        "point_splat_radius_px": 0,
        "visibility_query": "projected_material_point_center_pixel",
        "cached_tensors": [
            "track_xy_px",
            "track_depth_m",
            "track_visible",
            "source_frame_indices",
        ],
    }
    reuse_cache_path = None
    reuse_cache_hash = None
    reuse_records = {}
    reused_text_by_hash = {}
    reuse_artifact_root = root
    reuse_latents = False
    reuse_point_tracks = False
    reuse_track_correspondence = False
    if args.reuse_cache_manifest is not None:
        reuse_cache_path = (root / args.reuse_cache_manifest).resolve()
        reuse_cache_hash = sha256(reuse_cache_path)
        reuse_cache = json.loads(reuse_cache_path.read_text(encoding="utf-8"))
        if reuse_cache.get("schema") not in REUSABLE_CACHE_SCHEMAS:
            raise ValueError("reused cache uses an unsupported schema")
        reused_dataset_root = Path(reuse_cache.get("dataset_root", "")).resolve()
        if reused_dataset_root != dataset_root:
            raise ValueError("reused cache belongs to a different dataset root")
        if reuse_cache["source_manifest_sha256"] != manifest_hash:
            raise ValueError("reused cache belongs to a different training manifest")
        if reuse_cache.get("source_dataset_summary_sha256") != dataset_summary_hash:
            raise ValueError("reused cache belongs to a different dataset build")
        if Path(reuse_cache["checkpoint"]).resolve() != checkpoint:
            raise ValueError("reused cache belongs to a different Wan checkpoint")
        reuse_latents = reusable_latent_protocol_matches(
            reuse_cache.get("schema"),
            reuse_cache.get("preprocess"),
            preprocess,
        )
        reuse_artifact_root = resolve_cache_artifact_root(root, reuse_cache)
        reuse_records = {
            item["sample_id"]: item for item in reuse_cache["records"]
        }
        if point_manifest_path is not None:
            if reuse_cache.get("source_point_trajectory_manifest_sha256") != point_manifest_hash:
                raise ValueError("reused cache belongs to a different point trajectory manifest")
            if reuse_cache.get("point_track_preprocess") != point_track_preprocess:
                if cache_root.resolve() == reuse_cache_path.parent.resolve():
                    raise ValueError(
                        "point-track preprocessing changed; use a new cache root "
                        "so old point maps cannot be reused"
                    )
                # Text is independent of trajectory rasterization. Point maps
                # are rebuilt under the new transform; video latents are reused
                # only when their condition-frame protocol also matches.
            else:
                reuse_point_tracks = True
            if (
                reuse_cache.get("schema")
                in CENTER_PIXEL_TRACK_CORRESPONDENCE_CACHE_SCHEMAS
                and reuse_cache.get("point_correspondence_preprocess")
                == point_correspondence_preprocess
            ):
                reuse_track_correspondence = True
        reused_text_by_hash = index_reusable_text_contexts(reuse_records)

    selection = {
        "split": args.split,
        "base_scene_ids": sorted(args.base_scene_id),
        "sweep_axis": args.sweep_axis,
        "limit_base_scenes": args.limit_base_scenes,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    cache_manifest_name = (
        "manifest.json"
        if args.shard_count == 1
        else f"manifest.shard-{args.shard_index:05d}-of-{args.shard_count:05d}.json"
    )
    cache_manifest_path = cache_root / cache_manifest_name
    artifact_root = cache_root
    cached = {}
    if cache_manifest_path.is_file() and not args.overwrite:
        existing = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema") != CACHE_SCHEMA:
            raise ValueError(
                f"cache must use the current trajectory schema {CACHE_SCHEMA}"
            )
        existing_dataset_root = Path(existing.get("dataset_root", "")).resolve()
        if existing_dataset_root != dataset_root:
            raise ValueError("cache belongs to a different dataset root")
        if existing["source_manifest_sha256"] != manifest_hash:
            raise ValueError("cache belongs to a different training manifest")
        if existing.get("source_dataset_summary_sha256") != dataset_summary_hash:
            raise ValueError("cache belongs to a different dataset build")
        if Path(existing["checkpoint"]).resolve() != checkpoint:
            raise ValueError("cache belongs to a different Wan checkpoint")
        if existing.get("preprocess") != preprocess:
            raise ValueError("cache uses different video preprocessing settings")
        existing_artifact_root = resolve_cache_artifact_root(root, existing)
        if existing_artifact_root not in (root, cache_root):
            raise ValueError("cache manifest belongs to a different artifact root")
        artifact_root = existing_artifact_root
        if existing.get("selection") not in (None, selection):
            raise ValueError("cache shard uses different record selection settings")
        if point_manifest_path is not None:
            if existing.get("source_point_trajectory_manifest_sha256") != point_manifest_hash:
                raise ValueError("cache belongs to a different point trajectory manifest")
            if existing.get("point_track_preprocess") != point_track_preprocess:
                raise ValueError("cache uses different point-track preprocessing")
            if existing.get("point_correspondence_preprocess") != (
                point_correspondence_preprocess
            ):
                raise ValueError("cache uses different point-correspondence preprocessing")
        cached = {record["sample_id"]: record for record in existing["records"]}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    wan_repo = args.wan_repo.resolve()
    sys.path.insert(0, str(wan_repo))
    latent_root = cache_root / "latents"
    text_root = cache_root / "text"
    point_track_root = cache_root / "point_tracks"
    track_correspondence_root = cache_root / "track_correspondence"

    point_track_targets = {}
    track_correspondence_targets = {}
    if point_manifest_path is not None:
        expected_point_shape = expected_point_track_shape(
            args.trajectory_representation,
            args.frames,
            expected_latent_shape,
        )
        for record in selected:
            point_path = point_track_root / f"{record['sample_id']}.safetensors"
            cached_descriptor = cached.get(record["sample_id"], {}).get(
                "point_track"
            )
            reusable_descriptor = (
                reuse_records.get(record["sample_id"], {}).get("point_track")
                if reuse_point_tracks
                else None
            )
            if prepare_reusable_artifact(
                point_path,
                artifact_root,
                reuse_artifact_root,
                cached_descriptor,
                reusable_descriptor,
                args.overwrite,
            ):
                point_track_targets[record["sample_id"]] = point_path
            correspondence_path = (
                track_correspondence_root
                / f"{record['sample_id']}.safetensors"
            )
            current = cached.get(record["sample_id"], {}).get(
                "track_correspondence"
            )
            reusable = (
                reuse_records.get(record["sample_id"], {}).get(
                    "track_correspondence"
                )
                if reuse_track_correspondence
                else None
            )
            if prepare_reusable_artifact(
                correspondence_path,
                artifact_root,
                reuse_artifact_root,
                current,
                reusable,
                args.overwrite,
            ):
                track_correspondence_targets[record["sample_id"]] = (
                    correspondence_path
                )
        geometry_target_ids = set(point_track_targets) | set(
            track_correspondence_targets
        )
        geometry_targets = [
            record
            for record in selected
            if record["sample_id"] in geometry_target_ids
        ]
        for index, record in enumerate(geometry_targets, 1):
            sample_id = record["sample_id"]
            source = point_records[record["sample_id"]]
            source_path = (dataset_root / source["path"]).resolve()
            if (
                not source_path.is_relative_to(dataset_root)
                or not source_path.is_file()
                or sha256(source_path) != source.get("sha256")
            ):
                raise ValueError(
                    "point trajectory file/hash mismatch for "
                    f"{record['sample_id']}"
                )
            with np.load(source_path, allow_pickle=False) as archive:
                point_payload = {key: archive[key] for key in archive.files}
            source_frame_count = int(point_payload["points_world_m"].shape[0])
            video_path = dataset_root / record["target"]["video"]
            source_video = VideoReader(
                str(video_path), ctx=cpu(0), num_threads=1
            )
            source_video_frame_count = len(source_video)
            del source_video
            if source_frame_count != source_video_frame_count:
                raise ValueError(
                    "point trajectory and source video frame counts differ for "
                    f"{record['sample_id']}: {source_frame_count} != "
                    f"{source_video_frame_count}"
                )
            frame_indices = trajectory_frame_indices(
                args.trajectory_representation,
                source_frame_count,
                args.frames,
                expected_latent_shape[1],
            )
            static_points_camera0_m = None
            if (
                args.trajectory_representation == "das_3d_tracks"
                or sample_id in track_correspondence_targets
            ):
                scene_path = dataset_root / record["conditioning"]["scene"]
                with np.load(scene_path, allow_pickle=False) as scene_archive:
                    if "environment_xyz_camera_m" not in scene_archive:
                        raise ValueError(
                            f"scene condition has no static environment points: {scene_path}"
                        )
                    static_points_camera0_m = scene_archive[
                        "environment_xyz_camera_m"
                    ].astype(np.float32)
            point_path = point_track_targets.get(sample_id)
            correspondence_path = track_correspondence_targets.get(sample_id)
            point_track_map = None
            correspondence = None
            if args.trajectory_representation == "das_3d_tracks":
                if point_path is not None:
                    rendered = rasterize_das_3d_tracks(
                        point_payload,
                        (expected_latent_shape[3], expected_latent_shape[2]),
                        preprocess_size_px=(args.width, args.height),
                        frame_indices=frame_indices,
                        max_objects=3,
                        static_points_camera0_m=static_points_camera0_m,
                        point_radius_px=1,
                        return_correspondence=correspondence_path is not None,
                    )
                    if correspondence_path is None:
                        point_track_map = rendered
                    else:
                        point_track_map, correspondence = rendered
                elif correspondence_path is not None:
                    correspondence = build_das_track_correspondence(
                        point_payload,
                        preprocess_size_px=(args.width, args.height),
                        frame_indices=frame_indices,
                        static_points_camera0_m=static_points_camera0_m,
                        point_radius_px=0,
                    )
            else:
                if point_path is not None:
                    point_track_map = rasterize_projected_tracks(
                        point_payload,
                        (expected_latent_shape[3], expected_latent_shape[2]),
                        preprocess_size_px=(args.width, args.height),
                        frame_indices=frame_indices,
                        max_objects=3,
                    )
                if correspondence_path is not None:
                    correspondence = build_das_track_correspondence(
                        point_payload,
                        preprocess_size_px=(args.width, args.height),
                        frame_indices=trajectory_frame_indices(
                            "das_3d_tracks",
                            source_frame_count,
                            args.frames,
                            expected_latent_shape[1],
                        ),
                        static_points_camera0_m=static_points_camera0_m,
                        point_radius_px=0,
                    )
            if point_path is not None:
                if (
                    point_track_map is None
                    or tuple(point_track_map.shape) != expected_point_shape
                ):
                    actual_shape = (
                        None if point_track_map is None else tuple(point_track_map.shape)
                    )
                    raise ValueError(
                        f"unexpected point-track shape for {record['sample_id']}: "
                        f"{actual_shape} != {expected_point_shape}"
                    )
                atomic_safetensors(
                    {"point_track_map": torch.from_numpy(point_track_map)},
                    point_path,
                )
            if correspondence_path is not None:
                object_count = int(source["object_count"])
                expected_geometry_shape = (args.frames, object_count, 2048)
                if correspondence is None or (
                    tuple(correspondence["track_xy_px"].shape)
                    != expected_geometry_shape + (2,)
                    or tuple(correspondence["track_depth_m"].shape)
                    != expected_geometry_shape
                    or tuple(correspondence["track_visible"].shape)
                    != expected_geometry_shape
                    or tuple(correspondence["source_frame_indices"].shape)
                    != (args.frames,)
                ):
                    raise ValueError(
                        f"unexpected point-correspondence shape for {sample_id}"
                    )
                correspondence_tensors = validate_track_correspondence(
                    {
                        key: torch.from_numpy(correspondence[key])
                        for key in (
                            "track_xy_px",
                            "track_depth_m",
                            "track_visible",
                            "source_frame_indices",
                        )
                    },
                    preprocess_size_px=(args.width, args.height),
                    expected_frames=args.frames,
                )
                atomic_safetensors(correspondence_tensors, correspondence_path)
            outputs = []
            if point_path is not None:
                outputs.append("point-track")
            if correspondence_path is not None:
                outputs.append("track-correspondence")
            print(
                f"geometry[{'+'.join(outputs)}] {index}/{len(geometry_targets)} {sample_id}",
                flush=True,
            )

    latent_targets = []
    for record in selected:
        latent_path = latent_root / f"{record['sample_id']}.safetensors"
        reusable = (
            reuse_records.get(record["sample_id"], {}).get("latent")
            if reuse_latents
            else None
        )
        current = cached.get(record["sample_id"], {}).get("latent")
        if prepare_reusable_artifact(
            latent_path,
            artifact_root,
            reuse_artifact_root,
            current,
            reusable,
            args.overwrite,
        ):
            latent_targets.append((record, latent_path))

    peak_memory = 0
    if latent_targets:
        from wan.modules.vae2_2 import Wan2_2_VAE

        vae = Wan2_2_VAE(
            vae_pth=str(checkpoint / "Wan2.2_VAE.pth"),
            device=device,
        )
        for index, (record, latent_path) in enumerate(latent_targets, 1):
            video_path = dataset_root / record["target"]["video"]
            first_frame_path = dataset_root / record["conditioning"]["first_frame"]
            video = load_video(video_path, args.width, args.height, args.frames).to(device)
            canonical_first_frame = load_canonical_first_frame(
                first_frame_path, args.width, args.height
            ).to(device)
            bind_canonical_condition_frame(video, canonical_first_frame)
            with torch.inference_mode():
                latent = vae.encode([video])[0].to(torch.bfloat16).cpu().contiguous()
            if tuple(latent.shape) != expected_latent_shape:
                raise ValueError(
                    f"unexpected latent shape for {record['sample_id']}: "
                    f"{tuple(latent.shape)} != {expected_latent_shape}"
                )
            atomic_safetensors({"latent": latent}, latent_path)
            del video, canonical_first_frame, latent
            if device.type == "cuda":
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
                torch.cuda.empty_cache()
            print(f"latent {index}/{len(latent_targets)} {record['sample_id']}", flush=True)
        del vae
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    text_by_hash = {}
    for record in selected:
        prompt = record["conditioning"]["text"]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        text_by_hash[prompt_hash] = prompt
    cached_text_by_hash = {}
    for item in cached.values():
        descriptor = item.get("text_context")
        if isinstance(descriptor, dict) and descriptor.get("prompt_sha256"):
            cached_text_by_hash[descriptor["prompt_sha256"]] = descriptor
    missing_text = [
        (prompt_hash, prompt)
        for prompt_hash, prompt in text_by_hash.items()
        if prepare_text_artifact(
            prompt_hash,
            text_root,
            artifact_root,
            reuse_artifact_root,
            cached_text_by_hash,
            reused_text_by_hash,
            args.overwrite,
        )
    ]
    if missing_text:
        from wan.modules.t5 import T5EncoderModel

        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=str(checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(checkpoint / "google" / "umt5-xxl"),
        )
        for index, (prompt_hash, prompt) in enumerate(missing_text, 1):
            with torch.inference_mode():
                context = text_encoder([prompt], device)[0].to(torch.bfloat16).cpu().contiguous()
            if context.ndim != 2 or context.shape[1] != 4096 or context.shape[0] > 512:
                raise ValueError(
                    f"unexpected T5 context shape for {prompt_hash}: {tuple(context.shape)}"
                )
            atomic_safetensors({"context": context}, text_root / f"{prompt_hash}.safetensors")
            print(f"text {index}/{len(missing_text)} {prompt_hash[:12]}", flush=True)
        del text_encoder
        gc.collect()
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
            torch.cuda.empty_cache()

    for record in selected:
        prompt = record["conditioning"]["text"]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        latent_path = latent_root / f"{record['sample_id']}.safetensors"
        text_path = text_root / f"{prompt_hash}.safetensors"
        video_path = dataset_root / record["target"]["video"]
        first_frame_path = dataset_root / record["conditioning"]["first_frame"]
        scene_path = dataset_root / record["conditioning"]["scene"]
        if not latent_path.is_file() or not text_path.is_file():
            raise FileNotFoundError(
                f"cache artifacts were not materialized for {record['sample_id']}"
            )
        latent_descriptor = {
            "path": relative(latent_path, artifact_root),
            "sha256": sha256(latent_path),
            "shape": list(expected_latent_shape),
            "dtype": "bfloat16",
            "condition_frame_sha256": sha256(first_frame_path),
            "condition_frame_protocol": CANONICAL_CONDITION_FRAME_PROTOCOL,
        }
        text_descriptor = {
            "path": relative(text_path, artifact_root),
            "sha256": sha256(text_path),
            "prompt_sha256": prompt_hash,
        }
        cached_record = {
            "sample_id": record["sample_id"],
            "base_scene_id": record["base_scene_id"],
            "split": record["split"],
            "record": record,
            "latent": latent_descriptor,
            "text_context": text_descriptor,
            "source_video_sha256": sha256(video_path),
            "source_first_frame_sha256": sha256(first_frame_path),
            "source_scene_sha256": sha256(scene_path),
        }
        if point_manifest_path is not None:
            point_path = point_track_root / f"{record['sample_id']}.safetensors"
            source = point_records[record["sample_id"]]
            cached_record["point_track"] = {
                "path": relative(point_path, artifact_root),
                "sha256": sha256(point_path),
                "shape": list(expected_point_shape),
                "dtype": "float32",
                "source_point_trajectory": source["path"],
                "source_point_trajectory_sha256": source["sha256"],
                "object_count": source["object_count"],
                "object_ids": source["object_ids"],
            }
            correspondence_path = (
                track_correspondence_root
                / f"{record['sample_id']}.safetensors"
            )
            if not correspondence_path.is_file():
                raise FileNotFoundError(
                    "track correspondence was not materialized for "
                    f"{record['sample_id']}"
                )
            object_count = int(source["object_count"])
            cached_record["track_correspondence"] = {
                "path": relative(correspondence_path, artifact_root),
                "sha256": sha256(correspondence_path),
                "xy_shape": [args.frames, object_count, 2048, 2],
                "depth_shape": [args.frames, object_count, 2048],
                "visible_shape": [args.frames, object_count, 2048],
                "source_frame_indices_shape": [args.frames],
                "xy_dtype": "float32",
                "depth_dtype": "float32",
                "visible_dtype": "bool",
                "source_frame_indices_dtype": "int64",
                "object_count": object_count,
                "object_ids": source["object_ids"],
            }
        cached[record["sample_id"]] = cached_record

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_manifest = {
        "schema": CACHE_SCHEMA,
        "dataset_root": str(dataset_root),
        "source_manifest": relative(manifest_path, dataset_root),
        "source_manifest_sha256": manifest_hash,
        "source_dataset_summary": relative(dataset_summary_path, dataset_root),
        "source_dataset_summary_sha256": dataset_summary_hash,
        "checkpoint": str(checkpoint),
        "preprocess": preprocess,
        "selection": selection,
        "records": sorted(cached.values(), key=lambda item: order[item["sample_id"]]),
    }
    if artifact_root != root:
        cache_manifest.update(
            {
                "artifact_path_base": "cache_root",
                "artifact_root": str(artifact_root),
            }
        )
    if point_manifest_path is not None:
        cache_manifest.update(
            {
                "source_point_trajectory_manifest": relative(
                    point_manifest_path, dataset_root
                ),
                "source_point_trajectory_manifest_sha256": point_manifest_hash,
                "point_track_preprocess": point_track_preprocess,
                "point_correspondence_preprocess": point_correspondence_preprocess,
            }
        )
    if reuse_cache_path is not None:
        cache_manifest.update(
            {
                "reused_cache_manifest": portable_path(reuse_cache_path, root),
                "reused_cache_manifest_sha256": reuse_cache_hash,
            }
        )
    temporary = cache_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    temporary.replace(cache_manifest_path)
    report = {
        "passed": True,
        "cache_manifest": portable_path(cache_manifest_path, root),
        "selected_count": len(selected),
        "cached_count": len(cache_manifest["records"]),
        "new_latent_count": len(latent_targets),
        "new_text_count": len(missing_text),
        "new_point_track_count": len(point_track_targets),
        "new_track_correspondence_count": len(track_correspondence_targets),
        "reused_latent_count": len(selected) - len(latent_targets),
        "reused_text_count": len(text_by_hash) - len(missing_text),
        "peak_cuda_memory_gib": round(peak_memory / 1024**3, 3),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
