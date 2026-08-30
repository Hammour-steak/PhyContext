#!/usr/bin/env python3
"""Recompute stratified Wan latent and DaS point-map cache samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from safetensors.torch import load_file

from cache_contract import (
    CANONICAL_CONDITION_FRAME_PROTOCOL,
    CURRENT_CACHE_SCHEMA,
    resolve_cache_artifact_root,
    resolve_cache_dataset_root,
    validate_cache_artifact,
    validate_cache_source_manifest,
)
from cache_wan_inputs import (
    bind_canonical_condition_frame,
    expected_point_track_shape,
    load_canonical_first_frame,
    load_video,
    record_dynamic_object_ids,
    sha256,
    trajectory_frame_indices,
)
from point_trajectory import rasterize_das_3d_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, default=Path("external/Wan2.2"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pick_evenly(items: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if len(items) < count:
        raise ValueError(f"audit stratum has {len(items)} records, needs {count}")
    indices = np.rint(np.linspace(0, len(items) - 1, count)).astype(np.int64)
    return [items[int(index)] for index in indices]


def select_audit_records(
    records: list[dict], sample_count: int, shard_count: int
) -> list[dict]:
    """Select base/sweep and train/validation/test examples from every shard."""
    if sample_count <= 0 or shard_count <= 0 or sample_count % shard_count:
        raise ValueError("sample count must be positive and divisible by shard count")
    per_shard = sample_count // shard_count
    if per_shard < 6:
        raise ValueError("each audit shard needs at least six samples")
    base_scene_ids = list(dict.fromkeys(item["base_scene_id"] for item in records))
    shard_by_scene = {
        scene_id: index % shard_count
        for index, scene_id in enumerate(base_scene_ids)
    }
    selected = []
    for shard in range(shard_count):
        shard_records = [
            item
            for item in records
            if shard_by_scene[item["base_scene_id"]] == shard
        ]
        quotas = {
            ("train", "base"): 1,
            ("validation", "base"): 1,
            ("validation", "sweep"): 1,
            ("test", "base"): 1,
            ("test", "sweep"): 1,
            ("train", "sweep"): per_shard - 5,
        }
        shard_selected = []
        for (split, mode), count in quotas.items():
            pool = [
                item
                for item in shard_records
                if item["split"] == split
                and (
                    (mode == "base" and item["record"]["sweep"]["mode"] == "base")
                    or (
                        mode == "sweep"
                        and item["record"]["sweep"]["mode"] != "base"
                    )
                )
            ]
            shard_selected.extend(pick_evenly(pool, count))
        for item in shard_selected:
            selected.append({**item, "_audit_shard": shard})
    sample_ids = [item["sample_id"] for item in selected]
    if len(selected) != sample_count or len(set(sample_ids)) != sample_count:
        raise ValueError("stratified audit selection is not unique and complete")
    return selected


def load_single_tensor(path: Path, key: str) -> torch.Tensor:
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {key}:
        raise ValueError(f"{path} must contain only {key}")
    return tensors[key]


def validate_point_map(
    item: dict,
    source: dict,
    artifact_root: Path,
    dataset_root: Path,
    width: int,
    height: int,
    frames: int,
    latent_shape: tuple[int, int, int, int],
) -> dict:
    sample_id = item["sample_id"]
    record = item["record"]
    cached_path = validate_cache_artifact(
        artifact_root, item["point_track"], f"point track for {sample_id}"
    )
    cached = load_single_tensor(cached_path, "point_track_map")
    expected_shape = expected_point_track_shape("das_3d_tracks", frames, latent_shape)
    if tuple(cached.shape) != expected_shape or cached.dtype != torch.float32:
        raise ValueError(f"cached point-map contract mismatch: {sample_id}")
    if not torch.isfinite(cached).all():
        raise ValueError(f"cached point map is nonfinite: {sample_id}")
    object_ids = record_dynamic_object_ids(record)
    if source.get("object_ids") != object_ids:
        raise ValueError(f"point-source object binding mismatch: {sample_id}")
    if int(source.get("object_count", -1)) != len(object_ids):
        raise ValueError(f"point-source object count mismatch: {sample_id}")
    source_path = (dataset_root / source["path"]).resolve()
    if not source_path.is_relative_to(dataset_root):
        raise ValueError(f"point source escapes dataset root: {sample_id}")
    if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
        raise ValueError(f"point-source file/hash mismatch: {sample_id}")
    with np.load(source_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    source_frames = int(payload["points_world_m"].shape[0])
    video_path = (dataset_root / record["target"]["video"]).resolve()
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    video_frames = len(reader)
    del reader
    if source_frames != video_frames:
        raise ValueError(f"point/video frame-count mismatch: {sample_id}")
    indices = trajectory_frame_indices(
        "das_3d_tracks", source_frames, frames, latent_shape[1]
    )
    scene_path = (dataset_root / record["conditioning"]["scene"]).resolve()
    with np.load(scene_path, allow_pickle=False) as archive:
        static_points = archive["environment_xyz_camera_m"].astype(np.float32)
    recomputed = torch.from_numpy(
        rasterize_das_3d_tracks(
            payload,
            (latent_shape[3], latent_shape[2]),
            preprocess_size_px=(width, height),
            frame_indices=indices,
            max_objects=3,
            static_points_camera0_m=static_points,
            point_radius_px=1,
        )
    )
    difference = (cached - recomputed).abs()
    max_abs = float(difference.max())
    if not torch.equal(cached, recomputed):
        raise ValueError(f"point recomputation mismatch: {sample_id}, max={max_abs}")
    slots = cached.reshape(3, 4, *cached.shape[1:])
    visible_cells = int(slots[:, 3:4].sum())
    if visible_cells <= 0:
        raise ValueError(f"point condition has no visible cells: {sample_id}")
    return {
        "exact_recompute": True,
        "max_abs_error": max_abs,
        "visible_cells": visible_cells,
        "cached_sha256": item["point_track"]["sha256"],
        "source_sha256": source["sha256"],
    }


def validate_latent(
    item: dict,
    vae,
    artifact_root: Path,
    dataset_root: Path,
    width: int,
    height: int,
    frames: int,
    expected_shape: tuple[int, int, int, int],
    device: torch.device,
) -> dict:
    sample_id = item["sample_id"]
    record = item["record"]
    descriptor = item["latent"]
    if (
        descriptor.get("condition_frame_protocol")
        != CANONICAL_CONDITION_FRAME_PROTOCOL
        or descriptor.get("condition_frame_sha256")
        != item.get("source_first_frame_sha256")
    ):
        raise ValueError(f"latent condition-frame binding mismatch: {sample_id}")
    cached_path = validate_cache_artifact(
        artifact_root, descriptor, f"latent for {sample_id}"
    )
    cached = load_single_tensor(cached_path, "latent")
    if tuple(cached.shape) != expected_shape or cached.dtype != torch.bfloat16:
        raise ValueError(f"cached latent contract mismatch: {sample_id}")
    if not torch.isfinite(cached.float()).all():
        raise ValueError(f"cached latent is nonfinite: {sample_id}")
    video_path = (dataset_root / record["target"]["video"]).resolve()
    first_frame_path = (
        dataset_root / record["conditioning"]["first_frame"]
    ).resolve()
    video = load_video(video_path, width, height, frames).to(device)
    canonical = load_canonical_first_frame(first_frame_path, width, height).to(device)
    raw_frame_zero_difference = (video[:, :1] - canonical).abs()
    raw_max_abs = float(raw_frame_zero_difference.max().cpu())
    raw_mean_abs = float(raw_frame_zero_difference.mean().cpu())
    bind_canonical_condition_frame(video, canonical)
    if not torch.equal(video[:, :1], canonical):
        raise ValueError(f"canonical first-frame replacement failed: {sample_id}")
    with torch.inference_mode():
        recomputed = vae.encode([video])[0].to(torch.bfloat16).cpu().contiguous()
    del video, canonical
    difference = (cached.float() - recomputed.float()).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    if not torch.equal(cached, recomputed):
        raise ValueError(
            f"latent recomputation mismatch: {sample_id}, "
            f"max={max_abs}, mean={mean_abs}"
        )
    return {
        "exact_recompute": True,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "raw_target_vs_canonical_frame_zero_max_abs": raw_max_abs,
        "raw_target_vs_canonical_frame_zero_mean_abs": raw_mean_abs,
        "cached_sha256": descriptor["sha256"],
        "source_video_sha256": item["source_video_sha256"],
        "condition_frame_sha256": descriptor["condition_frame_sha256"],
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    cache_path = (root / args.cache_manifest).resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema") != CURRENT_CACHE_SCHEMA:
        raise ValueError(f"sample recomputation requires {CURRENT_CACHE_SCHEMA}")
    validate_cache_source_manifest(root, cache)
    artifact_root = resolve_cache_artifact_root(root, cache)
    dataset_root = resolve_cache_dataset_root(root, cache)
    preprocess = cache["preprocess"]
    width = int(preprocess["width"])
    height = int(preprocess["height"])
    frames = int(preprocess["frames"])
    latent_shape = (48, (frames - 1) // 4 + 1, height // 16, width // 16)
    selected = select_audit_records(
        cache["records"], args.sample_count, args.shard_count
    )
    point_manifest_path = (
        dataset_root / cache["source_point_trajectory_manifest"]
    ).resolve()
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    point_by_id = {item["sample_id"]: item for item in point_manifest["records"]}
    wan_repo = (root / args.wan_repo).resolve()
    checkpoint = (
        (root / args.checkpoint).resolve()
        if args.checkpoint is not None
        else Path(cache["checkpoint"]).resolve()
    )
    sys.path.insert(0, str(wan_repo))
    from wan.modules.vae2_2 import Wan2_2_VAE

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    vae = Wan2_2_VAE(
        vae_pth=str(checkpoint / "Wan2.2_VAE.pth"), device=device
    )
    started = time.time()
    results = []
    for index, item in enumerate(selected, 1):
        sample_id = item["sample_id"]
        result = {
            "sample_id": sample_id,
            "shard": item["_audit_shard"],
            "split": item["split"],
            "sweep_mode": item["record"]["sweep"]["mode"],
            "point_track": validate_point_map(
                item,
                point_by_id[sample_id],
                artifact_root,
                dataset_root,
                width,
                height,
                frames,
                latent_shape,
            ),
            "latent": validate_latent(
                item,
                vae,
                artifact_root,
                dataset_root,
                width,
                height,
                frames,
                latent_shape,
                device,
            ),
        }
        results.append(result)
        print(f"validated {index}/{len(selected)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "phycontext.wan_cache_sample_recompute.v1",
        "passed": True,
        "cache_manifest": str(cache_path),
        "sample_count": len(results),
        "selection": {
            "shards": dict(Counter(str(item["shard"]) for item in results)),
            "splits": dict(Counter(item["split"] for item in results)),
            "sweep_modes": dict(Counter(item["sweep_mode"] for item in results)),
        },
        "point_exact_recompute_count": len(results),
        "latent_exact_recompute_count": len(results),
        "max_point_abs_error": max(
            item["point_track"]["max_abs_error"] for item in results
        ),
        "max_latent_abs_error": max(
            item["latent"]["max_abs_error"] for item in results
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        output = (root / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
