#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from safetensors.torch import save_file

from point_trajectory import POINT_TRAJECTORY_SCHEMA, rasterize_projected_tracks
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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


def cover_center_crop(
    frames: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
) -> np.ndarray:
    source_height, source_width = frames.shape[1:3]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    resized = np.stack(
        [
            cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=interpolation,
            )
            for frame in frames
        ]
    )
    x0 = (resized_width - width) // 2
    y0 = (resized_height - height) // 2
    return np.ascontiguousarray(
        resized[:, y0 : y0 + height, x0 : x0 + width]
    )


def load_video(path: Path, width: int, height: int, frame_count: int) -> torch.Tensor:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=4)
    if len(reader) == 0:
        raise ValueError(f"video has no frames: {path}")
    indices = np.rint(np.linspace(0, len(reader) - 1, frame_count)).astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    cropped = cover_center_crop(frames, width, height, cv2.INTER_AREA)
    return torch.from_numpy(cropped).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


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
    if args.point_trajectory_manifest is None:
        raise ValueError("formal PhyContext caching requires dense point trajectories")
    if args.width % 32 or args.height % 32:
        raise ValueError("TI2V cache width and height must be divisible by 32")
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError("TI2V frame count must have the form 4n+1")
    root = args.project_root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    cache_root = (root / args.cache_root).resolve()
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
        point_records = {
            item["sample_id"]: item for item in point_manifest["records"]
        }
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
            != [record["conditioning"]["physics"]["object"]["object_id"]]
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
    }
    expected_latent_shape = (
        48,
        (args.frames - 1) // 4 + 1,
        args.height // 16,
        args.width // 16,
    )
    point_track_preprocess = {
        "source_schema": POINT_TRAJECTORY_SCHEMA,
        "channels_per_object": 6,
        "max_objects": 3,
        "frame_selection": "evenly_spaced_from_first_to_last",
        "spatial_grid": "Wan_latent_grid",
        "spatial_transform": "cover_then_center_crop_to_preprocess_size",
        "preprocess_size_px": [args.width, args.height],
    }
    reuse_cache_path = None
    reuse_cache_hash = None
    reuse_records = {}
    reused_text_by_hash = {}
    if args.reuse_cache_manifest is not None:
        reuse_cache_path = (root / args.reuse_cache_manifest).resolve()
        reuse_cache_hash = sha256(reuse_cache_path)
        reuse_cache = json.loads(reuse_cache_path.read_text(encoding="utf-8"))
        if reuse_cache.get("schema") != "phycontext.wan_ti2v_cache.v3":
            raise ValueError("reused cache must use the dense-point-track v3 schema")
        reused_dataset_root = Path(reuse_cache.get("dataset_root", "")).resolve()
        if reused_dataset_root != dataset_root:
            raise ValueError("reused cache belongs to a different dataset root")
        if reuse_cache["source_manifest_sha256"] != manifest_hash:
            raise ValueError("reused cache belongs to a different training manifest")
        if reuse_cache.get("source_dataset_summary_sha256") != dataset_summary_hash:
            raise ValueError("reused cache belongs to a different dataset build")
        if Path(reuse_cache["checkpoint"]).resolve() != checkpoint:
            raise ValueError("reused cache belongs to a different Wan checkpoint")
        if reuse_cache["preprocess"] != preprocess:
            raise ValueError("reused cache uses different video preprocessing")
        reuse_records = {
            item["sample_id"]: item for item in reuse_cache["records"]
        }
        missing_reuse = sorted(
            record["sample_id"]
            for record in selected
            if record["sample_id"] not in reuse_records
        )
        if missing_reuse:
            raise ValueError(f"reused cache is missing samples: {missing_reuse}")
        if point_manifest_path is not None:
            if reuse_cache.get("source_point_trajectory_manifest_sha256") != point_manifest_hash:
                raise ValueError("reused cache belongs to a different point trajectory manifest")
            if reuse_cache.get("point_track_preprocess") != point_track_preprocess:
                if cache_root.resolve() == reuse_cache_path.parent.resolve():
                    raise ValueError(
                        "point-track preprocessing changed; use a new cache root "
                        "so old point maps cannot be reused"
                    )
                # Latents and text are safe to reuse from the old cache.  Point
                # maps are written into the new cache root with the new transform.
        for item in reuse_records.values():
            descriptor = item["text_context"]
            reused_text_by_hash[descriptor["prompt_sha256"]] = descriptor

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
    cached = {}
    if cache_manifest_path.is_file() and not args.overwrite:
        existing = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema") != "phycontext.wan_ti2v_cache.v3":
            raise ValueError("cache must use the dense-point-track v3 schema")
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
        if existing.get("selection") not in (None, selection):
            raise ValueError("cache shard uses different record selection settings")
        if point_manifest_path is not None:
            if existing.get("source_point_trajectory_manifest_sha256") != point_manifest_hash:
                raise ValueError("cache belongs to a different point trajectory manifest")
            if existing.get("point_track_preprocess") != point_track_preprocess:
                raise ValueError("cache uses different point-track preprocessing")
        cached = {record["sample_id"]: record for record in existing["records"]}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    wan_repo = args.wan_repo.resolve()
    sys.path.insert(0, str(wan_repo))
    latent_root = cache_root / "latents"
    text_root = cache_root / "text"
    point_track_root = cache_root / "point_tracks"

    point_track_targets = []
    if point_manifest_path is not None:
        for record in selected:
            point_path = point_track_root / f"{record['sample_id']}.safetensors"
            if args.overwrite or not point_path.is_file():
                point_track_targets.append((record, point_path))
        frame_indices = np.rint(
            np.linspace(0, args.frames - 1, expected_latent_shape[1])
        ).astype(np.int64).tolist()
        expected_point_shape = (
            18,
            expected_latent_shape[1],
            expected_latent_shape[2],
            expected_latent_shape[3],
        )
        for index, (record, point_path) in enumerate(point_track_targets, 1):
            source_path = dataset_root / point_records[record["sample_id"]]["path"]
            with np.load(source_path, allow_pickle=False) as archive:
                point_payload = {key: archive[key] for key in archive.files}
            point_track_map = rasterize_projected_tracks(
                point_payload,
                (expected_latent_shape[3], expected_latent_shape[2]),
                preprocess_size_px=(args.width, args.height),
                frame_indices=frame_indices,
                max_objects=3,
            )
            if tuple(point_track_map.shape) != expected_point_shape:
                raise ValueError(
                    f"unexpected point-track shape for {record['sample_id']}: "
                    f"{tuple(point_track_map.shape)} != {expected_point_shape}"
                )
            atomic_safetensors(
                {"point_track_map": torch.from_numpy(point_track_map)},
                point_path,
            )
            print(
                f"point-track {index}/{len(point_track_targets)} {record['sample_id']}",
                flush=True,
            )

    latent_targets = []
    for record in selected:
        latent_path = latent_root / f"{record['sample_id']}.safetensors"
        reusable = reuse_records.get(record["sample_id"], {}).get("latent")
        if args.overwrite or (
            not latent_path.is_file()
            and not (
                reusable
                and (root / reusable["path"]).is_file()
                and sha256(root / reusable["path"]) == reusable["sha256"]
            )
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
            video = load_video(video_path, args.width, args.height, args.frames).to(device)
            with torch.inference_mode():
                latent = vae.encode([video])[0].to(torch.bfloat16).cpu().contiguous()
            if tuple(latent.shape) != expected_latent_shape:
                raise ValueError(
                    f"unexpected latent shape for {record['sample_id']}: "
                    f"{tuple(latent.shape)} != {expected_latent_shape}"
                )
            atomic_safetensors({"latent": latent}, latent_path)
            del video, latent
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
    missing_text = [
        (prompt_hash, prompt)
        for prompt_hash, prompt in text_by_hash.items()
        if args.overwrite
        or (
            not (text_root / f"{prompt_hash}.safetensors").is_file()
            and not (
                prompt_hash in reused_text_by_hash
                and (root / reused_text_by_hash[prompt_hash]["path"]).is_file()
                and sha256(root / reused_text_by_hash[prompt_hash]["path"])
                == reused_text_by_hash[prompt_hash]["sha256"]
            )
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
        reused = reuse_records.get(record["sample_id"], {})
        latent_descriptor = (
            {
                "path": relative(latent_path, root),
                "sha256": sha256(latent_path),
                "shape": list(expected_latent_shape),
                "dtype": "bfloat16",
            }
            if latent_path.is_file()
            else reused["latent"]
        )
        text_descriptor = (
            {
                "path": relative(text_path, root),
                "sha256": sha256(text_path),
                "prompt_sha256": prompt_hash,
            }
            if text_path.is_file()
            else reused_text_by_hash[prompt_hash]
        )
        cached_record = {
            "sample_id": record["sample_id"],
            "base_scene_id": record["base_scene_id"],
            "split": record["split"],
            "record": record,
            "latent": latent_descriptor,
            "text_context": text_descriptor,
            "source_video_sha256": sha256(video_path),
        }
        if point_manifest_path is not None:
            point_path = point_track_root / f"{record['sample_id']}.safetensors"
            source = point_records[record["sample_id"]]
            cached_record["point_track"] = {
                "path": relative(point_path, root),
                "sha256": sha256(point_path),
                "shape": [
                    18,
                    expected_latent_shape[1],
                    expected_latent_shape[2],
                    expected_latent_shape[3],
                ],
                "dtype": "float32",
                "source_point_trajectory": source["path"],
                "source_point_trajectory_sha256": source["sha256"],
                "object_count": source["object_count"],
                "object_ids": source["object_ids"],
            }
        cached[record["sample_id"]] = cached_record

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_manifest = {
        "schema": "phycontext.wan_ti2v_cache.v3",
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
    if point_manifest_path is not None:
        cache_manifest.update(
            {
                "source_point_trajectory_manifest": relative(
                    point_manifest_path, dataset_root
                ),
                "source_point_trajectory_manifest_sha256": point_manifest_hash,
                "point_track_preprocess": point_track_preprocess,
            }
        )
    if reuse_cache_path is not None:
        cache_manifest.update(
            {
                "reused_cache_manifest": relative(reuse_cache_path, root),
                "reused_cache_manifest_sha256": reuse_cache_hash,
            }
        )
    temporary = cache_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    temporary.replace(cache_manifest_path)
    report = {
        "passed": True,
        "cache_manifest": relative(cache_manifest_path, root),
        "selected_count": len(selected),
        "cached_count": len(cache_manifest["records"]),
        "new_latent_count": len(latent_targets),
        "new_text_count": len(missing_text),
        "new_point_track_count": len(point_track_targets),
        "reused_latent_count": len(selected) - len(latent_targets),
        "reused_text_count": len(text_by_hash) - len(missing_text),
        "peak_cuda_memory_gib": round(peak_memory / 1024**3, 3),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
