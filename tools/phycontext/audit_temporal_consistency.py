#!/usr/bin/env python3
"""Measure temporal instability separately on the moving object and background."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video has no readable frames: {path}")
    return np.stack(frames), fps


def cover_center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    x0 = (resized_width - width) // 2
    y0 = (resized_height - height) // 2
    return np.ascontiguousarray(resized[y0 : y0 + height, x0 : x0 + width])


def validate_decoded_video(
    project_root: Path,
    video_path: Path,
    video: np.ndarray,
    fps: float,
    report: dict,
) -> dict[str, float | int | bool | list[int]]:
    frames, height, width, _channels = video.shape
    if frames < 3:
        raise ValueError("decoded video audit requires at least three frames")
    expected_shape = (
        int(report.get("frames", frames)),
        int(report.get("height", height)),
        int(report.get("width", width)),
        3,
    )
    if video.shape != expected_shape:
        raise ValueError(
            f"decoded video shape differs from report: {video.shape} vs "
            f"{expected_shape}"
        )
    expected_hash = report.get("video_sha256")
    actual_hash = sha256(video_path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"video SHA256 differs from report: {video_path}")

    as_float = video.astype(np.float32)
    spatial_std = as_float.std(axis=(1, 2, 3))
    adjacent_mae = np.abs(np.diff(as_float, axis=0)).mean(axis=(1, 2, 3))
    exact_duplicates = int(
        np.count_nonzero(np.all(video[1:] == video[:-1], axis=(1, 2, 3)))
    )
    result: dict[str, float | int | bool | list[int]] = {
        "decoded_shape": list(video.shape),
        "decoded_fps": fps,
        "video_sha256_matches_report": expected_hash is None
        or actual_hash == expected_hash,
        "finite": bool(np.isfinite(as_float).all()),
        "spatial_std_min": float(spatial_std.min()),
        "spatial_std_max": float(spatial_std.max()),
        "adjacent_mae_min": float(adjacent_mae.min()),
        "adjacent_mae_median": float(np.median(adjacent_mae)),
        "adjacent_mae_p95": float(np.percentile(adjacent_mae, 95)),
        "adjacent_mae_max": float(adjacent_mae.max()),
        "exact_adjacent_duplicates": exact_duplicates,
        "near_black_fraction_max": float(
            (video <= 2).mean(axis=(1, 2, 3)).max()
        ),
        "near_white_fraction_max": float(
            (video >= 253).mean(axis=(1, 2, 3)).max()
        ),
    }
    first_frame_value = report.get("first_frame")
    if first_frame_value:
        first_frame_path = Path(first_frame_value)
        if not first_frame_path.is_absolute():
            first_frame_path = project_root / first_frame_path
        first_frame_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if first_frame_bgr is None:
            raise ValueError(f"cannot decode first frame: {first_frame_path}")
        first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
        first_frame = cover_center_crop(first_frame, width, height)
        result.update(
            {
                "first_frame_mae": float(
                    np.abs(as_float[0] - first_frame.astype(np.float32)).mean()
                ),
                "first_frame_psnr_db": float(cv2.PSNR(video[0], first_frame)),
            }
        )
    return result


def load_point_mask(
    project_root: Path,
    item: dict,
    frame_count: int,
    height: int,
    width: int,
    dilation_px: int,
) -> np.ndarray:
    point_path = project_root / item["point_track"]["path"]
    point_map = load_file(str(point_path), device="cpu")["point_track_map"].numpy()
    if point_map.ndim != 4 or point_map.shape[0] not in {12, 18}:
        raise ValueError(f"invalid point-track map: {point_path}")
    if point_map.shape[0] == 12:
        visibility = point_map[3::4]
        source = np.repeat(
            visibility[:, :1], visibility.shape[1], axis=1
        ).max(axis=0)
        current = visibility.max(axis=0)
        occupancy = np.maximum(source, current) > 0
        if frame_count != occupancy.shape[0]:
            raise ValueError(
                f"video/full-rate point-track temporal shape differs: "
                f"{frame_count} vs {occupancy.shape[0]} frames"
            )
        mask = occupancy.astype(np.uint8)
    else:
        source = point_map[0::6].max(axis=0)
        current = point_map[1::6].max(axis=0)
        occupancy = np.maximum(source, current) > 0
        latent_frames = occupancy.shape[0]
        if frame_count != 4 * (latent_frames - 1) + 1:
            raise ValueError(
                f"video/point-track temporal shape differs: {frame_count} vs "
                f"{latent_frames} latent frames"
            )
        mask = np.zeros(
            (frame_count, occupancy.shape[1], occupancy.shape[2]), dtype=np.uint8
        )
        mask[0] = occupancy[0]
        for latent_index in range(1, latent_frames):
            start = 1 + 4 * (latent_index - 1)
            mask[start : start + 4] = occupancy[latent_index]
    mask = np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            for frame in mask
        ]
    )
    if dilation_px > 0:
        kernel_size = 2 * dilation_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = np.stack(
            [cv2.dilate(frame, kernel, iterations=1) for frame in mask]
        )
    return mask.astype(bool)


def score(video: np.ndarray, object_mask: np.ndarray) -> dict[str, float | int]:
    if video.ndim != 3 or object_mask.shape != video.shape:
        raise ValueError("video and object mask must share F x H x W")
    if video.shape[0] < 3:
        raise ValueError("temporal audit requires at least three frames")
    second_difference = np.abs(
        video[2:] - 2.0 * video[1:-1] + video[:-2]
    )
    mask = object_mask[1:-1]
    background = ~mask
    return {
        "frames": int(video.shape[0]),
        "object_second_difference_mean": float(second_difference[mask].mean())
        if mask.any()
        else float("nan"),
        "background_second_difference_mean": float(second_difference[background].mean())
        if background.any()
        else float("nan"),
        "global_second_difference_mean": float(second_difference.mean()),
        "global_second_difference_p95": float(np.percentile(second_difference, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dilation-px", type=int, default=8)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    cache_manifest = args.cache_manifest.resolve()
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    records = {item["sample_id"]: item for item in cache["records"]}
    results = []
    for video_path in sorted(evaluation_root.rglob("*.mp4")):
        report_path = video_path.with_suffix(".json")
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sample_id = report["sample_id"]
        if sample_id not in records:
            raise KeyError(f"sample is missing from cache: {sample_id}")
        decoded_video, fps = read_video(video_path)
        video = np.stack(
            [
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                / 255.0
                for frame in decoded_video
            ]
        )
        object_mask = load_point_mask(
            project_root,
            records[sample_id],
            video.shape[0],
            video.shape[1],
            video.shape[2],
            args.dilation_px,
        )
        result = {
            "group": video_path.parent.name,
            "name": video_path.stem,
            "sample_id": sample_id,
            "video": str(video_path),
            **validate_decoded_video(
                project_root,
                video_path,
                decoded_video,
                fps,
                report,
            ),
            **score(video, object_mask),
        }
        results.append(result)
    payload = {
        "schema": "phycontext.temporal_consistency_audit.v2",
        "evaluation_root": str(evaluation_root),
        "cache_manifest": str(cache_manifest),
        "dilation_px": args.dilation_px,
        "results": results,
    }
    output = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
