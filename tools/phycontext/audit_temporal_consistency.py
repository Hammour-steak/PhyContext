#!/usr/bin/env python3
"""Measure temporal instability separately on the moving object and background."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from safetensors.torch import load_file


def read_gray(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        )
    capture.release()
    if not frames:
        raise ValueError(f"video has no readable frames: {path}")
    return np.stack(frames)


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
    if point_map.ndim != 4 or point_map.shape[0] % 6:
        raise ValueError(f"invalid point-track map: {point_path}")
    source = point_map[0::6].max(axis=0)
    current = point_map[1::6].max(axis=0)
    occupancy = np.maximum(source, current) > 0
    latent_frames = occupancy.shape[0]
    if frame_count != 4 * (latent_frames - 1) + 1:
        raise ValueError(
            f"video/point-track temporal shape differs: {frame_count} vs "
            f"{latent_frames} latent frames"
        )
    mask = np.zeros((frame_count, occupancy.shape[1], occupancy.shape[2]), dtype=np.uint8)
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
        video = read_gray(video_path)
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
            **score(video, object_mask),
        }
        results.append(result)
    payload = {
        "schema": "phycontext.temporal_consistency_audit.v1",
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
