#!/usr/bin/env python3
"""Estimate a visible object's 2D trajectory directly from a fixed-camera video.

The reference trajectory is used only for the optional red overlay and for the
first-frame object seed. After frame 0, tracking follows foreground components
from the previous observed position and velocity; it never searches around the
future reference trajectory. Missing detections remain missing in the report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, fps


def cover_center_crop_points(
    points: np.ndarray,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    output_width, output_height = output_size
    scale = max(output_width / source_width, output_height / source_height)
    rendered_width = max(output_width, round(source_width * scale))
    rendered_height = max(output_height, round(source_height * scale))
    crop_x = (rendered_width - output_width) // 2
    crop_y = (rendered_height - output_height) // 2
    return points * scale - np.asarray([crop_x, crop_y], dtype=np.float32)


def cover_center_crop_frame(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    output_width, output_height = output_size
    input_height, input_width = frame.shape[:2]
    scale = max(output_width / input_width, output_height / input_height)
    rendered_width = max(output_width, round(input_width * scale))
    rendered_height = max(output_height, round(input_height * scale))
    resized = cv2.resize(frame, (rendered_width, rendered_height), interpolation=cv2.INTER_AREA)
    crop_x = (rendered_width - output_width) // 2
    crop_y = (rendered_height - output_height) // 2
    return resized[crop_y : crop_y + output_height, crop_x : crop_x + output_width]


def load_reference(
    path: Path,
    key: str,
    object_index: int,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path} has no array named {key}")
        tracks = np.asarray(archive[key], dtype=np.float32)
    if tracks.ndim == 4:
        if tracks.shape[-1] != 2:
            raise ValueError(f"reference array must end in 2, got {tracks.shape}")
        if not 0 <= object_index < tracks.shape[1]:
            raise ValueError(f"object_index {object_index} is outside {tracks.shape[1]} objects")
        tracks = tracks[:, object_index]
        tracks = np.nanmean(tracks, axis=1)
    elif tracks.ndim == 3:
        if tracks.shape[-1] != 2:
            raise ValueError(f"reference array must end in 2, got {tracks.shape}")
        tracks = np.nanmean(tracks, axis=1)
    elif tracks.ndim != 2 or tracks.shape[-1] != 2:
        raise ValueError(f"reference array must have [T, N, 2] or [T, 2], got {tracks.shape}")
    return cover_center_crop_points(tracks, source_size, output_size)


def foreground_candidates(frame: np.ndarray, background_gray: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray, background_gray)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 50.0 or area > 12000.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = np.asarray(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
        x, y, width, height = cv2.boundingRect(contour)
        candidates.append(
            {"center": center, "area": area, "size": float(max(width, height))}
        )
    return candidates


def track_centers(
    frames: list[np.ndarray],
    seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    background = np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    centers: list[np.ndarray] = []
    areas: list[float] = []
    for frame_index, frame in enumerate(frames):
        candidates = foreground_candidates(frame, background_gray)
        valid_previous = [center for center in centers if np.isfinite(center).all()]
        if valid_previous:
            target = valid_previous[-1].copy()
            if len(valid_previous) >= 2:
                target += valid_previous[-1] - valid_previous[-2]
            gate = 150.0
            reference_area = areas[-1] if areas else 0.0
        else:
            # The first frame identifies the object; future frames do not use it.
            target = np.asarray(seed, dtype=np.float32)
            gate = 160.0
            reference_area = 0.0

        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            distance = float(np.linalg.norm(candidate["center"] - target))
            if distance > gate:
                continue
            area_penalty = 0.0
            if reference_area > 0.0:
                area_penalty = 0.15 * abs(
                    np.log(max(candidate["area"], 1.0) / max(reference_area, 1.0))
                )
            scored.append((distance + area_penalty * 20.0, candidate))
        if scored:
            scored.sort(key=lambda item: item[0])
            candidate = scored[0][1]
            centers.append(candidate["center"])
            areas.append(float(candidate["area"]))
        else:
            centers.append(np.asarray([np.nan, np.nan], dtype=np.float32))
            areas.append(0.0)
    return np.asarray(centers, dtype=np.float32), np.asarray(areas, dtype=np.float32)


def draw_segments(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    finite = np.isfinite(points).all(axis=1)
    start = None
    for index, is_finite in enumerate(np.r_[finite, False]):
        if is_finite and start is None:
            start = index
        elif not is_finite and start is not None:
            if index - start > 1:
                polyline = points[start:index].round().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [polyline], False, color, 2, cv2.LINE_AA)
            start = None


def write_overlay(
    frames: list[np.ndarray],
    observed: np.ndarray,
    reference: np.ndarray | None,
    output: Path,
    fps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for index, frame in enumerate(frames):
        canvas = frame.copy()
        if reference is not None:
            draw_segments(canvas, reference[: index + 1], (40, 40, 255))
            if np.isfinite(reference[index]).all():
                cv2.circle(canvas, tuple(reference[index].round().astype(int)), 7, (40, 40, 255), -1)
        draw_segments(canvas, observed[: index + 1], (40, 255, 80))
        if np.isfinite(observed[index]).all():
            cv2.circle(canvas, tuple(observed[index].round().astype(int)), 6, (40, 255, 80), -1)
        cv2.putText(
            canvas,
            "red=input green=observed pixel track",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    temporary.unlink(missing_ok=True)


def build_report(
    video: Path,
    frames: list[np.ndarray],
    fps: float,
    observed: np.ndarray,
    areas: np.ndarray,
    reference: np.ndarray | None,
) -> dict[str, Any]:
    detected = np.isfinite(observed).all(axis=1)
    report: dict[str, Any] = {
        "video": str(video),
        "frames": len(frames),
        "fps": fps,
        "image_size_px": [frames[0].shape[1], frames[0].shape[0]],
        "detected_frames": int(detected.sum()),
        "detection_coverage": round(float(detected.mean()), 6),
        "samples": {},
    }
    if reference is not None:
        comparable = detected & np.isfinite(reference).all(axis=1)
        distances = np.linalg.norm(observed[comparable] - reference[comparable], axis=1)
        report["mean_distance_px"] = round(float(distances.mean()), 4) if len(distances) else None
        report["median_distance_px"] = round(float(np.median(distances)), 4) if len(distances) else None
        report["p95_distance_px"] = round(float(np.percentile(distances, 95)), 4) if len(distances) else None
        report["rmse_px"] = round(float(np.sqrt(np.mean(distances**2))), 4) if len(distances) else None
        report["max_distance_px"] = round(float(distances.max()), 4) if len(distances) else None
    for index in [0, len(frames) // 2, len(frames) - 1]:
        report["samples"][str(index + 1)] = {
            "observed": observed[index].round(2).tolist(),
            "area": round(float(areas[index]), 2),
        }
        if reference is not None:
            report["samples"][str(index + 1)]["reference"] = reference[index].round(2).tolist()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--reference-key", default="tracks_xy_px")
    parser.add_argument("--reference-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--track-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--seed-output", nargs=2, type=float, metavar=("X", "Y"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, fps = read_video(args.video)
    if args.track_size is not None:
        track_size = tuple(args.track_size)
        frames = [cover_center_crop_frame(frame, track_size) for frame in frames]
    output_size = (frames[0].shape[1], frames[0].shape[0])
    reference = None
    if args.reference_npz is not None:
        if args.reference_size is None:
            raise ValueError("--reference-size WIDTH HEIGHT is required with --reference-npz")
        reference = load_reference(
            args.reference_npz,
            args.reference_key,
            args.object_index,
            tuple(args.reference_size),
            output_size,
        )
        seed = reference[0]
    elif args.seed_output is not None:
        seed = np.asarray(args.seed_output, dtype=np.float32)
    else:
        raise ValueError("provide --reference-npz or --seed-output X Y")
    if len(frames) != len(reference) if reference is not None else False:
        raise ValueError("reference length does not match video frame count")
    observed, areas = track_centers(frames, seed)
    write_overlay(frames, observed, reference, args.output_overlay, fps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(build_report(args.video, frames, fps, observed, areas, reference), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
