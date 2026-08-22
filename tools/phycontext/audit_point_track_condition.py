#!/usr/bin/env python3
"""Audit occupancy and representation-specific point-track channel semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def load_map(project_root: Path, item: dict) -> tuple[np.ndarray, str]:
    path = Path(item["point_track"]["path"])
    if not path.is_absolute():
        path = project_root / path
    value = load_file(str(path), device="cpu")["point_track_map"].numpy()
    if value.ndim != 4 or value.shape[0] not in {12, 18}:
        raise ValueError(f"invalid point-track map: {path}")
    representation = "das_3d_tracks" if value.shape[0] == 12 else "dense_point_tracks"
    return value.astype(np.float32, copy=False), representation


def summarize_object(
    value: np.ndarray,
    object_index: int,
    representation: str,
) -> dict[str, float | int | bool]:
    if representation == "dense_point_tracks":
        base = object_index * 6
        source = value[base]
        target = value[base + 1] > 0
        dx = value[base + 2]
        dy = value[base + 3]
        valid = value[base + 5] > 0
    else:
        base = object_index * 4
        target = value[base + 3] > 0
    active_counts = target.reshape(target.shape[0], -1).sum(axis=1)
    flip_rates: list[float] = []
    ious: list[float] = []
    displacement_jumps: list[float] = []
    displacement_p95: list[float] = []
    for frame_index in range(1, target.shape[0]):
        previous = target[frame_index - 1]
        current = target[frame_index]
        union = np.logical_or(previous, current)
        intersection = np.logical_and(previous, current)
        flip_rates.append(float(np.logical_xor(previous, current).mean()))
        ious.append(
            float(intersection.sum() / union.sum()) if union.any() else 1.0
        )
        if representation == "dense_point_tracks":
            shared = (
                (source[frame_index - 1] > 0)
                & (source[frame_index] > 0)
                & valid[frame_index - 1]
                & valid[frame_index]
            )
            delta = np.concatenate(
                [
                    np.abs(dx[frame_index][shared] - dx[frame_index - 1][shared]),
                    np.abs(dy[frame_index][shared] - dy[frame_index - 1][shared]),
                ]
            )
            if len(delta):
                displacement_jumps.append(float(delta.mean()))
                displacement_p95.append(float(np.percentile(delta, 95)))
    result = {
        "object_index": object_index,
        "slot_active": bool(active_counts.max() > 0),
        "active_cells_mean": float(active_counts.mean()),
        "active_cells_min": int(active_counts.min()),
        "active_cells_max": int(active_counts.max()),
        "target_flip_rate_mean": float(np.mean(flip_rates)) if flip_rates else 0.0,
        "target_flip_rate_max": float(np.max(flip_rates)) if flip_rates else 0.0,
        "target_iou_mean": float(np.mean(ious)) if ious else 1.0,
    }
    if representation == "dense_point_tracks":
        result.update(
            {
                "displacement_step_abs_mean": (
                    float(np.mean(displacement_jumps))
                    if displacement_jumps
                    else 0.0
                ),
                "displacement_step_abs_p95": (
                    float(np.percentile(displacement_jumps, 95))
                    if displacement_jumps
                    else 0.0
                ),
                "per_cell_displacement_abs_p95": (
                    float(np.percentile(displacement_p95, 95))
                    if displacement_p95
                    else 0.0
                ),
            }
        )
    else:
        rgb = value[base : base + 3]
        occupied_rgb = np.moveaxis(rgb, 0, -1)[target]
        result.update(
            {
                "identity_rgb_min": (
                    float(occupied_rgb.min()) if len(occupied_rgb) else 0.0
                ),
                "identity_rgb_max": (
                    float(occupied_rgb.max()) if len(occupied_rgb) else 0.0
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    manifest = json.loads(args.cache_manifest.resolve().read_text(encoding="utf-8"))
    records = {item["sample_id"]: item for item in manifest["records"]}
    sample_ids = []
    for report_path in sorted(args.evaluation_root.resolve().rglob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sample_id = report.get("sample_id")
        if sample_id and sample_id not in sample_ids:
            sample_ids.append(sample_id)
    results = []
    for sample_id in sample_ids:
        item = records[sample_id]
        point_map, representation = load_map(project_root, item)
        channels_per_object = 4 if representation == "das_3d_tracks" else 6
        object_results = [
            summarize_object(point_map, object_index, representation)
            for object_index in range(point_map.shape[0] // channels_per_object)
        ]
        results.append(
            {
                "sample_id": sample_id,
                "representation": representation,
                "channel_semantics": (
                    ["identity_r", "identity_g", "identity_b", "visibility"]
                    if representation == "das_3d_tracks"
                    else [
                        "source_occupancy",
                        "target_occupancy",
                        "dx",
                        "dy",
                        "depth_delta",
                        "validity",
                    ]
                ),
                "map_shape": list(point_map.shape),
                "grid_cell_px": [
                    float(manifest["preprocess"]["width"]) / point_map.shape[-1],
                    float(manifest["preprocess"]["height"]) / point_map.shape[-2],
                ],
                "objects": object_results,
            }
        )
    payload = {
        "schema": "phycontext.point_track_condition_audit.v2",
        "cache_manifest": str(args.cache_manifest.resolve()),
        "evaluation_root": str(args.evaluation_root.resolve()),
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
