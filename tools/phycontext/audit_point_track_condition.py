#!/usr/bin/env python3
"""Audit temporal jumps introduced by the rasterized point-track condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def load_map(project_root: Path, item: dict) -> np.ndarray:
    path = Path(item["point_track"]["path"])
    if not path.is_absolute():
        path = project_root / path
    value = load_file(str(path), device="cpu")["point_track_map"].numpy()
    if value.ndim != 4 or value.shape[0] % 6 != 0:
        raise ValueError(f"invalid point-track map: {path}")
    return value.astype(np.float32, copy=False)


def summarize_object(value: np.ndarray, object_index: int) -> dict[str, float | int]:
    base = object_index * 6
    source = value[base]
    target = value[base + 1] > 0
    dx = value[base + 2]
    dy = value[base + 3]
    valid = value[base + 5] > 0
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
        ious.append(float(intersection.sum() / max(union.sum(), 1)))
        shared = (
            (source[frame_index - 1] > 0)
            & (source[frame_index] > 0)
            & valid[frame_index - 1]
            & valid[frame_index]
        )
        delta = np.concatenate(
            [np.abs(dx[frame_index][shared] - dx[frame_index - 1][shared]),
             np.abs(dy[frame_index][shared] - dy[frame_index - 1][shared])]
        )
        if len(delta):
            displacement_jumps.append(float(delta.mean()))
            displacement_p95.append(float(np.percentile(delta, 95)))
    return {
        "object_index": object_index,
        "active_cells_mean": float(active_counts.mean()),
        "active_cells_min": int(active_counts.min()),
        "active_cells_max": int(active_counts.max()),
        "target_flip_rate_mean": float(np.mean(flip_rates)) if flip_rates else 0.0,
        "target_flip_rate_max": float(np.max(flip_rates)) if flip_rates else 0.0,
        "target_iou_mean": float(np.mean(ious)) if ious else 1.0,
        "displacement_step_abs_mean": float(np.mean(displacement_jumps)) if displacement_jumps else 0.0,
        "displacement_step_abs_p95": float(np.percentile(displacement_jumps, 95)) if displacement_jumps else 0.0,
        "per_cell_displacement_abs_p95": float(np.percentile(displacement_p95, 95)) if displacement_p95 else 0.0,
    }


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
        point_map = load_map(project_root, item)
        object_results = [
            summarize_object(point_map, object_index)
            for object_index in range(point_map.shape[0] // 6)
        ]
        results.append(
            {
                "sample_id": sample_id,
                "map_shape": list(point_map.shape),
                "latent_cell_px": [
                    640 / point_map.shape[-1],
                    352 / point_map.shape[-2],
                ],
                "objects": object_results,
            }
        )
    payload = {
        "schema": "phycontext.point_track_condition_audit.v1",
        "cache_manifest": str(args.cache_manifest.resolve()),
        "evaluation_root": str(args.evaluation_root.resolve()),
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
