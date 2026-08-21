#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from wan_training import SWEEP_AXES, select_sweep_endpoint_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("datasets/physweep_training/manifest.jsonl")
DEFAULT_OUTPUT = Path("datasets/physweep_training/sweep_response_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure low/high sweep response directly from simulation trajectories"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--limit-base-scenes", type=int)
    parser.add_argument("--active-threshold-object-extents", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def bound_metadata_path(root: Path, record: dict) -> Path:
    return root / record["target"]["metadata"]


def control_value(record: dict, axis: str) -> float:
    dynamic_object = record["conditioning"]["physics"]["object"]
    if axis == "mass_kg":
        return float(dynamic_object["mass_kg"])
    key = "friction" if axis == "contact_friction" else "restitution"
    return float(dynamic_object[key])


def load_positions(root: Path, item: dict) -> tuple[np.ndarray, float]:
    record = item["record"]
    metadata = json.loads(bound_metadata_path(root, record).read_text(encoding="utf-8"))
    if len(metadata["simulation"]["objects"]) != 1:
        raise ValueError("the one-object audit requires exactly one simulated object")
    object_id = str(metadata["simulation"]["objects"][0]["id"])
    trajectory = np.load(root / metadata["trajectory"]["path"], allow_pickle=False)
    positions = trajectory[f"{object_id}__position_m"].astype(np.float64)
    objects = {entry["id"]: entry for entry in metadata["simulation"]["objects"]}
    extent = max(float(value) for value in objects[object_id]["geometry"]["size_m"])
    if positions.ndim != 2 or positions.shape[1] != 3 or extent <= 0:
        raise ValueError(f"invalid trajectory geometry for {item['sample_id']}")
    return positions, extent


def measure_pair(root: Path, pair: dict, threshold: float) -> dict:
    low, low_extent = load_positions(root, pair["low"])
    high, high_extent = load_positions(root, pair["high"])
    if low.shape != high.shape:
        raise ValueError(f"trajectory shape mismatch for {pair['base_scene_id']}/{pair['axis']}")
    extent = max(low_extent, high_extent)
    separation = np.linalg.norm(low - high, axis=1)
    low_path = float(np.linalg.norm(np.diff(low, axis=0), axis=1).sum())
    high_path = float(np.linalg.norm(np.diff(high, axis=0), axis=1).sum())
    maximum_extents = float(separation.max() / extent)
    return {
        "base_scene_id": pair["base_scene_id"],
        "axis": pair["axis"],
        "low_sample_id": pair["low"]["sample_id"],
        "high_sample_id": pair["high"]["sample_id"],
        "low_value": control_value(pair["low"]["record"], pair["axis"]),
        "high_value": control_value(pair["high"]["record"], pair["axis"]),
        "frame_count": int(len(low)),
        "object_extent_m": extent,
        "initial_separation_m": float(separation[0]),
        "endpoint_separation_m": float(separation[-1]),
        "mean_separation_m": float(separation.mean()),
        "maximum_separation_m": float(separation.max()),
        "maximum_separation_object_extents": maximum_extents,
        "path_length_delta_m": abs(low_path - high_path),
        "active": maximum_extents >= threshold,
    }


def summarize_axis(rows: list[dict], top_k: int) -> dict:
    responses = np.asarray(
        [row["maximum_separation_object_extents"] for row in rows], dtype=np.float64
    )
    active_count = sum(bool(row["active"]) for row in rows)
    top = sorted(
        rows,
        key=lambda row: row["maximum_separation_object_extents"],
        reverse=True,
    )[:top_k]
    return {
        "group_count": len(rows),
        "active_count": active_count,
        "active_fraction": active_count / len(rows) if rows else 0.0,
        "median_maximum_separation_object_extents": (
            float(np.median(responses)) if len(responses) else 0.0
        ),
        "maximum_separation_object_extents": (
            float(responses.max()) if len(responses) else 0.0
        ),
        "top_groups": top,
    }


def main() -> None:
    args = parse_args()
    if args.limit_base_scenes is not None and args.limit_base_scenes <= 0:
        raise ValueError("limit-base-scenes must be positive")
    if args.active_threshold_object_extents <= 0 or args.top_k <= 0:
        raise ValueError("response threshold and top-k must be positive")
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.split is not None:
        manifest = [record for record in manifest if record["split"] == args.split]
    if not manifest:
        raise ValueError("sweep response selection is empty")
    wrapped = [
        {
            "sample_id": record["sample_id"],
            "base_scene_id": record["base_scene_id"],
            "record": record,
        }
        for record in manifest
    ]
    base_count = len(dict.fromkeys(item["base_scene_id"] for item in wrapped))
    if args.limit_base_scenes is not None:
        base_count = min(base_count, args.limit_base_scenes)
    pairs = select_sweep_endpoint_pairs(wrapped, base_scene_count=base_count)
    rows = [
        measure_pair(root, pair, args.active_threshold_object_extents)
        for pair in pairs
    ]
    by_axis = {
        axis: summarize_axis([row for row in rows if row["axis"] == axis], args.top_k)
        for axis in SWEEP_AXES
    }
    initial_error = max((row["initial_separation_m"] for row in rows), default=0.0)
    report = {
        "schema": "physweep.sweep_response_audit.v1",
        "manifest": args.manifest.as_posix(),
        "split": args.split,
        "base_scene_count": base_count,
        "pair_count": len(rows),
        "active_threshold_object_extents": args.active_threshold_object_extents,
        "maximum_initial_endpoint_separation_m": initial_error,
        "initial_state_invariant": initial_error <= 1e-5,
        "by_axis": by_axis,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"passed": report["initial_state_invariant"], **report}, indent=2))


if __name__ == "__main__":
    main()
