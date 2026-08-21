#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from schema import iter_jsonl
from project_defaults import DATASET_MANIFEST, DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("outputs/training_data_review/scene_conditions.png")
BODY_COLORS = {
    0: "#5c86bd",
    1: "#e84b57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render first-frame and scene-condition contact sheets"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        required=DATASET_ROOT is None,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _equal_limits(axis, points: np.ndarray) -> None:
    low = np.quantile(points, 0.005, axis=0)
    high = np.quantile(points, 0.995, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.58, 1e-4)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    base_records = {
        record["base_scene_id"]: record
        for record in iter_jsonl(dataset_root / DATASET_MANIFEST)
        if record["sweep"]["mode"] == "base"
    }
    missing = [scene_id for scene_id in args.scene_id if scene_id not in base_records]
    if missing:
        raise ValueError(f"unknown base scene ids: {missing}")

    rows = len(args.scene_id)
    figure = plt.figure(figsize=(14, rows * 4.2), facecolor="#17191c")
    reports = []
    for row, scene_id in enumerate(args.scene_id):
        record = base_records[scene_id]
        frame_path = dataset_root / record["conditioning"]["first_frame"]
        scene_path = dataset_root / record["conditioning"]["scene"]
        with np.load(scene_path, allow_pickle=False) as archive:
            xyz = archive["xyz_normalized"].astype(np.float32)
            body = archive["body_id"].astype(np.int16)
            radius = float(archive["context_radius"])
        oriented = np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])

        frame_axis = figure.add_subplot(rows, 2, row * 2 + 1)
        with Image.open(frame_path) as image:
            frame_axis.imshow(image.convert("RGB"))
        frame_axis.set_axis_off()
        frame_axis.set_title(scene_id, color="white", fontsize=8, loc="left", pad=6)

        scene_axis = figure.add_subplot(rows, 2, row * 2 + 2, projection="3d")
        scene_axis.set_facecolor("#17191c")
        for body_id in sorted(int(value) for value in np.unique(body)):
            mask = body == body_id
            points = oriented[mask]
            scene_axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=4.0 if body_id == 1 else 2.2,
                c=BODY_COLORS.get(body_id, "#9aa0a6"),
                alpha=0.88,
                linewidths=0,
                depthshade=False,
            )
        _equal_limits(scene_axis, oriented)
        scene_axis.view_init(elev=23, azim=-58)
        scene_axis.set_axis_off()
        scene_axis.set_title(
            f"{len(body)} points | object {(body == 1).sum()} | radius {radius:.3f} m",
            color="white",
            fontsize=9,
            loc="left",
            pad=4,
        )
        reports.append(
            {
                "scene_id": scene_id,
                "scene": record["conditioning"]["scene"],
                "first_frame": record["conditioning"]["first_frame"],
                "point_count": int(len(body)),
                "object_point_count": int((body == 1).sum()),
                "context_radius": radius,
            }
        )

    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015, hspace=0.16, wspace=0.02)
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps({"output": str(output), "scenes": reports}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), "scene_count": rows}, indent=2))


if __name__ == "__main__":
    main()
