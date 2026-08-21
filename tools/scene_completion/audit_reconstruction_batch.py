#!/usr/bin/env python3
"""Build a quantitative and visual audit for a reconstruction validation batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_interactive_html import build_html, build_scene


PANELS = (
    ("input_preprocessed.png", "input"),
    ("object_mask_overlay.png", "controlled object"),
    ("alignment_review.png", "mesh alignment"),
    ("components_review.png", "scene components"),
)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def failure_reasons(report: dict) -> list[str]:
    gate = report["quality_gate"]
    thresholds = gate["thresholds"]
    values = gate["measurements"]
    failed = []
    if values["aligned_mask_iou"] < thresholds["minimum_aligned_mask_iou"]:
        failed.append("mesh-mask")
    if (
        values["relative_depth_error"] is None
        or values["relative_depth_error"] > thresholds["maximum_relative_depth_error"]
    ):
        failed.append("depth")
    if values["environment_completion_fraction"] < thresholds["minimum_environment_completion_fraction"]:
        failed.append("environment-hole")
    if values["environment_surface_coverage"] < thresholds["minimum_environment_surface_coverage"]:
        failed.append("environment-coverage")
    if (
        values["depthlab_positive_completion_fraction"]
        < thresholds["minimum_depthlab_positive_completion_fraction"]
    ):
        failed.append("depthlab-missing")
    if values["depthlab_known_depth_error"] > thresholds["maximum_depthlab_known_depth_error"]:
        failed.append("depthlab-scale")
    if (
        values["environment_depth_seam_p90_error"]
        > thresholds["maximum_environment_depth_seam_p90_error"]
    ):
        failed.append("environment-seam")
    if not values["physical_proxy_verified"]:
        failed.append("support")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(path for path in root.glob("[0-9][0-9]_*") if path.is_dir())
    if not scene_dirs:
        raise ValueError(f"no reconstruction scenes found in {root}")

    panel_width, panel_height = 360, 230
    header_height = 62
    rows = []
    summaries = []
    for scene_dir in scene_dirs:
        report = json.loads((scene_dir / "report.json").read_text(encoding="utf-8"))
        measurements = report["quality_gate"]["measurements"]
        failed = failure_reasons(report)
        environment_passed = not any(
            reason.startswith("environment-") or reason.startswith("depthlab-")
            for reason in failed
        )
        summaries.append(
            {
                "scene": scene_dir.name,
                "passed": report["quality_gate"]["passed"],
                "environment_passed": environment_passed,
                "failures": failed,
                "gt_mask_iou": report.get("gt_mask_iou"),
                **measurements,
            }
        )
        panels = []
        for filename, label in PANELS:
            image = cv2.imread(str(scene_dir / filename), cv2.IMREAD_COLOR)
            if image is None:
                image = np.full((panel_height, panel_width, 3), 32, dtype=np.uint8)
                cv2.putText(image, f"missing {filename}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
            panel = fit_panel(image, panel_width, panel_height)
            cv2.putText(panel, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            panels.append(panel)
        body = np.concatenate(panels, axis=1)
        header = np.full((header_height, body.shape[1], 3), 20, dtype=np.uint8)
        status_color = (70, 210, 90) if not failed else (60, 100, 245)
        title = scene_dir.name
        cv2.putText(header, title[:150], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        depth = measurements["relative_depth_error"]
        metrics = (
            f"{'PASS' if not failed else 'REJECT'}  failures={','.join(failed) or 'none'}  "
            f"GT-mask={report.get('gt_mask_iou', float('nan')):.3f}  "
            f"mesh-mask={measurements['aligned_mask_iou']:.3f}  "
            f"rel-depth={depth if depth is not None else float('nan'):.4f}  "
            f"DepthLab-known={measurements['depthlab_known_depth_error']:.3f}  "
            f"env-seam-p90={measurements['environment_depth_seam_p90_error']:.3f}  "
            f"support={measurements['physical_proxy_status']}"
        )
        cv2.putText(header, metrics, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
        rows.append(np.concatenate([header, body], axis=0))

    sheet = np.concatenate(rows, axis=0)
    image_path = output / "reconstruction_audit10.png"
    cv2.imwrite(str(image_path), sheet)
    viewer_path = output / "reconstruction_audit10.html"
    viewer_path.write_text(
        build_html(
            [
                build_scene(scene_dir, scene_dir.name, 20260809 + index)
                for index, scene_dir in enumerate(scene_dirs)
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": "phycontext.scene_reconstruction_batch_audit.v1",
        "scene_count": len(summaries),
        "quality_pass_count": sum(item["passed"] for item in summaries),
        "environment_quality_pass_count": sum(
            item["environment_passed"] for item in summaries
        ),
        "failure_counts": {
            reason: sum(reason in item["failures"] for item in summaries)
            for reason in (
                "mesh-mask",
                "depth",
                "environment-hole",
                "environment-coverage",
                "depthlab-missing",
                "depthlab-scale",
                "environment-seam",
                "support",
            )
        },
        "review": str(image_path),
        "viewer": str(viewer_path),
        "scenes": summaries,
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
