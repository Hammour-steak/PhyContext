from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _scatter(ax, points: np.ndarray, colors: np.ndarray, title: str, seed: int) -> None:
    if len(points) > 28000:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), 28000, replace=False)
        points, colors = points[indices], colors[indices]
    ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colors / 255.0, s=0.7, linewidths=0)
    ax.set_title(title)
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    ranges = np.ptp(points, axis=0)
    center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    radius = max(float(np.max(ranges)) * 0.52, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(-center[1] - radius, -center[1] + radius)


def save_review(
    path: Path,
    rgb: np.ndarray,
    overlay: np.ndarray,
    inpainted_rgb: np.ndarray,
    environment_points: np.ndarray,
    environment_colors: np.ndarray,
    merged_points: np.ndarray,
    semantic_colors: np.ndarray,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    ax1 = figure.add_subplot(2, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Input first frame")
    ax1.axis("off")
    ax2 = figure.add_subplot(2, 3, 2)
    ax2.imshow(overlay)
    ax2.set_title("SAM2 selected object")
    ax2.axis("off")
    ax3 = figure.add_subplot(2, 3, 3)
    ax3.imshow(inpainted_rgb)
    ax3.set_title("Object removed in RGB")
    ax3.axis("off")
    ax4 = figure.add_subplot(2, 3, 4, projection="3d")
    _scatter(ax4, environment_points, environment_colors, "Visible environment only", seed)
    ax5 = figure.add_subplot(2, 3, 5, projection="3d")
    _scatter(ax5, merged_points, semantic_colors, "Completed scene by source", seed + 1)
    ax6 = figure.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.02,
        0.96,
        "blue: observed environment\ngreen: completed environment\nred: aligned InstantMesh object",
        va="top",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_component_review(
    path: Path,
    object_partial: np.ndarray,
    object_aligned: np.ndarray,
    environment_observed: np.ndarray,
    environment_completed: np.ndarray,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    partial_colors = np.tile([74, 144, 226], (len(object_partial), 1)).astype(np.uint8)
    object_points = np.concatenate([object_partial, object_aligned], axis=0)
    object_colors = np.concatenate(
        [partial_colors, np.tile([255, 74, 92], (len(object_aligned), 1))], axis=0
    ).astype(np.uint8)
    environment_points = np.concatenate([environment_observed, environment_completed], axis=0)
    environment_colors = np.concatenate(
        [
            np.tile([74, 144, 226], (len(environment_observed), 1)),
            np.tile([79, 190, 125], (len(environment_completed), 1)),
        ],
        axis=0,
    ).astype(np.uint8)

    ax1 = figure.add_subplot(1, 3, 1, projection="3d")
    _scatter(ax1, object_partial, partial_colors, "Object: visible VGGT", seed)
    ax2 = figure.add_subplot(1, 3, 2, projection="3d")
    _scatter(ax2, object_points, object_colors, "Object: visible + aligned mesh", seed + 1)
    ax3 = figure.add_subplot(1, 3, 3, projection="3d")
    _scatter(
        ax3,
        environment_points,
        environment_colors,
        "Environment: observed + occlusion completion",
        seed + 2,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_alignment_review(
    path: Path,
    rgb: np.ndarray,
    target_mask: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth: np.ndarray,
) -> None:
    overlay = rgb.copy()
    target_edge = cv2.Canny(target_mask.astype(np.uint8) * 255, 50, 150) > 0
    rendered_edge = cv2.Canny(rendered_mask.astype(np.uint8) * 255, 50, 150) > 0
    overlay[target_edge] = [255, 64, 64]
    overlay[rendered_edge] = [64, 255, 96]
    visible_depth = rendered_depth[rendered_depth > 0]
    if len(visible_depth):
        low, high = np.quantile(visible_depth, [0.02, 0.98])
        normalized = np.clip((rendered_depth - low) / max(high - low, 1e-8), 0, 1)
        depth_image = plt.get_cmap("viridis")(normalized)[..., :3]
        depth_image[rendered_depth <= 0] = 0
    else:
        depth_image = np.zeros_like(rgb, dtype=np.float32)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title("Target edge red / rendered edge green")
    axes[2].imshow(depth_image)
    axes[2].set_title("Aligned mesh depth")
    for axis in axes:
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)
