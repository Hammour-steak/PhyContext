from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneDecomposition:
    valid_mask: np.ndarray
    object_mask: np.ndarray
    object_core_mask: np.ndarray
    environment_mask: np.ndarray
    reliable_environment_mask: np.ndarray
    object_points: np.ndarray
    object_colors: np.ndarray
    environment_points: np.ndarray
    environment_colors: np.ndarray
    confidence_threshold: float
    object_confidence_threshold: float
    omitted_boundary_points: int


def decompose_scene(
    points: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    object_mask: np.ndarray,
    confidence_quantile: float,
    boundary_pixels: int = 2,
    content_mask: np.ndarray | None = None,
) -> SceneDecomposition:
    """Split a VGGT point map without leaking object-edge points into the environment."""
    if points.shape[:2] != object_mask.shape:
        raise ValueError("Point map and object mask must have matching image dimensions")
    if content_mask is None:
        content_mask = np.ones_like(object_mask, dtype=bool)
    else:
        content_mask = np.asarray(content_mask, dtype=bool)
        if content_mask.shape != object_mask.shape:
            raise ValueError("Content mask and object mask must have matching dimensions")
    finite = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & (depth > 0)
        & content_mask
    )
    if not finite.any():
        raise RuntimeError("VGGT did not produce any finite positive-depth points")
    kernel_size = boundary_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    object_core = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    object_exclusion = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    environment = ~object_exclusion

    environment_finite = finite & environment
    if environment_finite.sum() < 128:
        raise RuntimeError(
            f"Only {int(environment_finite.sum())} finite environment points are available"
        )
    confidence_threshold = float(
        np.quantile(confidence[environment_finite], confidence_quantile)
    )
    reliable_environment = environment_finite & (confidence >= confidence_threshold)

    object_finite = finite & object_core
    if object_finite.sum() < 64:
        object_finite = finite & object_mask
    if object_finite.sum() < 32:
        raise RuntimeError(
            f"Only {int(object_finite.sum())} finite object points are available"
        )
    object_confidence_threshold = float(
        np.quantile(confidence[object_finite], confidence_quantile)
    )
    object_valid = object_finite & (confidence >= object_confidence_threshold)
    if object_valid.sum() < 32:
        raise RuntimeError(
            f"Only {int(object_valid.sum())} valid object points survived local filtering"
        )

    assigned = object_valid | environment_finite
    return SceneDecomposition(
        valid_mask=assigned,
        object_mask=object_mask.astype(bool),
        object_core_mask=object_valid,
        environment_mask=environment_finite,
        reliable_environment_mask=reliable_environment,
        object_points=points[object_valid].astype(np.float32),
        object_colors=colors[object_valid].astype(np.uint8),
        environment_points=points[environment_finite].astype(np.float32),
        environment_colors=colors[environment_finite].astype(np.uint8),
        confidence_threshold=confidence_threshold,
        object_confidence_threshold=object_confidence_threshold,
        omitted_boundary_points=int((finite & ~assigned).sum()),
    )
