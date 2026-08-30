"""Shared spatial preprocessing for Wan video and first-frame inputs."""

from __future__ import annotations

import cv2
import numpy as np


def cover_center_crop_frames(
    frames: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize to cover ``width x height``, then take the centered crop."""
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] != 3 or value.shape[0] < 1:
        raise ValueError("frames must have shape [F, H, W, 3]")
    if width <= 0 or height <= 0:
        raise ValueError("output width and height must be positive")
    source_height, source_width = value.shape[1:3]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source frames must have positive spatial dimensions")
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    resized = np.stack(
        [
            cv2.resize(
                frame,
                (resized_width, resized_height),
                interpolation=interpolation,
            )
            for frame in value
        ]
    )
    x0 = (resized_width - width) // 2
    y0 = (resized_height - height) // 2
    return np.ascontiguousarray(
        resized[:, y0 : y0 + height, x0 : x0 + width]
    )


def cover_center_crop_intrinsics(
    camera_intrinsics: np.ndarray,
    source_size_px: tuple[int, int] | np.ndarray,
    target_size_px: tuple[int, int],
) -> np.ndarray:
    """Transform a pinhole matrix through the exact RGB resize/crop operation."""
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics must be a finite 3 x 3 matrix")
    source = np.asarray(source_size_px).reshape(-1)
    if (
        source.shape != (2,)
        or not np.isfinite(source).all()
        or np.any(source <= 0)
        or not np.allclose(source, np.rint(source))
    ):
        raise ValueError("source image size must contain positive integer width/height")
    source_width, source_height = (int(value) for value in source)
    target_width, target_height = (int(value) for value in target_size_px)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target image size must be positive")

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    crop_x = (resized_width - target_width) // 2
    crop_y = (resized_height - target_height) // 2

    transformed = intrinsics.copy()
    transformed[0, :] *= scale_x
    transformed[1, :] *= scale_y
    # OpenCV's half-pixel resize convention adds a translation that is not
    # represented by scaling K alone.
    transformed[0, 2] += 0.5 * scale_x - 0.5 - crop_x
    transformed[1, 2] += 0.5 * scale_y - 0.5 - crop_y
    return transformed
