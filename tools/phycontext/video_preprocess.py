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
