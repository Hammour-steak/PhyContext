#!/usr/bin/env python3
"""Validate and rasterize published PhysSweep point trajectories."""

from __future__ import annotations

import numpy as np


POINT_TRAJECTORY_SCHEMA = "physweep.point_trajectories.v1"
POINT_COUNT = 2048
MAX_OBJECTS = 3
TRACK_CHANNELS_PER_OBJECT = 6


def validate_point_trajectory(payload: dict[str, np.ndarray]) -> None:
    required = {
        "time_s",
        "object_ids",
        "points_world_m",
        "points_camera_m",
        "tracks_xy_px",
        "depth_m",
        "valid",
        "initial_points_camera_m",
        "camera_from_world",
        "camera_intrinsics",
        "image_size_px",
        "metadata_json",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"point trajectory is missing fields: {', '.join(missing)}")
    points_world = np.asarray(payload["points_world_m"])
    points_camera = np.asarray(payload["points_camera_m"])
    tracks = np.asarray(payload["tracks_xy_px"])
    depth = np.asarray(payload["depth_m"])
    valid = np.asarray(payload["valid"])
    if points_world.ndim != 4 or points_world.shape[-1] != 3:
        raise ValueError("points_world_m must have shape [T, O, 2048, 3]")
    if points_world.shape[2] != POINT_COUNT:
        raise ValueError("point trajectory must contain exactly 2048 points per object")
    if points_camera.shape != points_world.shape:
        raise ValueError("world and camera point trajectories must have equal shapes")
    expected_tracks = points_world.shape[:3] + (2,)
    if tracks.shape != expected_tracks:
        raise ValueError(f"tracks_xy_px must have shape {expected_tracks}")
    if depth.shape != points_world.shape[:3] or valid.shape != points_world.shape[:3]:
        raise ValueError("depth and valid arrays must match [T, O, 2048]")
    object_ids = np.asarray(payload["object_ids"]).reshape(-1)
    if len(object_ids) != points_world.shape[1] or len(set(map(str, object_ids))) != len(object_ids):
        raise ValueError("object_ids must be unique and match the object axis")
    if len(object_ids) > MAX_OBJECTS:
        raise ValueError(f"at most {MAX_OBJECTS} dynamic objects are supported")
    for name in ("points_world_m", "points_camera_m", "tracks_xy_px", "depth_m"):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if not np.isfinite(payload["time_s"]).all():
        raise ValueError("time_s contains non-finite values")


def _splat_max(
    output: np.ndarray, x: np.ndarray, y: np.ndarray, values: np.ndarray
) -> None:
    if not len(x):
        return
    height, width = output.shape
    ix = np.rint(x).astype(np.int64)
    iy = np.rint(y).astype(np.int64)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    if np.any(inside):
        np.maximum.at(output, (iy[inside], ix[inside]), values[inside].astype(output.dtype))


def _splat_mean(
    output: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> None:
    if not len(x):
        return
    height, width = output.shape
    ix = np.rint(x).astype(np.int64)
    iy = np.rint(y).astype(np.int64)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    if not np.any(inside):
        return
    sums = np.zeros_like(output, dtype=np.float32)
    counts = np.zeros_like(output, dtype=np.float32)
    np.add.at(sums, (iy[inside], ix[inside]), values[inside].astype(np.float32))
    np.add.at(counts, (iy[inside], ix[inside]), 1.0)
    occupied = counts > 0
    output[occupied] = sums[occupied] / counts[occupied]


def cover_center_crop_coordinates(
    coordinates_xy_px: np.ndarray,
    source_size_px: tuple[int, int],
    target_size_px: tuple[int, int],
) -> np.ndarray:
    """Apply the same cover-resize and center-crop transform as video inputs."""
    coordinates = np.asarray(coordinates_xy_px, dtype=np.float32)
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have a final dimension of 2")
    source_width, source_height = [int(value) for value in source_size_px]
    target_width, target_height = [int(value) for value in target_size_px]
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target sizes must be positive")
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    crop_x = (resized_width - target_width) // 2
    crop_y = (resized_height - target_height) // 2
    transformed = coordinates.copy()
    transformed[..., 0] = transformed[..., 0] * scale_x - crop_x
    transformed[..., 1] = transformed[..., 1] * scale_y - crop_y
    return transformed


def rasterize_projected_tracks(
    payload: dict[str, np.ndarray],
    output_size_px: tuple[int, int],
    *,
    preprocess_size_px: tuple[int, int],
    frame_indices: list[int] | None = None,
    max_objects: int = MAX_OBJECTS,
    depth_normalization: str = "sequence",
    spatial_transform: str = "cover_center_crop",
) -> np.ndarray:
    """Rasterize all point tracks into fixed-width per-object control channels.

    Each object contributes six channels: source-point occupancy, current-point
    occupancy, source-anchored dx, source-anchored dy, current depth, and
    current validity. Empty object slots are zero-filled.
    """
    validate_point_trajectory(payload)
    if max_objects <= 0 or max_objects > MAX_OBJECTS:
        raise ValueError("max_objects is outside the supported range")
    output_width, output_height = [int(value) for value in output_size_px]
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output size must be positive")
    tracks = np.asarray(payload["tracks_xy_px"], dtype=np.float32)
    depth = np.asarray(payload["depth_m"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool)
    frame_count, object_count, _, _ = tracks.shape
    if object_count > max_objects:
        raise ValueError("point trajectory has more objects than the conditioner budget")
    if frame_indices is None:
        frame_indices = list(range(frame_count))
    if not frame_indices or any(index < 0 or index >= frame_count for index in frame_indices):
        raise ValueError("frame_indices must select valid trajectory frames")
    image_width, image_height = [int(value) for value in payload["image_size_px"]]
    preprocess_width, preprocess_height = [int(value) for value in preprocess_size_px]
    if min(preprocess_width, preprocess_height) <= 0:
        raise ValueError("preprocess size must be positive")
    if depth_normalization not in {"sequence", "per_frame"}:
        raise ValueError("depth normalization must be sequence or per_frame")
    if spatial_transform not in {"cover_center_crop", "source_image"}:
        raise ValueError(
            "spatial transform must be cover_center_crop or source_image"
        )
    grid_scale_x = output_width / preprocess_width
    grid_scale_y = output_height / preprocess_height
    if spatial_transform == "source_image":
        if (preprocess_width, preprocess_height) != (image_width, image_height):
            raise ValueError(
                "source-image tracks require preprocess size to equal image size"
            )
        processed_tracks = tracks
    else:
        processed_tracks = cover_center_crop_coordinates(
            tracks,
            (image_width, image_height),
            (preprocess_width, preprocess_height),
        )
    channels = np.zeros(
        (max_objects * TRACK_CHANNELS_PER_OBJECT, len(frame_indices), output_height, output_width),
        dtype=np.float32,
    )
    if depth_normalization == "sequence":
        all_valid_depth = depth[valid & np.isfinite(depth)]
        if len(all_valid_depth):
            sequence_depth_low, sequence_depth_high = np.percentile(
                all_valid_depth, [1.0, 99.0]
            )
        else:
            sequence_depth_low, sequence_depth_high = 0.0, 1.0
        sequence_depth_span = max(
            float(sequence_depth_high - sequence_depth_low), 1.0e-6
        )
    for object_index in range(object_count):
        channel = object_index * TRACK_CHANNELS_PER_OBJECT
        source_xy = processed_tracks[0, object_index]
        source_valid = valid[0, object_index]
        source_x = source_xy[:, 0] * grid_scale_x
        source_y = source_xy[:, 1] * grid_scale_y
        source_mask = source_valid & np.isfinite(source_x) & np.isfinite(source_y)
        for output_index, frame_index in enumerate(frame_indices):
            current_xy = processed_tracks[frame_index, object_index]
            current_valid = valid[frame_index, object_index]
            current_x = current_xy[:, 0] * grid_scale_x
            current_y = current_xy[:, 1] * grid_scale_y
            shared = source_mask & current_valid
            source_occupancy = channels[channel, output_index]
            target_occupancy = channels[channel + 1, output_index]
            _splat_max(source_occupancy, source_x[source_mask], source_y[source_mask], np.ones(source_mask.sum()))
            _splat_max(target_occupancy, current_x[current_valid], current_y[current_valid], np.ones(current_valid.sum()))
            # Normalize in the same coordinate system used by the occupancy
            # channels.  Both points already come from `processed_tracks`,
            # which applies the same resize and crop as the video input.
            dx = (current_xy[:, 0] - source_xy[:, 0]) / max(
                float(preprocess_width), 1.0
            )
            dy = (current_xy[:, 1] - source_xy[:, 1]) / max(
                float(preprocess_height), 1.0
            )
            _splat_mean(
                channels[channel + 2, output_index],
                source_x[shared],
                source_y[shared],
                dx[shared],
            )
            _splat_mean(
                channels[channel + 3, output_index],
                source_x[shared],
                source_y[shared],
                dy[shared],
            )
            depth_values = depth[frame_index, object_index]
            finite_depth = current_valid & np.isfinite(depth_values)
            if np.any(finite_depth):
                if depth_normalization == "per_frame":
                    depth_low, depth_high = np.percentile(
                        depth_values[finite_depth], [1.0, 99.0]
                    )
                    depth_span = max(float(depth_high - depth_low), 1.0e-6)
                else:
                    depth_low = sequence_depth_low
                    depth_span = sequence_depth_span
                depth_norm = 1.0 - np.clip(
                    (depth_values - depth_low) / depth_span, 0.0, 1.0
                )
                _splat_mean(
                    channels[channel + 4, output_index],
                    current_x[finite_depth],
                    current_y[finite_depth],
                    depth_norm[finite_depth],
                )
            _splat_max(
                channels[channel + 5, output_index],
                current_x[current_valid],
                current_y[current_valid],
                np.ones(current_valid.sum()),
            )
    return channels
