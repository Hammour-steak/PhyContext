#!/usr/bin/env python3
"""Validate and rasterize published PhysSweep point trajectories."""

from __future__ import annotations

import json

import numpy as np


POINT_TRAJECTORY_SCHEMA = "physweep.point_trajectories.v1"
POINT_COUNT = 2048
MAX_OBJECTS = 3
TRACK_CHANNELS_PER_OBJECT = 6
DAS_TRACK_CHANNELS_PER_OBJECT = 4
DAS_TRACK_REPRESENTATION = "das_3d_tracks"
POINT_TRACK_DEFINITION = "perspective_projection_of_fixed_material_points"
POINT_VISIBILITY_DEFINITION = "in_frame_and_clip_validity;_not_a_z_buffer"


def _trajectory_metadata(payload: dict[str, np.ndarray]) -> dict:
    metadata_value = np.asarray(payload["metadata_json"])
    if metadata_value.size != 1:
        raise ValueError("metadata_json must contain one JSON object")
    metadata_text = metadata_value.reshape(()).item()
    try:
        if isinstance(metadata_text, bytes):
            metadata_text = metadata_text.decode("utf-8")
        metadata = json.loads(str(metadata_text))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode an object")
    return metadata


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
    initial_points_camera = np.asarray(payload["initial_points_camera_m"])
    if initial_points_camera.shape != points_world.shape[1:]:
        raise ValueError(
            "initial_points_camera_m must have shape [O, 2048, 3]"
        )
    expected_tracks = points_world.shape[:3] + (2,)
    if tracks.shape != expected_tracks:
        raise ValueError(f"tracks_xy_px must have shape {expected_tracks}")
    if depth.shape != points_world.shape[:3] or valid.shape != points_world.shape[:3]:
        raise ValueError("depth and valid arrays must match [T, O, 2048]")
    if not np.isin(valid, (0, 1)).all():
        raise ValueError("valid must contain only boolean or binary values")
    if not np.allclose(
        depth,
        points_camera[..., 2],
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError("depth_m must equal the camera-space z coordinate")
    if np.any(valid & (depth <= 1.0e-6)):
        raise ValueError("valid trajectory points must have positive depth")
    object_ids = np.asarray(payload["object_ids"]).reshape(-1)
    if len(object_ids) != points_world.shape[1] or len(set(map(str, object_ids))) != len(object_ids):
        raise ValueError("object_ids must be unique and match the object axis")
    if not 1 <= len(object_ids) <= MAX_OBJECTS:
        raise ValueError(
            f"point trajectory must contain one to {MAX_OBJECTS} dynamic objects"
        )
    for name in ("points_world_m", "points_camera_m", "tracks_xy_px", "depth_m"):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if not np.isfinite(payload["time_s"]).all():
        raise ValueError("time_s contains non-finite values")
    frame_count = points_world.shape[0]
    time_s = np.asarray(payload["time_s"]).reshape(-1)
    if time_s.shape[0] != frame_count:
        raise ValueError("time_s must match the trajectory frame count")
    if frame_count > 1 and not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")
    if not np.allclose(
        initial_points_camera,
        points_camera[0],
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "initial_points_camera_m must equal points_camera_m at frame zero"
        )
    camera_from_world = np.asarray(payload["camera_from_world"])
    if camera_from_world.shape not in {(4, 4), (frame_count, 4, 4)}:
        raise ValueError("camera_from_world must have shape [4, 4] or [T, 4, 4]")
    camera_intrinsics = np.asarray(payload["camera_intrinsics"])
    if camera_intrinsics.shape not in {(3, 3), (frame_count, 3, 3)}:
        raise ValueError("camera_intrinsics must have shape [3, 3] or [T, 3, 3]")
    if not np.isfinite(camera_from_world).all() or not np.isfinite(
        camera_intrinsics
    ).all():
        raise ValueError("camera matrices contain non-finite values")
    camera_sequence_for_contract = (
        camera_from_world[None]
        if camera_from_world.ndim == 2
        else camera_from_world
    )
    if not np.allclose(
        camera_sequence_for_contract[..., 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError("camera_from_world must contain homogeneous transforms")
    rotations = camera_sequence_for_contract[..., :3, :3].astype(np.float64)
    rotation_gram = np.einsum("tij,tkj->tik", rotations, rotations)
    rotation_determinants = np.linalg.det(rotations)
    if not np.allclose(
        rotation_gram,
        np.eye(3),
        rtol=1.0e-4,
        atol=1.0e-4,
    ) or not np.allclose(
        np.abs(rotation_determinants),
        1.0,
        rtol=1.0e-4,
        atol=1.0e-4,
    ):
        raise ValueError("camera_from_world rotations must be orthonormal")
    intrinsic_sequence_for_contract = (
        camera_intrinsics[None]
        if camera_intrinsics.ndim == 2
        else camera_intrinsics
    )
    if (
        np.any(intrinsic_sequence_for_contract[..., 0, 0] <= 0.0)
        or np.any(intrinsic_sequence_for_contract[..., 1, 1] <= 0.0)
        or not np.allclose(
            intrinsic_sequence_for_contract[..., 2, :],
            np.asarray([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise ValueError("camera_intrinsics must use positive focal lengths")
    image_size = np.asarray(payload["image_size_px"]).reshape(-1)
    if (
        image_size.shape != (2,)
        or not np.isfinite(image_size).all()
        or np.any(image_size <= 0)
        or not np.allclose(image_size, np.rint(image_size))
    ):
        raise ValueError(
            "image_size_px must contain an integer positive width and height"
        )
    metadata = _trajectory_metadata(payload)
    if metadata.get("schema", POINT_TRAJECTORY_SCHEMA) != POINT_TRAJECTORY_SCHEMA:
        raise ValueError("metadata_json uses an unsupported trajectory schema")
    if metadata.get(
        "coordinate_frame_camera", "camera_right_up_forward"
    ) != "camera_right_up_forward":
        raise ValueError("metadata_json uses an unsupported camera coordinate frame")
    if metadata.get("schema") == POINT_TRAJECTORY_SCHEMA:
        if metadata.get("coordinate_frame_world") != "pybullet_world_xyz":
            raise ValueError("metadata_json uses an unsupported world coordinate frame")
        # PhysSweep stores camera axes in right/up/forward row order. Relative
        # to PyBullet's right-handed world this basis is left-handed, so the
        # world-to-camera rotation has determinant -1 by construction.
        if np.any(rotation_determinants >= 0.0):
            raise ValueError(
                "camera_from_world handedness does not match camera_right_up_forward"
            )
        if int(metadata.get("point_count", -1)) != POINT_COUNT:
            raise ValueError("metadata_json point count does not match the payload")
        if int(metadata.get("object_count", -1)) != len(object_ids):
            raise ValueError("metadata_json object count does not match the payload")
        if [str(value) for value in metadata.get("object_ids", [])] != list(
            map(str, object_ids)
        ):
            raise ValueError("metadata_json object ids do not match the payload")
        if metadata.get("track_definition") != POINT_TRACK_DEFINITION:
            raise ValueError("metadata_json uses an unsupported track definition")
        if metadata.get("visibility_definition") != POINT_VISIBILITY_DEFINITION:
            raise ValueError("metadata_json uses an unsupported visibility definition")

        camera_sequence = camera_from_world
        if camera_sequence.ndim == 2:
            camera_sequence = np.broadcast_to(
                camera_sequence, (frame_count, 4, 4)
            )
        homogeneous_world = np.concatenate(
            (
                points_world.astype(np.float64),
                np.ones(points_world.shape[:-1] + (1,), dtype=np.float64),
            ),
            axis=-1,
        )
        expected_camera = np.einsum(
            "tij,tonj->toni", camera_sequence, homogeneous_world, optimize=True
        )[..., :3]
        if not np.allclose(
            points_camera,
            expected_camera,
            rtol=1.0e-4,
            atol=1.0e-5,
        ):
            raise ValueError(
                "points_camera_m does not match camera_from_world and points_world_m"
            )

        intrinsic_sequence = camera_intrinsics
        if intrinsic_sequence.ndim == 2:
            intrinsic_sequence = np.broadcast_to(
                intrinsic_sequence, (frame_count, 3, 3)
            )
        projection_points = points_camera.astype(np.float64, copy=True)
        projection_points[..., 1] *= -1.0
        projected = np.einsum(
            "tij,tonj->toni", intrinsic_sequence, projection_points, optimize=True
        )
        expected_tracks = np.zeros_like(tracks, dtype=np.float64)
        nonzero_projection = np.abs(projected[..., 2]) > 1.0e-12
        np.divide(
            projected[..., :2],
            projected[..., 2:3],
            out=expected_tracks,
            where=nonzero_projection[..., None],
        )
        positive_depth = points_camera[..., 2] > 1.0e-6
        if np.any(positive_depth) and not np.allclose(
            tracks[positive_depth],
            expected_tracks[positive_depth],
            rtol=1.0e-4,
            atol=1.0e-3,
        ):
            raise ValueError(
                "tracks_xy_px does not match PhysSweep camera projection"
            )
        clip_start = float(metadata.get("clip_start_m", 0.03))
        clip_end = float(metadata.get("clip_end_m", 100.0))
        if not 0.0 <= clip_start < clip_end:
            raise ValueError("metadata_json camera clip range is invalid")
        width, height = [int(value) for value in image_size]
        expected_valid = (
            np.isfinite(expected_tracks).all(axis=-1)
            & np.isfinite(points_camera).all(axis=-1)
            & (points_camera[..., 2] > clip_start)
            & (points_camera[..., 2] < clip_end)
            & (expected_tracks[..., 0] >= 0.0)
            & (expected_tracks[..., 0] < width)
            & (expected_tracks[..., 1] >= 0.0)
            & (expected_tracks[..., 1] < height)
        )
        if not np.array_equal(valid.astype(bool), expected_valid):
            raise ValueError(
                "valid does not match PhysSweep in-frame and clip validity"
            )


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
    # Preserve sub-pixel side information until the caller discretizes pixels.
    # Float32 can turn a value beside a half-integer into an exact rounding tie.
    coordinates = np.asarray(coordinates_xy_px, dtype=np.float64)
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
    # OpenCV uses half-pixel centers for resize sampling.  Applying the exact
    # forward transform keeps projected geometry aligned with the resized RGB.
    transformed[..., 0] = (
        (transformed[..., 0] + 0.5) * scale_x - 0.5 - crop_x
    )
    transformed[..., 1] = (
        (transformed[..., 1] + 0.5) * scale_y - 0.5 - crop_y
    )
    return transformed


def unproject_physweep_tracks(
    tracks_xy_px: np.ndarray,
    depth_m: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray:
    """Recover right/up/forward camera points from pixels and metric depth.

    This is the exact inverse of PhysSweep's projection for positive-depth
    points. It is intentionally public so audits can round-trip released tracks
    instead of only checking the forward projection implemented by the loader.
    """
    tracks = np.asarray(tracks_xy_px, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if tracks.ndim < 2 or tracks.shape[-1] != 2:
        raise ValueError("tracks_xy_px must end in a two-coordinate axis")
    if depth.shape != tracks.shape[:-1]:
        raise ValueError("depth_m must match the track prefix shape")
    if not np.isfinite(tracks).all() or not np.isfinite(depth).all():
        raise ValueError("tracks and depth must be finite")
    frame_count = tracks.shape[0]
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (frame_count, 3, 3))
    elif intrinsics.shape != (frame_count, 3, 3):
        raise ValueError("camera_intrinsics must have shape [3, 3] or [T, 3, 3]")
    if not np.isfinite(intrinsics).all():
        raise ValueError("camera_intrinsics contains non-finite values")
    try:
        inverse_intrinsics = np.linalg.inv(intrinsics)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera_intrinsics is singular") from exc
    homogeneous_pixels = np.concatenate(
        (tracks, np.ones(tracks.shape[:-1] + (1,), dtype=np.float64)),
        axis=-1,
    )
    projection_points = np.einsum(
        "tij,t...j->t...i", inverse_intrinsics, homogeneous_pixels, optimize=True
    )
    projection_points *= depth[..., None]
    camera_points = projection_points
    camera_points[..., 1] *= -1.0
    return camera_points.astype(np.float32)


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
    tracks = np.asarray(payload["tracks_xy_px"], dtype=np.float64)
    depth = np.asarray(payload["depth_m"], dtype=np.float64)
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


def _das_point_identity_colors(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode first-frame camera coordinates as persistent DaS RGB identities.

    DaS assigns every 3D point one color from its first-frame ``(x, y, 1/z)``
    coordinate and keeps that color fixed through time.  Normalization is shared
    across all object slots in a sample so equal 3D locations have equal colors.
    Camera x/y use sample-wide min/max normalization. Reciprocal depth uses the
    robust 2nd/98th percentiles from the released DaS renderer.
    """
    initial = np.asarray(payload["initial_points_camera_m"], dtype=np.float64)
    finite = np.isfinite(initial).all(axis=-1) & (initial[..., 2] > 1.0e-6)
    # Identity is defined by first-frame 3D position, not first-frame in-frame
    # status. A positive-depth point that enters the view later must retain a
    # usable fixed identity even if its frame-zero projection was outside view.
    anchor_valid = finite
    coordinates = np.stack(
        (
            initial[..., 0],
            initial[..., 1],
            np.reciprocal(np.maximum(initial[..., 2], 1.0e-6)),
        ),
        axis=-1,
    )
    colors = np.zeros_like(coordinates, dtype=np.float64)
    if not np.any(anchor_valid):
        return colors, anchor_valid
    values = coordinates[anchor_valid]
    low = np.asarray(
        [values[:, 0].min(), values[:, 1].min(), np.percentile(values[:, 2], 2.0)],
        dtype=np.float64,
    )
    high = np.asarray(
        [values[:, 0].max(), values[:, 1].max(), np.percentile(values[:, 2], 98.0)],
        dtype=np.float64,
    )
    span = high - low
    stable = span > 1.0e-6
    colors[..., stable] = np.clip(
        (coordinates[..., stable] - low[stable]) / span[stable],
        0.0,
        1.0,
    )
    colors[..., ~stable] = 0.5
    colors[~anchor_valid] = 0.0
    return colors.astype(np.float32, copy=False), anchor_valid


def _project_static_camera0_points(
    points_camera0_m: np.ndarray,
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project static scene points through the trajectory camera sequence."""
    points_camera0 = np.asarray(points_camera0_m, dtype=np.float64)
    if points_camera0.ndim != 2 or points_camera0.shape[-1] != 3:
        raise ValueError("static scene points must have shape [N, 3]")
    if not np.isfinite(points_camera0).all():
        raise ValueError("static scene points contain non-finite values")
    camera_from_world = np.asarray(payload["camera_from_world"], dtype=np.float64)
    frame_count = int(np.asarray(payload["points_world_m"]).shape[0])
    if camera_from_world.ndim == 2:
        camera_from_world = np.broadcast_to(
            camera_from_world, (frame_count, 4, 4)
        )
    try:
        world_from_camera0 = np.linalg.inv(camera_from_world[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError("first camera transform is singular") from exc
    homogeneous = np.concatenate(
        (points_camera0.astype(np.float64), np.ones((len(points_camera0), 1))),
        axis=1,
    )
    points_world = (world_from_camera0 @ homogeneous.T).T
    camera_points = np.einsum(
        "tij,nj->tni", camera_from_world, points_world, optimize=True
    )[..., :3]
    intrinsics = np.asarray(payload["camera_intrinsics"], dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics, (frame_count, 3, 3))
    # PhysSweep camera coordinates are right/up/forward, whereas image rows grow
    # downward.  Its trajectory exporter therefore projects v = cy - fy*y/z.
    # Flip camera y before applying K so static scene points use exactly the same
    # convention as the published dynamic tracks (including any K skew terms).
    projection_points = camera_points.copy()
    projection_points[..., 1] *= -1.0
    projected = np.einsum(
        "tij,tnj->tni", intrinsics, projection_points, optimize=True
    )
    depth = camera_points[..., 2]
    metadata = _trajectory_metadata(payload)
    clip_start = float(metadata.get("clip_start_m", 0.03))
    clip_end = float(metadata.get("clip_end_m", 100.0))
    if not 0.0 <= clip_start < clip_end:
        raise ValueError("metadata_json camera clip range is invalid")
    valid = (
        np.isfinite(camera_points).all(axis=-1)
        & np.isfinite(projected).all(axis=-1)
        & (depth > clip_start)
        & (depth < clip_end)
    )
    tracks = np.zeros((frame_count, len(points_camera0), 2), dtype=np.float64)
    np.divide(
        projected[..., :2],
        projected[..., 2:3],
        out=tracks,
        where=np.abs(projected[..., 2:3]) > 1.0e-12,
    )
    return tracks, depth.astype(np.float64, copy=False), valid


def _expand_pixel_splats(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    objects: np.ndarray,
    points: np.ndarray,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if radius_px < 0:
        raise ValueError("point splat radius must be non-negative")
    if not len(x) or radius_px == 0:
        return x, y, depth, objects, points
    offsets = np.arange(-radius_px, radius_px + 1, dtype=np.int64)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="xy")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)
    repeats = offset_x.shape[1]
    return (
        (x[:, None] + offset_x).reshape(-1),
        (y[:, None] + offset_y).reshape(-1),
        np.repeat(depth, repeats),
        np.repeat(objects, repeats),
        np.repeat(points, repeats),
    )


def rasterize_das_3d_tracks(
    payload: dict[str, np.ndarray],
    output_size_px: tuple[int, int],
    *,
    preprocess_size_px: tuple[int, int],
    frame_indices: list[int] | None = None,
    max_objects: int = MAX_OBJECTS,
    spatial_transform: str = "cover_center_crop",
    static_points_camera0_m: np.ndarray | None = None,
    point_radius_px: int = 1,
) -> np.ndarray:
    """Render DaS-style identity-preserving 3D tracking maps for Wan.

    Each object slot contributes ``R, G, B, visibility``. RGB is determined once
    from the point's first-frame camera coordinate ``(x, y, 1/z)``. At every
    selected frame all objects and optional static scene points compete in a
    full-preprocess-resolution z-buffer. Visible dynamic points are aggregated
    onto the requested output grid only after visibility has been resolved.
    """
    validate_point_trajectory(payload)
    if max_objects <= 0 or max_objects > MAX_OBJECTS:
        raise ValueError("max_objects is outside the supported range")
    output_width, output_height = [int(value) for value in output_size_px]
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output size must be positive")
    preprocess_width, preprocess_height = [int(value) for value in preprocess_size_px]
    if min(preprocess_width, preprocess_height) <= 0:
        raise ValueError("preprocess size must be positive")
    if spatial_transform not in {"cover_center_crop", "source_image"}:
        raise ValueError(
            "spatial transform must be cover_center_crop or source_image"
        )

    tracks = np.asarray(payload["tracks_xy_px"], dtype=np.float64)
    depth = np.asarray(payload["depth_m"], dtype=np.float64)
    valid = np.asarray(payload["valid"], dtype=bool)
    frame_count, object_count, point_count, _ = tracks.shape
    if object_count > max_objects:
        raise ValueError("point trajectory has more objects than the conditioner budget")
    if frame_indices is None:
        frame_indices = list(range(frame_count))
    if not frame_indices or any(index < 0 or index >= frame_count for index in frame_indices):
        raise ValueError("frame_indices must select valid trajectory frames")

    image_width, image_height = [int(value) for value in payload["image_size_px"]]
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

    colors, anchor_valid = _das_point_identity_colors(payload)
    channels = np.zeros(
        (
            max_objects * DAS_TRACK_CHANNELS_PER_OBJECT,
            len(frame_indices),
            output_height,
            output_width,
        ),
        dtype=np.float32,
    )
    object_indices = np.broadcast_to(
        np.arange(object_count, dtype=np.int64)[:, None],
        (object_count, point_count),
    )
    point_indices = np.broadcast_to(
        np.arange(point_count, dtype=np.int64)[None, :],
        (object_count, point_count),
    )
    static_tracks = static_depth = static_valid = None
    if static_points_camera0_m is not None:
        static_tracks, static_depth, static_valid = _project_static_camera0_points(
            static_points_camera0_m, payload
        )
        if spatial_transform == "cover_center_crop":
            static_tracks = cover_center_crop_coordinates(
                static_tracks,
                (image_width, image_height),
                (preprocess_width, preprocess_height),
            )

    for output_index, frame_index in enumerate(frame_indices):
        current_xy = processed_tracks[frame_index]
        current_depth = depth[frame_index]
        current_valid = (
            valid[frame_index]
            & anchor_valid
            & np.isfinite(current_xy).all(axis=-1)
            & np.isfinite(current_depth)
            & (current_depth > 1.0e-6)
        )
        x = np.rint(current_xy[..., 0]).astype(np.int64)
        y = np.rint(current_xy[..., 1]).astype(np.int64)
        inside = (
            current_valid
            & (x >= 0)
            & (x < preprocess_width)
            & (y >= 0)
            & (y < preprocess_height)
        )
        if not np.any(inside) and static_tracks is None:
            continue

        candidate_x = x[inside]
        candidate_y = y[inside]
        candidate_depth = current_depth[inside]
        candidate_objects = object_indices[inside]
        candidate_points = point_indices[inside]
        candidate_x, candidate_y, candidate_depth, candidate_objects, candidate_points = (
            _expand_pixel_splats(
                candidate_x,
                candidate_y,
                candidate_depth,
                candidate_objects,
                candidate_points,
                point_radius_px,
            )
        )
        if static_tracks is not None:
            static_x = np.rint(static_tracks[frame_index, :, 0]).astype(np.int64)
            static_y = np.rint(static_tracks[frame_index, :, 1]).astype(np.int64)
            static_inside = (
                static_valid[frame_index]
                & (static_x >= 0)
                & (static_x < preprocess_width)
                & (static_y >= 0)
                & (static_y < preprocess_height)
            )
            static_x, static_y, static_z, static_objects, static_points = (
                _expand_pixel_splats(
                    static_x[static_inside],
                    static_y[static_inside],
                    static_depth[frame_index, static_inside],
                    np.full(int(static_inside.sum()), -1, dtype=np.int64),
                    np.full(int(static_inside.sum()), -1, dtype=np.int64),
                    point_radius_px,
                )
            )
            candidate_x = np.concatenate((candidate_x, static_x))
            candidate_y = np.concatenate((candidate_y, static_y))
            candidate_depth = np.concatenate((candidate_depth, static_z))
            candidate_objects = np.concatenate((candidate_objects, static_objects))
            candidate_points = np.concatenate((candidate_points, static_points))
        full_inside = (
            (candidate_x >= 0)
            & (candidate_x < preprocess_width)
            & (candidate_y >= 0)
            & (candidate_y < preprocess_height)
        )
        candidate_x = candidate_x[full_inside]
        candidate_y = candidate_y[full_inside]
        candidate_depth = candidate_depth[full_inside]
        candidate_objects = candidate_objects[full_inside]
        candidate_points = candidate_points[full_inside]
        if not len(candidate_x):
            continue
        flat_pixels = candidate_y * preprocess_width + candidate_x
        order = np.lexsort(
            (
                candidate_points,
                candidate_objects,
                candidate_depth,
                flat_pixels,
            )
        )
        sorted_pixels = flat_pixels[order]
        nearest = np.concatenate(
            (np.array([True]), sorted_pixels[1:] != sorted_pixels[:-1])
        )
        visible = order[nearest]
        visible_x = candidate_x[visible]
        visible_y = candidate_y[visible]
        visible_objects = candidate_objects[visible]
        visible_points = candidate_points[visible]
        dynamic_visible = visible_objects >= 0
        visible_x = visible_x[dynamic_visible]
        visible_y = visible_y[dynamic_visible]
        visible_objects = visible_objects[dynamic_visible]
        visible_points = visible_points[dynamic_visible]
        if not len(visible_x):
            continue
        cell_x = np.floor(
            (visible_x.astype(np.float64) + 0.5) * output_width / preprocess_width
        ).astype(np.int64)
        cell_y = np.floor(
            (visible_y.astype(np.float64) + 0.5) * output_height / preprocess_height
        ).astype(np.int64)
        cell_x = np.clip(cell_x, 0, output_width - 1)
        cell_y = np.clip(cell_y, 0, output_height - 1)

        for object_index in np.unique(visible_objects):
            selected = visible_objects == object_index
            base = int(object_index) * DAS_TRACK_CHANNELS_PER_OBJECT
            selected_y = cell_y[selected]
            selected_x = cell_x[selected]
            selected_colors = colors[
                object_index, visible_points[selected]
            ]
            flat_cells = selected_y * output_width + selected_x
            counts = np.bincount(
                flat_cells, minlength=output_height * output_width
            ).astype(np.float32)
            occupied = counts > 0
            for color_channel in range(3):
                sums = np.bincount(
                    flat_cells,
                    weights=selected_colors[:, color_channel],
                    minlength=output_height * output_width,
                ).astype(np.float32)
                target = channels[base + color_channel, output_index].reshape(-1)
                target[occupied] = sums[occupied] / counts[occupied]
            channels[base + 3, output_index].reshape(-1)[occupied] = 1.0
    return channels
