#!/usr/bin/env python3
"""Round-trip a real PhysSweep sample through the DaS-style Wan condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_file

from point_trajectory import (
    DAS_TRACK_CHANNELS_PER_OBJECT,
    MAX_OBJECTS,
    _das_point_identity_colors,
    rasterize_das_3d_tracks,
    unproject_physweep_tracks,
    validate_point_trajectory,
)
from wan_training import TrajectoryPatchConditioner, validate_point_track_object_slots


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_cached_das_map(path: Path) -> np.ndarray:
    if path.suffix != ".safetensors":
        raise ValueError("point-track map must be a .safetensors cache artifact")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"point_track_map"}:
        raise ValueError("point-track cache must contain only point_track_map")
    tensor = tensors["point_track_map"]
    if tensor.dtype != torch.float32:
        raise ValueError("point_track_map must use float32")
    return tensor.numpy()


def scene_dynamic_surface_source(scene: dict[str, np.ndarray]) -> str:
    if "metadata_json" not in scene:
        return "unspecified"
    try:
        metadata = json.loads(str(np.asarray(scene["metadata_json"]).reshape(()).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("scene metadata_json is invalid") from exc
    if not isinstance(metadata, dict):
        raise ValueError("scene metadata_json must contain an object")
    return str(metadata.get("dynamic_surface_source", "unspecified"))


def load_nontrivial_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode == "L":
            alpha = image.copy()
        elif "A" in image.getbands():
            alpha = image.getchannel("A").copy()
        else:
            raise ValueError(f"instance mask is neither grayscale nor alpha: {path}")
    foreground = np.asarray(alpha) > 0
    if not foreground.any() or foreground.all():
        raise ValueError(f"instance mask must be nonempty and non-full: {path}")
    return alpha



def quaternion_wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def identity_colors_reference(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    initial = np.asarray(payload["initial_points_camera_m"], dtype=np.float64)
    anchor = np.isfinite(initial).all(axis=-1) & (initial[..., 2] > 1.0e-6)
    coordinates = np.stack(
        (initial[..., 0], initial[..., 1], 1.0 / initial[..., 2]), axis=-1
    )
    values = coordinates[anchor]
    low = np.asarray(
        [values[:, 0].min(), values[:, 1].min(), np.percentile(values[:, 2], 2)]
    )
    high = np.asarray(
        [values[:, 0].max(), values[:, 1].max(), np.percentile(values[:, 2], 98)]
    )
    span = high - low
    stable = span > 1.0e-6
    colors = np.zeros_like(coordinates)
    colors[..., stable] = np.clip(
        (coordinates[..., stable] - low[stable]) / span[stable], 0.0, 1.0
    )
    colors[..., ~stable] = 0.5
    colors[~anchor] = 0.0
    recovered = low + colors * span
    interior = anchor[..., None] & (coordinates >= low) & (coordinates <= high)
    inverse_error = float(np.max(np.abs(recovered - coordinates)[interior]))
    return colors.astype(np.float32), anchor, inverse_error


def broadcast_matrix(value: np.ndarray, frames: int, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == shape:
        return np.broadcast_to(matrix, (frames, *shape))
    if matrix.shape != (frames, *shape):
        raise ValueError(f"matrix must have shape {shape} or {(frames, *shape)}")
    return matrix


def cover_reference(
    coordinates: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    result = np.asarray(coordinates, dtype=np.float64).copy()
    result[..., 0] = (
        (result[..., 0] + 0.5) * resized_width / source_width
        - 0.5
        - (resized_width - target_width) // 2
    )
    result[..., 1] = (
        (result[..., 1] + 0.5) * resized_height / source_height
        - 0.5
        - (resized_height - target_height) // 2
    )
    return result


def static_projection_reference(
    static_camera0: np.ndarray,
    payload: dict[str, np.ndarray],
    preprocess_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = int(payload["points_world_m"].shape[0])
    camera = broadcast_matrix(payload["camera_from_world"], frames, (4, 4))
    intrinsics = broadcast_matrix(payload["camera_intrinsics"], frames, (3, 3))
    homogeneous = np.column_stack(
        [static_camera0.astype(np.float64), np.ones(len(static_camera0))]
    )
    points_world = (np.linalg.inv(camera[0]) @ homogeneous.T).T
    camera_points = np.einsum("tij,nj->tni", camera, points_world)[..., :3]
    projection_points = camera_points.copy()
    projection_points[..., 1] *= -1.0
    projected = np.einsum("tij,tnj->tni", intrinsics, projection_points)
    tracks = projected[..., :2] / projected[..., 2:3]
    tracks = cover_reference(
        tracks,
        tuple(int(value) for value in payload["image_size_px"]),
        preprocess_size,
    )
    depth = camera_points[..., 2]
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(()).item()))
    clip_start = float(metadata.get("clip_start_m", 0.03))
    clip_end = float(metadata.get("clip_end_m", 100.0))
    if not 0.0 <= clip_start < clip_end:
        raise ValueError("metadata_json camera clip range is invalid")
    valid = (
        np.isfinite(tracks).all(axis=-1)
        & np.isfinite(depth)
        & (depth > clip_start)
        & (depth < clip_end)
    )
    return tracks, depth, valid


def expand_splats(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    objects: np.ndarray,
    points: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="xy")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)
    count = offset_x.shape[1]
    return (
        (x[:, None] + offset_x).reshape(-1),
        (y[:, None] + offset_y).reshape(-1),
        np.repeat(depth, count),
        np.repeat(objects, count),
        np.repeat(points, count),
    )


def reference_frame_map(
    payload: dict[str, np.ndarray],
    static_camera0: np.ndarray,
    frame: int,
    colors: np.ndarray,
    anchor: np.ndarray,
    *,
    preprocess_size: tuple[int, int] = (832, 480),
    output_size: tuple[int, int] = (52, 30),
    radius: int = 1,
) -> np.ndarray:
    preprocess_width, preprocess_height = preprocess_size
    output_width, output_height = output_size
    tracks = cover_reference(
        payload["tracks_xy_px"],
        tuple(int(value) for value in payload["image_size_px"]),
        preprocess_size,
    )
    depth = np.asarray(payload["depth_m"], dtype=np.float64)
    valid = np.asarray(payload["valid"], dtype=bool)
    objects, points = tracks.shape[1:3]
    object_indices = np.broadcast_to(np.arange(objects)[:, None], (objects, points))
    point_indices = np.broadcast_to(np.arange(points)[None, :], (objects, points))
    xy = tracks[frame]
    x = np.rint(xy[..., 0]).astype(np.int64)
    y = np.rint(xy[..., 1]).astype(np.int64)
    inside = (
        valid[frame]
        & anchor
        & (depth[frame] > 1e-6)
        & (x >= 0)
        & (x < preprocess_width)
        & (y >= 0)
        & (y < preprocess_height)
    )
    candidates = expand_splats(
        x[inside],
        y[inside],
        depth[frame][inside],
        object_indices[inside],
        point_indices[inside],
        radius,
    )
    static_tracks, static_depth, static_valid = static_projection_reference(
        static_camera0, payload, preprocess_size
    )
    static_x = np.rint(static_tracks[frame, :, 0]).astype(np.int64)
    static_y = np.rint(static_tracks[frame, :, 1]).astype(np.int64)
    static_inside = (
        static_valid[frame]
        & (static_x >= 0)
        & (static_x < preprocess_width)
        & (static_y >= 0)
        & (static_y < preprocess_height)
    )
    static = expand_splats(
        static_x[static_inside],
        static_y[static_inside],
        static_depth[frame, static_inside],
        np.full(int(static_inside.sum()), -1, dtype=np.int64),
        np.full(int(static_inside.sum()), -1, dtype=np.int64),
        radius,
    )
    candidate_x, candidate_y, candidate_depth, candidate_objects, candidate_points = [
        np.concatenate([dynamic, environment])
        for dynamic, environment in zip(candidates, static)
    ]
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
    pixels = candidate_y * preprocess_width + candidate_x
    order = np.lexsort(
        (candidate_points, candidate_objects, candidate_depth, pixels)
    )
    sorted_pixels = pixels[order]
    nearest = np.concatenate(([True], sorted_pixels[1:] != sorted_pixels[:-1]))
    visible = order[nearest]
    dynamic = candidate_objects[visible] >= 0
    visible_x = candidate_x[visible][dynamic]
    visible_y = candidate_y[visible][dynamic]
    visible_objects = candidate_objects[visible][dynamic]
    visible_points = candidate_points[visible][dynamic]
    cell_x = np.floor((visible_x + 0.5) * output_width / preprocess_width).astype(int)
    cell_y = np.floor((visible_y + 0.5) * output_height / preprocess_height).astype(int)
    result = np.zeros(
        (MAX_OBJECTS * DAS_TRACK_CHANNELS_PER_OBJECT, output_height, output_width),
        dtype=np.float32,
    )
    for object_index in np.unique(visible_objects):
        selected = visible_objects == object_index
        flat = cell_y[selected] * output_width + cell_x[selected]
        counts = np.bincount(flat, minlength=output_height * output_width).astype(np.float32)
        occupied = counts > 0
        base = int(object_index) * DAS_TRACK_CHANNELS_PER_OBJECT
        selected_colors = colors[object_index, visible_points[selected]]
        for channel in range(3):
            sums = np.bincount(
                flat,
                weights=selected_colors[:, channel],
                minlength=output_height * output_width,
            ).astype(np.float32)
            result[base + channel].reshape(-1)[occupied] = sums[occupied] / counts[occupied]
        result[base + 3].reshape(-1)[occupied] = 1.0
    return result


def dilate_without_wrap(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            result[target_y, target_x] |= mask[source_y, source_x]
    return result


def raw_object_pose(
    raw: dict[str, np.ndarray], object_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read one pose sequence from either released or legacy trajectory layout."""
    has_position_axis = "position_m" in raw
    has_quaternion_axis = "quaternion_wxyz" in raw
    if has_position_axis != has_quaternion_axis:
        raise ValueError("raw trajectory has only one explicit-axis pose field")
    if has_position_axis:
        if "object_ids" not in raw:
            raise ValueError("explicit-axis raw trajectory has no object_ids")
        object_ids = [str(value) for value in np.asarray(raw["object_ids"]).reshape(-1)]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("explicit-axis raw trajectory has duplicate object_ids")
        if object_id not in object_ids:
            raise ValueError(f"raw trajectory does not contain object {object_id}")
        position = np.asarray(raw["position_m"], dtype=np.float64)
        quaternion = np.asarray(raw["quaternion_wxyz"], dtype=np.float64)
        if (
            position.ndim != 3
            or position.shape[1:] != (len(object_ids), 3)
            or quaternion.shape != (len(position), len(object_ids), 4)
        ):
            raise ValueError("explicit-axis raw trajectory pose arrays have invalid shapes")
        object_index = object_ids.index(object_id)
        return position[:, object_index], quaternion[:, object_index]

    position_key = f"{object_id}__position_m"
    quaternion_key = f"{object_id}__quaternion_wxyz"
    if position_key not in raw or quaternion_key not in raw:
        raise ValueError(f"legacy raw trajectory does not contain object {object_id}")
    position = np.asarray(raw[position_key], dtype=np.float64)
    quaternion = np.asarray(raw[quaternion_key], dtype=np.float64)
    if position.ndim != 2 or position.shape[1:] != (3,) or quaternion.shape != (
        len(position),
        4,
    ):
        raise ValueError("legacy raw trajectory pose arrays have invalid shapes")
    return position, quaternion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-trajectory", type=Path, required=True)
    parser.add_argument("--scene-condition", type=Path, required=True)
    parser.add_argument("--raw-trajectory", type=Path, required=True)
    parser.add_argument("--point-track-map", type=Path, required=True)
    parser.add_argument(
        "--mask-root",
        type=Path,
        required=True,
        help="Directory containing frame_####.png for the audited object.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frames", default="0,24,48,72,96")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_npz(args.point_trajectory.resolve())
    scene = load_npz(args.scene_condition.resolve())
    raw = load_npz(args.raw_trajectory.resolve())
    das_map = load_cached_das_map(args.point_track_map.resolve())
    validate_point_trajectory(payload)
    frame_count, object_count, point_count = payload["valid"].shape
    if das_map.shape != (12, frame_count, 30, 52):
        raise ValueError("DaS map shape does not match the trajectory")
    validate_point_track_object_slots(torch.from_numpy(das_map), "das_3d_tracks", object_count)

    camera_recovered = unproject_physweep_tracks(
        payload["tracks_xy_px"], payload["depth_m"], payload["camera_intrinsics"]
    ).astype(np.float64)
    camera_error = np.linalg.norm(
        camera_recovered - payload["points_camera_m"], axis=-1
    )
    camera_sequence = broadcast_matrix(payload["camera_from_world"], frame_count, (4, 4))
    world_from_camera = np.linalg.inv(camera_sequence)
    homogeneous_camera = np.concatenate(
        [camera_recovered, np.ones(camera_recovered.shape[:-1] + (1,))], axis=-1
    )
    world_recovered = np.einsum(
        "tij,tonj->toni", world_from_camera, homogeneous_camera, optimize=True
    )[..., :3]
    world_error = np.linalg.norm(world_recovered - payload["points_world_m"], axis=-1)

    object_ids = [str(value) for value in payload["object_ids"]]
    rigidity_errors = []
    quaternion_norm_errors = []
    for object_index, object_id in enumerate(object_ids):
        position, quaternion = raw_object_pose(raw, object_id)
        if len(position) != frame_count:
            raise ValueError("raw and point trajectory frame counts differ")
        rotations = np.stack([quaternion_wxyz_to_matrix(value) for value in quaternion])
        local = np.einsum(
            "tni,tij->tnj",
            payload["points_world_m"][:, object_index] - position[:, None, :],
            rotations,
            optimize=True,
        )
        rigidity_errors.append(float(np.linalg.norm(local - local[:1], axis=-1).max()))
        quaternion_norm_errors.append(
            float(np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0).max())
        )

    object_scene = np.asarray(scene["object_xyz_camera_m"], dtype=np.float32)
    if object_scene.ndim == 2:
        object_scene = object_scene[None]
    initial_error = np.linalg.norm(
        object_scene - payload["initial_points_camera_m"], axis=-1
    )
    normal_errors = {
        key: float(np.abs(np.linalg.norm(scene[key], axis=-1) - 1.0).max())
        for key in ("object_normal_camera", "environment_normal_camera")
    }
    colors, anchor, _ = identity_colors_reference(payload)
    production_colors, production_anchor = _das_point_identity_colors(payload)
    initial = np.asarray(payload["initial_points_camera_m"], dtype=np.float64)
    identity_coordinates = np.stack(
        (initial[..., 0], initial[..., 1], 1.0 / initial[..., 2]), axis=-1
    )
    identity_values = identity_coordinates[anchor]
    identity_low = np.asarray(
        [
            identity_values[:, 0].min(),
            identity_values[:, 1].min(),
            np.percentile(identity_values[:, 2], 2),
        ]
    )
    identity_high = np.asarray(
        [
            identity_values[:, 0].max(),
            identity_values[:, 1].max(),
            np.percentile(identity_values[:, 2], 98),
        ]
    )
    production_identity_recovered = identity_low + production_colors * (
        identity_high - identity_low
    )
    identity_interior = (
        anchor[..., None]
        & (identity_coordinates >= identity_low)
        & (identity_coordinates <= identity_high)
    )
    identity_inverse_error = float(
        np.abs(production_identity_recovered - identity_coordinates)[
            identity_interior
        ].max()
    )
    environment = np.asarray(scene["environment_xyz_camera_m"], dtype=np.float32)
    reference_frames = [int(value) for value in args.reference_frames.split(",")]
    reference_errors = []
    reference_occupancy_mismatches = []
    for frame in reference_frames:
        reference = reference_frame_map(payload, environment, frame, colors, anchor)
        reference_errors.append(float(np.abs(reference - das_map[:, frame]).max()))
        reference_occupancy_mismatches.append(
            int(
                np.count_nonzero(
                    reference[3::DAS_TRACK_CHANNELS_PER_OBJECT]
                    != das_map[3::DAS_TRACK_CHANNELS_PER_OBJECT, frame]
                )
            )
        )

    containment = []
    latent_precision = []
    centroid_errors = []
    mask_occupancies = []
    image_width, image_height = [int(value) for value in payload["image_size_px"]]
    for frame in range(frame_count):
        alpha = load_nontrivial_mask(
            args.mask_root.resolve() / f"frame_{frame + 1:04d}.png"
        )
        mask = np.asarray(alpha) > 0
        mask_occupancies.append(float(mask.mean()))
        dilated = np.asarray(alpha.filter(ImageFilter.MaxFilter(7))) > 0
        selected = np.asarray(payload["valid"][frame, 0], dtype=bool)
        xy = np.rint(payload["tracks_xy_px"][frame, 0, selected]).astype(int)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < image_width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < image_height)
        )
        xy = xy[inside]
        containment.append(float(dilated[xy[:, 1], xy[:, 0]].mean()))
        y, x = np.nonzero(mask)
        mask_center = np.asarray([x.mean(), y.mean()])
        point_center = np.median(payload["tracks_xy_px"][frame, 0, selected], axis=0)
        centroid_errors.append(float(np.linalg.norm(mask_center - point_center)))
        coordinates = cover_reference(
            np.stack([x, y], axis=-1),
            (image_width, image_height),
            (832, 480),
        )
        cell_x = np.floor((coordinates[:, 0] + 0.5) * 52 / 832).astype(int)
        cell_y = np.floor((coordinates[:, 1] + 0.5) * 30 / 480).astype(int)
        cell_inside = (cell_x >= 0) & (cell_x < 52) & (cell_y >= 0) & (cell_y < 30)
        latent_mask = np.zeros((30, 52), dtype=bool)
        latent_mask[cell_y[cell_inside], cell_x[cell_inside]] = True
        latent_mask = dilate_without_wrap(latent_mask, 1)
        occupied = das_map[3, frame] > 0
        latent_precision.append(float(latent_mask[occupied].mean()))

    conditioner = TrajectoryPatchConditioner(
        hidden_dim=8,
        patch_size=(1, 2, 2),
        rank=4,
        representation="das_3d_tracks",
    )
    tensor = torch.from_numpy(das_map).unsqueeze(0)
    temporal = torch.cat(
        [tensor[:, :, :1], conditioner.temporal_projection(tensor[:, :, 1:])],
        dim=2,
    )
    manual_temporal = torch.cat(
        [
            tensor[:, :, :1],
            tensor[:, :, 1:].reshape(1, 12, 24, 4, 30, 52).mean(dim=3),
        ],
        dim=2,
    )
    patch = conditioner.patch_projection(temporal)
    residual = conditioner.output_projection(torch.nn.functional.silu(patch))

    initial_points = np.asarray(
        payload["initial_points_camera_m"], dtype=np.float32
    ).reshape(-1, 3)
    raster_options = {
        "output_size_px": (52, 30),
        "preprocess_size_px": (832, 480),
        "frame_indices": [0],
    }
    counterfactual_baseline = rasterize_das_3d_tracks(payload, **raster_options)
    counterfactual_near = rasterize_das_3d_tracks(
        payload,
        static_points_camera0_m=initial_points * 0.5,
        **raster_options,
    )
    counterfactual_far = rasterize_das_3d_tracks(
        payload,
        static_points_camera0_m=initial_points * 2.0,
        **raster_options,
    )

    metrics = {
        "frame_count": frame_count,
        "object_count": object_count,
        "point_count": point_count,
        "raw_time_match_max_s": float(
            np.abs(np.asarray(raw["time_s"]) - payload["time_s"]).max()
        ),
        "quaternion_norm_max_error": max(quaternion_norm_errors),
        "rigid_local_coordinate_max_error_m": max(rigidity_errors),
        "pixel_depth_unprojection_max_error_m": float(camera_error.max()),
        "camera_inverse_world_max_error_m": float(world_error.max()),
        "scene_to_trajectory_initial_max_error_m": float(initial_error.max()),
        "normal_unit_max_error": max(normal_errors.values()),
        "environment_friction_min": float(scene["environment_friction"].min()),
        "environment_restitution_range": [
            float(scene["environment_restitution"].min()),
            float(scene["environment_restitution"].max()),
        ],
        "identity_coordinate_inverse_max_error": identity_inverse_error,
        "identity_anchor_mismatch_count": int(
            np.count_nonzero(production_anchor != anchor)
        ),
        "reference_raster_max_error": max(reference_errors),
        "reference_occupancy_mismatch_count": sum(
            reference_occupancy_mismatches
        ),
        "point_inside_dilated_mask_min": min(containment),
        "latent_occupancy_inside_dilated_mask_min": min(latent_precision),
        "mask_occupancy_fraction_range": [
            min(mask_occupancies),
            max(mask_occupancies),
        ],
        "mask_track_centroid_error_max_px": max(centroid_errors),
        "temporal_manual_average_max_error": float(
            (temporal - manual_temporal).abs().max().detach()
        ),
        "temporal_shape": list(temporal.shape),
        "patch_shape": list(patch.shape),
        "zero_initialized_residual_max": float(residual.abs().max().detach()),
        "counterfactual_baseline_visible_cells": int(
            counterfactual_baseline[3::DAS_TRACK_CHANNELS_PER_OBJECT].sum()
        ),
        "counterfactual_near_static_visible_cells": int(
            counterfactual_near[3::DAS_TRACK_CHANNELS_PER_OBJECT].sum()
        ),
        "counterfactual_far_static_max_error": float(
            np.abs(counterfactual_far - counterfactual_baseline).max()
        ),
    }
    dynamic_surface_source = scene_dynamic_surface_source(scene)
    source_mask_alignment = (
        metrics["point_inside_dilated_mask_min"] >= 0.98
        and metrics["mask_track_centroid_error_max_px"] <= 16.0
    )
    collision_proxy_surface = "collision_proxy" in dynamic_surface_source
    checks = {
        "raw_time": metrics["raw_time_match_max_s"] < 1.0e-7,
        "quaternion_norm": metrics["quaternion_norm_max_error"] < 1.0e-10,
        "rigid_identity": metrics["rigid_local_coordinate_max_error_m"] < 5.0e-6,
        "pixel_depth_inverse": metrics["pixel_depth_unprojection_max_error_m"] < 1.0e-5,
        "camera_inverse": metrics["camera_inverse_world_max_error_m"] < 1.0e-5,
        "scene_initial_identity": metrics["scene_to_trajectory_initial_max_error_m"] < 1.0e-6,
        "unit_normals": metrics["normal_unit_max_error"] < 1.0e-4,
        "environment_material_range": metrics["environment_friction_min"] >= 0.0
        and 0.0 <= metrics["environment_restitution_range"][0]
        <= metrics["environment_restitution_range"][1]
        <= 1.0,
        "identity_color_inverse": metrics["identity_coordinate_inverse_max_error"] < 1.0e-5
        and metrics["identity_anchor_mismatch_count"] == 0,
        "independent_reference_raster": metrics["reference_raster_max_error"] < 2.0e-6
        and metrics["reference_occupancy_mismatch_count"] == 0,
        "latent_mask_alignment": metrics["latent_occupancy_inside_dilated_mask_min"] >= 0.95,
        "temporal_window_semantics": metrics["temporal_manual_average_max_error"] < 1.0e-7,
        "zero_initialized_adapter": metrics["zero_initialized_residual_max"] == 0.0,
        "counterfactual_global_zbuffer": metrics[
            "counterfactual_baseline_visible_cells"
        ]
        > 0
        and metrics["counterfactual_near_static_visible_cells"] == 0
        and metrics["counterfactual_far_static_max_error"] == 0.0,
    }
    diagnostics = {}
    advisories = []
    if collision_proxy_surface:
        diagnostics["source_mask_alignment"] = {
            "passed_visual_surface_threshold": source_mask_alignment,
            "enforcement": "diagnostic_only_for_collision_proxy",
        }
        if not source_mask_alignment:
            advisories.append("collision_proxy_differs_from_unpublished_visual_surface")
    else:
        checks["source_mask_alignment"] = source_mask_alignment
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": "phycontext.das_roundtrip_audit.v1",
        "status": "pass" if not failures else "fail",
        "inputs": {
            "point_trajectory": str(args.point_trajectory.resolve()),
            "scene_condition": str(args.scene_condition.resolve()),
            "raw_trajectory": str(args.raw_trajectory.resolve()),
            "point_track_map": str(args.point_track_map.resolve()),
            "mask_root": str(args.mask_root.resolve()),
        },
        "reference_frames_zero_based": reference_frames,
        "dynamic_surface_source": dynamic_surface_source,
        "metrics": metrics,
        "checks": checks,
        "diagnostics": diagnostics,
        "advisories": advisories,
        "failures": failures,
        "lossy_boundary": (
            "The 97-to-25 temporal average and latent-cell RGB aggregation are intentionally "
            "many-to-one; they can be checked for conservation/window semantics but cannot "
            "reconstruct every source frame or point from the compressed condition alone."
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"round-trip audit failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
