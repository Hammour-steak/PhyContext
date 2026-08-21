from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d


@dataclass
class EnvironmentCompletion:
    points: np.ndarray
    colors: np.ndarray
    pixel_mask: np.ndarray
    dense_mask: np.ndarray
    depth: np.ndarray
    inpainted_rgb: np.ndarray
    planes: list[dict]
    diagnostics: dict


def _camera_rays(
    height: int,
    width: int,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels_x, pixels_y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([pixels_x, pixels_y, np.ones_like(pixels_x)], axis=-1).astype(
        np.float64
    )
    directions_camera = pixels @ np.linalg.inv(intrinsic).T

    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :4] = extrinsic
    world_from_camera = np.linalg.inv(world_from_camera)
    origin = world_from_camera[:3, 3]
    directions_world = directions_camera @ world_from_camera[:3, :3].T
    return origin, directions_world


def _fit_local_planes(
    point_map: np.ndarray,
    valid_environment: np.ndarray,
    object_mask: np.ndarray,
    max_planes: int,
    seed: int,
) -> list[dict]:
    """Fit nearby support candidates without using them as scene completion."""
    o3d.utility.random.seed(int(seed))
    height, width = object_mask.shape
    local_radius = max(18, int(round(min(height, width) * 0.075)))
    outside_distance = cv2.distanceTransform(
        (~object_mask).astype(np.uint8), cv2.DIST_L2, 5
    )
    local = valid_environment & (outside_distance > 2.0) & (
        outside_distance <= local_radius
    )
    local_points = point_map[local]
    if len(local_points) < 160:
        return []

    extent = np.quantile(np.ptp(local_points, axis=0), 0.8)
    distance_threshold = max(float(extent) * 0.006, 1e-5)
    remaining_points = local_points
    original_count = len(remaining_points)
    planes: list[dict] = []

    for _ in range(max_planes * 2):
        if len(planes) >= max_planes or len(remaining_points) < 100:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining_points))
        equation, inlier_indices = cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1600,
        )
        minimum_inliers = max(80, int(original_count * 0.07))
        if len(inlier_indices) < minimum_inliers:
            break
        inlier_indices = np.asarray(inlier_indices, dtype=np.int64)
        normal = np.asarray(equation[:3], dtype=np.float64)
        equation = np.asarray(equation, dtype=np.float64) / max(
            float(np.linalg.norm(normal)), 1e-12
        )
        planes.append(
            {
                "index": len(planes),
                "equation": equation.tolist(),
                "local_inlier_count": int(len(inlier_indices)),
                "distance_threshold": distance_threshold,
            }
        )
        keep = np.ones(len(remaining_points), dtype=bool)
        keep[inlier_indices] = False
        remaining_points = remaining_points[keep]
    return planes


def _depth_seam_diagnostics(
    completed_depth: np.ndarray,
    reference_depth: np.ndarray,
    completed_mask: np.ndarray,
    observed_environment: np.ndarray,
    neighborhood_radius_pixels: int = 8,
) -> dict:
    """Measure boundary continuity without crossing an unrelated depth edge."""
    if neighborhood_radius_pixels < 1:
        raise ValueError("Depth-seam neighborhood radius must be positive")
    height, width = completed_mask.shape
    radius = int(neighborhood_radius_pixels)
    padded_depth = np.pad(
        reference_depth,
        radius,
        mode="constant",
        constant_values=np.nan,
    )
    padded_observed = np.pad(
        observed_environment,
        radius,
        mode="constant",
        constant_values=False,
    )
    best_relative_error = np.full((height, width), np.inf, dtype=np.float32)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x == 0 and offset_y == 0:
                continue
            shifted_depth = padded_depth[
                radius + offset_y : radius + offset_y + height,
                radius + offset_x : radius + offset_x + width,
            ]
            shifted_observed = padded_observed[
                radius + offset_y : radius + offset_y + height,
                radius + offset_x : radius + offset_x + width,
            ]
            valid = completed_mask & shifted_observed
            relative_error = np.abs(completed_depth - shifted_depth) / np.maximum(
                shifted_depth,
                1e-8,
            )
            best_relative_error[valid] = np.minimum(
                best_relative_error[valid],
                relative_error[valid],
            )
    seam_mask = completed_mask & np.isfinite(best_relative_error)
    if int(seam_mask.sum()) < 32:
        raise RuntimeError("Too few completed pixels are available for depth-seam validation")
    relative_error = best_relative_error[seam_mask]
    return {
        "method": "minimum_relative_error_over_adjacent_observed_surfaces",
        "neighborhood_radius_pixels": radius,
        "pixel_count": int(seam_mask.sum()),
        "median_relative_error": float(np.median(relative_error)),
        "p90_relative_error": float(np.quantile(relative_error, 0.90)),
    }


def complete_environment(
    point_map: np.ndarray,
    inpainted_rgb: np.ndarray,
    reference_depth: np.ndarray,
    observed_environment: np.ndarray,
    reliable_environment: np.ndarray,
    object_mask: np.ndarray,
    content_mask: np.ndarray,
    completion_depth: np.ndarray,
    completion_mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    max_planes: int = 3,
    seed: int = 0,
) -> EnvironmentCompletion:
    """Add DepthLab geometry only where the original environment has a hole."""
    shape = object_mask.shape
    arrays = {
        "point map": point_map.shape[:2],
        "inpainted RGB": inpainted_rgb.shape[:2],
        "reference depth": reference_depth.shape,
        "observed environment": observed_environment.shape,
        "reliable environment": reliable_environment.shape,
        "content mask": content_mask.shape,
        "completion depth": completion_depth.shape,
        "completion mask": completion_mask.shape,
    }
    for name, current_shape in arrays.items():
        if current_shape != shape:
            raise ValueError(f"{name} dimensions do not match the object mask")

    object_mask = np.asarray(object_mask, dtype=bool)
    content_mask = np.asarray(content_mask, dtype=bool)
    observed_environment = np.asarray(observed_environment, dtype=bool) & content_mask
    reliable_environment = (
        np.asarray(reliable_environment, dtype=bool) & observed_environment
    )
    completion_mask = np.asarray(completion_mask, dtype=bool) & content_mask
    finite_completion = (
        completion_mask
        & np.isfinite(completion_depth)
        & (completion_depth > 0)
    )
    completed_mask = content_mask & ~observed_environment & finite_completion
    if int(completed_mask.sum()) < 32:
        raise RuntimeError("DepthLab supplied too few valid hidden-environment pixels")

    dense_mask = observed_environment | completed_mask
    completed_depth = np.zeros_like(reference_depth, dtype=np.float32)
    completed_depth[completed_mask] = completion_depth[completed_mask]
    seam_diagnostics = _depth_seam_diagnostics(
        completed_depth,
        reference_depth,
        completed_mask,
        observed_environment,
    )

    height, width = shape
    origin, directions = _camera_rays(height, width, intrinsic, extrinsic)
    completed_points = origin[None, None, :] + directions * completion_depth[..., None]
    points = completed_points[completed_mask].astype(np.float32)
    colors = inpainted_rgb[completed_mask].astype(np.uint8)
    planes = _fit_local_planes(
        point_map,
        reliable_environment,
        object_mask,
        max_planes,
        seed,
    )

    missing_environment = content_mask & ~observed_environment
    diagnostics = {
        "method": "lama_rgb_inpainting_then_depthlab_known_depth_completion",
        "object_mask_completion_fraction": float(
            (completed_mask & object_mask).sum() / max(int(object_mask.sum()), 1)
        ),
        "missing_environment_completion_fraction": float(
            completed_mask.sum() / max(int(missing_environment.sum()), 1)
        ),
        "content_surface_coverage": float(
            dense_mask.sum() / max(int(content_mask.sum()), 1)
        ),
        "observed_pixel_count": int(observed_environment.sum()),
        "reliable_observed_pixel_count": int(reliable_environment.sum()),
        "completed_pixel_count": int(completed_mask.sum()),
        "content_pixel_count": int(content_mask.sum()),
        "observed_geometry_policy": "immutable_original_vggt_points",
        "depth_seam": seam_diagnostics,
    }
    return EnvironmentCompletion(
        points=points,
        colors=colors,
        pixel_mask=completed_mask,
        dense_mask=dense_mask,
        depth=completed_depth,
        inpainted_rgb=inpainted_rgb,
        planes=planes,
        diagnostics=diagnostics,
    )
