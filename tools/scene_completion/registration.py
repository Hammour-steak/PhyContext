from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


@dataclass
class RegistrationResult:
    transform_world: np.ndarray
    transform_camera: np.ndarray
    aligned_points: np.ndarray
    aligned_colors: np.ndarray
    rendered_mask: np.ndarray
    rendered_depth: np.ndarray
    collision_vertices: np.ndarray
    collision_faces: np.ndarray
    candidate_montage: np.ndarray
    diagnostics: dict


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force=None, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"Mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise RuntimeError(f"Could not load a triangular mesh from {path}")
    return loaded


def _target_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    size: int,
    padding_fraction: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    pixels_y, pixels_x = np.nonzero(mask)
    object_width = int(pixels_x.max() - pixels_x.min() + 1)
    object_height = int(pixels_y.max() - pixels_y.min() + 1)
    padding = max(6, int(round(max(object_width, object_height) * padding_fraction)))
    height, width = mask.shape
    x0 = max(int(pixels_x.min()) - padding, 0)
    y0 = max(int(pixels_y.min()) - padding, 0)
    x1 = min(int(pixels_x.max()) + padding + 1, width)
    y1 = min(int(pixels_y.max()) + padding + 1, height)
    crop = cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(
        mask[y0:y1, x0:x1].astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    crop = crop.copy()
    crop[~crop_mask] = 0
    return crop, crop_mask, (x0, y0, x1, y1)


def _render_candidates(mesh_path: Path, device: torch.device, size: int):
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        MeshRasterizer,
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        look_at_view_transform,
    )

    mesh = load_objs_as_meshes([str(mesh_path)], device=device, load_textures=True)
    vertices = mesh.verts_packed()
    low = vertices.amin(dim=0)
    high = vertices.amax(dim=0)
    center = ((low + high) * 0.5).detach().cpu().numpy()
    radius = float(torch.linalg.norm(high - low).item())
    distance = max(radius * 1.45, 1e-3)
    azimuths = np.arange(0.0, 360.0, 30.0, dtype=np.float32)
    elevations = np.asarray([-25.0, 0.0, 25.0], dtype=np.float32)
    elev_grid, azim_grid = np.meshgrid(elevations, azimuths, indexing="ij")
    elevations_flat = elev_grid.reshape(-1).tolist()
    azimuths_flat = azim_grid.reshape(-1).tolist()
    count = len(azimuths_flat)
    at = torch.tensor(center, dtype=torch.float32, device=device).repeat(count, 1)
    R, T = look_at_view_transform(
        dist=[distance] * count,
        elev=elevations_flat,
        azim=azimuths_flat,
        at=at,
        device=device,
    )
    focal = 0.5 * size / np.tan(np.deg2rad(30.0))
    cameras = PerspectiveCameras(
        device=device,
        focal_length=torch.full((count, 2), float(focal), device=device),
        principal_point=torch.full((count, 2), float(size) * 0.5, device=device),
        image_size=torch.tensor([[size, size]], device=device).repeat(count, 1),
        R=R,
        T=T,
        in_ndc=False,
    )
    settings = RasterizationSettings(
        image_size=size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=max(int(mesh.num_faces_per_mesh()[0].item() * 1.25), 10000),
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=settings)
    batched_mesh = mesh.extend(count)
    fragments = rasterizer(batched_mesh)
    lights = PointLights(device=device, location=cameras.get_camera_center())
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    with torch.inference_mode():
        images = renderer(batched_mesh)[..., :3]
    return {
        "mesh": mesh,
        "images": np.clip(images.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8),
        "masks": (fragments.pix_to_face[..., 0] >= 0).detach().cpu().numpy(),
        "pix_to_face": fragments.pix_to_face[..., 0].detach().cpu().numpy(),
        "barycentric": fragments.bary_coords[..., 0, :].detach().cpu().numpy(),
        "azimuths": azimuths_flat,
        "elevations": elevations_flat,
    }


def _sift_matches(
    candidate: np.ndarray,
    candidate_mask: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
    sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=0.015, edgeThreshold=12)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    candidate_keypoints, candidate_descriptors = sift.detectAndCompute(
        candidate_gray, candidate_mask.astype(np.uint8) * 255
    )
    target_keypoints, target_descriptors = sift.detectAndCompute(
        target_gray, target_mask.astype(np.uint8) * 255
    )
    if candidate_descriptors is None or target_descriptors is None:
        return candidate_keypoints, target_keypoints, []
    raw_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        candidate_descriptors, target_descriptors, k=2
    )
    matches = [first for first, second in raw_matches if first.distance < 0.76 * second.distance]
    return candidate_keypoints, target_keypoints, matches


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def _contour_correspondences(
    candidate_mask: np.ndarray,
    target_mask: np.ndarray,
    count: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_contour = _largest_contour(candidate_mask)
    target_contour = _largest_contour(target_mask)
    if candidate_contour is None or target_contour is None:
        return np.empty((0, 2)), np.empty((0, 2))
    candidate_points = candidate_contour[:, 0, :].astype(np.float32)
    target_points = target_contour[:, 0, :].astype(np.float32)
    candidate_center = candidate_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    directions = np.column_stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, count, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, count, endpoint=False)),
        ]
    ).astype(np.float32)
    candidate_selected = []
    target_selected = []
    for direction in directions:
        candidate_selected.append(
            candidate_points[np.argmax((candidate_points - candidate_center) @ direction)]
        )
        target_selected.append(target_points[np.argmax((target_points - target_center) @ direction)])
    return np.asarray(candidate_selected), np.asarray(target_selected)


def _mesh_points_at_pixels(
    pixels: np.ndarray,
    view_index: int,
    pix_to_face: np.ndarray,
    barycentric: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_3d = []
    kept_pixels = []
    height, width = pix_to_face.shape[1:]
    for pixel in pixels:
        x = int(np.clip(round(float(pixel[0])), 0, width - 1))
        y = int(np.clip(round(float(pixel[1])), 0, height - 1))
        face_index = int(pix_to_face[view_index, y, x])
        if face_index < 0:
            found = False
            for radius in (1, 2, 3):
                y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
                x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
                valid_y, valid_x = np.nonzero(pix_to_face[view_index, y0:y1, x0:x1] >= 0)
                if len(valid_x):
                    y, x = y0 + int(valid_y[0]), x0 + int(valid_x[0])
                    face_index = int(pix_to_face[view_index, y, x])
                    found = True
                    break
            if not found:
                continue
        # PyTorch3D stores face indices in one packed array for the whole batch.
        # Candidate views repeat the same mesh, so map the packed index back to
        # the mesh-local face index before barycentric interpolation.
        face_index %= len(faces)
        weights = barycentric[view_index, y, x]
        points_3d.append(np.sum(vertices[faces[face_index]] * weights[:, None], axis=0))
        kept_pixels.append(pixel)
    return np.asarray(points_3d, dtype=np.float32), np.asarray(kept_pixels, dtype=np.float32)


def _map_crop_pixels_to_image(
    pixels: np.ndarray,
    crop_box: tuple[int, int, int, int],
    candidate_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    mapped = pixels.astype(np.float32).copy()
    mapped[:, 0] = x0 + mapped[:, 0] / candidate_size * (x1 - x0)
    mapped[:, 1] = y0 + mapped[:, 1] / candidate_size * (y1 - y0)
    return mapped


def _build_candidate_montage(
    images: np.ndarray,
    scores: list[tuple[int, int, float]],
    azimuths: list[float],
    elevations: list[float],
) -> np.ndarray:
    tiles = []
    for rank, (view_index, matches, shape_score) in enumerate(scores[:12]):
        tile = cv2.resize(images[view_index], (240, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"#{rank + 1} m={matches} shape={shape_score:.3f}",
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 255, 40),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"az={azimuths[view_index]:.0f} el={elevations[view_index]:.0f}",
            (7, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 255, 40),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    blank = np.full((240, 240, 3), 24, dtype=np.uint8)
    while len(tiles) < 12:
        tiles.append(blank.copy())
    return np.concatenate(
        [np.concatenate(tiles[row : row + 4], axis=1) for row in range(0, 12, 4)],
        axis=0,
    )


def _initial_pose(
    rendered: dict,
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    images = rendered["images"]
    masks = rendered["masks"]
    mesh = rendered["mesh"]
    vertices = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    target_contour = _largest_contour(target_mask)
    scored_views: list[tuple[int, int, float]] = []
    cached_matches: dict[int, tuple] = {}
    for view_index, (image, mask) in enumerate(zip(images, masks)):
        keypoints_candidate, keypoints_target, matches = _sift_matches(
            image, mask, target_rgb, target_mask
        )
        candidate_contour = _largest_contour(mask)
        shape_score = (
            float(cv2.matchShapes(candidate_contour, target_contour, cv2.CONTOURS_MATCH_I1, 0.0))
            if candidate_contour is not None and target_contour is not None
            else 1e6
        )
        scored_views.append((view_index, len(matches), shape_score))
        cached_matches[view_index] = (keypoints_candidate, keypoints_target, matches)
    scored_views.sort(key=lambda record: (-record[1], record[2]))

    best_solution = None
    for view_index, match_count, shape_score in scored_views[:10]:
        candidate_keypoints, target_keypoints, matches = cached_matches[view_index]
        if match_count >= 6:
            candidate_pixels = np.asarray(
                [candidate_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32
            )
            target_pixels = np.asarray(
                [target_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32
            )
            mode = "sift"
        else:
            candidate_pixels, target_pixels = _contour_correspondences(
                masks[view_index], target_mask
            )
            mode = "contour"
        object_points, kept_candidate_pixels = _mesh_points_at_pixels(
            candidate_pixels,
            view_index,
            rendered["pix_to_face"],
            rendered["barycentric"],
            vertices,
            faces,
        )
        if len(object_points) < 6:
            continue
        if len(kept_candidate_pixels) != len(candidate_pixels):
            lookup = cKDTree(candidate_pixels)
            kept_indices = lookup.query(kept_candidate_pixels, k=1)[1]
            target_pixels = target_pixels[kept_indices]
        image_points = _map_crop_pixels_to_image(target_pixels, crop_box, target_rgb.shape[0])
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            intrinsic,
            np.zeros(4, dtype=np.float32),
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=2000,
            reprojectionError=7.0,
            confidence=0.999,
        )
        if not success or inliers is None or len(inliers) < 5:
            continue
        inlier_indices = inliers[:, 0]
        success, rvec, tvec = cv2.solvePnP(
            object_points[inlier_indices],
            image_points[inlier_indices],
            intrinsic,
            np.zeros(4, dtype=np.float32),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            object_points[inlier_indices], rvec, tvec, intrinsic, np.zeros(4)
        )
        error = float(
            np.median(
                np.linalg.norm(projected[:, 0, :] - image_points[inlier_indices], axis=1)
            )
        )
        solution = {
            "rotation": rotation.astype(np.float32),
            "translation": tvec[:, 0].astype(np.float32),
            "object_points": object_points[inlier_indices],
            "image_points": image_points[inlier_indices],
            "view_index": view_index,
            "mode": mode,
            "inliers": int(len(inlier_indices)),
            "reprojection_error_px": error,
            "shape_score": shape_score,
        }
        if best_solution is None or (solution["inliers"], -error) > (
            best_solution["inliers"],
            -best_solution["reprojection_error_px"],
        ):
            best_solution = solution
    if best_solution is None:
        raise RuntimeError("No candidate view produced a valid PnP solution")
    montage = _build_candidate_montage(
        images, scored_views, rendered["azimuths"], rendered["elevations"]
    )
    return (
        best_solution["rotation"],
        best_solution["translation"],
        best_solution["object_points"],
        best_solution,
        montage,
    )


def _estimate_metric_scale(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    point_map: np.ndarray,
    extrinsic: np.ndarray,
    object_mask: np.ndarray,
) -> tuple[float, int]:
    height, width = object_mask.shape
    pixels = np.rint(image_points).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    observed_world = point_map[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(observed_world).all(axis=1) & object_mask[pixels[:, 1], pixels[:, 0]]
    if valid.sum() < 4:
        raise RuntimeError("Too few PnP correspondences overlap valid VGGT object points")
    observed_homogeneous = np.column_stack([observed_world[valid], np.ones(valid.sum())])
    observed_camera = observed_homogeneous @ extrinsic.T
    predicted_camera = object_points[valid] @ rotation.T + translation
    depth_ratios = observed_camera[:, 2] / predicted_camera[:, 2]
    depth_ratios = depth_ratios[np.isfinite(depth_ratios) & (depth_ratios > 0)]
    if len(depth_ratios) < 4:
        raise RuntimeError("VGGT/PnP correspondences do not yield a positive metric scale")
    low, high = np.quantile(depth_ratios, [0.15, 0.85])
    trimmed = depth_ratios[(depth_ratios >= low) & (depth_ratios <= high)]
    scale = float(np.median(trimmed))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Estimated object scale is invalid")
    return scale, int(len(trimmed))


def _initialize_metric_pose(
    vertices: np.ndarray,
    rotation: np.ndarray,
    object_mask: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[float, np.ndarray, dict]:
    """Resolve monocular scale and translation from VGGT depth and silhouette extent."""
    valid_depth = object_mask & np.isfinite(depth) & (depth > 0)
    if int(valid_depth.sum()) < 32:
        raise RuntimeError("Too few valid VGGT object-depth pixels for metric initialization")

    pixels_y, pixels_x = np.nonzero(object_mask)
    x_low, x_high = np.quantile(pixels_x, [0.01, 0.99])
    y_low, y_high = np.quantile(pixels_y, [0.01, 0.99])
    target_width = max(float(x_high - x_low), 2.0)
    target_height = max(float(y_high - y_low), 2.0)
    target_center = np.asarray([(x_low + x_high) * 0.5, (y_low + y_high) * 0.5])

    rotated = vertices @ rotation.T
    mesh_low, mesh_high = np.quantile(rotated, [0.005, 0.995], axis=0)
    mesh_extent = np.maximum(mesh_high - mesh_low, 1e-6)
    mesh_center = (mesh_low + mesh_high) * 0.5
    object_depth = depth[valid_depth].astype(np.float64)
    reference_depth = float(np.median(object_depth))
    scale_candidates = np.asarray(
        [
            target_width * reference_depth / (float(intrinsic[0, 0]) * mesh_extent[0]),
            target_height * reference_depth / (float(intrinsic[1, 1]) * mesh_extent[1]),
        ]
    )
    scale_candidates = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0)]
    if not len(scale_candidates):
        raise RuntimeError("Silhouette and VGGT depth did not yield a positive metric scale")
    scale = float(np.median(scale_candidates))

    target_front_depth = float(np.quantile(object_depth, 0.10))
    mesh_front = float(np.quantile(rotated[:, 2], 0.01))

    def translation_for(current_scale: float) -> np.ndarray:
        translation_z = target_front_depth - current_scale * mesh_front
        center_depth = current_scale * float(mesh_center[2]) + translation_z
        center_camera = np.asarray(
            [
                (target_center[0] - intrinsic[0, 2]) * center_depth / intrinsic[0, 0],
                (target_center[1] - intrinsic[1, 2]) * center_depth / intrinsic[1, 1],
                center_depth,
            ],
            dtype=np.float64,
        )
        return center_camera - current_scale * mesh_center

    def projected_bounds(current_scale: float, current_translation: np.ndarray) -> np.ndarray:
        transformed = current_scale * rotated + current_translation
        valid = transformed[:, 2] > 1e-6
        projected = np.column_stack(
            [
                intrinsic[0, 0] * transformed[valid, 0] / transformed[valid, 2]
                + intrinsic[0, 2],
                intrinsic[1, 1] * transformed[valid, 1] / transformed[valid, 2]
                + intrinsic[1, 2],
            ]
        )
        if len(projected) < 3:
            raise RuntimeError("Too few mesh vertices project in front of the camera")
        return np.quantile(projected, [0.001, 0.999], axis=0)

    projected_extent_initial: list[float] | None = None
    for _ in range(3):
        translation = translation_for(scale)
        projection_bounds = projected_bounds(scale, translation)
        projected_extent = np.maximum(projection_bounds[1] - projection_bounds[0], 1e-6)
        if projected_extent_initial is None:
            projected_extent_initial = projected_extent.tolist()
        correction = float(
            np.median([target_width / projected_extent[0], target_height / projected_extent[1]])
        )
        scale *= float(np.clip(correction, 0.70, 1.30))

    translation = translation_for(scale)
    for _ in range(2):
        projection_bounds = projected_bounds(scale, translation)
        projected_center = projection_bounds.mean(axis=0)
        center_depth = float(scale * mesh_center[2] + translation[2])
        pixel_delta = target_center - projected_center
        translation[0] += pixel_delta[0] * center_depth / intrinsic[0, 0]
        translation[1] += pixel_delta[1] * center_depth / intrinsic[1, 1]
    projection_bounds = projected_bounds(scale, translation)
    projected_extent_final = (projection_bounds[1] - projection_bounds[0]).tolist()
    diagnostics = {
        "method": "vggt_depth_and_silhouette_extent",
        "target_extent_px": [target_width, target_height],
        "target_center_px": target_center.tolist(),
        "reference_depth": reference_depth,
        "target_front_depth": target_front_depth,
        "mesh_extent": mesh_extent.tolist(),
        "scale_candidates": scale_candidates.tolist(),
        "projected_extent_initial_px": projected_extent_initial,
        "projected_extent_final_px": projected_extent_final,
        "scale": scale,
        "translation_camera": translation.tolist(),
    }
    return scale, translation.astype(np.float32), diagnostics


def _refine_pose(
    vertices: np.ndarray,
    faces: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    initial_scale: float,
    object_points: np.ndarray,
    image_points: np.ndarray,
    object_mask: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    device: torch.device,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    height, width = object_mask.shape
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(faces, dtype=torch.int64, device=device)
    target_mask_tensor = torch.tensor(object_mask, dtype=torch.bool, device=device)
    target_depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
    object_points_tensor = torch.tensor(object_points, dtype=torch.float32, device=device)
    image_points_tensor = torch.tensor(image_points, dtype=torch.float32, device=device)

    camera_rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device)).unsqueeze(0)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=((float(intrinsic[0, 0]), float(intrinsic[1, 1])),),
        principal_point=((float(intrinsic[0, 2]), float(intrinsic[1, 2])),),
        image_size=((height, width),),
        R=camera_rotation,
        T=torch.zeros((1, 3), device=device),
        in_ndc=False,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=max(int(len(faces) * 1.25), 10000),
        ),
    )
    median_depth = float(np.median(depth[object_mask & np.isfinite(depth) & (depth > 0)]))
    image_diagonal = float(np.hypot(height, width))

    def mask_bounds(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = torch.nonzero(mask, as_tuple=False).to(dtype=torch.float32)
        if len(pixels) < 4:
            raise RuntimeError("Rendered object mask is empty during raster calibration")
        low = pixels.amin(dim=0)
        high = pixels.amax(dim=0)
        return low, high

    def state_rotation(state: np.ndarray) -> torch.Tensor:
        return axis_angle_to_matrix(
            torch.tensor(state[:3], dtype=torch.float32, device=device)
        )

    def render_state(state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation = state_rotation(state)
        scale = float(np.exp(state[3]))
        translation = torch.tensor(state[4:7], dtype=torch.float32, device=device)
        transformed = scale * (vertices_tensor @ rotation.T) + translation
        mesh = Meshes(verts=[transformed], faces=[faces_tensor])
        fragments = rasterizer(mesh)
        rendered_depth = fragments.zbuf[0, ..., 0]
        rendered_mask = fragments.pix_to_face[0, ..., 0] >= 0
        return rendered_mask, rendered_depth, rotation

    def evaluate_state(state: np.ndarray, evaluation: int) -> dict:
        with torch.no_grad():
            rendered_mask, rendered_depth, rotation = render_state(state)
            rendered_area = int(rendered_mask.sum().cpu())
            if rendered_area < 4:
                return {
                    "evaluation": evaluation,
                    "loss": float("inf"),
                    "mask_iou": 0.0,
                    "rendered_mask_area": rendered_area,
                }
            intersection = int((rendered_mask & target_mask_tensor).sum().cpu())
            union = int((rendered_mask | target_mask_tensor).sum().cpu())
            mask_iou = intersection / max(union, 1)

            overlap = (
                rendered_mask
                & target_mask_tensor
                & (rendered_depth > 0)
                & (target_depth_tensor > 0)
            )
            if int(overlap.sum()) >= 32:
                depth_relative_error = float(
                    torch.median(
                        torch.abs(rendered_depth[overlap] - target_depth_tensor[overlap])
                    ).cpu()
                    / median_depth
                )
            else:
                depth_relative_error = 1.0

            scale = float(np.exp(state[3]))
            translation = torch.tensor(state[4:7], dtype=torch.float32, device=device)
            matched_camera = scale * (object_points_tensor @ rotation.T) + translation
            valid_projection = matched_camera[:, 2] > 1e-6
            if int(valid_projection.sum()) >= 4:
                projected_x = (
                    intrinsic[0, 0]
                    * matched_camera[valid_projection, 0]
                    / matched_camera[valid_projection, 2]
                    + intrinsic[0, 2]
                )
                projected_y = (
                    intrinsic[1, 1]
                    * matched_camera[valid_projection, 1]
                    / matched_camera[valid_projection, 2]
                    + intrinsic[1, 2]
                )
                projected = torch.stack([projected_x, projected_y], dim=1)
                reprojection_error = float(
                    torch.mean(
                        torch.linalg.norm(
                            projected - image_points_tensor[valid_projection], dim=1
                        )
                    ).cpu()
                    / image_diagonal
                )
            else:
                reprojection_error = 1.0

            rendered_low, rendered_high = mask_bounds(rendered_mask)
            rendered_center = (rendered_low + rendered_high) * 0.5
            rendered_extent = rendered_high - rendered_low + 1.0
            center_error = float(
                torch.linalg.norm(rendered_center - target_center).cpu() / image_diagonal
            )
            extent_error = float(
                torch.mean(torch.abs(torch.log(rendered_extent / target_extent))).cpu()
            )
            loss = (
                1.50 * (1.0 - mask_iou)
                + 0.35 * depth_relative_error
                + 0.10 * reprojection_error
                + 0.05 * center_error
                + 0.05 * extent_error
            )
            return {
                "evaluation": evaluation,
                "loss": float(loss),
                "mask_iou": float(mask_iou),
                "depth_relative_error": depth_relative_error,
                "reprojection_error_normalized": reprojection_error,
                "center_error_normalized": center_error,
                "extent_log_error": extent_error,
                "rendered_mask_area": rendered_area,
                "rendered_extent_px": rendered_extent.cpu().tolist()[::-1],
            }

    initial_rotation_tensor = torch.tensor(initial_rotation, dtype=torch.float32, device=device)
    initial_axis_angle = matrix_to_axis_angle(initial_rotation_tensor).cpu().numpy()
    state = np.concatenate(
        [
            initial_axis_angle.astype(np.float64),
            np.asarray([np.log(initial_scale)], dtype=np.float64),
            np.asarray(initial_translation, dtype=np.float64),
        ]
    )
    target_low, target_high = mask_bounds(target_mask_tensor)
    target_center = (target_low + target_high) * 0.5
    target_extent = target_high - target_low + 1.0
    target_area = float(target_mask_tensor.sum().cpu())
    raster_calibration = []
    with torch.no_grad():
        for calibration_step in range(3):
            rendered_mask, rendered_depth, _ = render_state(state)
            rendered_low, rendered_high = mask_bounds(rendered_mask)
            rendered_extent = rendered_high - rendered_low + 1.0
            rendered_area = float(rendered_mask.sum().cpu())
            correction = float(np.clip(np.sqrt(target_area / rendered_area), 0.85, 1.15))
            state[3] += np.log(correction)
            rendered_center = (rendered_low + rendered_high) * 0.5
            pixel_delta_yx = target_center - rendered_center
            visible_depth = rendered_depth[rendered_mask]
            center_depth = float(torch.median(visible_depth).clamp_min(1e-6).cpu())
            state[4] += float(pixel_delta_yx[1].cpu()) * center_depth / float(
                intrinsic[0, 0]
            )
            state[5] += float(pixel_delta_yx[0].cpu()) * center_depth / float(
                intrinsic[1, 1]
            )
            raster_calibration.append(
                {
                    "step": calibration_step,
                    "target_extent_px": target_extent.cpu().tolist()[::-1],
                    "rendered_extent_px": rendered_extent.cpu().tolist()[::-1],
                    "pixel_delta_xy": pixel_delta_yx.cpu().tolist()[::-1],
                    "scale_correction": correction,
                }
            )

    calibrated_scale = float(np.exp(state[3]))
    calibrated_state = state.copy()
    metric_extent = max(
        float(np.linalg.norm(np.ptp(vertices, axis=0))) * calibrated_scale,
        1e-6,
    )
    angle_limit = np.deg2rad(15.0)
    lower = calibrated_state.copy()
    upper = calibrated_state.copy()
    lower[:3] -= angle_limit
    upper[:3] += angle_limit
    lower[3] = np.log(calibrated_scale * 0.75)
    upper[3] = np.log(calibrated_scale * 1.30)
    translation_limit = np.asarray(
        [metric_extent * 0.30, metric_extent * 0.30, max(metric_extent * 0.30, median_depth * 0.08)]
    )
    lower[4:7] -= translation_limit
    upper[4:7] += translation_limit
    steps = np.asarray(
        [
            np.deg2rad(4.0),
            np.deg2rad(4.0),
            np.deg2rad(4.0),
            np.log(1.04),
            median_depth * 2.0 / float(intrinsic[0, 0]),
            median_depth * 2.0 / float(intrinsic[1, 1]),
            max(metric_extent * 0.025, median_depth * 0.006),
        ],
        dtype=np.float64,
    )
    parameter_names = ["rotation_x", "rotation_y", "rotation_z", "scale", "x", "y", "z"]
    parameter_order = [4, 5, 3, 6, 0, 1, 2]
    evaluation_budget = max(int(iterations), 1)
    evaluation_count = 1
    current = calibrated_state.copy()
    current_record = evaluate_state(current, evaluation_count)
    initial_record = dict(current_record)
    accepted_steps = []
    rounds = 0

    while evaluation_count < evaluation_budget and rounds < 12:
        improved = False
        for parameter_index in parameter_order:
            candidates = []
            for direction in (-1.0, 1.0):
                if evaluation_count >= evaluation_budget:
                    break
                candidate = current.copy()
                candidate[parameter_index] += direction * steps[parameter_index]
                candidate = np.clip(candidate, lower, upper)
                if np.allclose(candidate, current):
                    continue
                evaluation_count += 1
                record = evaluate_state(candidate, evaluation_count)
                candidates.append((record["loss"], candidate, record, direction))
            if not candidates:
                continue
            candidate_loss, candidate_state, candidate_record, direction = min(
                candidates, key=lambda item: item[0]
            )
            if candidate_loss + 1e-7 < current_record["loss"]:
                current = candidate_state
                current_record = candidate_record
                improved = True
                accepted_steps.append(
                    {
                        "round": rounds,
                        "parameter": parameter_names[parameter_index],
                        "direction": direction,
                        "step": float(steps[parameter_index]),
                        "evaluation": candidate_record["evaluation"],
                        "loss": candidate_record["loss"],
                        "mask_iou": candidate_record["mask_iou"],
                    }
                )
        steps *= 0.70 if improved else 0.50
        rounds += 1

    final_rotation = state_rotation(current).cpu().numpy()
    final_translation = current[4:7].astype(np.float32)
    final_scale = float(np.exp(current[3]))
    diagnostics = {
        "method": "hard_raster_coordinate_search",
        "evaluation_budget": evaluation_budget,
        "evaluations": evaluation_count,
        "rounds": rounds,
        "initial_scale": initial_scale,
        "raster_calibrated_scale": calibrated_scale,
        "raster_calibration": raster_calibration,
        "final_scale": final_scale,
        "initial": initial_record,
        "best": current_record,
        "accepted_steps": accepted_steps,
    }
    return final_rotation, final_translation, final_scale, diagnostics


def _render_aligned(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    height, width = image_shape
    transformed = np.column_stack([vertices, np.ones(len(vertices))]) @ transform_camera.T
    transformed = transformed[:, :3]
    mesh = Meshes(
        verts=[torch.tensor(transformed, dtype=torch.float32, device=device)],
        faces=[torch.tensor(faces, dtype=torch.int64, device=device)],
    )
    camera_rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device)).unsqueeze(0)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=((float(intrinsic[0, 0]), float(intrinsic[1, 1])),),
        principal_point=((float(intrinsic[0, 2]), float(intrinsic[1, 2])),),
        image_size=((height, width),),
        R=camera_rotation,
        T=torch.zeros((1, 3), device=device),
        in_ndc=False,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=max(int(len(faces) * 1.25), 10000),
        ),
    )
    fragments = rasterizer(mesh)
    depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
    mask = depth > 0
    depth[~mask] = 0
    return mask, depth.astype(np.float32)


def _sample_aligned_mesh(
    mesh: trimesh.Trimesh,
    transform_world: np.ndarray,
    visible_points: np.ndarray,
    visible_colors: np.ndarray,
    seed: int,
    count: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        sampled, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        sampled, _ = trimesh.sample.sample_surface(mesh, count)
        np.random.set_state(state)
    aligned = np.column_stack([sampled, np.ones(len(sampled))]) @ transform_world.T
    aligned = aligned[:, :3].astype(np.float32)
    nearest = cKDTree(visible_points).query(aligned, k=1)[1]
    return aligned, visible_colors[nearest].astype(np.uint8)


def _resolve_support_relation(
    vertices: np.ndarray,
    transform_world: np.ndarray,
    planes: list[dict],
    support_object_points: np.ndarray | None = None,
) -> dict:
    if not planes:
        return {
            "resolved": False,
            "status": "no_environment_plane",
            "visual_transform_modified": False,
        }
    transformed = np.column_stack([vertices, np.ones(len(vertices))]) @ transform_world.T
    transformed = transformed[:, :3]
    support_points = (
        np.asarray(support_object_points, dtype=np.float64)
        if support_object_points is not None and len(support_object_points) >= 64
        else transformed
    )
    object_extent = float(np.linalg.norm(np.ptp(transformed, axis=0)))
    tolerance = max(object_extent * 0.004, 1e-6)
    candidates = []
    for record in planes:
        equation = np.asarray(record["equation"], dtype=np.float64)
        equation_norm = max(float(np.linalg.norm(equation[:3])), 1e-12)
        normal = equation[:3] / equation_norm
        offset = float(equation[3]) / equation_norm
        mesh_distances = transformed @ normal + offset
        side = 1.0 if float(np.median(mesh_distances)) >= 0 else -1.0
        support_distances = (support_points @ normal + offset) * side
        oriented_mesh = mesh_distances * side
        lower = float(np.quantile(support_distances, 0.005))
        near = float(np.quantile(support_distances, 0.05))
        median = float(np.median(support_distances))
        bulk_intersection = median < object_extent * 0.03
        candidates.append(
            (
                abs(near) + (object_extent if bulk_intersection else 0.0),
                lower,
                near,
                median,
                float(np.quantile(oriented_mesh, 0.005)),
                normal * side,
                offset * side,
                record["index"],
            )
        )
    candidates.sort(key=lambda item: item[0])
    _, lower, near, median, mesh_lower, oriented_normal, oriented_offset, plane_index = candidates[0]
    common = {
        "plane_index": int(plane_index),
        "support_lower_distance": lower,
        "support_near_distance": near,
        "support_median_distance": median,
        "full_mesh_lower_distance": mesh_lower,
        "tolerance": tolerance,
        "oriented_plane_equation": [*oriented_normal.tolist(), oriented_offset],
        "support_basis": "full_vggt_object_mask",
        "visual_transform_modified": False,
    }
    if median < object_extent * 0.03:
        return {
            "resolved": False,
            "status": "support_plane_intersects_object_bulk",
            **common,
        }
    return {
        "resolved": True,
        "status": (
            "observed_clear"
            if lower >= -tolerance
            else "observed_depth_boundary_uncertain"
        ),
        **common,
    }


def _build_collision_proxy(
    vertices: np.ndarray,
    transform_world: np.ndarray,
    contact: dict,
) -> tuple[np.ndarray, dict]:
    equation = contact.get("oriented_plane_equation")
    if equation is None:
        return vertices.astype(np.float32), {
            "status": "unmodified_no_support_plane",
            "safe": False,
            "translation_distance": 0.0,
            "translation_fraction_of_object_extent": 0.0,
        }
    equation = np.asarray(equation, dtype=np.float64)
    normal, offset = equation[:3], float(equation[3])
    transformed = np.column_stack([vertices, np.ones(len(vertices))]) @ transform_world.T
    transformed = transformed[:, :3]
    distances = transformed @ normal + offset
    tolerance = float(contact["tolerance"])
    lower = float(np.quantile(distances, 0.005))
    required = max(tolerance - lower, 0.0)
    object_extent = max(float(np.linalg.norm(np.ptp(transformed, axis=0))), 1e-8)
    safe = required <= object_extent * 0.10
    if required:
        transformed += normal[None, :] * required
    local_from_world = np.linalg.inv(transform_world)
    local = np.column_stack([transformed, np.ones(len(transformed))]) @ local_from_world.T
    local = local[:, :3]
    return local.astype(np.float32), {
        "status": "support_translated" if required else "unmodified_clear",
        "safe": safe,
        "plane_index": contact["plane_index"],
        "support_lower_distance": lower,
        "translation_world": (normal * required).tolist(),
        "translation_distance": required,
        "translation_fraction_of_object_extent": required / object_extent,
    }


def register_object_mesh(
    mesh_path: Path,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    point_map: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    visible_object_points: np.ndarray,
    visible_object_colors: np.ndarray,
    environment_planes: list[dict],
    device: torch.device,
    seed: int,
    candidate_size: int = 384,
    refinement_iterations: int = 48,
) -> RegistrationResult:
    target_rgb, target_mask, crop_box = _target_crop(rgb, object_mask, candidate_size)
    rendered = _render_candidates(mesh_path, device, candidate_size)
    rotation, translation, matched_object_points, pnp, montage = _initial_pose(
        rendered, target_rgb, target_mask, crop_box, intrinsic
    )
    image_points = pnp["image_points"]
    mesh = _load_trimesh(mesh_path)
    try:
        pnp_metric_scale, scale_inliers = _estimate_metric_scale(
            matched_object_points,
            image_points,
            rotation,
            translation,
            point_map,
            extrinsic,
            object_mask,
        )
        pnp_scale_diagnostics = {
            "scale": pnp_metric_scale,
            "trimmed_correspondences": scale_inliers,
        }
    except RuntimeError as error:
        pnp_scale_diagnostics = {"error": str(error), "trimmed_correspondences": 0}
    scale, metric_translation, metric_initialization = _initialize_metric_pose(
        np.asarray(mesh.vertices, dtype=np.float32),
        rotation,
        object_mask,
        depth,
        intrinsic,
    )
    refined_rotation, refined_translation, refined_scale, refinement = _refine_pose(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int64),
        rotation,
        metric_translation,
        scale,
        matched_object_points,
        image_points,
        object_mask,
        depth,
        intrinsic,
        device,
        refinement_iterations,
    )
    transform_camera = np.eye(4, dtype=np.float32)
    transform_camera[:3, :3] = refined_scale * refined_rotation
    transform_camera[:3, 3] = refined_translation
    world_from_camera = np.eye(4, dtype=np.float32)
    world_from_camera[:3, :4] = extrinsic
    world_from_camera = np.linalg.inv(world_from_camera)
    transform_world = world_from_camera @ transform_camera
    support_mask = object_mask & np.isfinite(point_map).all(axis=-1) & (depth > 0)
    support_points = point_map[support_mask]
    support_relation = _resolve_support_relation(
        np.asarray(mesh.vertices, dtype=np.float32),
        transform_world,
        environment_planes,
        support_points,
    )
    collision_vertices, collision_proxy = _build_collision_proxy(
        np.asarray(mesh.vertices, dtype=np.float32),
        transform_world,
        support_relation,
    )
    camera_from_world = np.eye(4, dtype=np.float32)
    camera_from_world[:3, :4] = extrinsic
    transform_camera = camera_from_world @ transform_world
    rendered_mask, rendered_depth = _render_aligned(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int64),
        transform_camera,
        intrinsic,
        object_mask.shape,
        device,
    )
    aligned_points, aligned_colors = _sample_aligned_mesh(
        mesh,
        transform_world,
        visible_object_points,
        visible_object_colors,
        seed,
    )
    intersection = np.logical_and(rendered_mask, object_mask).sum()
    union = np.logical_or(rendered_mask, object_mask).sum()
    overlap = rendered_mask & object_mask & np.isfinite(depth) & (depth > 0)
    depth_mae = (
        float(np.median(np.abs(rendered_depth[overlap] - depth[overlap])))
        if overlap.any()
        else None
    )
    diagnostics = {
        "pnp": {
            key: value
            for key, value in pnp.items()
            if key not in {"rotation", "translation", "object_points", "image_points"}
        },
        "pnp_metric_scale": pnp_scale_diagnostics,
        "metric_initialization": metric_initialization,
        "refinement": refinement,
        "final_mask_iou": float(intersection / max(union, 1)),
        "final_depth_median_absolute_error": depth_mae,
        "crop_box_xyxy": list(crop_box),
        "support_relation": support_relation,
        "collision_proxy": collision_proxy,
    }
    return RegistrationResult(
        transform_world=transform_world.astype(np.float32),
        transform_camera=transform_camera.astype(np.float32),
        aligned_points=aligned_points,
        aligned_colors=aligned_colors,
        rendered_mask=rendered_mask,
        rendered_depth=rendered_depth,
        collision_vertices=collision_vertices,
        collision_faces=np.asarray(mesh.faces, dtype=np.int64),
        candidate_montage=montage,
        diagnostics=diagnostics,
    )
