from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decomposition import decompose_scene
from depthlab_frontend import run_depthlab_completion
from environment_completion import complete_environment
from instantmesh_frontend import copy_mesh_bundle, run_instantmesh, save_object_crop
from lama_frontend import run_lama_inpainting
from outputs import (
    save_alignment_review,
    save_component_review,
    save_image,
    save_json,
    save_ply,
    save_review,
)
from registration import register_object_mesh
from segmentation import segmentation_from_mask, segment_primary_object
from vggt_frontend import preprocess_binary_mask_for_vggt_pad, run_vggt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a structured scene from one RGB first frame"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--depthlab-device", type=int)
    parser.add_argument("--instantmesh-device", type=int)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--confidence-quantile", type=float, default=0.25)
    parser.add_argument("--gt-mask", type=Path)
    parser.add_argument(
        "--object-mask",
        type=Path,
        help="explicit controlled-object mask; inference should obtain this from user selection",
    )
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument("--reuse-object-mesh", action="store_true")
    parser.add_argument("--instantmesh-diffusion-steps", type=int, default=50)
    parser.add_argument("--depthlab-denoise-steps", type=int, default=50)
    parser.add_argument("--candidate-render-size", type=int, default=384)
    parser.add_argument(
        "--pose-refinement-iterations",
        type=int,
        default=48,
        help="Maximum hard-raster pose evaluations after metric initialization",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/scene_completion/vggt/model.pt",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/scene_completion/sam2/sam2.1_hiera_small.pt",
    )
    parser.add_argument(
        "--lama-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/scene_completion/lama/big-lama.pt",
    )
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument(
        "--instantmesh-repository",
        type=Path,
        default=PROJECT_ROOT / "external/scene_completion/InstantMesh",
    )
    parser.add_argument(
        "--instantmesh-python",
        type=Path,
        default=PROJECT_ROOT / "envs/phycontext-instantmesh/bin/python",
    )
    parser.add_argument(
        "--instantmesh-cache",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/scene_completion/instantmesh",
    )
    parser.add_argument(
        "--depthlab-repository",
        type=Path,
        default=PROJECT_ROOT / "external/scene_completion/DepthLab",
    )
    parser.add_argument(
        "--depthlab-python",
        type=Path,
        default=PROJECT_ROOT / "envs/phycontext-depthlab/bin/python",
    )
    parser.add_argument(
        "--depthlab-checkpoints",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/scene_completion/depthlab",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binary_mask(target_path: Path) -> np.ndarray:
    target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise FileNotFoundError(target_path)
    if target.ndim == 3 and target.shape[2] == 4:
        alpha = target[..., 3]
        if np.unique(alpha).size >= 2:
            target = alpha
        else:
            target = target[..., :3]
    if target.ndim == 3:
        if np.max(np.ptp(target, axis=2)) > 0:
            raise ValueError(f"GT mask is an RGB image rather than a binary mask: {target_path}")
        target = target[..., 0]
    target = target > 0
    if np.unique(target).size < 2:
        raise ValueError(f"GT mask is constant and therefore invalid: {target_path}")
    return target


def _mask_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.shape != target.shape:
        raise ValueError(
            f"mask shapes must match before IoU evaluation: {predicted.shape} != {target.shape}"
        )
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    return float(intersection / max(union, 1))


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("The reconstruction pipeline currently requires a CUDA device")
    return device.index if device.index is not None else torch.cuda.current_device()


def _artifact_records(output_dir: Path, names: list[str]) -> dict[str, dict]:
    records = {}
    for name in names:
        path = output_dir / name
        if path.is_file():
            records[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return records


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    required_files = [
        args.image,
        args.vggt_checkpoint,
        args.lama_checkpoint,
        args.depthlab_repository / "infer.py",
        args.depthlab_python,
        args.depthlab_checkpoints / "marigold-depth-v1-0/model_index.json",
        args.depthlab_checkpoints / "CLIP-ViT-H-14-laion2B-s32B-b79K/config.json",
        args.depthlab_checkpoints
        / "CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors",
        args.depthlab_checkpoints / "DepthLab/denoising_unet.pth",
        args.depthlab_checkpoints / "DepthLab/reference_unet.pth",
        args.depthlab_checkpoints / "DepthLab/mapping_layer.pth",
    ]
    if args.object_mask is None:
        required_files.append(args.sam_checkpoint)
    else:
        required_files.append(args.object_mask)
    if args.object_mesh is not None:
        required_files.append(args.object_mesh)
    else:
        required_files.extend([args.instantmesh_repository / "run.py", args.instantmesh_python])
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    gt_mask = _load_binary_mask(args.gt_mask) if args.gt_mask else None
    provided_object_mask = (
        _load_binary_mask(args.object_mask) if args.object_mask else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    depthlab_device = (
        args.depthlab_device if args.depthlab_device is not None else _device_index(device)
    )
    instantmesh_device = (
        args.instantmesh_device if args.instantmesh_device is not None else _device_index(device)
    )
    timings: dict[str, float] = {}

    started = time.perf_counter()
    vggt = run_vggt(args.image, args.vggt_checkpoint, device)
    timings["vggt_seconds"] = time.perf_counter() - started
    save_image(args.output_dir / "input_preprocessed.png", vggt.rgb)
    if gt_mask is not None:
        gt_mask = preprocess_binary_mask_for_vggt_pad(
            gt_mask,
            vggt.source_shape_hw,
            vggt.rgb.shape[:2],
        )
    if provided_object_mask is not None:
        provided_object_mask = preprocess_binary_mask_for_vggt_pad(
            provided_object_mask,
            vggt.source_shape_hw,
            vggt.rgb.shape[:2],
        )

    started = time.perf_counter()
    if provided_object_mask is not None:
        segmentation = segmentation_from_mask(vggt.rgb, provided_object_mask)
        segmentation_source = "provided_object_mask"
    else:
        segmentation = segment_primary_object(
            vggt.rgb,
            vggt.depth,
            args.sam_checkpoint,
            args.sam_config,
            device,
        )
        segmentation_source = "automatic_sam2"
    timings["segmentation_seconds"] = time.perf_counter() - started
    timings["sam2_seconds"] = (
        timings["segmentation_seconds"] if segmentation_source == "automatic_sam2" else 0.0
    )
    save_image(args.output_dir / "object_mask_overlay.png", segmentation.overlay)
    save_image(args.output_dir / "object_mask.png", segmentation.mask.astype(np.uint8) * 255)
    save_image(args.output_dir / "sam2_candidates.png", segmentation.candidate_montage)

    decomposition = decompose_scene(
        vggt.points,
        vggt.rgb,
        vggt.depth,
        vggt.confidence,
        segmentation.mask,
        args.confidence_quantile,
        content_mask=vggt.content_mask,
    )
    save_ply(
        args.output_dir / "environment_observed.ply",
        decomposition.environment_points,
        decomposition.environment_colors,
    )
    save_ply(
        args.output_dir / "object_visible.ply",
        decomposition.object_points,
        decomposition.object_colors,
    )

    started = time.perf_counter()
    lama = run_lama_inpainting(
        vggt.rgb,
        segmentation.mask,
        vggt.content_mask,
        args.lama_checkpoint,
        device,
    )
    inpainted_rgb = lama.rgb
    inpainted_path = args.output_dir / "environment_inpainted.png"
    save_image(inpainted_path, inpainted_rgb)
    save_image(
        args.output_dir / "environment_inpainting_mask.png",
        lama.inpaint_mask.astype(np.uint8) * 255,
    )
    timings["environment_inpainting_seconds"] = time.perf_counter() - started

    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    requested_completion_mask = vggt.content_mask & (
        lama.inpaint_mask | ~decomposition.environment_mask
    )
    depthlab = run_depthlab_completion(
        inpainted_rgb=inpainted_rgb,
        reference_depth=vggt.depth,
        requested_completion_mask=requested_completion_mask,
        content_mask=vggt.content_mask,
        output_dir=args.output_dir,
        repository=args.depthlab_repository,
        python_executable=args.depthlab_python,
        checkpoint_root=args.depthlab_checkpoints,
        device_index=depthlab_device,
        seed=args.seed,
        denoise_steps=args.depthlab_denoise_steps,
    )
    timings["depthlab_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    environment_completion = complete_environment(
        vggt.points,
        inpainted_rgb,
        vggt.depth,
        decomposition.environment_mask,
        decomposition.reliable_environment_mask,
        segmentation.mask,
        vggt.content_mask,
        depthlab.depth,
        depthlab.completion_mask,
        vggt.intrinsic,
        vggt.extrinsic,
        seed=args.seed,
    )
    timings["environment_fusion_seconds"] = time.perf_counter() - started
    timings["environment_completion_seconds"] = sum(
        timings[key]
        for key in (
            "environment_inpainting_seconds",
            "depthlab_seconds",
            "environment_fusion_seconds",
        )
    )
    save_image(
        args.output_dir / "environment_completion_mask.png",
        environment_completion.pixel_mask.astype(np.uint8) * 255,
    )
    np.save(args.output_dir / "environment_completion_depth.npy", environment_completion.depth)
    save_ply(
        args.output_dir / "environment_completed.ply",
        environment_completion.points,
        environment_completion.colors,
    )
    environment_dense_points = np.concatenate(
        [decomposition.environment_points, environment_completion.points], axis=0
    )
    environment_dense_colors = np.concatenate(
        [decomposition.environment_colors, environment_completion.colors], axis=0
    )
    save_ply(
        args.output_dir / "environment_dense.ply",
        environment_dense_points,
        environment_dense_colors,
    )
    gc.collect()
    torch.cuda.empty_cache()

    crop = save_object_crop(
        vggt.rgb,
        segmentation.mask,
        args.output_dir / "object_crop_rgba.png",
    )
    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    if args.object_mesh is not None:
        mesh_path = args.output_dir / "object_mesh.obj"
        material_path, texture_path = copy_mesh_bundle(args.object_mesh, mesh_path)
        imported_artifacts = {mesh_path.name: _sha256(mesh_path)}
        if material_path is not None:
            imported_artifacts[material_path.name] = _sha256(material_path)
        if texture_path is not None:
            imported_artifacts[texture_path.name] = _sha256(texture_path)
        object_reconstruction = {
            "schema": "phycontext.object_reconstruction.v1",
            "generator": "external_mesh",
            "provenance_status": "recorded",
            "source_mesh": str(args.object_mesh.resolve()),
            "source_mesh_sha256": _sha256(args.object_mesh),
            "artifacts": imported_artifacts,
        }
        save_json(args.output_dir / "object_mesh_manifest.json", object_reconstruction)
        object_mesh_reused = False
    else:
        instantmesh = run_instantmesh(
            crop.path,
            args.output_dir,
            args.instantmesh_repository,
            args.instantmesh_python,
            args.instantmesh_cache,
            instantmesh_device,
            args.seed,
            args.instantmesh_diffusion_steps,
            args.reuse_object_mesh,
        )
        mesh_path = instantmesh.mesh_path
        object_reconstruction = instantmesh.metadata
        object_mesh_reused = instantmesh.reused
    timings["instantmesh_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    registration = register_object_mesh(
        mesh_path=mesh_path,
        rgb=vggt.rgb,
        object_mask=segmentation.mask,
        point_map=vggt.points,
        depth=vggt.depth,
        intrinsic=vggt.intrinsic,
        extrinsic=vggt.extrinsic,
        visible_object_points=decomposition.object_points,
        visible_object_colors=decomposition.object_colors,
        environment_planes=environment_completion.planes,
        device=device,
        seed=args.seed,
        candidate_size=args.candidate_render_size,
        refinement_iterations=args.pose_refinement_iterations,
    )
    timings["registration_seconds"] = time.perf_counter() - started
    save_image(args.output_dir / "pose_candidates.png", registration.candidate_montage)
    save_alignment_review(
        args.output_dir / "alignment_review.png",
        vggt.rgb,
        segmentation.mask,
        registration.rendered_mask,
        registration.rendered_depth,
    )
    save_image(
        args.output_dir / "aligned_object_mask.png",
        registration.rendered_mask.astype(np.uint8) * 255,
    )
    np.save(args.output_dir / "aligned_object_depth.npy", registration.rendered_depth)
    collision_mesh = trimesh.Trimesh(
        vertices=registration.collision_vertices,
        faces=registration.collision_faces,
        process=False,
    )
    collision_mesh.export(args.output_dir / "object_collision_proxy.obj")
    save_ply(
        args.output_dir / "object_mesh_aligned.ply",
        registration.aligned_points,
        registration.aligned_colors,
    )

    scene_points = np.concatenate(
        [
            decomposition.environment_points,
            environment_completion.points,
            registration.aligned_points,
        ],
        axis=0,
    )
    scene_colors = np.concatenate(
        [
            decomposition.environment_colors,
            environment_completion.colors,
            registration.aligned_colors,
        ],
        axis=0,
    )
    source = np.concatenate(
        [
            np.zeros(len(decomposition.environment_points), dtype=np.uint8),
            np.ones(len(environment_completion.points), dtype=np.uint8),
            np.full(len(registration.aligned_points), 2, dtype=np.uint8),
        ]
    )
    body_id = np.concatenate(
        [
            np.zeros(len(decomposition.environment_points), dtype=np.int16),
            np.zeros(len(environment_completion.points), dtype=np.int16),
            np.ones(len(registration.aligned_points), dtype=np.int16),
        ]
    )
    semantic_palette = np.asarray([[74, 144, 226], [79, 190, 125], [255, 74, 92]], dtype=np.uint8)
    semantic_colors = semantic_palette[source]
    support_relation = registration.diagnostics["support_relation"]
    support_plane_index = int(support_relation.get("plane_index", -1))
    support_plane_record = next(
        (
            record
            for record in environment_completion.planes
            if int(record["index"]) == support_plane_index
        ),
        None,
    )
    support_plane_equation = np.asarray(
        support_relation.get("oriented_plane_equation", [np.nan] * 4),
        dtype=np.float32,
    )
    support_plane_valid = bool(
        support_relation.get("resolved")
        and support_plane_record is not None
        and support_plane_equation.shape == (4,)
        and np.isfinite(support_plane_equation).all()
    )
    support_plane_threshold = float(
        support_plane_record["distance_threshold"]
        if support_plane_record is not None
        else np.nan
    )
    save_ply(args.output_dir / "scene.ply", scene_points, scene_colors)
    save_ply(args.output_dir / "scene_sources.ply", scene_points, semantic_colors)
    np.savez_compressed(
        args.output_dir / "scene.npz",
        xyz=scene_points.astype(np.float32),
        rgb=scene_colors.astype(np.uint8),
        source=source,
        body_id=body_id,
        intrinsic=vggt.intrinsic,
        extrinsic=vggt.extrinsic,
        point_map_world=vggt.points,
        depth=vggt.depth,
        confidence=vggt.confidence,
        pixel_body_id=segmentation.mask.astype(np.int16),
        object_mask=segmentation.mask,
        aligned_object_mask=registration.rendered_mask,
        aligned_object_depth=registration.rendered_depth,
        environment_mask=decomposition.environment_mask,
        reliable_environment_mask=decomposition.reliable_environment_mask,
        environment_completion_mask=environment_completion.pixel_mask,
        environment_dense_mask=environment_completion.dense_mask,
        environment_completion_depth=environment_completion.depth,
        environment_depthlab_mask=depthlab.completion_mask,
        environment_depthlab_raw_depth=depthlab.raw_depth,
        content_mask=vggt.content_mask,
        object_transform_world=registration.transform_world,
        object_transform_camera=registration.transform_camera,
        collision_proxy_vertices=registration.collision_vertices,
        collision_proxy_faces=registration.collision_faces,
        support_plane_valid=np.uint8(support_plane_valid),
        support_plane_index=np.int16(support_plane_index),
        support_plane_equation_world=support_plane_equation,
        support_plane_distance_threshold=np.float32(support_plane_threshold),
        source_labels=np.asarray(
            ["observed_environment", "completed_environment", "reconstructed_object"]
        ),
        body_labels=np.asarray(["environment", "object_0"]),
    )

    save_json(
        args.output_dir / "camera.json",
        {
            "intrinsic": vggt.intrinsic.tolist(),
            "extrinsic_world_to_camera": vggt.extrinsic.tolist(),
            "image_size_hw": list(segmentation.mask.shape),
        },
    )
    save_json(
        args.output_dir / "object_transform.json",
        {
            "object_id": "object_0",
            "mesh": "object_mesh.obj",
            "camera_from_object_sim3": registration.transform_camera.tolist(),
            "world_from_object_sim3": registration.transform_world.tolist(),
            "registration": registration.diagnostics,
        },
    )
    save_json(
        args.output_dir / "physics.json",
        {
            "object_id": "object_0",
            "assignment": "user_controlled",
            "collision_proxy": "object_collision_proxy.obj",
            "collision_proxy_frame": "object_local",
            "parameters": {
                "mass_kg": None,
                "contact_friction": None,
                "restitution": None,
            },
            "support": {
                "support_id": "principal_support",
                "resolved": support_plane_valid,
                "plane_equation_world": support_plane_equation.tolist(),
                "distance_threshold_scene_units": support_plane_threshold,
            },
        },
    )
    save_review(
        args.output_dir / "review.png",
        vggt.rgb,
        segmentation.overlay,
        environment_completion.inpainted_rgb,
        decomposition.environment_points,
        decomposition.environment_colors,
        scene_points,
        semantic_colors,
        args.seed,
    )
    save_component_review(
        args.output_dir / "components_review.png",
        decomposition.object_points,
        registration.aligned_points,
        decomposition.environment_points,
        environment_completion.points,
        args.seed,
    )

    object_depth_values = vggt.depth[
        segmentation.mask & np.isfinite(vggt.depth) & (vggt.depth > 0)
    ]
    object_reference_depth = (
        float(np.median(object_depth_values)) if len(object_depth_values) else None
    )
    aligned_depth_error = registration.diagnostics[
        "final_depth_median_absolute_error"
    ]
    relative_depth_error = (
        float(aligned_depth_error / object_reference_depth)
        if aligned_depth_error is not None and object_reference_depth
        else None
    )
    completion_fraction = environment_completion.diagnostics[
        "missing_environment_completion_fraction"
    ]
    content_surface_coverage = environment_completion.diagnostics[
        "content_surface_coverage"
    ]
    depthlab_positive_fraction = depthlab.diagnostics[
        "positive_completion_fraction"
    ]
    depthlab_known_depth_error = depthlab.diagnostics[
        "raw_known_depth_median_relative_error"
    ]
    depthlab_known_depth_p90_error = depthlab.diagnostics[
        "raw_known_depth_p90_relative_error"
    ]
    completion_seam_p90_error = environment_completion.diagnostics["depth_seam"][
        "p90_relative_error"
    ]
    support_diagnostics = registration.diagnostics["support_relation"]
    collision_diagnostics = registration.diagnostics["collision_proxy"]
    physical_proxy_verified = bool(
        support_diagnostics.get("resolved") and collision_diagnostics.get("safe")
    )
    physical_proxy_status = " / ".join(
        [
            str(support_diagnostics.get("status")),
            str(collision_diagnostics.get("status")),
        ]
    )
    quality_gate = {
        "passed": bool(
            registration.diagnostics["final_mask_iou"] >= 0.65
            and relative_depth_error is not None
            and relative_depth_error <= 0.03
            and completion_fraction >= 0.80
            and content_surface_coverage >= 0.95
            and depthlab_positive_fraction >= 0.99
            and depthlab_known_depth_error <= 0.08
            and completion_seam_p90_error <= 0.10
            and physical_proxy_verified
        ),
        "thresholds": {
            "minimum_aligned_mask_iou": 0.65,
            "maximum_relative_depth_error": 0.03,
            "minimum_environment_completion_fraction": 0.80,
            "minimum_environment_surface_coverage": 0.95,
            "minimum_depthlab_positive_completion_fraction": 0.99,
            "maximum_depthlab_known_depth_error": 0.08,
            "maximum_environment_depth_seam_p90_error": 0.10,
            "require_safe_physical_proxy": True,
        },
        "measurements": {
            "aligned_mask_iou": registration.diagnostics["final_mask_iou"],
            "relative_depth_error": relative_depth_error,
            "environment_completion_fraction": completion_fraction,
            "environment_surface_coverage": content_surface_coverage,
            "depthlab_positive_completion_fraction": depthlab_positive_fraction,
            "depthlab_known_depth_error": depthlab_known_depth_error,
            "depthlab_known_depth_p90_error": depthlab_known_depth_p90_error,
            "environment_depth_seam_p90_error": completion_seam_p90_error,
            "physical_proxy_verified": physical_proxy_verified,
            "physical_proxy_status": physical_proxy_status,
            "collision_proxy_translation_fraction": collision_diagnostics.get(
                "translation_fraction_of_object_extent"
            ),
        },
    }

    report = {
        "schema": "phycontext.image_to_scene.v1",
        "input_image": str(args.image.resolve()),
        "device": str(device),
        "seed": args.seed,
        "confidence_quantile": args.confidence_quantile,
        "confidence_threshold": decomposition.confidence_threshold,
        "object_confidence_threshold": decomposition.object_confidence_threshold,
        "point_counts": {
            "observed_object": int(len(decomposition.object_points)),
            "observed_environment": int(len(decomposition.environment_points)),
            "completed_environment": int(len(environment_completion.points)),
            "dense_environment": int(len(environment_dense_points)),
            "reconstructed_object": int(len(registration.aligned_points)),
            "scene": int(len(scene_points)),
            "omitted_mixed_boundary": decomposition.omitted_boundary_points,
        },
        "environment": {
            "planes": environment_completion.planes,
            "rgb_inpainting": {
                "model": "big-lama",
                "mask_expansion_pixels": lama.expansion_pixels,
            },
            "depth_completion": depthlab.diagnostics,
            **environment_completion.diagnostics,
        },
        "registration": registration.diagnostics,
        "sam2": {
            "source": segmentation_source,
            "selected_candidate_index": segmentation.selected_index,
            "top_candidates": segmentation.candidates,
        },
        "object_reconstruction": {
            **object_reconstruction,
            "reused": object_mesh_reused,
            "object_crop_bbox_xyxy": list(crop.bbox_xyxy),
        },
        "timings": timings,
        "checkpoints": {
            "vggt": {"path": str(args.vggt_checkpoint), "sha256": _sha256(args.vggt_checkpoint)},
            "lama": {"path": str(args.lama_checkpoint), "sha256": _sha256(args.lama_checkpoint)},
            "depthlab": {
                "repository": str(args.depthlab_repository),
                "repository_revision": depthlab.diagnostics["repository_revision"],
                "checkpoint_root": str(args.depthlab_checkpoints),
                "model_sources": depthlab.diagnostics["model_sources"],
            },
            "sam2": (
                {"path": str(args.sam_checkpoint), "sha256": _sha256(args.sam_checkpoint)}
                if args.object_mask is None
                else None
            ),
        },
        "gt_mask_iou": _mask_iou(segmentation.mask, gt_mask) if gt_mask is not None else None,
        "quality_gate": quality_gate,
        "source_semantics": {
            "0": "observed VGGT environment",
            "1": "DepthLab completed hidden environment",
            "2": "aligned InstantMesh object",
        },
    }
    save_json(args.output_dir / "report.json", report)

    try:
        from build_interactive_html import build_html, build_scene

        viewer = build_html([build_scene(args.output_dir, args.image.stem, args.seed)])
        (args.output_dir / "viewer.html").write_text(viewer, encoding="utf-8")
    except Exception as error:
        report["viewer_error"] = str(error)
    save_json(args.output_dir / "report.json", report)

    artifact_names = [
        "input_preprocessed.png",
        "object_mask.png",
        "aligned_object_mask.png",
        "aligned_object_depth.npy",
        "environment_observed.ply",
        "environment_completed.ply",
        "environment_dense.ply",
        "environment_inpainted.png",
        "environment_inpainting_mask.png",
        "environment_depthlab_rgb.png",
        "environment_depthlab_known_depth.npy",
        "environment_depthlab_mask.png",
        "environment_depthlab_raw_depth.npy",
        "depthlab_manifest.json",
        "depthlab.log",
        "environment_completion_mask.png",
        "environment_completion_depth.npy",
        "object_visible.ply",
        "object_mesh.obj",
        "object_mesh.mtl",
        "object_mesh.png",
        "object_mesh_manifest.json",
        "object_mesh_aligned.ply",
        "object_collision_proxy.obj",
        "object_transform.json",
        "camera.json",
        "physics.json",
        "scene.ply",
        "scene_sources.ply",
        "scene.npz",
        "report.json",
        "viewer.html",
    ]
    scene_manifest = {
        "schema": "phycontext.structured_scene_package.v1",
        "input": {
            "path": str(args.image.resolve()),
            "sha256": _sha256(args.image),
            "preprocessed_image": "input_preprocessed.png",
            "image_size_hw": list(segmentation.mask.shape),
            "object_selection": {
                "source": segmentation_source,
                "mask_path": str(args.object_mask.resolve()) if args.object_mask else None,
                "mask_sha256": _sha256(args.object_mask) if args.object_mask else None,
            },
        },
        "coordinate_system": {
            "world_frame": "VGGT world frame",
            "camera_extrinsic_convention": "world_to_camera",
            "units": "vggt_scene_units",
            "si_scale_status": "requires an external metric calibration for monocular RGB",
        },
        "camera": "camera.json",
        "environment": {
            "observed_geometry": "environment_observed.ply",
            "completed_hidden_geometry": "environment_completed.ply",
            "dense_camera_frustum_geometry": "environment_dense.ply",
            "completion_scope": "camera-visible content after removing object_0, including low-confidence holes",
            "completion_method": "big_lama_object_removal_then_depthlab_known_depth_completion",
        },
        "objects": [
            {
                "object_id": "object_0",
                "body_id": 1,
                "visible_surface": "object_visible.ply",
                "complete_mesh": "object_mesh.obj",
                "mesh_generation": "object_mesh_manifest.json",
                "aligned_surface": "object_mesh_aligned.ply",
                "collision_proxy": "object_collision_proxy.obj",
                "transform": "object_transform.json",
                "physics": "physics.json",
                "pixel_mask": "object_mask.png",
                "aligned_pixel_mask": "aligned_object_mask.png",
                "aligned_depth": "aligned_object_depth.npy",
            }
        ],
        "pixel_correspondence": {
            "archive": "scene.npz",
            "fields": [
                "point_map_world",
                "depth",
                "confidence",
                "pixel_body_id",
                "object_mask",
                "aligned_object_mask",
                "aligned_object_depth",
                "environment_mask",
                "reliable_environment_mask",
                "environment_completion_mask",
                "environment_dense_mask",
                "environment_completion_depth",
                "environment_depthlab_mask",
                "environment_depthlab_raw_depth",
                "content_mask",
            ],
        },
        "scene_archive": "scene.npz",
        "quality_report": "report.json",
        "quality_gate": quality_gate,
        "artifacts": _artifact_records(args.output_dir, artifact_names),
    }
    save_json(args.output_dir / "scene_manifest.json", scene_manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
