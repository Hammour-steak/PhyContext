#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "xyz",
    "rgb",
    "source",
    "body_id",
    "intrinsic",
    "extrinsic",
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
    "object_transform_world",
    "object_transform_camera",
    "collision_proxy_vertices",
    "collision_proxy_faces",
    "support_plane_valid",
    "support_plane_index",
    "support_plane_equation_world",
    "support_plane_distance_threshold",
    "source_labels",
    "body_labels",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(scene_dir: Path) -> dict:
    scene_dir = scene_dir.resolve()
    errors: list[str] = []
    quality_gate_passed = False
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        return {
            "passed": False,
            "integrity_passed": False,
            "quality_gate_passed": False,
            "scene_dir": str(scene_dir),
            "errors": ["missing scene_manifest.json"],
        }

    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "phycontext.structured_scene_package.v1":
        errors.append("unexpected manifest schema")

    artifacts = manifest.get("artifacts", {})
    for name, expected in artifacts.items():
        path = (scene_dir / name).resolve()
        if path.parent != scene_dir:
            errors.append(f"artifact escapes scene directory: {name}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"size mismatch: {name}")
        if _sha256(path) != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {name}")

    report_path = scene_dir / manifest.get("quality_report", "report.json")
    if not report_path.is_file():
        errors.append("missing quality report")
    else:
        report = _load_json(report_path)
        quality_gate_passed = bool(report.get("quality_gate", {}).get("passed"))

    physics_path = scene_dir / "physics.json"
    if not physics_path.is_file():
        errors.append("missing physics.json")
    else:
        physics = _load_json(physics_path)
        proxy_name = physics.get("collision_proxy")
        if proxy_name != "object_collision_proxy.obj" or not (scene_dir / proxy_name).is_file():
            errors.append("physics.json does not bind the exported collision proxy")

    archive_path = scene_dir / manifest.get("scene_archive", "scene.npz")
    if not archive_path.is_file():
        errors.append("missing scene archive")
    else:
        with np.load(archive_path, allow_pickle=False) as archive:
            missing = sorted(REQUIRED_ARRAYS - set(archive.files))
            if missing:
                errors.append(f"scene.npz missing fields: {', '.join(missing)}")
            else:
                xyz, rgb = archive["xyz"], archive["rgb"]
                source, body_id = archive["source"], archive["body_id"]
                depth = archive["depth"]
                if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                    errors.append("xyz must be finite N x 3")
                if rgb.shape != xyz.shape or rgb.dtype != np.uint8:
                    errors.append("rgb must be uint8 and match xyz")
                if source.shape != (len(xyz),) or body_id.shape != (len(xyz),):
                    errors.append("source/body_id length does not match xyz")
                if not set(np.unique(source)).issubset({0, 1, 2}):
                    errors.append("source contains unknown labels")
                if archive["point_map_world"].shape[:2] != depth.shape:
                    errors.append("point_map_world and depth image shapes differ")
                for name in (
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
                ):
                    if archive[name].shape != depth.shape:
                        errors.append(f"{name} does not match depth image shape")
                environment = archive["environment_mask"].astype(bool)
                reliable_environment = archive["reliable_environment_mask"].astype(bool)
                completed = archive["environment_completion_mask"].astype(bool)
                dense = archive["environment_dense_mask"].astype(bool)
                content = archive["content_mask"].astype(bool)
                completion_depth = archive["environment_completion_depth"]
                depthlab_mask = archive["environment_depthlab_mask"].astype(bool)
                depthlab_raw_depth = archive["environment_depthlab_raw_depth"]
                if np.any(environment & completed):
                    errors.append("completion replaces observed environment pixels")
                if not np.array_equal(dense, environment | completed):
                    errors.append("dense environment mask is not observed union completed")
                if np.any(dense & ~content):
                    errors.append("dense environment includes preprocessing padding")
                if np.any(depthlab_mask & ~content):
                    errors.append("DepthLab completion mask includes preprocessing padding")
                if np.any(completed & ~depthlab_mask):
                    errors.append("environment completion escapes the DepthLab mask")
                if np.any(reliable_environment & ~environment):
                    errors.append("reliable environment is not a subset of observed environment")
                if np.any(completion_depth[completed] <= 0) or not np.isfinite(
                    completion_depth[completed]
                ).all():
                    errors.append("completed environment depth must be finite and positive")
                if np.any(completion_depth[~completed] != 0):
                    errors.append("completion depth is nonzero outside the completion mask")
                if np.any(depthlab_raw_depth[~content] != 0):
                    errors.append("DepthLab raw depth is nonzero in preprocessing padding")
                if np.any(depthlab_raw_depth[depthlab_mask] <= 0) or not np.isfinite(
                    depthlab_raw_depth[depthlab_mask]
                ).all():
                    errors.append("DepthLab masked depth must be finite and positive")
                for name in ("object_transform_world", "object_transform_camera"):
                    value = archive[name]
                    if value.shape != (4, 4) or not np.isfinite(value).all():
                        errors.append(f"{name} must be a finite 4 x 4 transform")
                vertices = archive["collision_proxy_vertices"]
                faces = archive["collision_proxy_faces"]
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    errors.append("collision proxy vertices must be finite N x 3")
                if faces.ndim != 2 or faces.shape[1] != 3:
                    errors.append("collision proxy faces must be M x 3")
                elif len(vertices) and (faces.min(initial=0) < 0 or faces.max(initial=0) >= len(vertices)):
                    errors.append("collision proxy face index is out of bounds")
                if bool(archive["support_plane_valid"]):
                    equation = archive["support_plane_equation_world"]
                    threshold = float(archive["support_plane_distance_threshold"])
                    if equation.shape != (4,) or not np.isfinite(equation).all():
                        errors.append("resolved support plane must contain a finite equation")
                    if not np.isfinite(threshold) or threshold <= 0:
                        errors.append("resolved support plane must contain a positive threshold")

    integrity_passed = not errors
    return {
        "passed": integrity_passed and quality_gate_passed,
        "integrity_passed": integrity_passed,
        "quality_gate_passed": quality_gate_passed,
        "scene_dir": str(scene_dir),
        "artifact_count": len(artifacts),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one immutable PhyContext scene package")
    parser.add_argument("scene_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.scene_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
