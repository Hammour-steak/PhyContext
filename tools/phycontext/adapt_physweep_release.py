#!/usr/bin/env python3
"""Adapt a PhysSweep one-object release into PhysContext training inputs.

This adapter is deliberately a consumer-side, derived-data operation.  It reads
the immutable release layout (base samples plus twelve one-factor sweeps per
group) and writes a separate PhysContext dataset containing:

* one canonical first frame and one simulation-proxy scene per group;
* one schema-v4 training record and one fixed-material-point trajectory per
  sample; and
* hash-bound manifests and an audit report.

The release does not publish the rendered dynamic-object mesh.  Dynamic surface
points therefore come from the exact collision proxy used by the simulator, and
the scene metadata names that limitation explicitly.  No visual mesh is
invented and no source file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import trimesh

from point_trajectory import (
    POINT_COUNT,
    POINT_TRACK_DEFINITION,
    POINT_TRAJECTORY_SCHEMA,
    POINT_VISIBILITY_DEFINITION,
    validate_point_trajectory,
)
from schema import MANIFEST_SCHEMA, SAMPLE_SCHEMA, SWEEP_AXES, validate_manifest


ADAPTER_SCHEMA = "phycontext.physweep_release_adapter.v1"
SCENE_SCHEMA = "phycontext.release_physics_proxy_scene.v1"
POINT_MANIFEST_SCHEMA = "physweep.point_trajectory_manifest.v1"
ENVIRONMENT_POINT_COUNT = 8192
IMAGE_SIZE_PX = (1280, 720)
BASE_LEVEL_INDEX = 2
LEVEL_COUNT = 5


@dataclass(frozen=True)
class SurfaceComponent:
    component_id: str
    vertices_world_m: np.ndarray
    faces: np.ndarray
    friction: float
    restitution: float

    @property
    def area_m2(self) -> float:
        triangles = self.vertices_world_m[self.faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="PhysSweep repository/data root; every emitted path is relative to it.",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("outputs/one_object"),
        help="Immutable release directory, relative to --dataset-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/physweep_training"),
        help="New derived dataset directory, relative to --dataset-root.",
    )
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a completed output from this adapter; unrelated directories are refused.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def _relative_input(root: Path, value: Path, context: str) -> Path:
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{context} must be relative to the dataset root: {value}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{context} escapes the dataset root: {value}")
    return path


def resolve_roots(dataset_root: Path, release_root: Path, output_root: Path) -> tuple[Path, Path, Path]:
    dataset = dataset_root.resolve()
    release = _relative_input(dataset, release_root, "release root")
    output = _relative_input(dataset, output_root, "output root")
    if not release.is_dir():
        raise FileNotFoundError(f"release root does not exist: {release}")
    if output == release or output.is_relative_to(release) or release.is_relative_to(output):
        raise ValueError("output root and release root must be disjoint")
    return dataset, release, output


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def quaternion_matrix_wxyz(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError("wxyz quaternion must contain four values")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion is invalid")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_matrix_xyzw(quaternion: Any) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return quaternion_matrix_wxyz([w, x, y, z])


def euler_xyz_matrix_degrees(euler: Any) -> np.ndarray:
    x, y, z = np.deg2rad(np.asarray(euler, dtype=np.float64))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def camera_contract(camera: dict[str, Any], image_size_px: tuple[int, int] = IMAGE_SIZE_PX) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(camera["position_m"], dtype=np.float64)
    target = np.asarray(camera["target_m"], dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1.0e-8:
        raise ValueError("camera view direction is parallel to world up")
    right /= right_norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    world_to_camera_rotation = np.stack((right, up, forward), axis=0)
    if not np.isclose(np.linalg.det(world_to_camera_rotation), -1.0, atol=1.0e-7):
        raise ValueError("camera_right_up_forward basis must be left-handed")
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :3] = world_to_camera_rotation
    camera_from_world[:3, 3] = -world_to_camera_rotation @ position

    width, height = image_size_px
    focal = float(camera["focal_length_mm"]) / float(camera["sensor_width_mm"]) * width
    intrinsics = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return camera_from_world, intrinsics


def _apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nj->ni", rotation, points, optimize=True) + translation


def _mesh_arrays(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh must be triangular")
    if not len(vertices) or not len(faces):
        raise ValueError("surface mesh is empty")
    return vertices, faces


def _primitive_mesh(record: dict[str, Any]) -> trimesh.Trimesh | None:
    spec = _primitive_spec(record)
    if spec is None:
        return None
    shape, dimensions = spec
    if shape == "box":
        return trimesh.creation.box(extents=dimensions)
    if shape == "sphere":
        (radius,) = dimensions
        return trimesh.creation.icosphere(subdivisions=3, radius=radius)
    radius, height = dimensions
    return trimesh.creation.cylinder(radius=radius, height=height, sections=48)


def _primitive_spec(record: dict[str, Any]) -> tuple[str, tuple[float, ...]] | None:
    shape = str(record.get("primitive", record.get("shape", record.get("type", "")))).lower()
    if shape in {"box", "cuboid"}:
        if "half_extents_m" in record:
            extents = 2.0 * np.asarray(record["half_extents_m"], dtype=np.float64)
        elif "size_m" in record:
            extents = np.asarray(record["size_m"], dtype=np.float64)
        else:
            return None
        if extents.shape != (3,) or np.any(extents <= 0.0):
            raise ValueError("box dimensions must be positive xyz values")
        return "box", tuple(float(value) for value in extents)
    if shape == "sphere":
        radius = float(record.get("radius_m", 0.0))
        return ("sphere", (radius,)) if radius > 0.0 else None
    if shape == "cylinder":
        if "radius_m" in record:
            radius = float(record["radius_m"])
            height = float(record.get("length_m", record.get("height_m", 0.0)))
        elif "size_m" in record:
            size = np.asarray(record["size_m"], dtype=np.float64)
            if size.shape != (3,):
                raise ValueError("cylinder size_m must contain xyz dimensions")
            radius = float(max(size[0], size[1]) * 0.5)
            height = float(size[2])
        else:
            return None
        return ("cylinder", (radius, height)) if radius > 0.0 and height > 0.0 else None
    return None


def _analytic_surface_area(record: dict[str, Any]) -> float:
    spec = _primitive_spec(record)
    if spec is None:
        raise ValueError(f"unsupported analytic primitive: {record}")
    shape, dimensions = spec
    if shape == "box":
        x, y, z = dimensions
        return 2.0 * (x * y + x * z + y * z)
    if shape == "sphere":
        (radius,) = dimensions
        return 4.0 * math.pi * radius * radius
    radius, height = dimensions
    return 2.0 * math.pi * radius * (height + radius)


def _sample_analytic_primitive(
    record: dict[str, Any],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    spec = _primitive_spec(record)
    if spec is None:
        raise ValueError(f"unsupported analytic primitive: {record}")
    shape, dimensions = spec
    if shape == "sphere":
        (radius,) = dimensions
        z = rng.uniform(-1.0, 1.0, size=count)
        angle = rng.uniform(0.0, 2.0 * math.pi, size=count)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        normals = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))
        return radius * normals, normals
    if shape == "box":
        x, y, z = dimensions
        pair_areas = np.asarray([y * z, y * z, x * z, x * z, x * y, x * y])
        faces = rng.choice(6, size=count, p=pair_areas / pair_areas.sum())
        points = rng.uniform(-0.5, 0.5, size=(count, 3)) * np.asarray([x, y, z])
        normals = np.zeros((count, 3), dtype=np.float64)
        for face, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))):
            selected = faces == face
            points[selected, axis] = sign * dimensions[axis] * 0.5
            normals[selected, axis] = sign
        return points, normals

    radius, height = dimensions
    lateral_area = 2.0 * math.pi * radius * height
    cap_area = math.pi * radius * radius
    regions = rng.choice(
        3,
        size=count,
        p=np.asarray([lateral_area, cap_area, cap_area]) / (lateral_area + 2.0 * cap_area),
    )
    angle = rng.uniform(0.0, 2.0 * math.pi, size=count)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    points = np.zeros((count, 3), dtype=np.float64)
    normals = np.zeros((count, 3), dtype=np.float64)
    lateral = regions == 0
    points[lateral, 0] = radius * cos_angle[lateral]
    points[lateral, 1] = radius * sin_angle[lateral]
    points[lateral, 2] = rng.uniform(-height * 0.5, height * 0.5, size=int(lateral.sum()))
    normals[lateral, 0] = cos_angle[lateral]
    normals[lateral, 1] = sin_angle[lateral]
    for region, sign in ((1, -1.0), (2, 1.0)):
        selected = regions == region
        radial = radius * np.sqrt(rng.random(int(selected.sum())))
        points[selected, 0] = radial * cos_angle[selected]
        points[selected, 1] = radial * sin_angle[selected]
        points[selected, 2] = sign * height * 0.5
        normals[selected, 2] = sign
    return points, normals


def _surface_sample(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-14)
    if not np.any(valid):
        raise ValueError("mesh has no finite non-degenerate triangles")
    triangles = triangles[valid]
    cross = cross[valid]
    double_area = double_area[valid]
    selected = rng.choice(len(triangles), size=count, p=double_area / double_area.sum())
    chosen = triangles[selected]
    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = (
        (1.0 - root_u)[:, None] * chosen[:, 0]
        + (root_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (root_u * v)[:, None] * chosen[:, 2]
    )
    normals = cross[selected]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _allocate_counts(areas: np.ndarray, total: int, minimum: int = 0) -> np.ndarray:
    if total <= 0 or len(areas) == 0 or np.any(areas <= 0.0):
        raise ValueError("surface allocation requires positive areas and budget")
    minimum = min(minimum, total // len(areas))
    remaining = total - minimum * len(areas)
    raw = areas / areas.sum() * remaining
    counts = np.floor(raw).astype(np.int64) + minimum
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def sample_dynamic_proxy(proxy: dict[str, Any], rng: np.random.Generator, count: int = POINT_COUNT) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    proxy_type = str(proxy.get("type", ""))
    if proxy_type == "compound":
        records = list(proxy.get("colliders", []))
        if not records:
            raise ValueError("compound collision proxy has no colliders")
    else:
        records = [proxy]
    records_with_transforms: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    ids: list[str] = []
    for index, record in enumerate(records):
        if _primitive_spec(record) is None:
            raise ValueError(f"unsupported dynamic collision proxy component: {record}")
        rotation = euler_xyz_matrix_degrees(record.get("rotation_euler_degrees", [0.0, 0.0, 0.0]))
        translation = np.asarray(record.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        records_with_transforms.append((record, rotation, translation))
        ids.append(str(record.get("id", f"proxy_{index:02d}")))
    areas = np.asarray([_analytic_surface_area(record) for record in records], dtype=np.float64)
    counts = _allocate_counts(areas, count, minimum=1)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for (record, rotation, translation), component_count in zip(
        records_with_transforms, counts, strict=True
    ):
        sampled_points, sampled_normals = _sample_analytic_primitive(
            record, int(component_count), rng
        )
        points.append(_apply_transform(sampled_points, rotation, translation))
        normals.append(np.einsum("ij,nj->ni", rotation, sampled_normals, optimize=True))
    return (
        np.concatenate(points, axis=0),
        np.concatenate(normals, axis=0),
        {
            "proxy_type": proxy_type,
            "component_ids": ids,
            "component_point_counts": counts.tolist(),
            "point_count": int(count),
            "sampling": "exact_analytic_component_surfaces_area_weighted",
        },
    )


def _fixture_material(fixture: dict[str, Any], component_kind: str) -> tuple[float, float]:
    physical = fixture.get("physical", {})
    candidates: list[dict[str, Any]] = []
    if component_kind == "static_prop":
        candidates.append(physical.get("static_dynamics", {}).get("static_prop", {}))
    elif component_kind == "mesh":
        candidates.append(physical.get("fixture", {}).get("mesh_material", {}))
    else:
        candidates.append(physical.get("fixture", {}).get("analytic_material", {}))
    candidates.extend(
        [
            physical.get("fixture", {}).get("material", {}),
            physical.get("support_dynamics", {}),
            physical.get("support", {}).get("dynamics", {}),
            physical.get("static_dynamics", {}).get("support", {}),
            physical.get("static_dynamics", {}).get("static_prop", {}),
            physical.get("static_dynamics", {}).get("ground", {}),
        ]
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        friction = candidate.get("contact_friction", candidate.get("lateral_friction"))
        restitution = candidate.get("contact_restitution", candidate.get("restitution"))
        if friction is not None and restitution is not None:
            return float(friction), float(restitution)
    raise ValueError("fixture does not expose friction and restitution for its static geometry")


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene contains no triangle geometry: {path}")
        # Bake every scene-graph instance transform before the release binding
        # transform. Concatenating raw geometry would silently drop node poses.
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh payload: {path}")
    return loaded


def fixture_components(fixture: dict[str, Any], release_root: Path) -> list[SurfaceComponent]:
    components: list[SurfaceComponent] = []
    seen_mesh_bindings: set[tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = set()

    physical = fixture.get("physical", {})
    prop_binding = physical.get("static_prop_binding")
    prop_record = physical.get("static_prop_record")
    if (prop_binding is None) != (prop_record is None):
        raise ValueError("static prop record and placement binding must appear together")
    if prop_binding is not None:
        colliders = prop_record.get("proxy", {}).get("colliders", [])
        if not colliders:
            raise ValueError("bound static prop has no collision-proxy components")
        parent_position = np.asarray(prop_binding["position_m"], dtype=np.float64)
        parent_rotation = euler_xyz_matrix_degrees([0.0, 0.0, float(prop_binding["yaw_degrees"])])
        friction, restitution = _fixture_material(fixture, "static_prop")
        for index, collider in enumerate(colliders):
            mesh = _primitive_mesh(collider)
            if mesh is None:
                raise ValueError(f"unsupported bound static-prop collider: {collider}")
            vertices, faces = _mesh_arrays(mesh)
            local_rotation = euler_xyz_matrix_degrees(
                collider.get("rotation_euler_degrees", [0.0, 0.0, 0.0])
            )
            local_position = np.asarray(collider.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)
            local_vertices = _apply_transform(vertices, local_rotation, local_position)
            components.append(
                SurfaceComponent(
                    f"physical/static_prop/{collider.get('id', index)}",
                    _apply_transform(local_vertices, parent_rotation, parent_position),
                    faces,
                    friction,
                    restitution,
                )
            )

    def walk(node: Any, context: dict[str, Any], tokens: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, context, (*tokens, str(index)))
            return
        if not isinstance(node, dict):
            return
        current = dict(context)
        if "base_position_m" in node:
            current["position"] = node["base_position_m"]
        elif "position_m" in node:
            current["position"] = node["position_m"]
        if "base_orientation_quaternion_xyzw" in node:
            current["quaternion"] = node["base_orientation_quaternion_xyzw"]
            current.pop("euler", None)
        elif "orientation_quaternion_xyzw" in node:
            current["quaternion"] = node["orientation_quaternion_xyzw"]
            current.pop("euler", None)
        elif "rotation_euler_degrees" in node:
            current["euler"] = node["rotation_euler_degrees"]
            current.pop("quaternion", None)
        if "mesh_scale" in node:
            current["scale"] = node["mesh_scale"]
        elif "scale" in node and isinstance(node["scale"], list):
            current["scale"] = node["scale"]

        if node.get("collision_enabled") is False or node.get("visible") is False:
            return
        primitive = _primitive_mesh(node)
        if primitive is not None:
            vertices, faces = _mesh_arrays(primitive)
            rotation = (
                quaternion_matrix_xyzw(current["quaternion"])
                if "quaternion" in current
                else euler_xyz_matrix_degrees(current.get("euler", [0.0, 0.0, 0.0]))
            )
            position = np.asarray(current.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            friction, restitution = _fixture_material(fixture, "analytic")
            components.append(
                SurfaceComponent(
                    "/".join(tokens) or f"analytic_{len(components)}",
                    _apply_transform(vertices, rotation, position),
                    faces,
                    friction,
                    restitution,
                )
            )
            return

        mesh_value = node.get("path")
        if isinstance(mesh_value, str) and Path(mesh_value).suffix.lower() in {".obj", ".ply", ".stl"}:
            relative = Path(mesh_value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"fixture mesh path is not release-relative: {mesh_value}")
            path = (release_root / "base" / relative).resolve()
            base_root = (release_root / "base").resolve()
            if not path.is_relative_to(base_root) or not path.is_file():
                raise FileNotFoundError(f"fixture mesh is missing: {path}")
            expected_hash = node.get("sha256")
            if expected_hash and sha256_file(path) != expected_hash:
                raise ValueError(f"fixture mesh hash mismatch: {path}")
            position = tuple(float(value) for value in current.get("position", [0.0, 0.0, 0.0]))
            scale = tuple(float(value) for value in current.get("scale", [1.0, 1.0, 1.0]))
            quaternion = tuple(float(value) for value in current.get("quaternion", [0.0, 0.0, 0.0, 1.0]))
            binding = (str(relative), position, scale, quaternion)
            if binding not in seen_mesh_bindings:
                seen_mesh_bindings.add(binding)
                mesh = _load_trimesh(path)
                vertices, faces = _mesh_arrays(mesh)
                vertices = vertices * np.asarray(scale, dtype=np.float64)
                rotation = (
                    quaternion_matrix_xyzw(quaternion)
                    if "quaternion" in current
                    else euler_xyz_matrix_degrees(current.get("euler", [0.0, 0.0, 0.0]))
                )
                friction, restitution = _fixture_material(fixture, "mesh")
                components.append(
                    SurfaceComponent(
                        "/".join(tokens),
                        _apply_transform(vertices, rotation, np.asarray(position)),
                        faces,
                        friction,
                        restitution,
                    )
                )
            return

        for key, value in node.items():
            if key in {"static_prop_binding", "static_prop_record"}:
                continue
            walk(value, current, (*tokens, str(key)))

    walk(physical, {}, ("physical",))
    if not components:
        raise ValueError("fixture contains no supported visible static geometry")
    if any(not np.isfinite(component.area_m2) or component.area_m2 <= 0.0 for component in components):
        raise ValueError("fixture contains an invalid surface component")
    return components


def _project_camera(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = points_camera_m.copy()
    projected[..., 1] *= -1.0
    homogeneous = np.einsum("ij,...j->...i", intrinsics, projected, optimize=True)
    tracks = np.zeros(homogeneous.shape[:-1] + (2,), dtype=np.float64)
    np.divide(homogeneous[..., :2], homogeneous[..., 2:3], out=tracks, where=np.abs(homogeneous[..., 2:3]) > 1.0e-12)
    return tracks


def _camera_points(points_world_m: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,...j->...i",
        camera_from_world[:3, :3],
        points_world_m,
        optimize=True,
    ) + camera_from_world[:3, 3]


def sample_environment(
    components: list[SurfaceComponent],
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    clip_start_m: float,
    clip_end_m: float,
    rng: np.random.Generator,
    count: int = ENVIRONMENT_POINT_COUNT,
    image_size_px: tuple[int, int] = IMAGE_SIZE_PX,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    areas = np.asarray([component.area_m2 for component in components], dtype=np.float64)
    width, height = image_size_px
    candidate_budget = max(65536, count * 8)
    retained: dict[str, np.ndarray] | None = None
    attempts = 0
    while candidate_budget <= 1048576:
        attempts += 1
        allocations = _allocate_counts(areas, candidate_budget, minimum=256)
        xyz_world: list[np.ndarray] = []
        normal_world: list[np.ndarray] = []
        friction: list[np.ndarray] = []
        restitution: list[np.ndarray] = []
        part_id: list[np.ndarray] = []
        for component_index, (component, component_count) in enumerate(zip(components, allocations, strict=True)):
            points, normals = _surface_sample(
                component.vertices_world_m,
                component.faces,
                int(component_count),
                rng,
            )
            xyz_world.append(points)
            normal_world.append(normals)
            friction.append(np.full((len(points), 1), component.friction, dtype=np.float64))
            restitution.append(np.full((len(points), 1), component.restitution, dtype=np.float64))
            part_id.append(np.full(len(points), component_index, dtype=np.int64))
        world = np.concatenate(xyz_world)
        normals_world = np.concatenate(normal_world)
        camera = _camera_points(world, camera_from_world)
        tracks = _project_camera(camera, intrinsics)
        rounded_x = np.rint(tracks[:, 0]).astype(np.int64)
        rounded_y = np.rint(tracks[:, 1]).astype(np.int64)
        valid = (
            np.isfinite(tracks).all(axis=1)
            & np.isfinite(camera).all(axis=1)
            & (camera[:, 2] > clip_start_m)
            & (camera[:, 2] < clip_end_m)
            & (rounded_x >= 0)
            & (rounded_x < width)
            & (rounded_y >= 0)
            & (rounded_y < height)
        )
        candidate_indices = np.flatnonzero(valid)
        if len(candidate_indices):
            pixel = rounded_y[candidate_indices] * width + rounded_x[candidate_indices]
            order = np.lexsort((camera[candidate_indices, 2], pixel))
            sorted_indices = candidate_indices[order]
            sorted_pixels = pixel[order]
            first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
            visible_indices = sorted_indices[first]
        else:
            visible_indices = candidate_indices
        if len(visible_indices) >= count:
            all_part_ids = np.concatenate(part_id)
            visible_part_ids = all_part_ids[visible_indices]
            reserved: list[np.ndarray] = []
            for component_index in sorted(set(visible_part_ids.tolist())):
                choices = visible_indices[visible_part_ids == component_index]
                take = min(8, len(choices))
                reserved.append(rng.choice(choices, size=take, replace=False))
            reserved_indices = np.unique(np.concatenate(reserved)) if reserved else np.empty(0, np.int64)
            pool = np.setdiff1d(visible_indices, reserved_indices, assume_unique=False)
            selected = np.concatenate(
                [reserved_indices, rng.choice(pool, size=count - len(reserved_indices), replace=False)]
            )
            all_friction = np.concatenate(friction)
            all_restitution = np.concatenate(restitution)
            retained = {
                "xyz_world": world[selected],
                "xyz_camera": camera[selected],
                "normal_camera": np.einsum(
                    "ij,nj->ni", camera_from_world[:3, :3], normals_world[selected], optimize=True
                ),
                "friction": all_friction[selected],
                "restitution": all_restitution[selected],
            }
            break
        candidate_budget *= 2
    if retained is None:
        raise ValueError(
            f"fixture has fewer than {count} z-buffer-visible surface pixels after {attempts} attempts"
        )
    return retained, {
        "component_count": len(components),
        "visible_component_count": len(set(visible_part_ids.tolist())),
        "candidate_budget": int(candidate_budget),
        "zbuffer_visible_candidates": int(len(visible_indices)),
        "output_point_count": int(count),
        "selection": "full_resolution_global_zbuffer_then_surface_sample_with_component_retention",
    }


def _find_object(metadata: dict[str, Any], object_id: str) -> dict[str, Any]:
    matches = [item for item in metadata["physics"]["objects"] if str(item.get("object_id")) == object_id]
    if len(matches) != 1:
        raise ValueError(f"metadata must contain exactly one object {object_id}")
    return matches[0]


def _validate_release_group_sample(
    base_metadata: dict[str, Any],
    metadata: dict[str, Any],
    descriptor: dict[str, Any],
    group: dict[str, Any],
    object_id: str,
    is_base: bool,
) -> None:
    group_id = str(group["group_id"])
    expected_kind = "base" if is_base else "sweep"
    if (
        metadata.get("scene_id") != descriptor.get("scene_id")
        or metadata.get("group_id") != group_id
        or metadata.get("family") != group.get("family")
        or metadata.get("sample_kind") != expected_kind
    ):
        raise ValueError(f"release sample identity binding differs: {descriptor.get('scene_id')}")
    for key in ("visual", "text", "semantics", "seed"):
        if metadata.get(key) != base_metadata.get(key):
            raise ValueError(f"{key} changes inside release group {group_id}")
    for key in ("backend", "fixture", "solver", "time", "world"):
        if metadata["physics"].get(key) != base_metadata["physics"].get(key):
            raise ValueError(f"physics.{key} changes inside release group {group_id}")
    if metadata["physics"]["time"] != {
        "duration_s": 4.0,
        "output_fps": 24,
        "simulation_hz": metadata["physics"]["time"]["simulation_hz"],
    }:
        raise ValueError("release sample must use the 4 s, 24 fps closed-endpoint protocol")

    base_object = _find_object(base_metadata, object_id)
    target_object = _find_object(metadata, object_id)
    stable_object_fields = (set(base_object) | set(target_object)) - {
        "material",
        "inertia_diagonal_kg_m2",
    }
    for key in stable_object_fields:
        if target_object.get(key) != base_object.get(key):
            raise ValueError(f"physics object field {key} changes inside group {group_id}")
    if is_base:
        if "sweep" in metadata:
            raise ValueError("base metadata must not declare a sweep")
        return

    sweep = metadata.get("sweep")
    if not isinstance(sweep, dict) or set(sweep) != {
        "level_index",
        "parameter",
        "target_object_id",
        "value",
    }:
        raise ValueError("sweep metadata uses an unsupported descriptor")
    if (
        sweep["parameter"] != descriptor["parameter"]
        or int(sweep["level_index"]) != int(descriptor["level_index"])
        or sweep["target_object_id"] != object_id
    ):
        raise ValueError(f"group and sample sweep descriptors differ: {metadata['scene_id']}")
    axis = str(sweep["parameter"])
    if axis not in SWEEP_AXES or int(sweep["level_index"]) not in {0, 1, 3, 4}:
        raise ValueError(f"invalid one-factor sweep: {sweep}")
    base_material = base_object["material"]
    target_material = target_object["material"]
    if set(base_material) != set(target_material):
        raise ValueError("sweep changes the material field set")
    for key in base_material:
        if key == axis:
            if not np.isclose(float(target_material[key]), float(sweep["value"]), rtol=0.0, atol=1.0e-12):
                raise ValueError("sweep value does not match the target material field")
        elif target_material[key] != base_material[key]:
            raise ValueError(f"one-factor sweep unexpectedly changes material.{key}")
    base_inertia = np.asarray(base_object["inertia_diagonal_kg_m2"], dtype=np.float64)
    target_inertia = np.asarray(target_object["inertia_diagonal_kg_m2"], dtype=np.float64)
    expected_inertia = (
        base_inertia * float(target_material["mass_kg"]) / float(base_material["mass_kg"])
        if axis == "mass_kg"
        else base_inertia
    )
    if not np.allclose(target_inertia, expected_inertia, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("sweep inertia is inconsistent with its one-factor mass change")


def _trajectory(path: Path, expected_object_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    required = {
        "schema_version",
        "object_ids",
        "time_s",
        "position_m",
        "quaternion_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
        "contact_count",
    }
    if required - set(payload):
        raise ValueError(f"release trajectory is missing fields: {sorted(required - set(payload))}")
    schema = str(np.asarray(payload["schema_version"]).reshape(()).item())
    if schema != "physweep_object_trajectory_v4":
        raise ValueError(f"unsupported release trajectory schema: {schema}")
    object_ids = [str(value) for value in np.asarray(payload["object_ids"]).reshape(-1)]
    if object_ids != [expected_object_id]:
        raise ValueError(f"trajectory object binding differs: {object_ids} != {[expected_object_id]}")
    time_s = np.asarray(payload["time_s"], dtype=np.float64)
    position = np.asarray(payload["position_m"], dtype=np.float64)
    quaternion = np.asarray(payload["quaternion_wxyz"], dtype=np.float64)
    linear_velocity = np.asarray(payload["linear_velocity_m_s"], dtype=np.float64)
    angular_velocity = np.asarray(payload["angular_velocity_rad_s"], dtype=np.float64)
    contact_count = np.asarray(payload["contact_count"])
    if (
        time_s.shape != (97,)
        or position.shape != (97, 1, 3)
        or quaternion.shape != (97, 1, 4)
        or linear_velocity.shape != (97, 1, 3)
        or angular_velocity.shape != (97, 1, 3)
        or contact_count.shape != (97, 1)
    ):
        raise ValueError("release trajectory must have 97 frames and one object")
    for name, value in (
        ("time_s", time_s),
        ("position_m", position),
        ("quaternion_wxyz", quaternion),
        ("linear_velocity_m_s", linear_velocity),
        ("angular_velocity_rad_s", angular_velocity),
        ("contact_count", contact_count),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"release trajectory contains non-finite {name}")
    if np.any(contact_count < 0) or not np.allclose(contact_count, np.rint(contact_count)):
        raise ValueError("release trajectory contact counts must be non-negative integers")
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("release trajectory quaternions must be unit length")
    if not np.all(np.diff(time_s) > 0.0) or not np.isclose(time_s[0], 0.0, atol=1.0e-10) or not np.isclose(time_s[-1], 4.0, atol=1.0e-8):
        raise ValueError("release trajectory time grid must contain closed 0..4 s endpoints")
    return payload


def _quaternion_sign_invariant_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(min(np.max(np.abs(actual - expected)), np.max(np.abs(actual + expected))))


def _validate_t0(metadata: dict[str, Any], trajectory: dict[str, np.ndarray], object_id: str) -> None:
    obj = _find_object(metadata, object_id)
    initial = obj["initial_state"]
    position = np.asarray(trajectory["position_m"], dtype=np.float64)[0, 0]
    quaternion = np.asarray(trajectory["quaternion_wxyz"], dtype=np.float64)[0, 0]
    if not np.allclose(position, initial["position_m"], rtol=1.0e-7, atol=1.0e-7):
        raise ValueError("metadata and trajectory initial positions differ")
    if _quaternion_sign_invariant_error(quaternion, np.asarray(initial["quaternion_wxyz"])) > 1.0e-7:
        raise ValueError("metadata and trajectory initial orientations differ")
    for key, trajectory_key in (
        ("linear_velocity_m_s", "linear_velocity_m_s"),
        ("angular_velocity_rad_s", "angular_velocity_rad_s"),
    ):
        if trajectory_key in trajectory and not np.allclose(
            np.asarray(trajectory[trajectory_key], dtype=np.float64)[0, 0],
            initial[key],
            rtol=1.0e-7,
            atol=1.0e-7,
        ):
            raise ValueError(f"metadata and trajectory initial {key} differ")


def build_point_trajectory_payload(
    trajectory: dict[str, np.ndarray],
    local_points_m: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    camera: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    time_s = np.asarray(trajectory["time_s"], dtype=np.float64)
    object_ids = np.asarray(trajectory["object_ids"])
    position = np.asarray(trajectory["position_m"], dtype=np.float64)
    quaternion = np.asarray(trajectory["quaternion_wxyz"], dtype=np.float64)
    frame_count, object_count, _ = position.shape
    if local_points_m.shape != (object_count, POINT_COUNT, 3):
        raise ValueError(f"local points must have shape {(object_count, POINT_COUNT, 3)}")
    rotations = np.empty((frame_count, object_count, 3, 3), dtype=np.float64)
    for frame in range(frame_count):
        for object_index in range(object_count):
            rotations[frame, object_index] = quaternion_matrix_wxyz(quaternion[frame, object_index])
    points_world = np.einsum("toij,onj->toni", rotations, local_points_m, optimize=True) + position[:, :, None, :]
    points_camera = _camera_points(points_world, camera_from_world)
    tracks = _project_camera(points_camera, intrinsics)
    depth = points_camera[..., 2]
    width, height = IMAGE_SIZE_PX
    clip_start = float(camera["clip_start_m"])
    clip_end = float(camera["clip_end_m"])
    valid = (
        np.isfinite(tracks).all(axis=-1)
        & np.isfinite(points_camera).all(axis=-1)
        & (depth > clip_start)
        & (depth < clip_end)
        & (tracks[..., 0] >= 0.0)
        & (tracks[..., 0] < width)
        & (tracks[..., 1] >= 0.0)
        & (tracks[..., 1] < height)
    )
    recovered_local = np.einsum(
        "toij,toni->tonj",
        rotations,
        points_world - position[:, :, None, :],
        optimize=True,
    )
    rigid_roundtrip_error = float(np.max(np.abs(recovered_local - local_points_m[None])))
    inverse_rotation = camera_from_world[:3, :3].T
    recovered_world = np.einsum(
        "ij,...j->...i",
        inverse_rotation,
        points_camera - camera_from_world[:3, 3],
        optimize=True,
    )
    camera_roundtrip_error = float(np.max(np.abs(recovered_world - points_world)))
    metadata = {
        "schema": POINT_TRAJECTORY_SCHEMA,
        "point_count": POINT_COUNT,
        "object_count": int(object_count),
        "object_ids": [str(value) for value in object_ids],
        "coordinate_frame_world": "pybullet_world_xyz",
        "coordinate_frame_camera": "camera_right_up_forward",
        "track_definition": POINT_TRACK_DEFINITION,
        "visibility_definition": POINT_VISIBILITY_DEFINITION,
        "clip_start_m": clip_start,
        "clip_end_m": clip_end,
        "surface_source": "physweep_release_collision_proxy_fixed_material_points",
        "initial_alignment_error_m": 0.0,
        "rigid_roundtrip_max_abs_error_m": rigid_roundtrip_error,
        "camera_roundtrip_max_abs_error_m": camera_roundtrip_error,
    }
    payload = {
        "time_s": time_s.astype(np.float32),
        "object_ids": object_ids,
        "points_world_m": points_world.astype(np.float32),
        "points_camera_m": points_camera.astype(np.float32),
        "tracks_xy_px": tracks.astype(np.float32),
        "depth_m": depth.astype(np.float32),
        "valid": valid,
        "initial_points_camera_m": points_camera[0].astype(np.float32),
        "camera_from_world": camera_from_world.astype(np.float32),
        "camera_intrinsics": intrinsics.astype(np.float32),
        "image_size_px": np.asarray(IMAGE_SIZE_PX, dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    validate_point_trajectory(payload)
    if rigid_roundtrip_error > 1.0e-10 or camera_roundtrip_error > 1.0e-10:
        raise ValueError("point trajectory failed its inverse-geometry audit")
    return payload, {
        "rigid_roundtrip_max_abs_error_m": rigid_roundtrip_error,
        "camera_roundtrip_max_abs_error_m": camera_roundtrip_error,
    }


def _extract_first_frame(video: Path, output: Path, ffmpeg: str, overwrite: bool) -> None:
    if output.is_file() and not overwrite:
        with Image.open(output) as image:
            if image.size != IMAGE_SIZE_PX or image.mode != "RGB":
                raise ValueError(f"existing canonical first frame has the wrong contract: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    command = [ffmpeg, "-v", "error", "-y", "-i", str(video), "-frames:v", "1", str(temporary)]
    try:
        subprocess.run(command, check=True)
        with Image.open(temporary) as image:
            if image.size != IMAGE_SIZE_PX:
                raise ValueError(f"decoded first frame has size {image.size}, expected {IMAGE_SIZE_PX}")
            converted = image.convert("RGB") if image.mode != "RGB" else None
        if converted is not None:
            converted.save(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_paths(sample_root: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "video": sample_root / "video.mp4",
        "trajectory": sample_root / "trajectory.npz",
        "mask_manifest": sample_root / "mask_manifest.json",
        "metadata": sample_root / "metadata.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"sample artifact is missing ({name}): {path}")
    expected = {
        "video": metadata["artifacts"]["video"]["sha256"],
        "trajectory": metadata["artifacts"]["trajectory"]["sha256"],
        "mask_manifest": metadata["artifacts"]["masks"]["manifest_sha256"],
    }
    for name, expected_hash in expected.items():
        if sha256_file(paths[name]) != expected_hash:
            raise ValueError(f"release artifact hash mismatch ({name}): {paths[name]}")
    return paths


def _relative(dataset_root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(f"output path escapes the dataset root: {path}")
    return resolved.relative_to(dataset_root).as_posix()


def _training_prompt(metadata: dict[str, Any], object_id: str) -> str:
    caption = str(metadata.get("text", {}).get("caption", "")).strip()
    semantic_records = metadata.get("semantics", {}).get("objects", [])
    labels = [str(record.get("semantic_label", "")).strip() for record in semantic_records if str(record.get("object_id")) == object_id]
    label = labels[0] if labels and labels[0] else "object"
    if not caption:
        raise ValueError("release metadata has no training caption")
    if caption[-1] not in ".!?":
        caption += "."
    return (
        f"{caption} Static-camera video of exactly one {label}; the same object, scene geometry, "
        "appearance, lighting, and background remain consistent throughout."
    )


def _physics_condition(metadata: dict[str, Any], object_id: str, camera_from_world: np.ndarray) -> dict[str, Any]:
    obj = _find_object(metadata, object_id)
    material = obj["material"]
    initial = obj["initial_state"]
    world_to_camera = camera_from_world[:3, :3]
    object_to_world = quaternion_matrix_wxyz(initial["quaternion_wxyz"])
    object_to_camera = world_to_camera @ object_to_world
    inertia_camera = object_to_camera @ np.diag(np.asarray(obj["inertia_diagonal_kg_m2"], dtype=np.float64)) @ object_to_camera.T
    linear_velocity = world_to_camera @ np.asarray(initial["linear_velocity_m_s"], dtype=np.float64)
    # Angular velocity is axial.  The right/up/forward camera basis is a
    # reflection, so a determinant factor is required for its coordinates.
    angular_velocity = np.linalg.det(world_to_camera) * world_to_camera @ np.asarray(initial["angular_velocity_rad_s"], dtype=np.float64)
    gravity = world_to_camera @ np.asarray(metadata["physics"]["world"]["gravity_m_s2"], dtype=np.float64)
    return {
        "source": "simulation_gt",
        "coordinate_frame": "camera_right_up_forward",
        "object": {
            "object_id": object_id,
            "mass_kg": float(material["mass_kg"]),
            "inertia_tensor_camera_kg_m2": inertia_camera.tolist(),
            "friction": float(material["contact_friction"]),
            "restitution": float(material["contact_restitution"]),
            "rolling_friction": float(material["rolling_friction"]),
            "spinning_friction": float(material["spinning_friction"]),
            "linear_damping": float(material["linear_damping"]),
            "angular_damping": float(material["angular_damping"]),
            "initial_state": {
                "linear_velocity_camera_m_s": linear_velocity.tolist(),
                "angular_velocity_camera_rad_s": angular_velocity.tolist(),
            },
        },
        "world": {"gravity_camera_m_s2": gravity.tolist()},
    }


def _mask_projection_audit(mask_manifest_path: Path, tracks_t0: np.ndarray, valid_t0: np.ndarray) -> dict[str, Any]:
    manifest = _json(mask_manifest_path)
    objects = manifest.get("objects")
    if int(manifest.get("frame_count", -1)) != 97 or not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("one-object mask manifest must bind one object and 97 frames")
    object_record = objects[0]
    object_id = str(object_record.get("object_id", ""))
    hashes = object_record.get("frame_sha256")
    if not object_id or not isinstance(hashes, list) or len(hashes) != 97:
        raise ValueError("mask manifest object record is incomplete")
    mask_path = (mask_manifest_path.parent / "masks" / object_id / "frame_0001.png").resolve()
    if not mask_path.is_relative_to(mask_manifest_path.parent.resolve()) or not mask_path.is_file():
        raise FileNotFoundError(f"first-frame mask is missing: {mask_path}")
    if sha256_file(mask_path) != hashes[0]:
        raise ValueError(f"first-frame mask hash mismatch: {mask_path}")
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != (IMAGE_SIZE_PX[1], IMAGE_SIZE_PX[0]):
        raise ValueError(f"mask has wrong image size: {mask.shape}")
    points = tracks_t0[valid_t0]
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    if not np.any(inside):
        return {"valid_projected_points": 0, "inside_mask_ratio": 0.0, "mask_nonzero_pixels": int(np.count_nonzero(mask))}
    covered = mask[y[inside], x[inside]] > 0
    return {
        "valid_projected_points": int(np.count_nonzero(inside)),
        "inside_mask_ratio": float(np.mean(covered)),
        "mask_nonzero_pixels": int(np.count_nonzero(mask)),
        "interpretation": "diagnostic_only_collision_proxy_vs_rendered_object_silhouette",
    }


def _sample_descriptor(record: dict[str, Any], is_base: bool) -> dict[str, Any]:
    if is_base:
        return {
            "mode": "base",
            "axis": None,
            "level_index": BASE_LEVEL_INDEX,
            "level_count": LEVEL_COUNT,
            "base_level_index": BASE_LEVEL_INDEX,
            "base_level_indices": {axis: BASE_LEVEL_INDEX for axis in SWEEP_AXES},
            "source_axis": "mass_kg",
            "source_value": float(record["conditioning"]["physics"]["object"]["mass_kg"]),
        }
    sweep = record.pop("_raw_sweep")
    axis = str(sweep["parameter"])
    level_index = int(sweep["level_index"])
    if axis not in SWEEP_AXES or level_index not in {0, 1, 3, 4}:
        raise ValueError(f"invalid release sweep descriptor: {sweep}")
    return {
        "mode": "one_factor",
        "axis": axis,
        "level_index": level_index,
        "level_count": LEVEL_COUNT,
        "base_level_index": BASE_LEVEL_INDEX,
        "base_level_indices": {name: BASE_LEVEL_INDEX for name in SWEEP_AXES},
        "source_axis": axis,
        "source_value": float(sweep["value"]),
    }


def validate_scene_payload(payload: dict[str, np.ndarray]) -> None:
    expected_shapes = {
        "object_xyz_camera_m": (1, POINT_COUNT, 3),
        "object_normal_camera": (1, POINT_COUNT, 3),
        "environment_xyz_camera_m": (ENVIRONMENT_POINT_COUNT, 3),
        "environment_normal_camera": (ENVIRONMENT_POINT_COUNT, 3),
        "environment_friction": (ENVIRONMENT_POINT_COUNT, 1),
        "environment_restitution": (ENVIRONMENT_POINT_COUNT, 1),
        "camera_intrinsics_normalized": (4,),
        "object_local_points_m": (1, POINT_COUNT, 3),
        "object_local_normal": (1, POINT_COUNT, 3),
        "environment_xyz_world_m": (ENVIRONMENT_POINT_COUNT, 3),
        "camera_from_world": (4, 4),
        "camera_intrinsics": (3, 3),
        "image_size_px": (2,),
    }
    missing = sorted((set(expected_shapes) | {"metadata_json"}) - set(payload))
    if missing:
        raise ValueError(f"scene payload is missing fields: {missing}")
    for name, shape in expected_shapes.items():
        value = np.asarray(payload[name])
        if value.shape != shape:
            raise ValueError(f"scene {name} has shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"scene {name} contains non-finite values")
    for name in ("object_normal_camera", "environment_normal_camera", "object_local_normal"):
        lengths = np.linalg.norm(np.asarray(payload[name]), axis=-1)
        if not np.allclose(lengths, 1.0, rtol=1.0e-4, atol=1.0e-4):
            raise ValueError(f"scene {name} must contain unit normals")
    friction = np.asarray(payload["environment_friction"])
    restitution = np.asarray(payload["environment_restitution"])
    if np.any(friction < 0.0) or np.any((restitution < 0.0) | (restitution > 1.0)):
        raise ValueError("scene environment physics is outside its valid range")
    transform = np.asarray(payload["camera_from_world"], dtype=np.float64)
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-5)
        or not np.isclose(np.linalg.det(rotation), -1.0, atol=1.0e-5)
    ):
        raise ValueError("scene camera transform violates camera_right_up_forward")
    intrinsics = np.asarray(payload["camera_intrinsics"], dtype=np.float64)
    width, height = np.asarray(payload["image_size_px"], dtype=np.int64)
    expected_normalized = np.asarray(
        [
            intrinsics[0, 0] / width,
            intrinsics[1, 1] / height,
            intrinsics[0, 2] / width,
            intrinsics[1, 2] / height,
        ]
    )
    if not np.allclose(payload["camera_intrinsics_normalized"], expected_normalized, atol=1.0e-6):
        raise ValueError("scene normalized intrinsics do not match pixel intrinsics")
    metadata_value = np.asarray(payload["metadata_json"])
    if metadata_value.size != 1:
        raise ValueError("scene metadata_json must contain one JSON object")
    try:
        metadata = json.loads(str(metadata_value.reshape(()).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("scene metadata_json is invalid") from exc
    if metadata.get("schema") != SCENE_SCHEMA:
        raise ValueError("scene metadata_json uses an unsupported schema")


def _load_groups(path: Path, selected_ids: set[str], limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(path)
    records = manifest.get("records")
    if not isinstance(records, list) or int(manifest.get("group_count", -1)) != len(records):
        raise ValueError("release group manifest has inconsistent group_count")
    if manifest.get("path_base") != "release_parent":
        raise ValueError("release group manifest uses an unsupported path base")
    if selected_ids:
        groups = [record for record in records if str(record.get("group_id")) in selected_ids]
        missing = sorted(selected_ids - {str(record.get("group_id")) for record in groups})
        if missing:
            raise ValueError(f"unknown group ids: {missing}")
    else:
        groups = records
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit-groups must be positive")
        groups = groups[:limit]
    if not groups:
        raise ValueError("no release groups selected")
    return manifest, groups


def _record_path(release_root: Path, descriptor: dict[str, Any]) -> Path:
    relative = Path(str(descriptor["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"sample path is not release-relative: {relative}")
    path = (release_root / relative).resolve()
    if not path.is_relative_to(release_root) or not path.is_dir():
        raise FileNotFoundError(f"sample directory is missing: {path}")
    return path


def adapt(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root, release_root, output_root = resolve_roots(args.dataset_root, args.release_root, args.output_root)
    group_manifest_path = release_root / "sweep" / "group_manifest.json"
    group_manifest, groups = _load_groups(group_manifest_path, set(args.group_id), args.limit_groups)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"derived output already contains files; use --overwrite deliberately: {output_root}"
            )
        previous_summary_path = output_root / "summary.json"
        previous_summary = _json(previous_summary_path) if previous_summary_path.is_file() else {}
        if previous_summary.get("adapter_schema") != ADAPTER_SCHEMA:
            raise ValueError(
                "refusing to overwrite a non-empty directory that is not a completed output "
                f"of {ADAPTER_SCHEMA}: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    training_records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    group_audits: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, 1):
        group_id = str(group["group_id"])
        object_id = str(group["target_object_id"])
        sweeps = list(group.get("sweeps", []))
        if len(sweeps) != 12 or {(str(item["parameter"]), int(item["level_index"])) for item in sweeps} != {
            (axis, level) for axis in SWEEP_AXES for level in (0, 1, 3, 4)
        }:
            raise ValueError(f"group does not contain the required 12 sweeps: {group_id}")
        descriptors = [(group["base"], True), *[(item, False) for item in sweeps]]
        base_root = _record_path(release_root, group["base"])
        base_metadata_path = base_root / "metadata.json"
        if sha256_file(base_metadata_path) != group["base"]["metadata_sha256"]:
            raise ValueError(f"group manifest base metadata hash mismatch: {group_id}")
        base_metadata = _json(base_metadata_path)
        _validate_release_group_sample(
            base_metadata,
            base_metadata,
            group["base"],
            group,
            object_id,
            True,
        )
        base_artifacts = _artifact_paths(base_root, base_metadata)
        camera = base_metadata["visual"]["camera"]
        camera_from_world, intrinsics = camera_contract(camera)
        base_trajectory = _trajectory(base_artifacts["trajectory"], object_id)
        _validate_t0(base_metadata, base_trajectory, object_id)

        group_rng = np.random.default_rng(stable_seed(group_id, args.seed))
        object_record = _find_object(base_metadata, object_id)
        local_points, local_normals, proxy_report = sample_dynamic_proxy(
            object_record["collision_proxy"], group_rng
        )
        fixture_sha = str(base_metadata["physics"]["fixture"]["sha256"])
        fixture_path = release_root / "base" / "fixtures" / f"{fixture_sha}.json"
        if not fixture_path.is_file() or sha256_file(fixture_path) != fixture_sha:
            raise ValueError(f"fixture payload is missing or hash-invalid: {fixture_path}")
        fixture = _json(fixture_path)
        components = fixture_components(fixture, release_root)
        environment, environment_report = sample_environment(
            components,
            camera_from_world,
            intrinsics,
            float(camera["clip_start_m"]),
            float(camera["clip_end_m"]),
            group_rng,
        )
        initial_rotation = quaternion_matrix_wxyz(np.asarray(base_trajectory["quaternion_wxyz"])[0, 0])
        initial_position = np.asarray(base_trajectory["position_m"], dtype=np.float64)[0, 0]
        object_world = _apply_transform(local_points, initial_rotation, initial_position)
        object_camera = _camera_points(object_world, camera_from_world)
        object_normal_camera = np.einsum(
            "ij,nj->ni",
            camera_from_world[:3, :3],
            np.einsum("ij,nj->ni", initial_rotation, local_normals, optimize=True),
            optimize=True,
        )

        first_frame_path = output_root / "first_frames" / f"{group_id}.png"
        _extract_first_frame(base_artifacts["video"], first_frame_path, args.ffmpeg, args.overwrite)
        scene_path = output_root / "scenes" / f"{group_id}.npz"
        scene_metadata = {
            "schema": SCENE_SCHEMA,
            "group_id": group_id,
            "object_ids": [object_id],
            "dynamic_surface_source": "simulation_collision_proxy_not_rendered_visual_mesh",
            "environment_surface_source": "simulation_static_fixture_collision_geometry",
            "coordinate_frame": "camera_right_up_forward",
            "object_point_count": POINT_COUNT,
            "environment_point_count": ENVIRONMENT_POINT_COUNT,
            "fixture_sha256": fixture_sha,
            "base_metadata_sha256": group["base"]["metadata_sha256"],
            "canonical_first_frame_sha256": sha256_file(first_frame_path),
            "seed": int(args.seed),
            "proxy_sampling": proxy_report,
            "environment_sampling": environment_report,
        }
        scene_payload = {
            "object_xyz_camera_m": object_camera[None].astype(np.float32),
            "object_normal_camera": object_normal_camera[None].astype(np.float32),
            "environment_xyz_camera_m": environment["xyz_camera"].astype(np.float32),
            "environment_normal_camera": environment["normal_camera"].astype(np.float32),
            "environment_friction": environment["friction"].astype(np.float32),
            "environment_restitution": environment["restitution"].astype(np.float32),
            "camera_intrinsics_normalized": np.asarray(
                [intrinsics[0, 0] / IMAGE_SIZE_PX[0], intrinsics[1, 1] / IMAGE_SIZE_PX[1], intrinsics[0, 2] / IMAGE_SIZE_PX[0], intrinsics[1, 2] / IMAGE_SIZE_PX[1]],
                dtype=np.float32,
            ),
            "object_local_points_m": local_points[None].astype(np.float32),
            "object_local_normal": local_normals[None].astype(np.float32),
            "environment_xyz_world_m": environment["xyz_world"].astype(np.float32),
            "camera_from_world": camera_from_world.astype(np.float32),
            "camera_intrinsics": intrinsics.astype(np.float32),
            "image_size_px": np.asarray(IMAGE_SIZE_PX, dtype=np.int32),
            "metadata_json": np.asarray(json.dumps(scene_metadata, sort_keys=True)),
        }
        validate_scene_payload(scene_payload)
        _atomic_npz(scene_path, scene_payload)

        base_point_payload: dict[str, np.ndarray] | None = None
        for descriptor, is_base in descriptors:
            sample_root = _record_path(release_root, descriptor)
            metadata_path = sample_root / "metadata.json"
            if sha256_file(metadata_path) != descriptor["metadata_sha256"]:
                raise ValueError(f"group manifest sample metadata hash mismatch: {sample_root}")
            metadata = _json(metadata_path)
            _validate_release_group_sample(
                base_metadata,
                metadata,
                descriptor,
                group,
                object_id,
                is_base,
            )
            artifacts = _artifact_paths(sample_root, metadata)
            trajectory = _trajectory(artifacts["trajectory"], object_id)
            _validate_t0(metadata, trajectory, object_id)
            point_payload, roundtrip = build_point_trajectory_payload(
                trajectory,
                local_points[None],
                camera_from_world,
                intrinsics,
                camera,
            )
            initial_alignment_error = float(
                np.max(np.abs(point_payload["initial_points_camera_m"] - scene_payload["object_xyz_camera_m"]))
            )
            if initial_alignment_error > 1.0e-6:
                raise ValueError(f"scene and point trajectory t0 differ: {descriptor['scene_id']}")
            point_metadata = json.loads(str(point_payload["metadata_json"].item()))
            point_metadata["initial_alignment_error_m"] = initial_alignment_error
            point_metadata["source_trajectory_sha256"] = metadata["artifacts"]["trajectory"]["sha256"]
            point_payload["metadata_json"] = np.asarray(json.dumps(point_metadata, sort_keys=True))
            validate_point_trajectory(point_payload)
            point_path = output_root / "point_trajectories" / str(descriptor["scene_id"]) / "point_trajectory.npz"
            _atomic_npz(point_path, point_payload)
            if is_base:
                base_point_payload = point_payload

            physics = _physics_condition(metadata, object_id, camera_from_world)
            training_record: dict[str, Any] = {
                "schema": SAMPLE_SCHEMA,
                "sample_id": str(metadata["scene_id"]),
                "base_scene_id": group_id,
                "split": split_for_group(group_id),
                "conditioning": {
                    "first_frame": _relative(dataset_root, first_frame_path),
                    "scene": _relative(dataset_root, scene_path),
                    "scene_source": "simulation_gt",
                    "text": _training_prompt(metadata, object_id),
                    "physics": physics,
                },
                "target": {
                    "video": _relative(dataset_root, artifacts["video"]),
                    "metadata": _relative(dataset_root, artifacts["metadata"]),
                    "duration_seconds": float(metadata["physics"]["time"]["duration_s"]),
                    "fps": int(metadata["physics"]["time"]["output_fps"]),
                },
                "provenance": {
                    "adapter_schema": ADAPTER_SCHEMA,
                    "release_group_manifest": _relative(dataset_root, group_manifest_path),
                    "release_group_id": group_id,
                    "release_family": str(group["family"]),
                    "release_metadata_sha256": descriptor["metadata_sha256"],
                    "trajectory": _relative(dataset_root, artifacts["trajectory"]),
                    "trajectory_sha256": metadata["artifacts"]["trajectory"]["sha256"],
                    "mask_manifest": _relative(dataset_root, artifacts["mask_manifest"]),
                    "mask_manifest_sha256": metadata["artifacts"]["masks"]["manifest_sha256"],
                    "scene_geometry_scope": "simulation_physics_proxy_not_visual_mesh",
                },
            }
            if not is_base:
                training_record["_raw_sweep"] = metadata["sweep"]
            training_record["sweep"] = _sample_descriptor(training_record, is_base)
            training_records.append(training_record)
            point_records.append(
                {
                    "sample_id": str(metadata["scene_id"]),
                    "path": _relative(dataset_root, point_path),
                    "sha256": sha256_file(point_path),
                    "schema": POINT_TRAJECTORY_SCHEMA,
                    "point_count": POINT_COUNT,
                    "object_count": 1,
                    "object_ids": [object_id],
                    "shape": list(point_payload["points_world_m"].shape),
                    "source_scene": _relative(dataset_root, scene_path),
                    "source_trajectory": _relative(dataset_root, artifacts["trajectory"]),
                    "initial_alignment_error_m": initial_alignment_error,
                    **roundtrip,
                }
            )
        if base_point_payload is None:
            raise AssertionError("group did not emit a base point trajectory")
        mask_audit = _mask_projection_audit(
            base_artifacts["mask_manifest"],
            base_point_payload["tracks_xy_px"][0, 0],
            base_point_payload["valid"][0, 0],
        )
        group_audits.append(
            {
                "group_id": group_id,
                "family": str(group["family"]),
                "sample_count": 13,
                "scene_sha256": sha256_file(scene_path),
                "canonical_first_frame_sha256": sha256_file(first_frame_path),
                "mask_projection": mask_audit,
                "proxy_sampling": proxy_report,
                "environment_sampling": environment_report,
            }
        )
        print(f"adapt {group_index}/{len(groups)} {group_id}", flush=True)

    manifest_path = output_root / "manifest.jsonl"
    _atomic_jsonl(manifest_path, training_records)
    validation = validate_manifest(manifest_path, dataset_root, check_files=True)
    point_manifest_path = output_root / "point_trajectories" / "manifest.json"
    point_manifest = {
        "schema": POINT_MANIFEST_SCHEMA,
        "source_manifest": _relative(dataset_root, manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "point_count": POINT_COUNT,
        "object_axis": "[T, O, 2048, ...]",
        "record_count": len(point_records),
        "records": point_records,
    }
    _atomic_json(point_manifest_path, point_manifest)
    audit_path = output_root / "adapter_audit.json"
    audit = {
        "schema": ADAPTER_SCHEMA,
        "status": "passed",
        "group_count": len(groups),
        "sample_count": len(training_records),
        "groups": group_audits,
        "invariants": {
            "adapter_write_scope_excludes_release_root": True,
            "canonical_first_frame_shared_per_group": True,
            "scene_shared_per_group": True,
            "fixed_material_point_identity": True,
            "trajectory_inverse_roundtrip_checked": True,
            "scene_trajectory_t0_alignment_checked": True,
            "full_resolution_projection_checked": True,
            "visual_mesh_available": False,
            "collision_proxy_visual_overlap_is_diagnostic_only": True,
        },
    }
    _atomic_json(audit_path, audit)
    summary_path = output_root / "summary.json"
    summary = {
        "schema": MANIFEST_SCHEMA,
        "adapter_schema": ADAPTER_SCHEMA,
        "path_base": "physweep_project_root",
        "manifest": _relative(dataset_root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "point_trajectory_manifest": _relative(dataset_root, point_manifest_path),
        "point_trajectory_manifest_sha256": sha256_file(point_manifest_path),
        "adapter_audit": _relative(dataset_root, audit_path),
        "adapter_audit_sha256": sha256_file(audit_path),
        "release_root": _relative(dataset_root, release_root),
        "release_group_manifest": _relative(dataset_root, group_manifest_path),
        "release_group_manifest_sha256": sha256_file(group_manifest_path),
        "release_group_count": int(group_manifest["group_count"]),
        "selected_group_count": len(groups),
        "seed": int(args.seed),
        "validation": validation,
        "ready_for_wan_cache": True,
        "training_was_run": False,
    }
    _atomic_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = adapt(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
