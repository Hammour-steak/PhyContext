from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class ObjectCrop:
    path: Path
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class InstantMeshResult:
    mesh_path: Path
    material_path: Path | None
    texture_path: Path | None
    log_path: Path
    manifest_path: Path
    metadata: dict
    reused: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    temporary.replace(path)


def save_object_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    padding_fraction: float = 0.12,
) -> ObjectCrop:
    pixels_y, pixels_x = np.nonzero(mask)
    if not len(pixels_x):
        raise RuntimeError("Cannot crop an empty object mask")
    height, width = mask.shape
    object_width = int(pixels_x.max() - pixels_x.min() + 1)
    object_height = int(pixels_y.max() - pixels_y.min() + 1)
    padding = max(8, int(round(max(object_width, object_height) * padding_fraction)))
    x0 = max(int(pixels_x.min()) - padding, 0)
    y0 = max(int(pixels_y.min()) - padding, 0)
    x1 = min(int(pixels_x.max()) + padding + 1, width)
    y1 = min(int(pixels_y.max()) + padding + 1, height)

    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    rgba = np.dstack([crop_rgb, crop_mask])
    crop_height, crop_width = rgba.shape[:2]
    side = max(crop_height, crop_width)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    offset_y = (side - crop_height) // 2
    offset_x = (side - crop_width) // 2
    canvas[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = rgba

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGBA").save(output_path)
    return ObjectCrop(path=output_path, bbox_xyxy=(x0, y0, x1, y1))


def copy_mesh_bundle(source: Path, destination: Path) -> tuple[Path | None, Path | None]:
    source_stem = source.stem
    destination_stem = destination.stem
    destination.parent.mkdir(parents=True, exist_ok=True)
    obj_text = source.read_text(encoding="utf-8").replace(
        f"{source_stem}.mtl", f"{destination_stem}.mtl"
    )
    destination.write_text(obj_text, encoding="utf-8")

    source_mtl = source.with_suffix(".mtl")
    source_texture = source.with_suffix(".png")
    destination_mtl = destination.with_suffix(".mtl")
    destination_texture = destination.with_suffix(".png")
    material_path: Path | None = None
    texture_path: Path | None = None
    if source_mtl.is_file():
        mtl_text = source_mtl.read_text(encoding="utf-8").replace(
            f"{source_stem}.png", f"{destination_stem}.png"
        )
        destination_mtl.write_text(mtl_text, encoding="utf-8")
        material_path = destination_mtl
    if source_texture.is_file():
        shutil.copy2(source_texture, destination_texture)
        texture_path = destination_texture
    return material_path, texture_path


def run_instantmesh(
    crop_path: Path,
    output_dir: Path,
    repository: Path,
    python_executable: Path,
    checkpoint_cache: Path,
    device_index: int,
    seed: int,
    diffusion_steps: int = 50,
    reuse_existing: bool = False,
) -> InstantMeshResult:
    crop_path = crop_path.resolve()
    output_dir = output_dir.resolve()
    repository = repository.resolve()
    # Keep the venv launcher symlink intact; resolving it can bypass the venv.
    python_executable = python_executable.absolute()
    checkpoint_cache = checkpoint_cache.resolve()
    destination = output_dir / "object_mesh.obj"
    log_path = output_dir / "instantmesh.log"
    manifest_path = output_dir / "object_mesh_manifest.json"
    if reuse_existing and destination.is_file():
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"cannot reuse InstantMesh output without provenance: {manifest_path}"
            )
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        return InstantMeshResult(
            mesh_path=destination,
            material_path=destination.with_suffix(".mtl") if destination.with_suffix(".mtl").is_file() else None,
            texture_path=destination.with_suffix(".png") if destination.with_suffix(".png").is_file() else None,
            log_path=log_path,
            manifest_path=manifest_path,
            metadata=metadata,
            reused=True,
        )
    for path in (crop_path, repository / "run.py", python_executable):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_output = output_dir / "instantmesh_raw"
    config = repository / "configs/instant-mesh-large.yaml"
    command = [
        str(python_executable),
        str(repository / "run.py"),
        str(config),
        str(crop_path),
        "--output_path",
        str(raw_output),
        "--diffusion_steps",
        str(diffusion_steps),
        "--seed",
        str(seed),
        "--no_rembg",
        "--export_texmap",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
    environment["HF_HOME"] = str(checkpoint_cache)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["DIFFUSERS_OFFLINE"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        raise RuntimeError(f"InstantMesh failed with exit code {result.returncode}:\n{tail}")

    generated = raw_output / "instant-mesh-large/meshes" / f"{crop_path.stem}.obj"
    if not generated.is_file():
        raise FileNotFoundError(f"InstantMesh did not produce the expected mesh: {generated}")
    material_path, texture_path = copy_mesh_bundle(generated, destination)
    artifacts = {"object_mesh.obj": _sha256(destination)}
    if material_path is not None:
        artifacts[material_path.name] = _sha256(material_path)
    if texture_path is not None:
        artifacts[texture_path.name] = _sha256(texture_path)
    metadata = {
        "schema": "phycontext.object_reconstruction.v1",
        "generator": "InstantMesh",
        "provenance_status": "recorded",
        "repository": str(repository),
        "repository_revision": _git_revision(repository),
        "config": str(config),
        "config_sha256": _sha256(config),
        "input_crop": str(crop_path),
        "input_crop_sha256": _sha256(crop_path),
        "seed": int(seed),
        "diffusion_steps": int(diffusion_steps),
        "artifacts": artifacts,
    }
    _write_json_atomic(manifest_path, metadata)
    return InstantMeshResult(
        mesh_path=destination,
        material_path=material_path,
        texture_path=texture_path,
        log_path=log_path,
        manifest_path=manifest_path,
        metadata=metadata,
        reused=False,
    )
