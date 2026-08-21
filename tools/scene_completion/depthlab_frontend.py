from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class DepthLabInput:
    rgb: np.ndarray
    known_depth: np.ndarray
    completion_mask: np.ndarray
    bounds_yxyx: tuple[int, int, int, int]


@dataclass
class DepthLabResult:
    depth: np.ndarray
    raw_depth: np.ndarray
    completion_mask: np.ndarray
    diagnostics: dict
    log_path: Path


def _content_bounds(content_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(content_mask)
    if not len(rows):
        raise RuntimeError("VGGT content mask is empty")
    return (
        int(rows.min()),
        int(rows.max()) + 1,
        int(columns.min()),
        int(columns.max()) + 1,
    )


def prepare_depthlab_input(
    inpainted_rgb: np.ndarray,
    reference_depth: np.ndarray,
    requested_completion_mask: np.ndarray,
    content_mask: np.ndarray,
) -> DepthLabInput:
    """Crop VGGT padding and form DepthLab's RGB, known-depth, and hole inputs."""
    inpainted_rgb = np.asarray(inpainted_rgb, dtype=np.uint8)
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    requested_completion_mask = np.asarray(requested_completion_mask, dtype=bool)
    content_mask = np.asarray(content_mask, dtype=bool)
    shape = reference_depth.shape
    if inpainted_rgb.shape != (*shape, 3):
        raise ValueError("DepthLab RGB and reference depth dimensions do not match")
    for name, value in (
        ("requested completion mask", requested_completion_mask),
        ("content mask", content_mask),
    ):
        if value.shape != shape:
            raise ValueError(f"DepthLab {name} dimensions do not match reference depth")

    valid_depth = np.isfinite(reference_depth) & (reference_depth > 0)
    effective_mask = content_mask & (requested_completion_mask | ~valid_depth)
    known_mask = content_mask & ~effective_mask & valid_depth
    if int(effective_mask.sum()) < 32:
        raise RuntimeError("DepthLab completion mask contains fewer than 32 pixels")
    if int(known_mask.sum()) < 128:
        raise RuntimeError("DepthLab has fewer than 128 known positive-depth pixels")

    y0, y1, x0, x1 = _content_bounds(content_mask)
    cropped_content = content_mask[y0:y1, x0:x1]
    if not cropped_content.all():
        raise RuntimeError("VGGT content mask is not a single rectangular image region")
    cropped_mask = effective_mask[y0:y1, x0:x1]
    cropped_depth = reference_depth[y0:y1, x0:x1].copy()
    cropped_depth[cropped_mask] = 0.0
    return DepthLabInput(
        rgb=inpainted_rgb[y0:y1, x0:x1].copy(),
        known_depth=cropped_depth,
        completion_mask=cropped_mask,
        bounds_yxyx=(y0, y1, x0, x1),
    )


def merge_depthlab_prediction(
    reference_depth: np.ndarray,
    prediction: np.ndarray,
    completion_mask: np.ndarray,
    bounds_yxyx: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Insert predicted hole depths while preserving every known depth exactly."""
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    completion_mask = np.asarray(completion_mask, dtype=bool)
    y0, y1, x0, x1 = bounds_yxyx
    expected_shape = (y1 - y0, x1 - x0)
    if prediction.shape != expected_shape or completion_mask.shape != expected_shape:
        raise ValueError(
            "DepthLab prediction/mask shape does not match the cropped content region"
        )
    valid_prediction = completion_mask & np.isfinite(prediction) & (prediction > 0)
    if int(valid_prediction.sum()) < 32:
        raise RuntimeError("DepthLab produced fewer than 32 positive completion pixels")

    full_raw = np.zeros_like(reference_depth, dtype=np.float32)
    full_raw[y0:y1, x0:x1] = prediction
    full_mask = np.zeros_like(reference_depth, dtype=bool)
    full_mask[y0:y1, x0:x1] = valid_prediction
    fused = reference_depth.copy()
    fused[full_mask] = full_raw[full_mask]
    return fused, full_mask


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_depthlab_completion(
    inpainted_rgb: np.ndarray,
    reference_depth: np.ndarray,
    requested_completion_mask: np.ndarray,
    content_mask: np.ndarray,
    output_dir: Path,
    repository: Path,
    python_executable: Path,
    checkpoint_root: Path,
    device_index: int,
    seed: int,
    denoise_steps: int = 50,
    processing_resolution: int = 768,
) -> DepthLabResult:
    """Run the frozen official DepthLab implementation in its isolated runtime."""
    output_dir = output_dir.resolve()
    repository = repository.resolve()
    python_executable = python_executable.absolute()
    checkpoint_root = checkpoint_root.resolve()
    files = {
        "runner": repository / "infer.py",
        "python": python_executable,
        "marigold": checkpoint_root / "marigold-depth-v1-0",
        "image_encoder": checkpoint_root / "CLIP-ViT-H-14-laion2B-s32B-b79K",
        "denoising_unet": checkpoint_root / "DepthLab/denoising_unet.pth",
        "reference_unet": checkpoint_root / "DepthLab/reference_unet.pth",
        "mapping_layer": checkpoint_root / "DepthLab/mapping_layer.pth",
    }
    for name, path in files.items():
        if name in {"marigold", "image_encoder"}:
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    if denoise_steps < 1:
        raise ValueError("DepthLab denoise steps must be positive")
    if processing_resolution < 64:
        raise ValueError("DepthLab processing resolution must be at least 64 pixels")

    prepared = prepare_depthlab_input(
        inpainted_rgb,
        reference_depth,
        requested_completion_mask,
        content_mask,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "environment_depthlab_rgb.png"
    known_depth_path = output_dir / "environment_depthlab_known_depth.npy"
    mask_path = output_dir / "environment_depthlab_mask.png"
    raw_depth_path = output_dir / "environment_depthlab_raw_depth.npy"
    log_path = output_dir / "depthlab.log"
    Image.fromarray(prepared.rgb, mode="RGB").save(rgb_path)
    np.save(known_depth_path, prepared.known_depth)
    Image.fromarray(prepared.completion_mask.astype(np.uint8) * 255, mode="L").save(
        mask_path
    )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="depthlab_", dir=output_dir) as temporary:
        runtime_output = Path(temporary) / "output"
        command = [
            str(python_executable),
            str(files["runner"]),
            "--seed",
            str(seed),
            "--denoise_steps",
            str(denoise_steps),
            "--processing_res",
            str(processing_resolution),
            "--normalize_scale",
            "1",
            "--strength",
            "0.8",
            "--pretrained_model_name_or_path",
            str(files["marigold"]),
            "--image_encoder_path",
            str(files["image_encoder"]),
            "--denoising_unet_path",
            str(files["denoising_unet"]),
            "--reference_unet_path",
            str(files["reference_unet"]),
            "--mapping_path",
            str(files["mapping_layer"]),
            "--output_dir",
            str(runtime_output),
            "--input_image_paths",
            str(rgb_path),
            "--known_depth_paths",
            str(known_depth_path),
            "--masks_paths",
            str(mask_path),
            "--blend",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device_index),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            )
            raise RuntimeError(
                f"DepthLab failed with exit code {result.returncode}:\n{tail}"
            )
        generated = (
            runtime_output
            / "depth_npy"
            / f"{rgb_path.stem}_pred.npy"
        )
        if not generated.is_file():
            raise FileNotFoundError(f"DepthLab did not produce the expected depth: {generated}")
        prediction = np.load(generated, allow_pickle=False).astype(np.float32)

    fused_depth, full_mask = merge_depthlab_prediction(
        reference_depth,
        prediction,
        prepared.completion_mask,
        prepared.bounds_yxyx,
    )
    y0, y1, x0, x1 = prepared.bounds_yxyx
    full_raw = np.zeros_like(reference_depth, dtype=np.float32)
    full_raw[y0:y1, x0:x1] = prediction
    np.save(raw_depth_path, full_raw)

    known = ~prepared.completion_mask & np.isfinite(prepared.known_depth) & (
        prepared.known_depth > 0
    )
    known_relative_error = np.abs(
        prediction[known] - prepared.known_depth[known]
    ) / np.maximum(prepared.known_depth[known], 1e-8)
    requested_count = int(prepared.completion_mask.sum())
    completed_count = int(full_mask.sum())
    diagnostics = {
        "method": "official_depthlab_known_depth_completion",
        "repository": str(repository),
        "repository_revision": _git_revision(repository),
        "model_sources": {
            "depthlab": {
                "repository": "Johanan0528/DepthLab",
                "revision": "ff9bba42b9ec458ac25acade326cf3007627f46d",
            },
            "marigold": {
                "repository": "prs-eth/marigold-depth-v1-0",
                "revision": "f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c",
            },
            "image_encoder": {
                "repository": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
            },
        },
        "seed": int(seed),
        "denoise_steps": int(denoise_steps),
        "processing_resolution": int(processing_resolution),
        "normalize_scale": 1.0,
        "strength": 0.8,
        "blend": True,
        "content_bounds_yxyx": list(prepared.bounds_yxyx),
        "known_pixel_count": int(known.sum()),
        "requested_completion_pixel_count": requested_count,
        "positive_completion_pixel_count": completed_count,
        "positive_completion_fraction": float(completed_count / max(requested_count, 1)),
        "raw_known_depth_median_relative_error": float(
            np.median(known_relative_error)
        ),
        "raw_known_depth_p90_relative_error": float(
            np.quantile(known_relative_error, 0.90)
        ),
        "known_depth_lock": "exact_original_values_restored_outside_completion_mask",
        "inference_seconds": float(time.perf_counter() - started),
        "inputs": {
            "rgb": rgb_path.name,
            "known_depth": known_depth_path.name,
            "mask": mask_path.name,
        },
        "raw_output": raw_depth_path.name,
    }
    _write_json(output_dir / "depthlab_manifest.json", diagnostics)
    return DepthLabResult(
        depth=fused_depth,
        raw_depth=full_raw,
        completion_mask=full_mask,
        diagnostics=diagnostics,
        log_path=log_path,
    )
