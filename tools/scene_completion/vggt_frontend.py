from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


@dataclass
class VGGTResult:
    rgb: np.ndarray
    points: np.ndarray
    confidence: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    source_shape_hw: tuple[int, int]
    content_mask: np.ndarray


def preprocess_binary_mask_for_vggt_pad(
    mask: np.ndarray,
    source_shape_hw: tuple[int, int],
    output_shape_hw: tuple[int, int],
    patch_size: int = 14,
) -> np.ndarray:
    """Apply VGGT's aspect-preserving ``mode="pad"`` geometry to a mask."""
    mask = np.asarray(mask, dtype=bool)
    source_height, source_width = (int(value) for value in source_shape_hw)
    output_height, output_width = (int(value) for value in output_shape_hw)
    if mask.ndim != 2:
        raise ValueError("VGGT object mask must be a two-dimensional binary image")
    if mask.shape != (source_height, source_width):
        raise ValueError(
            "VGGT object mask shape does not match the source image: "
            f"mask={mask.shape}, image={(source_height, source_width)}"
        )
    if output_height != output_width:
        raise ValueError(f'VGGT mode="pad" expects a square output, got {output_shape_hw}')
    if patch_size <= 0:
        raise ValueError("VGGT patch size must be positive")

    target_size = output_width
    if source_width >= source_height:
        resized_width = target_size
        resized_height = round(
            source_height * (resized_width / source_width) / patch_size
        ) * patch_size
    else:
        resized_height = target_size
        resized_width = round(
            source_width * (resized_height / source_height) / patch_size
        ) * patch_size
    if not (0 < resized_height <= output_height and 0 < resized_width <= output_width):
        raise ValueError(
            "VGGT mask resize produced invalid dimensions: "
            f"{(resized_height, resized_width)}"
        )

    resized = cv2.resize(
        mask.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    pad_top = (output_height - resized_height) // 2
    pad_left = (output_width - resized_width) // 2
    padded = np.zeros((output_height, output_width), dtype=bool)
    padded[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return padded


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("VGGT checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "base_model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def run_vggt(image_path: Path, checkpoint_path: Path, device: torch.device) -> VGGTResult:
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    source_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if source_image is None:
        raise FileNotFoundError(image_path)
    source_shape_hw = tuple(int(value) for value in source_image.shape[:2])
    images = load_and_preprocess_images([str(image_path)], mode="pad").to(device)
    model = VGGT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
    model.eval().to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )

    numpy_predictions: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            numpy_predictions[key] = value.detach().float().cpu().numpy().squeeze(0)

    extrinsic_np = extrinsic.detach().float().cpu().numpy().squeeze(0)
    intrinsic_np = intrinsic.detach().float().cpu().numpy().squeeze(0)
    depth = numpy_predictions["depth"]
    points = unproject_depth_map_to_point_map(depth, extrinsic_np, intrinsic_np)

    rgb = images[0].detach().float().cpu().permute(1, 2, 0).numpy()
    confidence = numpy_predictions["depth_conf"]
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    content_mask = preprocess_binary_mask_for_vggt_pad(
        np.ones(source_shape_hw, dtype=bool),
        source_shape_hw,
        rgb.shape[:2],
    )
    return VGGTResult(
        rgb=rgb,
        points=np.asarray(points[0], dtype=np.float32),
        confidence=np.squeeze(confidence[0]).astype(np.float32),
        depth=np.squeeze(depth[0]).astype(np.float32),
        intrinsic=np.asarray(intrinsic_np[0], dtype=np.float32),
        extrinsic=np.asarray(extrinsic_np[0], dtype=np.float32),
        source_shape_hw=source_shape_hw,
        content_mask=content_mask,
    )
