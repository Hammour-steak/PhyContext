from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class LaMaResult:
    rgb: np.ndarray
    inpaint_mask: np.ndarray
    expansion_pixels: int


def build_inpainting_mask(
    object_mask: np.ndarray,
    content_mask: np.ndarray,
    expansion_ratio: float = 0.15,
    minimum_expansion_pixels: int = 3,
    maximum_expansion_pixels: int = 32,
) -> tuple[np.ndarray, int]:
    """Expand an object mask proportionally to remove edge and shadow remnants."""
    object_mask = np.asarray(object_mask, dtype=bool)
    content_mask = np.asarray(content_mask, dtype=bool)
    if object_mask.shape != content_mask.shape:
        raise ValueError("Object and content masks must have matching dimensions")
    rows, columns = np.nonzero(object_mask)
    if not len(rows):
        raise ValueError("Object mask is empty")
    if expansion_ratio < 0:
        raise ValueError("Inpainting mask expansion ratio must be non-negative")

    object_extent = max(
        int(rows.max() - rows.min() + 1),
        int(columns.max() - columns.min() + 1),
    )
    expansion = int(round(object_extent * expansion_ratio))
    expansion = int(
        np.clip(expansion, minimum_expansion_pixels, maximum_expansion_pixels)
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expansion * 2 + 1, expansion * 2 + 1),
    )
    expanded = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )
    return expanded & content_mask, expansion


def run_lama_inpainting(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    content_mask: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    expansion_ratio: float = 0.15,
) -> LaMaResult:
    """Remove the selected object with the frozen Big-LaMa TorchScript model."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("LaMa input must be an HxWx3 RGB image")
    if rgb.shape[:2] != object_mask.shape:
        raise ValueError("RGB image and object mask must have matching dimensions")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    inpaint_mask, expansion = build_inpainting_mask(
        object_mask,
        content_mask,
        expansion_ratio=expansion_ratio,
    )
    height, width = rgb.shape[:2]
    pad_bottom = (-height) % 8
    pad_right = (-width) % 8
    padded_rgb = np.pad(
        rgb,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="reflect",
    )
    padded_mask = np.pad(
        inpaint_mask,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
    )
    image_tensor = (
        torch.from_numpy(padded_rgb.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    mask_tensor = (
        torch.from_numpy(padded_mask.copy())
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    model = torch.jit.load(str(checkpoint_path), map_location=device).eval()
    with torch.inference_mode():
        output = model(image_tensor, mask_tensor)
    generated = (
        output[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()[:height, :width]
    )
    result = rgb.copy()
    result[inpaint_mask] = generated[inpaint_mask]
    return LaMaResult(
        rgb=result,
        inpaint_mask=inpaint_mask,
        expansion_pixels=expansion,
    )
