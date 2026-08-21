from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class SegmentationResult:
    mask: np.ndarray
    overlay: np.ndarray
    candidate_montage: np.ndarray
    candidates: list[dict[str, float]]
    selected_index: int


def segmentation_from_mask(rgb: np.ndarray, mask: np.ndarray) -> SegmentationResult:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("provided object mask must match the preprocessed RGB image")
    if not mask.any() or mask.all():
        raise ValueError("provided object mask must contain foreground and background")
    overlay = rgb.copy()
    tint = np.array([255, 58, 80], dtype=np.float32)
    overlay[mask] = np.clip(0.55 * overlay[mask] + 0.45 * tint, 0, 255).astype(np.uint8)
    tile = cv2.resize(overlay, (259, 259), interpolation=cv2.INTER_AREA)
    montage = np.concatenate([np.concatenate([tile] * 4, axis=1)] * 3, axis=0)
    return SegmentationResult(
        mask=mask,
        overlay=overlay,
        candidate_montage=montage,
        candidates=[{"candidate_index": -1.0, "area_fraction": float(mask.mean())}],
        selected_index=-1,
    )


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, abs(median) * 1e-3, 1e-6)


def _candidate_score(record: dict, depth: np.ndarray) -> tuple[float, dict[str, float]]:
    mask = np.asarray(record["segmentation"], dtype=bool)
    height, width = mask.shape
    area_fraction = float(mask.mean())
    x, y, box_width, box_height = [float(value) for value in record["bbox"]]
    box_fraction = (box_width * box_height) / float(width * height)
    compactness = area_fraction / max(box_fraction, 1e-6)
    aspect_ratio = max(box_width / max(box_height, 1.0), box_height / max(box_width, 1.0))
    aspect_penalty = max(float(np.log(max(aspect_ratio, 1.0) / 2.0)), 0.0)
    center_x = (x + box_width * 0.5) / width
    center_y = (y + box_height * 0.5) / height
    center_distance = np.hypot(center_x - 0.5, center_y - 0.5) / np.sqrt(0.5)

    border = np.zeros_like(mask)
    border[:2] = True
    border[-2:] = True
    border[:, :2] = True
    border[:, -2:] = True
    border_fraction = float((mask & border).sum()) / max(float(mask.sum()), 1.0)

    valid_depth = np.isfinite(depth) & (depth > 0)
    scene_values = depth[valid_depth]
    mask_values = depth[mask & valid_depth]
    scene_median, scene_scale = _robust_scale(scene_values)
    if mask_values.size:
        object_median, object_scale = _robust_scale(mask_values)
        foreground = np.tanh(max((scene_median - object_median) / scene_scale, -2.0))
        depth_compactness = np.exp(-object_scale / max(abs(object_median), 1e-6))
    else:
        foreground = -1.0
        depth_compactness = 0.0

    if not 0.001 <= area_fraction <= 0.22:
        area_score = -4.0
    else:
        area_score = -abs(np.log(max(area_fraction, 1e-6) / 0.025)) * 0.35

    score = (
        1.6 * float(record["predicted_iou"])
        + 1.2 * float(record["stability_score"])
        + 0.6 * compactness
        + 0.45 * float(foreground)
        + 0.35 * float(depth_compactness)
        + area_score
        - 1.2 * float(center_distance)
        - 0.55 * aspect_penalty
        - 2.0 * border_fraction
    )
    metrics = {
        "score": float(score),
        "area_fraction": area_fraction,
        "compactness": float(compactness),
        "aspect_ratio": float(aspect_ratio),
        "center_distance": float(center_distance),
        "border_fraction": border_fraction,
        "foreground_score": float(foreground),
        "predicted_iou": float(record["predicted_iou"]),
        "stability_score": float(record["stability_score"]),
    }
    return float(score), metrics


def segment_primary_object(
    rgb: np.ndarray,
    depth: np.ndarray,
    checkpoint_path: Path,
    model_config: str,
    device: torch.device,
) -> SegmentationResult:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    # SAM2's optional CUDA connected-components extension is not needed here and can
    # poison the CUDA context when its build ABI differs from the active PyTorch ABI.
    # Mask cleanup below is deterministic and handled with OpenCV instead.
    model = build_sam2(
        model_config,
        str(checkpoint_path),
        device=str(device),
        apply_postprocessing=False,
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.92,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    records = generator.generate(rgb)
    if not records:
        raise RuntimeError("SAM2 did not produce any object candidates")

    scored: list[tuple[float, int, dict[str, float]]] = []
    for index, record in enumerate(records):
        score, metrics = _candidate_score(record, depth)
        metrics["candidate_index"] = float(index)
        scored.append((score, index, metrics))
    scored.sort(reverse=True, key=lambda item: item[0])
    _, selected_index, _ = scored[0]
    mask = np.asarray(records[selected_index]["segmentation"], dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    overlay = rgb.copy()
    tint = np.array([255, 58, 80], dtype=np.float32)
    overlay[mask] = np.clip(0.55 * overlay[mask] + 0.45 * tint, 0, 255).astype(np.uint8)

    tiles: list[np.ndarray] = []
    for rank, (score, index, _) in enumerate(scored[:12]):
        candidate = np.asarray(records[index]["segmentation"], dtype=bool)
        tile = cv2.resize(rgb, (259, 259), interpolation=cv2.INTER_AREA)
        candidate_small = cv2.resize(
            candidate.astype(np.uint8), (259, 259), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        tile[candidate_small] = np.clip(
            0.5 * tile[candidate_small] + 0.5 * tint, 0, 255
        ).astype(np.uint8)
        cv2.putText(
            tile,
            f"rank {rank + 1}  score {score:.2f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 255, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    blank = np.full_like(tiles[0], 245)
    while len(tiles) < 12:
        tiles.append(blank.copy())
    candidate_montage = np.concatenate(
        [np.concatenate(tiles[row : row + 4], axis=1) for row in range(0, 12, 4)], axis=0
    )

    return SegmentationResult(
        mask=mask,
        overlay=overlay,
        candidate_montage=candidate_montage,
        candidates=[item[2] for item in scored[:20]],
        selected_index=selected_index,
    )
