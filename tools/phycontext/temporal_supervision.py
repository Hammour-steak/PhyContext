"""Exact-time decoded-video supervision for PhyContext training.

Wan's causal VAE maps latent frame zero to source frame zero and every later
latent frame ``k`` to source frames ``4k-3 .. 4k``.  The helpers in this module
decode two adjacent four-frame chunks with their exact causal prefix, then apply
two residual-matching objectives:

* the RGB change of the same visible material point must match the target; and
* the RGB second difference of stable background pixels must match the target.

Neither objective assumes that a moving object or the background is constant.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from point_trajectory import (
    cover_center_crop_coordinates,
    validate_point_trajectory,
)


def source_frames_for_latent_window(
    latent_window_start: int,
    latent_window_chunks: int = 2,
) -> list[int]:
    """Return source frames for adjacent generated latent chunks.

    Two chunks are used by formal training so transitions across VAE chunk
    boundaries receive the same supervision as transitions inside a chunk.
    """
    if latent_window_start <= 0 or latent_window_chunks <= 0:
        raise ValueError("temporal supervision must exclude latent frame zero")
    first = 4 * latent_window_start - 3
    return list(range(first, first + 4 * latent_window_chunks))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_dataset_path(dataset_root: Path, relative_value: str, label: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be dataset-root-relative")
    root = dataset_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _evenly_spaced_frame_indices(source_frames: int, output_frames: int) -> list[int]:
    if source_frames < output_frames or output_frames <= 0:
        raise ValueError("source sequence cannot satisfy temporal supervision")
    return np.rint(
        np.linspace(0, source_frames - 1, output_frames)
    ).astype(np.int64).tolist()


def temporal_decoded_size(
    source_size_px: tuple[int, int], decoded_long_edge: int
) -> tuple[int, int]:
    """Preserve aspect ratio on the integer Wan latent grid."""
    if decoded_long_edge < 16 or decoded_long_edge % 16:
        raise ValueError("temporal decoded long edge must be a multiple of 16")
    source_width, source_height = (int(value) for value in source_size_px)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("temporal source size must be positive")
    maximum_extent = decoded_long_edge // 16
    scale = maximum_extent / max(source_width, source_height)
    latent_width = max(1, round(source_width * scale))
    latent_height = max(1, round(source_height * scale))
    return latent_width * 16, latent_height * 16


def _cover_center_crop_masks(
    masks: np.ndarray,
    target_size_px: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(masks, dtype=np.uint8)
    if value.ndim != 4 or value.shape[0] < 1:
        raise ValueError("instance masks must have shape F x O x H x W")
    target_width, target_height = (int(item) for item in target_size_px)
    source_height, source_width = value.shape[-2:]
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    crop_x = (resized_width - target_width) // 2
    crop_y = (resized_height - target_height) // 2
    processed = np.empty(
        (value.shape[0], value.shape[1], target_height, target_width),
        dtype=np.uint8,
    )
    for frame_index in range(value.shape[0]):
        for object_index in range(value.shape[1]):
            resized = cv2.resize(
                value[frame_index, object_index],
                (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST,
            )
            processed[frame_index, object_index] = resized[
                crop_y : crop_y + target_height,
                crop_x : crop_x + target_width,
            ]
    return processed


def dynamic_zbuffer_visibility_from_masks(
    tracks_xy_px: np.ndarray,
    depth_m: np.ndarray,
    valid: np.ndarray,
    instance_masks: np.ndarray,
    *,
    point_radius_px: int = 1,
) -> np.ndarray:
    """Select high-confidence visible material points.

    Dynamic points compete in one depth buffer across all object slots.  A
    winner is retained only when both its winning splat and projected centre
    fall inside the eroded renderer instance mask.  The mask removes static
    scene occlusion that a dynamic-only depth buffer cannot observe.
    """
    tracks = np.asarray(tracks_xy_px, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    masks = np.asarray(instance_masks, dtype=bool)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks must have shape F x O x N x 2")
    if depth.shape != tracks.shape[:-1] or validity.shape != depth.shape:
        raise ValueError("depth and validity must match track axes")
    if masks.shape[:2] != tracks.shape[:2] or masks.ndim != 4:
        raise ValueError("instance masks must match track frame/object axes")
    if point_radius_px < 0:
        raise ValueError("point radius must be nonnegative")
    frame_count, object_count, point_count = depth.shape
    height, width = masks.shape[-2:]
    visibility = np.zeros_like(validity, dtype=bool)
    object_indices = np.broadcast_to(
        np.arange(object_count, dtype=np.int64)[:, None],
        (object_count, point_count),
    )
    point_indices = np.broadcast_to(
        np.arange(point_count, dtype=np.int64)[None, :],
        (object_count, point_count),
    )
    offsets = np.arange(-point_radius_px, point_radius_px + 1, dtype=np.int64)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="xy")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)

    for frame_index in range(frame_count):
        xy = tracks[frame_index]
        z = depth[frame_index]
        centre_x = np.rint(xy[..., 0]).astype(np.int64)
        centre_y = np.rint(xy[..., 1]).astype(np.int64)
        inside = (
            validity[frame_index]
            & np.isfinite(xy).all(axis=-1)
            & np.isfinite(z)
            & (z > 0.0)
            & (centre_x >= 0)
            & (centre_x < width)
            & (centre_y >= 0)
            & (centre_y < height)
        )
        if not np.any(inside):
            continue
        x = centre_x[inside]
        y = centre_y[inside]
        candidate_depth = z[inside]
        candidate_objects = object_indices[inside]
        candidate_points = point_indices[inside]
        repeat_count = offset_x.shape[1]
        x = (x[:, None] + offset_x).reshape(-1)
        y = (y[:, None] + offset_y).reshape(-1)
        candidate_depth = np.repeat(candidate_depth, repeat_count)
        candidate_objects = np.repeat(candidate_objects, repeat_count)
        candidate_points = np.repeat(candidate_points, repeat_count)
        splat_inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[splat_inside]
        y = y[splat_inside]
        candidate_depth = candidate_depth[splat_inside]
        candidate_objects = candidate_objects[splat_inside]
        candidate_points = candidate_points[splat_inside]
        flat_pixels = y * width + x
        order = np.lexsort(
            (candidate_points, candidate_objects, candidate_depth, flat_pixels)
        )
        sorted_pixels = flat_pixels[order]
        nearest = np.concatenate(
            (np.asarray([True]), sorted_pixels[1:] != sorted_pixels[:-1])
        )
        winners = order[nearest]
        winner_objects = candidate_objects[winners]
        winner_points = candidate_points[winners]
        winner_x = x[winners]
        winner_y = y[winners]
        splat_in_mask = masks[
            frame_index, winner_objects, winner_y, winner_x
        ]
        centre_in_mask = masks[
            frame_index,
            winner_objects,
            centre_y[winner_objects, winner_points],
            centre_x[winner_objects, winner_points],
        ]
        keep = splat_in_mask & centre_in_mask
        visibility[
            frame_index, winner_objects[keep], winner_points[keep]
        ] = True
    return visibility


def load_temporal_supervision(
    cached_record: dict,
    dataset_root: Path,
    *,
    preprocess_size_px: tuple[int, int],
    output_frame_count: int,
    latent_window_start: int,
    latent_window_chunks: int = 2,
    decoded_resolution: int,
    mask_erosion_px: int,
) -> dict[str, torch.Tensor | int | str]:
    """Load one exact eight-frame track/mask supervision window from PhysSweep."""
    if decoded_resolution < 16 or decoded_resolution % 16:
        raise ValueError("temporal decoded long edge must be a multiple of 16")
    if mask_erosion_px < 0:
        raise ValueError("temporal mask erosion must be nonnegative")
    logical_frames = source_frames_for_latent_window(
        latent_window_start, latent_window_chunks
    )
    if logical_frames[-1] >= output_frame_count:
        raise ValueError("latent chunk lies outside the preprocessed video")
    descriptor = cached_record.get("point_track")
    if not isinstance(descriptor, dict):
        raise ValueError("temporal supervision requires a point-track descriptor")
    point_path = _safe_dataset_path(
        dataset_root,
        descriptor.get("source_point_trajectory", ""),
        "source point trajectory",
    )
    point_bytes = point_path.read_bytes()
    if _sha256_bytes(point_bytes) != descriptor.get(
        "source_point_trajectory_sha256"
    ):
        raise ValueError("source point trajectory hash mismatch")
    with np.load(io.BytesIO(point_bytes), allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_point_trajectory(payload)
    source_frame_count = int(payload["tracks_xy_px"].shape[0])
    output_to_source = _evenly_spaced_frame_indices(
        source_frame_count, output_frame_count
    )
    selected_source_frames = [output_to_source[index] for index in logical_frames]

    record = cached_record["record"]
    provenance = record.get("provenance", {})
    mask_manifest_path = _safe_dataset_path(
        dataset_root,
        provenance.get("mask_manifest", ""),
        "instance-mask manifest",
    )
    manifest_bytes = mask_manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != provenance.get("mask_manifest_sha256"):
        raise ValueError("instance-mask manifest hash mismatch")
    mask_manifest = json.loads(manifest_bytes)
    manifest_objects = mask_manifest.get("objects")
    object_ids = [str(value) for value in descriptor.get("object_ids", [])]
    if (
        mask_manifest.get("frame_count") != source_frame_count
        or not isinstance(manifest_objects, list)
        or [str(item.get("object_id")) for item in manifest_objects] != object_ids
    ):
        raise ValueError("instance masks do not match the point trajectory")

    source_width, source_height = (
        int(value) for value in np.asarray(payload["image_size_px"]).reshape(-1)
    )
    temporal_frame_count = len(selected_source_frames)
    raw_masks = np.empty(
        (temporal_frame_count, len(object_ids), source_height, source_width),
        dtype=np.uint8,
    )
    for local_frame, source_frame in enumerate(selected_source_frames):
        for object_index, object_record in enumerate(manifest_objects):
            mask_path = (
                mask_manifest_path.parent
                / "masks"
                / object_ids[object_index]
                / f"frame_{source_frame + 1:04d}.png"
            ).resolve()
            if (
                not mask_path.is_relative_to(mask_manifest_path.parent.resolve())
                or not mask_path.is_file()
            ):
                raise FileNotFoundError(f"instance mask is missing: {mask_path}")
            encoded = mask_path.read_bytes()
            expected_hashes = object_record.get("frame_sha256", [])
            if (
                len(expected_hashes) != source_frame_count
                or _sha256_bytes(encoded) != expected_hashes[source_frame]
            ):
                raise ValueError(f"instance mask hash mismatch: {mask_path}")
            decoded = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if decoded is None or decoded.shape != (source_height, source_width):
                raise ValueError(f"instance mask has the wrong image size: {mask_path}")
            raw_masks[local_frame, object_index] = decoded > 0

    processed_masks = _cover_center_crop_masks(raw_masks, preprocess_size_px)
    if mask_erosion_px:
        kernel_size = mask_erosion_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        interior_masks = np.stack(
            [
                np.stack(
                    [
                        cv2.erode(mask, kernel, iterations=1)
                        for mask in frame_masks
                    ]
                )
                for frame_masks in processed_masks
            ]
        )
    else:
        interior_masks = processed_masks

    tracks = cover_center_crop_coordinates(
        payload["tracks_xy_px"][selected_source_frames],
        (source_width, source_height),
        preprocess_size_px,
    )
    depth = payload["depth_m"][selected_source_frames]
    valid = payload["valid"][selected_source_frames]
    high_confidence_visible = dynamic_zbuffer_visibility_from_masks(
        tracks,
        depth,
        valid,
        interior_masks,
        point_radius_px=1,
    )
    preprocess_width, preprocess_height = preprocess_size_px
    normalized_tracks = np.empty_like(tracks, dtype=np.float32)
    normalized_tracks[..., 0] = (
        2.0 * (tracks[..., 0] + 0.5) / preprocess_width - 1.0
    )
    normalized_tracks[..., 1] = (
        2.0 * (tracks[..., 1] + 0.5) / preprocess_height - 1.0
    )
    normalized_tracks[~np.isfinite(normalized_tracks)] = 0.0

    decoded_width, decoded_height = temporal_decoded_size(
        preprocess_size_px, decoded_resolution
    )
    decoded_masks = np.empty(
        (
            temporal_frame_count,
            len(object_ids),
            decoded_height,
            decoded_width,
        ),
        dtype=bool,
    )
    for frame_index in range(temporal_frame_count):
        for object_index in range(len(object_ids)):
            decoded_masks[frame_index, object_index] = cv2.resize(
                processed_masks[frame_index, object_index].astype(np.float32),
                (decoded_width, decoded_height),
                # A conservative area reduction keeps a one-pixel distant
                # object in the exclusion mask instead of dropping it through
                # nearest-neighbour sampling.
                interpolation=cv2.INTER_AREA,
            ) > 0
    return {
        "sample_id": str(cached_record["sample_id"]),
        "latent_window_start": int(latent_window_start),
        "latent_window_chunks": int(latent_window_chunks),
        "source_frame_indices": torch.tensor(
            selected_source_frames, dtype=torch.int64
        ),
        "track_grid": torch.from_numpy(normalized_tracks),
        "track_visible": torch.from_numpy(high_confidence_visible),
        "object_masks": torch.from_numpy(decoded_masks),
    }


def _unpatchify_2x(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] % 4:
        raise ValueError("Wan VAE decoder output cannot be unpatchified")
    batch, channels4, frames, height, width = value.shape
    channels = channels4 // 4
    return (
        value.view(batch, channels, 2, 2, frames, height, width)
        .permute(0, 1, 4, 5, 3, 6, 2)
        .reshape(batch, channels, frames, height * 2, width * 2)
    )


def decode_wan_causal_window(
    vae,
    latent: torch.Tensor,
    latent_window_start: int,
    decoded_resolution: int,
    latent_window_chunks: int = 2,
) -> torch.Tensor:
    """Decode exact adjacent Wan chunks while bounding the autograd graph.

    Prefix chunks populate the frozen causal decoder's feature caches under
    ``no_grad``.  The selected chunk retains gradients, so its forward values
    are identical to a full-sequence decode without retaining 97 output frames.
    """
    if latent.ndim != 4:
        raise ValueError("Wan latent must have shape C x F x H x W")
    latent_window_end = latent_window_start + latent_window_chunks - 1
    if (
        latent_window_chunks <= 0
        or not 0 < latent_window_start <= latent_window_end < latent.shape[1]
    ):
        raise ValueError("Wan temporal window lies outside generated frames")
    if decoded_resolution < 16 or decoded_resolution % 16:
        raise ValueError("temporal decoded long edge must be a multiple of 16")
    maximum_extent = decoded_resolution // 16
    spatial_scale = maximum_extent / max(latent.shape[-2:])
    latent_height = max(1, round(latent.shape[-2] * spatial_scale))
    latent_width = max(1, round(latent.shape[-1] * spatial_scale))
    resized = F.interpolate(
        latent.float(),
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )
    model = vae.model
    if not all(hasattr(model, name) for name in ("clear_cache", "conv2", "decoder")):
        raise TypeError("temporal decoding requires the Wan2.2 causal VAE")
    if tuple(getattr(model.conv2, "kernel_size", ())) != (1, 1, 1):
        raise TypeError("Wan temporal streaming requires a pointwise VAE conv2")
    scale = vae.scale
    channel_count = int(latent.shape[0])
    mean = scale[0].view(1, channel_count, 1, 1, 1)
    inverse_std = scale[1].view(1, channel_count, 1, 1, 1)
    retain_selected_graph = torch.is_grad_enabled() and latent.requires_grad
    selected_outputs = []
    model.clear_cache()
    try:
        for index in range(latent_window_end + 1):
            context = (
                nullcontext()
                if retain_selected_graph and index >= latent_window_start
                else torch.no_grad()
            )
            with context, torch.autocast(
                device_type=latent.device.type,
                dtype=getattr(vae, "dtype", torch.bfloat16),
                enabled=latent.device.type == "cuda",
            ):
                normalized = resized[:, index : index + 1].unsqueeze(0)
                denormalized = normalized / inverse_std + mean
                features = model.conv2(denormalized)
                model._conv_idx = [0]
                output = model.decoder(
                    features,
                    feat_cache=model._feat_map,
                    feat_idx=model._conv_idx,
                    first_chunk=index == 0,
                )
                if index >= latent_window_start:
                    selected_outputs.append(output)
        if len(selected_outputs) != latent_window_chunks:
            raise RuntimeError("Wan temporal decoder produced an incomplete window")
        decoded = _unpatchify_2x(torch.cat(selected_outputs, dim=2)).squeeze(0).float()
        expected_shape = (
            3,
            4 * latent_window_chunks,
            latent_height * 16,
            latent_width * 16,
        )
        if decoded.shape != expected_shape:
            raise ValueError(
                "Wan temporal decoder produced an unexpected window shape: "
                f"{tuple(decoded.shape)}"
            )
        return decoded.clamp(-1.0, 1.0)
    finally:
        model.clear_cache()


def _sample_tracks(video: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or grid.ndim != 4 or grid.shape[0] != video.shape[1]:
        raise ValueError("video and track grid must share the frame axis")
    frame_count, object_count, point_count, _ = grid.shape
    frames = video.permute(1, 0, 2, 3)
    samples = F.grid_sample(
        frames,
        grid.reshape(frame_count, object_count * point_count, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return samples.squeeze(-1).permute(0, 2, 1).reshape(
        frame_count, object_count, point_count, video.shape[0]
    )


def object_track_temporal_residual_loss(
    predicted_video: torch.Tensor,
    target_video: torch.Tensor,
    track_grid: torch.Tensor,
    track_visible: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match RGB changes at the same point when it is visible in both frames."""
    if predicted_video.shape != target_video.shape:
        raise ValueError("object temporal videos must have equal shapes")
    if track_grid.shape[:-1] != track_visible.shape:
        raise ValueError("track coordinates and visibility axes differ")
    if beta <= 0:
        raise ValueError("object temporal Smooth-L1 beta must be positive")
    prediction_samples = _sample_tracks(predicted_video.float(), track_grid.float())
    target_samples = _sample_tracks(
        target_video.detach().float(), track_grid.float()
    )
    prediction_delta = prediction_samples[1:] - prediction_samples[:-1]
    target_delta = target_samples[1:] - target_samples[:-1]
    valid_pairs = track_visible[1:] & track_visible[:-1]
    pair_count = valid_pairs.sum()
    zero = predicted_video.sum() * 0.0
    if not bool(pair_count):
        return zero, pair_count.to(dtype=torch.float32)
    error = F.smooth_l1_loss(
        prediction_delta,
        target_delta,
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    return error[valid_pairs].mean(), pair_count.to(dtype=torch.float32)


def background_temporal_residual_loss(
    predicted_video: torch.Tensor,
    target_video: torch.Tensor,
    object_masks: torch.Tensor,
    *,
    dilation_px: int,
    stability_threshold: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match second temporal differences outside three-frame object sweeps."""
    if predicted_video.shape != target_video.shape:
        raise ValueError("background temporal videos must have equal shapes")
    if object_masks.ndim != 4 or object_masks.shape[0] != predicted_video.shape[1]:
        raise ValueError("object masks must have shape F x O x H x W")
    if object_masks.shape[-2:] != predicted_video.shape[-2:]:
        raise ValueError("background masks and decoded video sizes differ")
    if dilation_px < 0 or not 0 < stability_threshold <= 2 or beta <= 0:
        raise ValueError("background temporal dilation/threshold/beta is invalid")
    prediction = predicted_video.float()
    target = target_video.detach().float()
    prediction_second = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    target_second = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    object_union = object_masks.bool().any(dim=1)
    swept = torch.stack(
        [
            object_union[index : index + 3].any(dim=0)
            for index in range(object_union.shape[0] - 2)
        ]
    )
    if dilation_px:
        kernel = dilation_px * 2 + 1
        swept = F.max_pool2d(
            swept.float().unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=dilation_px,
        ).squeeze(1).bool()
    target_pair_change = (target[:, 1:] - target[:, :-1]).abs().amax(dim=0)
    target_dynamic = torch.stack(
        [
            target_pair_change[index : index + 2].amax(dim=0)
            > stability_threshold
            for index in range(target_pair_change.shape[0] - 1)
        ]
    )
    # The mask is target-only: shadows, reflections, disocclusions, and other
    # genuinely changing regions cannot be mislabeled as stable background,
    # while a bad prediction cannot hide itself by altering the mask.
    stable_background = ~(swept | target_dynamic)
    pixel_count = stable_background.sum()
    zero = predicted_video.sum() * 0.0
    if not bool(pixel_count):
        return zero, pixel_count.to(dtype=torch.float32)
    error = F.smooth_l1_loss(
        prediction_second,
        target_second,
        beta=beta,
        reduction="none",
    ).mean(dim=0)
    return error[stable_background].mean(), pixel_count.to(dtype=torch.float32)


def clean_video_temporal_losses(
    predicted_clean: list[torch.Tensor],
    target_clean: list[torch.Tensor],
    supervision: list[dict[str, torch.Tensor | int | str]],
    vae,
    *,
    decoded_resolution: int,
    object_beta: float,
    background_beta: float,
    background_dilation_px: int,
    background_stability_threshold: float,
) -> dict[str, torch.Tensor]:
    """Decode exact chunks and average the two temporal objectives over a batch."""
    if not (
        len(predicted_clean) == len(target_clean) == len(supervision)
        and predicted_clean
    ):
        raise ValueError("temporal prediction, target, and supervision batches differ")
    object_losses = []
    background_losses = []
    object_pair_counts = []
    background_pixel_counts = []
    decoded_targets = {}
    for predicted, target, sample in zip(
        predicted_clean, target_clean, supervision
    ):
        window_start = int(sample["latent_window_start"])
        window_chunks = int(sample["latent_window_chunks"])
        decoded_prediction = decode_wan_causal_window(
            vae,
            predicted,
            window_start,
            decoded_resolution,
            window_chunks,
        )
        target_key = (str(sample["sample_id"]), window_start, window_chunks)
        if target_key not in decoded_targets:
            with torch.no_grad():
                decoded_targets[target_key] = decode_wan_causal_window(
                    vae,
                    target.detach(),
                    window_start,
                    decoded_resolution,
                    window_chunks,
                )
        decoded_target = decoded_targets[target_key]
        object_loss, object_pairs = object_track_temporal_residual_loss(
            decoded_prediction,
            decoded_target,
            sample["track_grid"],
            sample["track_visible"],
            beta=object_beta,
        )
        background_loss, background_pixels = background_temporal_residual_loss(
            decoded_prediction,
            decoded_target,
            sample["object_masks"],
            dilation_px=background_dilation_px,
            stability_threshold=background_stability_threshold,
            beta=background_beta,
        )
        object_losses.append(object_loss)
        background_losses.append(background_loss)
        object_pair_counts.append(object_pairs)
        background_pixel_counts.append(background_pixels)
    valid_object_losses = [
        loss
        for loss, count in zip(object_losses, object_pair_counts)
        if bool(count)
    ]
    valid_background_losses = [
        loss
        for loss, count in zip(background_losses, background_pixel_counts)
        if bool(count)
    ]
    zero = predicted_clean[0].sum() * 0.0
    return {
        "object": (
            torch.stack(valid_object_losses).mean()
            if valid_object_losses
            else zero
        ),
        "background": (
            torch.stack(valid_background_losses).mean()
            if valid_background_losses
            else zero
        ),
        "object_pairs": torch.stack(object_pair_counts).mean(),
        "background_pixels": torch.stack(background_pixel_counts).mean(),
    }
