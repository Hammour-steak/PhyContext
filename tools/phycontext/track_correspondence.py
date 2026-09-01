#!/usr/bin/env python3
"""Track4Gen-style feature correspondence for exact PhysSweep 3D tracks."""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


TRACK_CORRESPONDENCE_ARCHITECTURE = "track4gen_swept_latent_v2"


def latent_rgb_windows(rgb_frames: int, latent_frames: int) -> list[tuple[int, int]]:
    """Map Wan causal latent frames to half-open RGB-frame windows."""
    if rgb_frames < 1 or latent_frames < 1 or rgb_frames != 4 * (latent_frames - 1) + 1:
        raise ValueError("Wan RGB/latent frame counts must satisfy R=4*(L-1)+1")
    return [(0, 1)] + [(4 * index - 3, 4 * index + 1) for index in range(1, latent_frames)]


def validate_track_correspondence(
    value: dict[str, torch.Tensor],
    *,
    preprocess_size_px: tuple[int, int] | None = None,
    expected_frames: int | None = None,
    require_background: bool = True,
) -> dict[str, torch.Tensor]:
    required = {
        "track_xy_px",
        "track_depth_m",
        "track_visible",
        "source_frame_indices",
    }
    background_keys = {
        "background_track_xy_px",
        "background_track_depth_m",
        "background_track_visible",
        "background_point_indices",
    }
    if require_background:
        required.update(background_keys)
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"track correspondence is missing tensors: {missing}")
    xy = value["track_xy_px"]
    depth = value["track_depth_m"]
    visible = value["track_visible"]
    source_frames = value["source_frame_indices"]
    present_background_keys = background_keys & set(value)
    if present_background_keys and present_background_keys != background_keys:
        raise ValueError("background track tensors are incomplete")
    has_background = present_background_keys == background_keys
    background_xy = value.get(
        "background_track_xy_px", xy.new_empty((xy.shape[0], 0, 2))
    )
    background_depth = value.get(
        "background_track_depth_m", depth.new_empty((depth.shape[0], 0))
    )
    background_visible = value.get(
        "background_track_visible",
        visible.new_empty((visible.shape[0], 0)),
    )
    background_indices = value.get(
        "background_point_indices",
        source_frames.new_empty((0,)),
    )
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError("track_xy_px must have shape F x O x P x 2")
    if depth.shape != xy.shape[:-1] or visible.shape != depth.shape:
        raise ValueError("track depth and visibility must match F x O x P")
    if source_frames.ndim != 1 or source_frames.shape[0] != xy.shape[0]:
        raise ValueError("source frame indices must match the correspondence frames")
    if expected_frames is not None and xy.shape[0] != expected_frames:
        raise ValueError("track correspondence frame count differs from the video")
    if xy.shape[1] < 1 or xy.shape[1] > 3 or xy.shape[2] != 2048:
        raise ValueError("track correspondence must contain 1-3 objects and 2048 points")
    if background_xy.ndim != 3 or background_xy.shape[0] != xy.shape[0] or background_xy.shape[-1] != 2:
        raise ValueError("background_track_xy_px must have shape F x S x 2")
    if background_depth.shape != background_xy.shape[:-1] or background_visible.shape != background_depth.shape:
        raise ValueError("background track depth and visibility must match F x S")
    if has_background and not 1 <= background_xy.shape[1] <= 256:
        raise ValueError("track correspondence must contain 1-256 background points")
    if has_background and not bool(background_visible[0].all()):
        raise ValueError("every cached background query must be visible in frame zero")
    if background_indices.ndim != 1 or background_indices.shape[0] != background_xy.shape[1]:
        raise ValueError("background point indices must match the background point axis")
    if background_indices.dtype != torch.int64:
        raise ValueError("background point indices must be int64")
    if has_background and (
        bool((background_indices < 0).any())
        or len(torch.unique(background_indices)) != len(background_indices)
    ):
        raise ValueError("background point indices must be unique and nonnegative")
    if xy.dtype != torch.float32 or depth.dtype != torch.float32 or background_xy.dtype != torch.float32 or background_depth.dtype != torch.float32:
        raise ValueError("track coordinates and depth must be float32")
    if visible.dtype != torch.bool or background_visible.dtype != torch.bool:
        raise ValueError("track visibility must be boolean")
    if source_frames.dtype != torch.int64:
        raise ValueError("source frame indices must be int64")
    if not bool(torch.isfinite(xy).all()) or not bool(torch.isfinite(depth).all()) or not bool(torch.isfinite(background_xy).all()) or not bool(torch.isfinite(background_depth).all()):
        raise ValueError("track correspondence contains non-finite geometry")
    if bool((visible & (depth <= 0)).any()):
        raise ValueError("visible correspondence points must have positive depth")
    if bool((background_visible & (background_depth <= 0)).any()):
        raise ValueError("visible background correspondence points must have positive depth")
    if source_frames.numel() > 1 and not bool((source_frames[1:] > source_frames[:-1]).all()):
        raise ValueError("source frame indices must be strictly increasing")
    if preprocess_size_px is not None:
        width, height = [int(item) for item in preprocess_size_px]
        if width <= 0 or height <= 0:
            raise ValueError("preprocess size must be positive")
        visible_xy = torch.cat((xy[visible], background_xy[background_visible]), dim=0)
        # DaS visibility is resolved after projecting continuous pixel-center
        # coordinates to raster pixels with np.rint.  A center such as -0.49
        # therefore belongs to boundary pixel 0 and must not be rejected here.
        # Validate the same raster-space contract instead of imposing the
        # stricter continuous interval [0, size), which disagrees at borders.
        visible_pixels = torch.round(visible_xy)
        if visible_xy.numel() and bool(
            (
                (visible_pixels[:, 0] < 0)
                | (visible_pixels[:, 0] >= width)
                | (visible_pixels[:, 1] < 0)
                | (visible_pixels[:, 1] >= height)
            ).any()
        ):
            raise ValueError(
                "visible correspondence rasterizes outside the preprocessed frame"
            )
    return value


class _SpatialRefinerBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = math.gcd(channels, 32)
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        # Track4Gen's spatial refiner starts as an identity map. The branch can
        # learn correspondence without perturbing pretrained Wan features.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(F.silu(self.depthwise(F.silu(self.norm(value)))))
        return value + residual


class TrackCorrespondenceAdapter(nn.Module):
    """Refine one Wan block's features and feed them back through a zero bridge."""

    def __init__(self, hidden_dim: int, feature_dim: int = 256, refiner_blocks: int = 4):
        super().__init__()
        if hidden_dim <= 0 or feature_dim <= 0 or refiner_blocks <= 0:
            raise ValueError("track correspondence dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.refiner_blocks = int(refiner_blocks)
        self.spatial_upsample_factor = 2
        self.architecture = TRACK_CORRESPONDENCE_ARCHITECTURE
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, feature_dim)
        self.refiner = nn.Sequential(
            *[_SpatialRefinerBlock(feature_dim) for _ in range(refiner_blocks)]
        )
        self.feedback = nn.Linear(feature_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.feedback.weight)
        self._features: list[torch.Tensor] | None = None

    def _refine(
        self, tokens: torch.Tensor, grid_sizes: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Wan track features must have shape B x L x hidden_dim")
        if grid_sizes.ndim != 2 or grid_sizes.shape != (tokens.shape[0], 3):
            raise ValueError("Wan grid sizes must have shape B x 3")
        projected = self.input_projection(self.input_norm(tokens))
        refined_tokens = torch.zeros_like(projected)
        feature_maps = []
        for batch_index, grid in enumerate(grid_sizes.tolist()):
            frames, height, width = [int(item) for item in grid]
            token_count = frames * height * width
            if token_count <= 0 or token_count > tokens.shape[1]:
                raise ValueError("Wan grid size is incompatible with its token sequence")
            frame_maps = (
                projected[batch_index, :token_count]
                .view(frames, height, width, self.feature_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            high_resolution_maps = F.interpolate(
                frame_maps,
                scale_factor=self.spatial_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            high_resolution_maps = self.refiner(high_resolution_maps)
            feedback_maps = F.interpolate(
                high_resolution_maps,
                size=(height, width),
                mode="area",
            )
            refined = feedback_maps.permute(0, 2, 3, 1).reshape(token_count, self.feature_dim)
            refined_tokens[batch_index, :token_count] = refined
            feature_maps.append(
                high_resolution_maps.permute(1, 0, 2, 3).contiguous()
            )
        return refined_tokens, feature_maps

    def forward(self, tokens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        refined, feature_maps = self._refine(tokens, grid_sizes)
        self._features = feature_maps
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            feedback = self.feedback(refined.detach().float())
        return tokens + feedback.to(dtype=tokens.dtype)

    def consume_features(self) -> list[torch.Tensor]:
        if self._features is None:
            raise RuntimeError("Wan did not produce track correspondence features")
        features = self._features
        self._features = None
        return features


def inject_track_correspondence(
    model: nn.Module,
    *,
    block_index: int,
    feature_dim: int = 256,
    refiner_blocks: int = 4,
) -> TrackCorrespondenceAdapter:
    if hasattr(model, "phycontext_track_correspondence"):
        raise ValueError("track correspondence is already injected")
    if not 0 <= block_index < len(model.blocks):
        raise ValueError("track correspondence block index is out of range")
    adapter = TrackCorrespondenceAdapter(
        int(model.dim), feature_dim=feature_dim, refiner_blocks=refiner_blocks
    )
    adapter.block_index = int(block_index)
    model.add_module("phycontext_track_correspondence", adapter)
    block = model.blocks[block_index]
    if hasattr(block, "_phycontext_track_base_forward"):
        raise ValueError("selected Wan block is already wrapped for correspondence")
    block._phycontext_track_base_forward = block.forward

    def track_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        hidden = self._phycontext_track_base_forward(
            x, e, seq_lens, grid_sizes, freqs, context, context_lens
        )
        return adapter(hidden, grid_sizes)

    block.forward = types.MethodType(track_forward, block)
    return adapter


def track_correspondence_feature_parameters(model: nn.Module):
    """Parameters trained by exact material-point feature correspondence."""
    adapter = getattr(model, "phycontext_track_correspondence", None)
    if adapter is not None:
        yield from adapter.input_norm.parameters()
        yield from adapter.input_projection.parameters()
        yield from adapter.refiner.parameters()


def track_correspondence_feedback_parameters(model: nn.Module):
    """Zero bridge trained only by generation-side objectives."""
    adapter = getattr(model, "phycontext_track_correspondence", None)
    if adapter is not None:
        yield from adapter.feedback.parameters()


def consume_track_correspondence_features(model: nn.Module) -> list[torch.Tensor]:
    adapter = getattr(model, "phycontext_track_correspondence", None)
    if adapter is None:
        raise RuntimeError("track correspondence is not injected")
    return adapter.consume_features()


def _pixel_to_feature(
    xy: torch.Tensor,
    *,
    preprocess_size_px: tuple[int, int],
    feature_size: tuple[int, int],
) -> torch.Tensor:
    width, height = [float(item) for item in preprocess_size_px]
    feature_height, feature_width = [int(item) for item in feature_size]
    result = xy.float().clone()
    result[..., 0] = (result[..., 0] + 0.5) * feature_width / width - 0.5
    result[..., 1] = (result[..., 1] + 0.5) * feature_height / height - 0.5
    return result


def _feature_to_grid_sample(
    xy: torch.Tensor, feature_size: tuple[int, int]
) -> torch.Tensor:
    height, width = [int(item) for item in feature_size]
    result = xy.float().clone()
    result[..., 0] = 2.0 * (result[..., 0] + 0.5) / width - 1.0
    result[..., 1] = 2.0 * (result[..., 1] + 0.5) / height - 1.0
    return result


def _sample_speed_balanced_candidates(
    xy: torch.Tensor,
    visible: torch.Tensor,
    *,
    latent_frames: int,
    feature_size: tuple[int, int],
    preprocess_size_px: tuple[int, int],
    maximum_pairs: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if xy.ndim != 3 or xy.shape[-1] != 2 or visible.shape != xy.shape[:-1]:
        raise ValueError("candidate tracks must have shape F x P x 2")
    windows = latent_rgb_windows(int(xy.shape[0]), latent_frames)
    anchor_visible = visible[0]
    anchor_xy = xy[0]
    target_frames = []
    point_indices = []
    displacements = []
    for latent_index, (start, end) in enumerate(windows[1:], 1):
        window_visible = visible[start:end]
        shared = anchor_visible & window_visible.any(dim=0)
        points = torch.where(shared)[0]
        if not len(points):
            continue
        selected_visible = window_visible[:, points]
        selected_xy = xy[start:end, points]
        counts = selected_visible.sum(dim=0).clamp_min(1).unsqueeze(-1)
        target_xy = (
            selected_xy * selected_visible.unsqueeze(-1)
        ).sum(dim=0) / counts
        source_feature = _pixel_to_feature(
            anchor_xy[points],
            preprocess_size_px=preprocess_size_px,
            feature_size=feature_size,
        )
        target_feature = _pixel_to_feature(
            target_xy,
            preprocess_size_px=preprocess_size_px,
            feature_size=feature_size,
        )
        displacement = torch.linalg.vector_norm(target_feature - source_feature, dim=-1)
        # One material-point query per source feature cell and target chunk avoids
        # contradictory labels when many 3D samples rasterize to the same token.
        source_cell = torch.floor(source_feature + 0.5).long()
        source_cell[:, 0].clamp_(0, feature_size[1] - 1)
        source_cell[:, 1].clamp_(0, feature_size[0] - 1)
        # One query per source cell and target chunk avoids contradictory
        # labels. Within a collision retain the fastest material point.
        keys = (
            latent_index * feature_size[0] * feature_size[1]
            + source_cell[:, 1] * feature_size[1]
            + source_cell[:, 0]
        )
        speed_order = torch.argsort(
            displacement, descending=True, stable=True
        )
        order = speed_order[
            torch.argsort(keys[speed_order], stable=True)
        ]
        ordered_keys = keys[order]
        keep = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=xy.device),
                ordered_keys[1:] != ordered_keys[:-1],
            )
        )
        selected = order[keep]
        count = len(selected)
        target_frames.append(
            torch.full((count,), latent_index, dtype=torch.long, device=xy.device)
        )
        point_indices.append(points[selected])
        displacements.append(displacement[selected])
    if not target_frames:
        empty = torch.empty(0, dtype=torch.long, device=xy.device)
        return empty, empty, empty.float()
    target_frames = torch.cat(target_frames)
    point_indices = torch.cat(point_indices)
    displacements = torch.cat(displacements)
    bins = (
        displacements.lt(1.0),
        displacements.ge(1.0) & displacements.lt(3.0),
        displacements.ge(3.0),
    )
    quota = max(1, maximum_pairs // 3)
    chosen = []
    for mask in bins:
        candidates = torch.where(mask)[0]
        if len(candidates):
            order = torch.randperm(len(candidates), device=xy.device, generator=generator)
            chosen.append(candidates[order[: min(quota, len(candidates))]])
    selected = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=xy.device)
    if len(selected) < maximum_pairs:
        available_mask = torch.ones(len(target_frames), dtype=torch.bool, device=xy.device)
        if len(selected):
            available_mask[selected] = False
        available = torch.where(available_mask)[0]
        if len(available):
            order = torch.randperm(len(available), device=xy.device, generator=generator)
            selected = torch.cat(
                (selected, available[order[: maximum_pairs - len(selected)]])
            )
    if len(selected) > maximum_pairs:
        selected = selected[:maximum_pairs]
    return (
        target_frames[selected],
        point_indices[selected],
        displacements[selected],
    )


def track4gen_correspondence_loss(
    feature_maps: list[torch.Tensor],
    correspondences: list[dict[str, torch.Tensor]],
    *,
    preprocess_size_px: tuple[int, int],
    maximum_pairs: int,
    temperature: float,
    gaussian_sigma: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Track foreground and background material features without trajectory input."""
    if len(feature_maps) != len(correspondences) or not feature_maps:
        raise ValueError("feature and correspondence batches must be equal and nonempty")
    if maximum_pairs <= 0 or temperature <= 0 or gaussian_sigma <= 0:
        raise ValueError("Track4Gen sampling and temperature values must be positive")
    zero = feature_maps[0].sum() * 0.0
    sample_losses = []
    total_pairs = 0
    total_displacement = zero.detach()
    total_kl = zero.detach()
    total_epe = zero.detach()
    pck_1_hits = 0
    fast_pairs = 0
    fast_epe = zero.detach()
    fast_pck_1_hits = 0
    foreground_pairs = 0
    background_pairs = 0
    foreground_pck_1_hits = 0
    background_pck_1_hits = 0
    for features, correspondence in zip(feature_maps, correspondences):
        if features.ndim != 4:
            raise ValueError("refined feature map must have shape D x F x H x W")
        _, latent_frames, feature_height, feature_width = features.shape
        correspondence = validate_track_correspondence(
            correspondence,
            preprocess_size_px=preprocess_size_px,
            expected_frames=4 * (latent_frames - 1) + 1,
        )
        source_frames = correspondence["source_frame_indices"]
        expected_source_frames = torch.arange(
            source_frames.shape[0],
            dtype=source_frames.dtype,
            device=source_frames.device,
        )
        if not torch.equal(source_frames, expected_source_frames):
            raise ValueError(
                "Track4Gen requires consecutive source frames starting at zero"
            )
        dynamic_xy = correspondence["track_xy_px"].to(features.device)
        dynamic_visible = correspondence["track_visible"].to(features.device)
        dynamic_xy = dynamic_xy.flatten(1, 2)
        dynamic_visible = dynamic_visible.flatten(1, 2)
        background_xy = correspondence["background_track_xy_px"].to(features.device)
        background_visible = correspondence["background_track_visible"].to(
            features.device
        )
        foreground_limit = maximum_pairs // 2
        background_limit = maximum_pairs - foreground_limit
        foreground = _sample_speed_balanced_candidates(
            dynamic_xy,
            dynamic_visible,
            latent_frames=latent_frames,
            feature_size=(feature_height, feature_width),
            preprocess_size_px=preprocess_size_px,
            maximum_pairs=foreground_limit,
            generator=generator,
        )
        background = _sample_speed_balanced_candidates(
            background_xy,
            background_visible,
            latent_frames=latent_frames,
            feature_size=(feature_height, feature_width),
            preprocess_size_px=preprocess_size_px,
            maximum_pairs=background_limit,
            generator=generator,
        )
        if len(foreground[0]) and len(background[0]):
            foreground_source = _pixel_to_feature(
                dynamic_xy[0, foreground[1]],
                preprocess_size_px=preprocess_size_px,
                feature_size=(feature_height, feature_width),
            )
            background_source = _pixel_to_feature(
                background_xy[0, background[1]],
                preprocess_size_px=preprocess_size_px,
                feature_size=(feature_height, feature_width),
            )

            def source_keys(frames: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
                cells = torch.floor(source + 0.5).long()
                cells[:, 0].clamp_(0, feature_width - 1)
                cells[:, 1].clamp_(0, feature_height - 1)
                return (
                    frames * feature_height * feature_width
                    + cells[:, 1] * feature_width
                    + cells[:, 0]
                )

            # A feature token has no foreground/background identity axis. Keep
            # foreground when both categories land in the same source token so
            # one query cannot be assigned contradictory destinations.
            keep_background = ~torch.isin(
                source_keys(background[0], background_source),
                source_keys(foreground[0], foreground_source),
            )
            background = tuple(value[keep_background] for value in background)
        if not len(foreground[0]) or not len(background[0]):
            raise ValueError(
                "each Track4Gen video must have foreground and background pairs"
            )
        target_frames = torch.cat((foreground[0], background[0]))
        points = torch.cat((foreground[1], background[1]))
        displacement = torch.cat((foreground[2], background[2]))
        is_background = torch.cat(
            (
                torch.zeros_like(foreground[0], dtype=torch.bool),
                torch.ones_like(background[0], dtype=torch.bool),
            )
        )
        anchor_xy = torch.empty(
            len(points), 2, dtype=dynamic_xy.dtype, device=features.device
        )
        anchor_xy[~is_background] = dynamic_xy[0, points[~is_background]]
        anchor_xy[is_background] = background_xy[0, points[is_background]]
        anchor_feature_xy = _pixel_to_feature(
            anchor_xy,
            preprocess_size_px=preprocess_size_px,
            feature_size=(feature_height, feature_width),
        )
        query_grid = _feature_to_grid_sample(
            anchor_feature_xy, (feature_height, feature_width)
        ).view(1, -1, 1, 2)
        with torch.autocast(device_type=features.device.type, enabled=False):
            query = F.grid_sample(
                features[:, :1].permute(1, 0, 2, 3).float(),
                query_grid.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0, :, :, 0].T
            query = F.normalize(query, dim=-1, eps=1.0e-6)
            y_grid, x_grid = torch.meshgrid(
                torch.arange(feature_height, device=features.device, dtype=torch.float32),
                torch.arange(feature_width, device=features.device, dtype=torch.float32),
                indexing="ij",
            )
            feature_grid = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)
            windows = latent_rgb_windows(int(dynamic_xy.shape[0]), latent_frames)
            sample_foreground_loss = query.sum() * 0.0
            sample_background_loss = query.sum() * 0.0
            sample_foreground_pairs = 0
            sample_background_pairs = 0
            sample_pairs = 0
            for latent_index in target_frames.unique(sorted=True).tolist():
                selection = torch.where(target_frames == latent_index)[0]
                target_features = (
                    features[:, latent_index].permute(1, 2, 0).reshape(-1, features.shape[0]).float()
                )
                target_features = F.normalize(target_features, dim=-1, eps=1.0e-6)
                logits = query[selection] @ target_features.T / temperature
                start, end = windows[latent_index]
                selected_points = points[selection]
                selected_background = is_background[selection]
                path_xy = torch.empty(
                    end - start,
                    len(selection),
                    2,
                    dtype=dynamic_xy.dtype,
                    device=features.device,
                )
                path_visible = torch.empty(
                    end - start,
                    len(selection),
                    dtype=torch.bool,
                    device=features.device,
                )
                if bool((~selected_background).any()):
                    chosen = torch.where(~selected_background)[0]
                    path_xy[:, chosen] = dynamic_xy[
                        start:end, selected_points[chosen]
                    ]
                    path_visible[:, chosen] = dynamic_visible[
                        start:end, selected_points[chosen]
                    ]
                if bool(selected_background.any()):
                    chosen = torch.where(selected_background)[0]
                    path_xy[:, chosen] = background_xy[
                        start:end, selected_points[chosen]
                    ]
                    path_visible[:, chosen] = background_visible[
                        start:end, selected_points[chosen]
                    ]
                path_feature_xy = _pixel_to_feature(
                    path_xy,
                    preprocess_size_px=preprocess_size_px,
                    feature_size=(feature_height, feature_width),
                )
                distance_squared = (
                    path_feature_xy[:, :, None, :] - feature_grid[None, None, :, :]
                ).square().sum(dim=-1)
                target = torch.exp(
                    -distance_squared / (2.0 * gaussian_sigma * gaussian_sigma)
                ) * path_visible[:, :, None].float()
                target = target.sum(dim=0)
                target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
                log_probability = F.log_softmax(logits, dim=-1)
                classification = -(target * log_probability).sum(dim=-1)
                target_entropy = -(
                    target * target.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                probability = log_probability.exp()
                predicted_coordinate = probability @ feature_grid
                target_coordinate = target @ feature_grid
                coordinate = F.smooth_l1_loss(
                    predicted_coordinate,
                    target_coordinate,
                    beta=1.0,
                    reduction="none",
                ).mean(dim=-1)
                # Track4Gen supervises the soft-argmax coordinate with Huber.
                # Cross-entropy is retained only as a diagnostic; optimizing it
                # would turn the task into coarse cell classification again.
                foreground_selection = ~selected_background
                if bool(foreground_selection.any()):
                    sample_foreground_loss = sample_foreground_loss + coordinate[
                        foreground_selection
                    ].sum()
                    sample_foreground_pairs += int(foreground_selection.sum().item())
                if bool(selected_background.any()):
                    sample_background_loss = sample_background_loss + coordinate[
                        selected_background
                    ].sum()
                    sample_background_pairs += int(selected_background.sum().item())
                predicted_feature_coordinate = predicted_coordinate
                target_feature_coordinate = target_coordinate
                endpoint_error = torch.linalg.vector_norm(
                    predicted_feature_coordinate - target_feature_coordinate,
                    dim=-1,
                )
                fast = displacement[selection].ge(3.0)
                total_kl = total_kl + (
                    classification - target_entropy
                ).clamp_min(0.0).sum().detach()
                total_epe = total_epe + endpoint_error.sum().detach()
                pck_1_hits += int(endpoint_error.le(1.0).sum().item())
                background_hits = selected_background & endpoint_error.le(1.0)
                foreground_hits = (~selected_background) & endpoint_error.le(1.0)
                background_pck_1_hits += int(background_hits.sum().item())
                foreground_pck_1_hits += int(foreground_hits.sum().item())
                if bool(fast.any()):
                    fast_epe = fast_epe + endpoint_error[fast].sum().detach()
                    fast_pck_1_hits += int(
                        endpoint_error[fast].le(1.0).sum().item()
                    )
                sample_pairs += len(selection)
            if sample_foreground_pairs <= 0 or sample_background_pairs <= 0:
                raise RuntimeError("Track4Gen lost a foreground/background category")
            sample_loss = 0.5 * (
                sample_foreground_loss / sample_foreground_pairs
                + sample_background_loss / sample_background_pairs
            )
            sample_losses.append(sample_loss)
            total_pairs += sample_pairs
            total_displacement = total_displacement + displacement.sum().detach()
            fast_pairs += int(displacement.ge(3.0).sum().item())
            foreground_pairs += int((~is_background).sum().item())
            background_pairs += int(is_background.sum().item())
    if len(sample_losses) != len(feature_maps) or total_pairs <= 0:
        raise RuntimeError("Track4Gen did not produce one loss for every video")
    # Every video contributes equally. With a fixed local batch size, DDP's
    # rank average is therefore exactly the global per-video objective instead
    # of being biased by the number of visible points on each rank.
    loss = torch.stack(sample_losses).mean()
    mean_displacement = total_displacement / total_pairs
    mean_kl = total_kl / total_pairs
    mean_epe = total_epe / total_pairs
    pck_1 = loss.new_tensor(float(pck_1_hits) / total_pairs)
    if fast_pairs:
        mean_fast_epe = fast_epe / fast_pairs
        fast_pck_1 = loss.new_tensor(float(fast_pck_1_hits) / fast_pairs)
    else:
        mean_fast_epe = loss.new_zeros(())
        fast_pck_1 = loss.new_zeros(())
    return {
        "loss": loss,
        "pairs": loss.new_tensor(float(total_pairs)),
        "fast_pairs": loss.new_tensor(float(fast_pairs)),
        "mean_displacement_tokens": mean_displacement.to(loss.device),
        "kl": mean_kl.to(loss.device),
        "epe_tokens": mean_epe.to(loss.device),
        "pck_1": pck_1,
        "fast_epe_tokens": mean_fast_epe.to(loss.device),
        "fast_pck_1": fast_pck_1,
        "foreground_pairs": loss.new_tensor(float(foreground_pairs)),
        "background_pairs": loss.new_tensor(float(background_pairs)),
        "foreground_pck_1": loss.new_tensor(
            float(foreground_pck_1_hits) / foreground_pairs
        ),
        "background_pck_1": loss.new_tensor(
            float(background_pck_1_hits) / background_pairs
        ),
    }
