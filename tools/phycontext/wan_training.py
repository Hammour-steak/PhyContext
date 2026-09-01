import json
import math
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from track_correspondence import inject_track_correspondence


SWEEP_AXES = ("mass_kg", "contact_friction", "contact_restitution")
TRAJECTORY_INPUT_SOURCES = ("target", "nominal_base")
FORMAL_RESPONSE_AXIS_CYCLE = (
    "contact_friction",
    "contact_restitution",
    "contact_friction",
    "contact_restitution",
    "mass_kg",
)


def index_nominal_trajectory_records(records: list[dict]) -> dict[str, dict]:
    """Index the single canonical-base trajectory record for each scene."""
    grouped: dict[str, list[dict]] = {}
    for item in records:
        if item["record"]["sweep"]["mode"] == "base":
            grouped.setdefault(item["base_scene_id"], []).append(item)
    scene_ids = {item["base_scene_id"] for item in records}
    invalid = sorted(
        scene_id for scene_id in scene_ids if len(grouped.get(scene_id, [])) != 1
    )
    if invalid:
        raise ValueError(
            "each base scene must contain exactly one nominal trajectory record: "
            + ", ".join(invalid[:5])
        )
    return {scene_id: items[0] for scene_id, items in grouped.items()}


def select_trajectory_input_records(
    targets: list[dict],
    nominal_records: dict[str, dict],
    source: str,
) -> list[dict]:
    """Resolve a sample-bound trajectory, or an explicit fixed-path ablation."""
    if source not in TRAJECTORY_INPUT_SOURCES:
        raise ValueError(f"unsupported trajectory input source: {source}")
    if source == "target":
        return list(targets)
    missing = sorted(
        {item["base_scene_id"] for item in targets} - set(nominal_records)
    )
    if missing:
        raise ValueError(
            "missing nominal trajectory records for: " + ", ".join(missing[:5])
        )
    return [nominal_records[item["base_scene_id"]] for item in targets]


def select_sweep_endpoint_pairs(
    records: list[dict],
    base_scene_count: int,
) -> list[dict]:
    """Select the low/high endpoint pair for every axis in complete base groups."""
    if base_scene_count <= 0:
        raise ValueError("base-scene count must be positive")
    groups: dict[str, list[dict]] = {}
    for item in records:
        scene_id = item["base_scene_id"]
        if scene_id not in groups:
            if len(groups) == base_scene_count:
                continue
            groups[scene_id] = []
        groups[scene_id].append(item)

    pairs = []
    for scene_id, group in groups.items():
        base_items = [
            item for item in group if item["record"]["sweep"]["mode"] == "base"
        ]
        if len(base_items) != 1:
            raise ValueError(f"{scene_id} must contain exactly one canonical base record")
        base_item = base_items[0]
        base_levels = base_item["record"]["sweep"]["base_level_indices"]
        for axis in SWEEP_AXES:
            levels = {int(base_levels[axis]): base_item}
            for item in group:
                sweep = item["record"]["sweep"]
                if sweep.get("axis") != axis:
                    continue
                level = int(sweep["level_index"])
                if level in levels:
                    raise ValueError(f"duplicate {scene_id}/{axis} level {level}")
                levels[level] = item
            expected_count = int(base_item["record"]["sweep"]["level_count"])
            if sorted(levels) != list(range(expected_count)):
                raise ValueError(
                    f"{scene_id}/{axis} is incomplete: levels={sorted(levels)}"
                )
            pairs.append(
                {
                    "base_scene_id": scene_id,
                    "axis": axis,
                    "low": levels[min(levels)],
                    "high": levels[max(levels)],
                }
            )
    return pairs


def is_formal_response_step(step: int) -> bool:
    """Use two response updates for every three ordinary reconstruction updates."""
    if step < 0:
        raise ValueError("training step must be non-negative")
    return step % 5 < 2


def formal_response_step_index(step: int) -> int:
    """Return the zero-based response-update index at a response step."""
    if not is_formal_response_step(step):
        raise ValueError("ordinary steps do not have a response index")
    periods, offset = divmod(step, 5)
    return periods * 2 + offset


def formal_ordinary_step_index(step: int) -> int:
    """Return the zero-based ordinary-update index at an ordinary step."""
    if is_formal_response_step(step):
        raise ValueError("response steps do not have an ordinary index")
    response_before = (step // 5) * 2 + min(step % 5, 2)
    return step - response_before


def _cyclic_permuted_item(items: list, index: int, seed: int):
    if not items or index < 0:
        raise ValueError("cyclic selection requires items and a non-negative index")
    cycle, offset = divmod(index, len(items))
    generator = torch.Generator().manual_seed(seed + cycle)
    permutation = torch.randperm(len(items), generator=generator).tolist()
    return items[permutation[offset]]


def formal_response_axis_occurrence_index(
    response_micro_index: int,
    axis: str,
) -> int:
    """Return the axis-local index for one response microbatch.

    The global response schedule interleaves axes with unequal frequencies.
    Pair permutations must therefore advance by the number of prior occurrences
    of the selected axis, rather than by the global response index; otherwise
    each axis skips offsets and repeats pairs before covering its own dataset.
    """
    if response_micro_index < 0:
        raise ValueError("response microbatch index must be non-negative")
    if axis not in SWEEP_AXES:
        raise ValueError(f"unsupported response axis: {axis}")
    cycle_count, cycle_offset = divmod(
        response_micro_index, len(FORMAL_RESPONSE_AXIS_CYCLE)
    )
    if FORMAL_RESPONSE_AXIS_CYCLE[cycle_offset] != axis:
        raise ValueError("response axis does not match the formal axis cycle")
    return (
        cycle_count * FORMAL_RESPONSE_AXIS_CYCLE.count(axis)
        + FORMAL_RESPONSE_AXIS_CYCLE[:cycle_offset].count(axis)
    )


def make_formal_training_batch(
    records: list[dict],
    endpoint_pairs: list[dict],
    step: int,
    accumulation_index: int,
    gradient_accumulation: int,
    rank: int,
    world_size: int,
    seed: int,
    response_updates: bool = True,
) -> tuple[list[dict], str, str | None]:
    """Build one deterministic two-video microbatch for formal DDP training."""
    if gradient_accumulation <= 0 or not 0 <= accumulation_index < gradient_accumulation:
        raise ValueError("gradient accumulation index is invalid")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("distributed rank is invalid")
    if not records:
        raise ValueError("formal training requires records")

    if response_updates and is_formal_response_step(step):
        response_index = formal_response_step_index(step)
        response_micro_index = response_index * gradient_accumulation + accumulation_index
        axis = FORMAL_RESPONSE_AXIS_CYCLE[
            response_micro_index % len(FORMAL_RESPONSE_AXIS_CYCLE)
        ]
        axis_pairs = [pair for pair in endpoint_pairs if pair["axis"] == axis]
        axis_index = formal_response_axis_occurrence_index(
            response_micro_index, axis
        )
        draw_index = axis_index * world_size + rank
        pair = _cyclic_permuted_item(
            axis_pairs,
            draw_index,
            seed + 10_000 * SWEEP_AXES.index(axis),
        )
        return [pair["low"], pair["high"]], "response", axis

    ordinary_index = (
        formal_ordinary_step_index(step) if response_updates else step
    )
    ordinary_micro_index = ordinary_index * gradient_accumulation + accumulation_index
    start = (ordinary_micro_index * world_size + rank) * 2
    batch = [
        _cyclic_permuted_item(records, start + offset, seed + 100_000)
        for offset in range(2)
    ]
    return batch, "ordinary", None


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            delta = self.up(self.down(self.dropout(inputs).float()))
        return base_output + delta.to(base_output.dtype) * self.scaling


TRAJECTORY_REPRESENTATION_CHANNELS = {
    "dense_point_tracks": 18,
    "das_3d_tracks": 12,
}
TRAJECTORY_CHANNELS_PER_OBJECT = {
    "dense_point_tracks": 6,
    "das_3d_tracks": 4,
}
TRAJECTORY_CHANNEL_SEMANTICS = {
    "dense_point_tracks": (
        "source_occupancy",
        "current_occupancy",
        "delta_x",
        "delta_y",
        "depth",
        "validity",
    ),
    "das_3d_tracks": (
        "identity_r",
        "identity_g",
        "identity_b",
        "visibility",
    ),
}

def canonical_trajectory_representation(representation: str) -> str:
    if representation not in TRAJECTORY_REPRESENTATION_CHANNELS:
        raise ValueError(f"unsupported trajectory representation: {representation}")
    return representation

def trajectory_channel_names(
    representation: str,
    object_slots: int = 3,
) -> list[str]:
    representation = canonical_trajectory_representation(representation)
    if object_slots <= 0:
        raise ValueError("trajectory object-slot count must be positive")
    return [
        f"object_{slot}_{channel}"
        for slot in range(object_slots)
        for channel in TRAJECTORY_CHANNEL_SEMANTICS[representation]
    ]

def validate_point_track_maps(
    point_track_maps: list[torch.Tensor] | None,
    representation: str,
) -> list[torch.Tensor]:
    canonical_trajectory_representation(representation)
    if point_track_maps is None:
        raise ValueError(
            f"{representation} requires pre-rasterized point tracks"
        )
    expected_channels = TRAJECTORY_REPRESENTATION_CHANNELS[representation]
    validated = []
    for condition in point_track_maps:
        if condition.ndim != 4 or condition.shape[0] != expected_channels:
            raise ValueError(
                "point-track condition must have shape "
                f"[{expected_channels}, trajectory_frames, latent_height, latent_width]"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("point-track condition contains non-finite values")
        if representation == "das_3d_tracks":
            slots = condition.reshape(
                expected_channels // 4, 4, *condition.shape[1:]
            )
            rgb = slots[:, :3]
            visibility = slots[:, 3:4]
            if bool((condition < -1.0e-6).any()) or bool(
                (condition > 1.0 + 1.0e-6).any()
            ):
                raise ValueError("das_3d_tracks values must lie in [0, 1]")
            binary_visibility = torch.logical_or(
                visibility == 0, visibility == 1
            )
            if not bool(binary_visibility.all()):
                raise ValueError("das_3d_tracks visibility must be binary")
            background_rgb = rgb.masked_select(visibility.expand_as(rgb) == 0)
            if len(background_rgb) and bool((background_rgb.abs() > 1.0e-6).any()):
                raise ValueError(
                    "das_3d_tracks RGB must be zero outside visible cells"
                )
        validated.append(condition)
    return validated


def validate_point_track_object_slots(
    point_track_map: torch.Tensor,
    representation: str,
    object_count: int,
) -> torch.Tensor:
    """Validate fixed-slot padding against the sample's bound object count."""
    representation = canonical_trajectory_representation(representation)
    if not 1 <= int(object_count) <= 3:
        raise ValueError("point-track object count must be between one and three")
    point_track_map = validate_point_track_maps(
        [point_track_map], representation
    )[0]
    channels_per_object = TRAJECTORY_CHANNELS_PER_OBJECT[representation]
    unused = point_track_map[int(object_count) * channels_per_object :]
    if unused.numel() and bool((unused.abs() > 1.0e-6).any()):
        raise ValueError("unused point-track object slots must be zero")
    return point_track_map


def motion_mask_from_point_track_map(
    point_track_map: torch.Tensor,
    representation: str | None = None,
) -> torch.Tensor:
    """Build latent-rate current occupancy from point-track visibility.

    The trajectory auxiliary losses need the object's position at each target
    frame, not the union of that position with frame zero.  Reconstruction and
    response losses expand this mask with ``source_target_motion_envelope`` at
    their call sites when they need to cover both removal and arrival regions.
    """
    if point_track_map.ndim != 4:
        raise ValueError("point-track map must have shape C x F x H x W")
    if representation is None:
        matches = [
            name
            for name, channel_count in TRAJECTORY_REPRESENTATION_CHANNELS.items()
            if point_track_map.shape[0] == channel_count
        ]
        if len(matches) != 1:
            raise ValueError("point-track map has an unexpected channel count")
        representation = matches[0]
    representation = canonical_trajectory_representation(representation)
    if point_track_map.shape[0] != TRAJECTORY_REPRESENTATION_CHANNELS[
        representation
    ]:
        raise ValueError("point-track map does not match its representation")
    point_track_map = validate_point_track_maps(
        [point_track_map], representation
    )[0]
    if representation == "dense_point_tracks":
        current = point_track_map[1::6].amax(dim=0)
    else:
        visibility = point_track_map[3::4]
        full_current = visibility.amax(dim=0)
        frame_count = int(full_current.shape[0])
        if frame_count < 1 or (frame_count - 1) % 4:
            raise ValueError(
                "das_3d_tracks must contain 4n+1 full-rate trajectory frames"
            )
        if frame_count == 1:
            current = full_current
        else:
            current = torch.cat(
                (
                    full_current[:1],
                    full_current[1:].reshape(
                        (frame_count - 1) // 4, 4, *full_current.shape[1:]
                    ).amax(dim=1),
                ),
                dim=0,
            )
    return current.gt(0).unsqueeze(0).to(dtype=point_track_map.dtype)


class TrajectoryPatchConditioner(nn.Module):
    """Project DaS-inspired fixed-slot trajectories into Wan patch features."""

    def __init__(
        self,
        hidden_dim: int,
        patch_size: tuple[int, int, int],
        rank: int = 32,
        representation: str = "dense_point_tracks",
        architecture: str | None = None,
    ):
        super().__init__()
        if hidden_dim <= 0 or rank <= 0 or len(patch_size) != 3:
            raise ValueError("trajectory conditioner dimensions are invalid")
        if any(int(value) <= 0 for value in patch_size):
            raise ValueError("trajectory patch dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.rank = int(rank)
        representation = canonical_trajectory_representation(representation)
        if representation not in TRAJECTORY_REPRESENTATION_CHANNELS:
            raise ValueError("unsupported trajectory representation")
        self.representation = representation
        self.input_channels = TRAJECTORY_REPRESENTATION_CHANNELS[representation]
        if architecture is None:
            architecture = (
                "full_frame_causal_patch_v2"
                if representation == "das_3d_tracks"
                else "framewise_patch"
            )
        allowed_architectures = {"framewise_patch"}
        if representation == "das_3d_tracks":
            allowed_architectures.add("full_frame_causal_patch_v2")
        if architecture not in allowed_architectures:
            raise ValueError(
                f"unsupported trajectory conditioner architecture: {architecture}"
            )
        self.architecture = architecture
        if architecture == "full_frame_causal_patch_v2":
            self.temporal_projection = nn.Conv3d(
                self.input_channels,
                self.input_channels,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
                bias=False,
                dtype=torch.float32,
            )
            nn.init.zeros_(self.temporal_projection.weight)
            with torch.no_grad():
                for channel in range(self.input_channels):
                    self.temporal_projection.weight[channel, channel, :, 0, 0] = 0.25
        self.patch_projection = nn.Conv3d(
            self.input_channels,
            rank,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
            dtype=torch.float32,
        )
        self.output_projection = nn.Conv3d(
            rank,
            hidden_dim,
            kernel_size=1,
            bias=False,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.patch_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        self._condition_maps: list[torch.Tensor] | None = None
        self._enabled = False
        self._cursor = 0
        self._expected_samples = 0

    def set_condition(
        self,
        point_track_maps: list[torch.Tensor],
        *,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._condition_maps = None
            return
        self._condition_maps = validate_point_track_maps(
            point_track_maps,
            representation=self.representation,
        )

    def begin_forward(self, sample_count: int) -> None:
        if sample_count <= 0:
            raise ValueError("trajectory-conditioned Wan batch cannot be empty")
        if self._enabled and (
            self._condition_maps is None or len(self._condition_maps) != sample_count
        ):
            raise RuntimeError("trajectory condition batch was not bound to Wan")
        self._cursor = 0
        self._expected_samples = int(sample_count)

    def add_residual(
        self,
        video: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        if self._cursor >= self._expected_samples:
            raise RuntimeError("Wan consumed more trajectory samples than expected")
        sample_index = self._cursor
        self._cursor += 1
        if not self._enabled:
            return patch_features
        condition_map = self._condition_maps[sample_index]
        if condition_map.shape[2:] != video.shape[3:]:
            raise ValueError("trajectory and Wan spatial latent grids do not match")
        condition_map = condition_map.unsqueeze(0).to(
            device=video.device,
            dtype=torch.float32,
        )
        if self.architecture == "full_frame_causal_patch_v2":
            expected_frames = 1 + 4 * (int(video.shape[2]) - 1)
            if condition_map.shape[2] != expected_frames:
                raise ValueError(
                    "full-rate trajectory must contain exactly 4*(latent_frames-1)+1 frames"
                )
            first = condition_map[:, :, :1]
            condition_map = (
                first
                if expected_frames == 1
                else torch.cat(
                    (first, self.temporal_projection(condition_map[:, :, 1:])),
                    dim=2,
                )
            )
        elif condition_map.shape[2] != video.shape[2]:
            raise ValueError("trajectory and Wan temporal latent grids do not match")
        residual = self.output_projection(
            torch.nn.functional.silu(self.patch_projection(condition_map))
        )
        if residual.shape != patch_features.shape:
            raise ValueError("trajectory projection and Wan patch grids do not match")
        return patch_features + residual.to(dtype=patch_features.dtype)

    def end_forward(self) -> None:
        if self._cursor != self._expected_samples:
            raise RuntimeError("Wan did not consume every bound trajectory sample")


class TrajectoryConditionedPatchEmbedding(nn.Module):
    def __init__(
        self,
        base: nn.Conv3d,
        conditioner: TrajectoryPatchConditioner,
    ):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_conditioner", conditioner)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.base(video)
        return self._conditioner.add_residual(video, features)


def inject_trajectory_conditioning(
    model: nn.Module,
    rank: int = 32,
    representation: str = "dense_point_tracks",
    architecture: str | None = None,
) -> TrajectoryPatchConditioner:
    """Attach a zero-initialized spatial-temporal trajectory branch to Wan."""
    if hasattr(model, "phycontext_trajectory_conditioner"):
        raise ValueError("trajectory conditioning is already injected")
    if not isinstance(model.patch_embedding, nn.Conv3d):
        raise TypeError("Wan patch embedding must be a Conv3d before injection")
    conditioner = TrajectoryPatchConditioner(
        hidden_dim=int(model.dim),
        patch_size=tuple(model.patch_size),
        rank=rank,
        representation=representation,
        architecture=architecture,
    )
    model.add_module("phycontext_trajectory_conditioner", conditioner)
    model.patch_embedding = TrajectoryConditionedPatchEmbedding(
        model.patch_embedding,
        conditioner,
    )
    model._phycontext_trajectory_base_forward = model.forward

    def trajectory_forward(self, x, *args, **kwargs):
        self.phycontext_trajectory_conditioner.begin_forward(len(x))
        output = self._phycontext_trajectory_base_forward(x, *args, **kwargs)
        self.phycontext_trajectory_conditioner.end_forward()
        return output

    model.forward = types.MethodType(trajectory_forward, model)
    return conditioner


def set_trajectory_condition(
    model: nn.Module,
    point_track_maps: list[torch.Tensor],
    *,
    enabled: bool = True,
) -> bool:
    conditioner = getattr(model, "phycontext_trajectory_conditioner", None)
    if conditioner is None:
        return False
    conditioner.set_condition(
        point_track_maps,
        enabled=enabled,
    )
    return True


def trajectory_conditioner_parameters(model: nn.Module):
    conditioner = getattr(model, "phycontext_trajectory_conditioner", None)
    if conditioner is not None:
        yield from conditioner.parameters()


class DirectConditionModulator(nn.Module):
    """Map structured physics/state values into every Wan block's AdaLN slots."""

    def __init__(
        self,
        hidden_dim: int,
        layer_count: int,
        input_dim: int = 12,
        object_slots: int = 1,
        rank: int = 32,
        alpha: float = 32.0,
    ):
        super().__init__()
        if (
            hidden_dim <= 0
            or layer_count <= 0
            or input_dim <= 0
            or object_slots <= 0
            or rank <= 0
        ):
            raise ValueError("direct modulation dimensions must be positive")
        if input_dim != 12 * object_slots:
            raise ValueError("direct modulation input must contain 12 values per object")
        self.hidden_dim = int(hidden_dim)
        self.layer_count = int(layer_count)
        self.input_dim = int(input_dim)
        self.object_slots = int(object_slots)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.architecture = "per_layer_v2"
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(input_dim, rank, bias=True, dtype=torch.float32)
        self.up = nn.ModuleList(
            nn.Linear(
                rank,
                6 * hidden_dim,
                bias=False,
                dtype=torch.float32,
            )
            for _ in range(layer_count)
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        for head in self.up:
            nn.init.zeros_(head.weight)
        self._structured_condition = None
        self._enabled = True

    def set_condition(self, condition: torch.Tensor, enabled: bool = True) -> None:
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != self.input_dim:
            raise ValueError("structured direct condition has an incompatible shape")
        self._structured_condition = condition.float()
        self._enabled = bool(enabled)

    def forward(self, layer_index: int, batch_size: int) -> torch.Tensor:
        if not 0 <= layer_index < self.layer_count:
            raise IndexError("direct modulation layer index is out of range")
        condition = self._structured_condition
        if condition is None:
            raise RuntimeError("structured direct condition was not set")
        if condition.shape[0] == 1 and batch_size > 1:
            condition = condition.expand(batch_size, -1)
        if condition.shape[0] != batch_size:
            raise ValueError("direct condition batch does not match Wan context")
        if not self._enabled:
            return torch.zeros(
                batch_size,
                1,
                6,
                self.hidden_dim,
                device=condition.device,
                dtype=torch.float32,
            )
        with torch.autocast(device_type=condition.device.type, enabled=False):
            hidden = torch.nn.functional.silu(self.down(condition))
            return self.up[layer_index](hidden).view(
                -1, 1, 6, self.hidden_dim
            ) * self.scaling


def inject_direct_condition_modulation(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    input_dim: int = 12,
    object_slots: int = 1,
) -> DirectConditionModulator:
    """Attach zero-initialized direct physical modulation without editing Wan."""
    if hasattr(model, "phycontext_direct_modulator"):
        raise ValueError("direct condition modulation is already injected")
    hidden_dim = int(model.dim)
    layer_count = len(model.blocks)
    modulator = DirectConditionModulator(
        hidden_dim=hidden_dim,
        layer_count=layer_count,
        input_dim=input_dim,
        object_slots=object_slots,
        rank=rank,
        alpha=alpha,
    )
    model.add_module("phycontext_direct_modulator", modulator)

    def make_forward(layer_index: int):
        def direct_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        ):
            direct = modulator(layer_index, context.shape[0])
            return self._phycontext_direct_base_forward(
                x,
                e + direct,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
            )

        return direct_forward

    for layer_index, block in enumerate(model.blocks):
        if hasattr(block, "_phycontext_direct_base_forward"):
            raise ValueError(f"direct modulation already wraps block {layer_index}")
        block._phycontext_direct_base_forward = block.forward
        block.forward = types.MethodType(make_forward(layer_index), block)
    return modulator


def set_direct_condition(
    model: nn.Module,
    controls: torch.Tensor,
    initial_state: torch.Tensor,
    enabled: bool = True,
) -> bool:
    """Bind normalized structured values for subsequent Wan forward calls."""
    modulator = getattr(model, "phycontext_direct_modulator", None)
    if modulator is None:
        return False
    condition = structured_direct_condition(
        controls,
        initial_state,
        object_slots=modulator.object_slots,
    )
    modulator.set_condition(condition, enabled=enabled)
    return True


def structured_direct_condition(
    controls: torch.Tensor,
    initial_state: torch.Tensor,
    object_slots: int = 1,
) -> torch.Tensor:
    """Normalize physical controls and initial state into the direct input."""
    if object_slots <= 0:
        raise ValueError("object_slots must be positive")
    if controls.ndim == 1:
        controls = controls.unsqueeze(0).unsqueeze(0)
    elif controls.ndim == 2:
        controls = controls.unsqueeze(1)
    if initial_state.ndim == 1:
        initial_state = initial_state.unsqueeze(0).unsqueeze(0)
    elif initial_state.ndim == 2:
        initial_state = initial_state.unsqueeze(1)
    if controls.ndim != 3 or controls.shape[-1] != 3:
        raise ValueError("direct controls must contain mass, friction, restitution")
    if initial_state.ndim != 3 or initial_state.shape[-1] != 9:
        raise ValueError("direct initial state must contain three 3D vectors")
    if controls.shape[:2] != initial_state.shape[:2]:
        raise ValueError("direct condition batches differ")
    batch, object_count, _ = controls.shape
    if object_count > object_slots:
        raise ValueError(
            f"direct condition has {object_count} objects; maximum is {object_slots}"
        )
    present = torch.ones(
        batch,
        object_count,
        1,
        dtype=controls.dtype,
        device=controls.device,
    )
    padded_controls = torch.zeros(
        batch, object_slots, 3, dtype=controls.dtype, device=controls.device
    )
    padded_state = torch.zeros(
        batch, object_slots, 9, dtype=initial_state.dtype, device=initial_state.device
    )
    padded_controls[:, :object_count] = controls
    padded_state[:, :object_count] = initial_state
    padded_present = torch.zeros(
        batch,
        object_slots,
        1,
        dtype=controls.dtype,
        device=controls.device,
    )
    padded_present[:, :object_count] = present
    normalized_controls = padded_controls.float().clone()
    normalized_controls[..., 0] = torch.log(
        normalized_controls[..., 0].clamp_min(1e-6)
    )
    normalized_controls[..., 1] = torch.log(
        normalized_controls[..., 1].clamp_min(1e-4)
    )
    normalized_controls[..., 2] = (
        2.0 * normalized_controls[..., 2].clamp(0.0, 1.0) - 1.0
    )
    vectors = padded_state.float().reshape(batch, object_slots, 3, 3).clone()
    vectors[..., 0, :] = torch.asinh(vectors[..., 0, :])
    vectors[..., 1, :] = torch.asinh(vectors[..., 1, :] / 5.0)
    vectors[..., 2, :] = vectors[..., 2, :] / 9.81
    normalized_controls = normalized_controls * padded_present
    vectors = vectors * padded_present.unsqueeze(-1)
    per_object = torch.cat([normalized_controls, vectors.flatten(2)], dim=-1)
    return per_object.flatten(1)


def direct_modulation_parameters(model: nn.Module):
    modulator = getattr(model, "phycontext_direct_modulator", None)
    if modulator is not None:
        yield from modulator.parameters()


def inject_cross_attention_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    replaced = []
    for block_index, block in enumerate(model.blocks):
        attention = block.cross_attn
        for name in ("q", "k", "v", "o"):
            layer = getattr(attention, name)
            if isinstance(layer, LoRALinear):
                raise ValueError(f"LoRA already injected at block {block_index}.{name}")
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"cross-attention {name} is not a linear layer")
            setattr(attention, name, LoRALinear(layer, rank, alpha, dropout))
            replaced.append(f"blocks.{block_index}.cross_attn.{name}")
    return replaced


def inject_self_attention_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    *,
    block_start: int = 0,
    block_end: int | None = None,
) -> list[str]:
    """Add LoRA to Wan's joint spatiotemporal attention projections."""
    end = len(model.blocks) if block_end is None else int(block_end)
    if not 0 <= block_start < end <= len(model.blocks):
        raise ValueError("self-attention LoRA block range is invalid")
    replaced = []
    for block_index in range(int(block_start), end):
        attention = model.blocks[block_index].self_attn
        for name in ("q", "k", "v", "o"):
            layer = getattr(attention, name)
            if isinstance(layer, LoRALinear):
                raise ValueError(
                    f"self-attention LoRA already injected at block {block_index}.{name}"
                )
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"self-attention {name} is not a linear layer")
            setattr(attention, name, LoRALinear(layer, rank, alpha, dropout))
            replaced.append(f"blocks.{block_index}.self_attn.{name}")
    return replaced


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.down.parameters()
            yield from module.up.parameters()


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            result[f"{name}.down.weight"] = module.down.weight.detach().cpu()
            result[f"{name}.up.weight"] = module.up.weight.detach().cpu()
    return result


def save_condition_checkpoint(
    output_dir: Path,
    model: nn.Module,
    condition_encoder: nn.Module,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"wan_lora.{key}": value.contiguous()
        for key, value in lora_state_dict(model).items()
    }
    tensors.update(
        {
            f"condition_encoder.{key}": value.detach().cpu().contiguous()
            for key, value in condition_encoder.state_dict().items()
        }
    )
    modulator = getattr(model, "phycontext_direct_modulator", None)
    if modulator is not None:
        tensors.update(
            {
                f"direct_modulation.{key}": value.detach().cpu().contiguous()
                for key, value in modulator.state_dict().items()
            }
        )
    trajectory_conditioner = getattr(
        model, "phycontext_trajectory_conditioner", None
    )
    if trajectory_conditioner is not None:
        tensors.update(
            {
                f"trajectory_conditioning.{key}": value.detach().cpu().contiguous()
                for key, value in trajectory_conditioner.state_dict().items()
            }
        )
    track_adapter = getattr(model, "phycontext_track_correspondence", None)
    if track_adapter is not None:
        tensors.update(
            {
                f"track_correspondence.{key}": value.detach().cpu().contiguous()
                for key, value in track_adapter.state_dict().items()
            }
        )
    save_file(tensors, output_dir / "adapter.safetensors")
    (output_dir / "adapter.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_condition_checkpoint(
    adapter_dir: Path,
    model: nn.Module,
    condition_encoder: nn.Module,
) -> dict:
    metadata = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
    if metadata.get("schema") not in {
        "phycontext.wan_condition_adapter.v2",
        "phycontext.wan_condition_adapter.v3",
        "phycontext.wan_condition_adapter.v4",
        "phycontext.wan_condition_adapter.v5",
    }:
        raise ValueError("adapter uses an unsupported conditioning schema")
    lora = metadata["lora"]
    replaced = inject_cross_attention_lora(
        model,
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
    )
    if len(replaced) != int(lora["module_count"]):
        raise ValueError("adapter LoRA module count does not match the Wan model")
    self_attention_config = metadata.get("self_attention_lora", {})
    self_attention_replaced = []
    if self_attention_config.get("enabled", False):
        self_attention_replaced = inject_self_attention_lora(
            model,
            rank=int(self_attention_config["rank"]),
            alpha=float(self_attention_config["alpha"]),
            block_start=int(self_attention_config["block_start"]),
            block_end=int(self_attention_config["block_end"]),
        )
        if len(self_attention_replaced) != int(
            self_attention_config["module_count"]
        ):
            raise ValueError(
                "adapter self-attention LoRA module count does not match Wan"
            )
    tensors = load_file(str(adapter_dir / "adapter.safetensors"), device="cpu")
    modules = dict(model.named_modules())
    consumed = set()
    for name in replaced + self_attention_replaced:
        module = modules[name]
        for projection in ("down", "up"):
            key = f"wan_lora.{name}.{projection}.weight"
            if key not in tensors:
                raise KeyError(f"missing adapter tensor: {key}")
            target = getattr(module, projection).weight
            target.data.copy_(tensors[key].to(dtype=target.dtype))
            consumed.add(key)
    condition_prefix = "condition_encoder."
    condition_state = {
        key[len(condition_prefix) :]: value
        for key, value in tensors.items()
        if key.startswith(condition_prefix)
    }
    condition_encoder.load_state_dict(condition_state, strict=True)
    consumed.update(key for key in tensors if key.startswith(condition_prefix))
    direct_config = metadata.get("direct_modulation", {})
    if direct_config.get("enabled", False):
        if direct_config.get("architecture") != "per_layer_v2":
            raise ValueError("adapter direct modulation architecture is unsupported")
        modulator = inject_direct_condition_modulation(
            model,
            rank=int(direct_config["rank"]),
            alpha=float(direct_config["alpha"]),
            input_dim=int(direct_config["input_dim"]),
            object_slots=int(
                direct_config.get(
                    "object_slots", int(direct_config["input_dim"]) // 12
                )
            ),
        )
        direct_prefix = "direct_modulation."
        direct_state = {
            key[len(direct_prefix) :]: value
            for key, value in tensors.items()
            if key.startswith(direct_prefix)
        }
        modulator.load_state_dict(direct_state, strict=True)
        consumed.update(key for key in tensors if key.startswith(direct_prefix))
    trajectory_config = metadata.get("trajectory_conditioning", {})
    if trajectory_config.get("enabled", False):
        representation = trajectory_config.get("representation", "dense_point_tracks")
        architecture = trajectory_config.get("architecture", "framewise_patch")
        conditioner = inject_trajectory_conditioning(
            model,
            rank=int(trajectory_config["rank"]),
            representation=representation,
            architecture=architecture,
        )
        expected_input_channels = int(
            trajectory_config.get("input_channels", conditioner.input_channels)
        )
        if conditioner.input_channels != expected_input_channels:
            raise ValueError("adapter trajectory input channels differ from representation")
        expected_patch_size = tuple(
            int(value) for value in trajectory_config["patch_size"]
        )
        if conditioner.patch_size != expected_patch_size:
            raise ValueError("adapter trajectory patch size differs from Wan")
        trajectory_prefix = "trajectory_conditioning."
        trajectory_state = {
            key[len(trajectory_prefix) :]: value
            for key, value in tensors.items()
            if key.startswith(trajectory_prefix)
        }
        conditioner.load_state_dict(trajectory_state, strict=True)
        consumed.update(
            key for key in tensors if key.startswith(trajectory_prefix)
        )
    track_config = metadata.get("track_correspondence", {})
    if track_config.get("enabled", False):
        adapter = inject_track_correspondence(
            model,
            block_index=int(track_config["block_index"]),
            feature_dim=int(track_config["feature_dim"]),
            refiner_blocks=int(track_config["refiner_blocks"]),
        )
        if adapter.architecture != track_config.get("architecture"):
            raise ValueError("adapter track-correspondence architecture is unsupported")
        track_prefix = "track_correspondence."
        track_state = {
            key[len(track_prefix) :]: value
            for key, value in tensors.items()
            if key.startswith(track_prefix)
        }
        adapter.load_state_dict(track_state, strict=True)
        consumed.update(key for key in tensors if key.startswith(track_prefix))
    unexpected = sorted(set(tensors) - consumed)
    if unexpected:
        raise ValueError(f"unexpected adapter tensors: {unexpected}")
    return metadata


def enable_block_checkpointing(model: nn.Module) -> int:
    def checkpointed_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        def run(x_value, e_value, context_value):
            return self._phycontext_forward(
                x_value,
                e_value,
                seq_lens,
                grid_sizes,
                freqs,
                context_value,
                context_lens,
            )

        return checkpoint(run, x, e, context, use_reentrant=False)

    track_adapter = getattr(model, "phycontext_track_correspondence", None)

    def checkpointed_track_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        def run(x_value, e_value, context_value):
            return self._phycontext_forward(
                x_value,
                e_value,
                seq_lens,
                grid_sizes,
                freqs,
                context_value,
                context_lens,
            )

        hidden = checkpoint(run, x, e, context, use_reentrant=False)
        return track_adapter(hidden, grid_sizes)

    count = 0
    for block in model.blocks:
        if hasattr(block, "_phycontext_forward"):
            continue
        if hasattr(block, "_phycontext_track_base_forward"):
            if track_adapter is None:
                raise RuntimeError("track-wrapped block has no correspondence adapter")
            block._phycontext_forward = block._phycontext_track_base_forward
            block.forward = types.MethodType(checkpointed_track_forward, block)
        else:
            block._phycontext_forward = block.forward
            block.forward = types.MethodType(checkpointed_forward, block)
        count += 1
    return count


def shifted_uniform_sigmas(
    batch_size: int,
    device: torch.device,
    shift: float = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("flow shift must be positive")
    uniform = torch.rand(batch_size, device=device, generator=generator)
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


def make_ti2v_flow_batch(
    latents: list[torch.Tensor],
    sigmas: torch.Tensor,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    generator: torch.Generator | None = None,
    shared_noise: bool = False,
) -> dict:
    if len(latents) != len(sigmas):
        raise ValueError("latent and sigma batch sizes differ")
    noisy_latents = []
    targets = []
    loss_masks = []
    timestep_rows = []
    sequence_lengths = []
    common_noise = None
    for latent, sigma in zip(latents, sigmas):
        if latent.ndim != 4:
            raise ValueError("each latent must have shape C x F x H x W")
        if latent.shape[2] % patch_size[1] or latent.shape[3] % patch_size[2]:
            raise ValueError("latent spatial dimensions must divide the Wan patch size")
        if shared_noise and common_noise is not None:
            if (
                latent.shape != common_noise.shape
                or latent.dtype != common_noise.dtype
                or latent.device != common_noise.device
            ):
                raise ValueError("shared-noise latents must have identical tensor layouts")
            noise = common_noise
        else:
            noise = torch.randn(
                latent.shape,
                dtype=latent.dtype,
                device=latent.device,
                generator=generator,
            )
            if shared_noise:
                common_noise = noise
        mask = torch.ones_like(latent)
        mask[:, 0] = 0
        sigma_value = sigma.to(dtype=latent.dtype)
        interpolated = (1.0 - sigma_value) * latent + sigma_value * noise
        noisy_latents.append((1.0 - mask) * latent + mask * interpolated)
        targets.append(noise - latent)
        loss_masks.append(mask)
        patch_mask = mask[0, :: patch_size[0], :: patch_size[1], :: patch_size[2]]
        timestep_rows.append(patch_mask.flatten().float() * sigma.float() * 1000.0)
        sequence_lengths.append(int(patch_mask.numel()))
    if len(set(sequence_lengths)) != 1:
        raise ValueError("the current trainer requires one latent shape per batch")
    return {
        "noisy_latents": noisy_latents,
        "targets": targets,
        "loss_masks": loss_masks,
        "timesteps": torch.stack(timestep_rows),
        "seq_len": sequence_lengths[0],
    }


def recover_clean_latents(
    noisy_latents: list[torch.Tensor],
    predictions: list[torch.Tensor],
    sigmas: torch.Tensor,
) -> list[torch.Tensor]:
    """Recover clean TI2V latents while preserving the supplied condition frame."""
    if not (len(noisy_latents) == len(predictions) == len(sigmas)):
        raise ValueError("clean-latent inputs have different batch sizes")
    clean = []
    for noisy, prediction, sigma in zip(noisy_latents, predictions, sigmas):
        if noisy.shape != prediction.shape or noisy.ndim != 4:
            raise ValueError("noisy and predicted latents must share C x F x H x W")
        estimate = noisy.float() - sigma.float() * prediction.float()
        clean.append(torch.cat((noisy.float()[:, :1], estimate[:, 1:]), dim=1))
    return clean


def latent_correspondence_motion(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    motion_mask: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Match target-frame object latents and return centers plus distributions."""
    if predicted_clean.ndim != 4 or predicted_clean.shape != target_clean.shape:
        raise ValueError("trajectory latents must share C x F x H x W")
    if motion_mask.ndim != 4 or motion_mask.shape[0] != 1:
        raise ValueError("trajectory motion mask must have shape 1 x F x H x W")
    if motion_mask.shape[1:] != predicted_clean.shape[1:]:
        raise ValueError("trajectory latent and motion-mask grids differ")
    if temperature <= 0:
        raise ValueError("trajectory temperature must be positive")

    _, frame_count, height, width = predicted_clean.shape
    coordinate_dtype = predicted_clean.dtype
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=predicted_clean.device),
        torch.linspace(-1.0, 1.0, width, device=predicted_clean.device),
        indexing="ij",
    )
    coordinates = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
    coordinates = coordinates.to(dtype=coordinate_dtype)

    predicted_centers = []
    target_centers = []
    predicted_distributions = []
    target_distributions = []
    valid_frames = []
    detached_target = target_clean.detach().float()
    predicted = predicted_clean.float()
    mask = motion_mask[0].gt(0)
    for frame in range(frame_count):
        frame_mask = mask[frame].reshape(-1)
        valid = bool(frame_mask.any()) and frame > 0
        valid_frames.append(valid)
        if not valid:
            zero = predicted.new_zeros(2)
            zero_distribution = predicted.new_zeros(height * width)
            predicted_centers.append(zero)
            target_centers.append(zero)
            predicted_distributions.append(zero_distribution)
            target_distributions.append(zero_distribution)
            continue

        predicted_features = predicted[:, frame].reshape(predicted.shape[0], -1).T
        predicted_features = predicted_features - predicted_features.mean(
            dim=0, keepdim=True
        )
        predicted_features = F.normalize(predicted_features, dim=-1, eps=1e-6)
        target_features = detached_target[:, frame].reshape(
            detached_target.shape[0], -1
        ).T
        target_features = target_features - target_features.mean(
            dim=0, keepdim=True
        )
        target_tokens = F.normalize(
            target_features[frame_mask], dim=-1, eps=1e-6
        )
        similarity = predicted_features @ target_tokens.T
        spatial_score = similarity.max(dim=-1).values
        spatial_probability = torch.softmax(spatial_score / temperature, dim=0)
        target_probability = frame_mask.float()
        target_probability = target_probability / target_probability.sum().clamp_min(1.0)
        predicted_centers.append(
            (spatial_probability.unsqueeze(-1) * coordinates.float()).sum(dim=0)
        )
        target_centers.append(
            (target_probability.unsqueeze(-1) * coordinates.float()).sum(dim=0)
        )
        predicted_distributions.append(spatial_probability)
        target_distributions.append(target_probability)

    return (
        torch.stack(predicted_centers),
        torch.stack(target_centers),
        torch.stack(predicted_distributions).reshape(frame_count, height, width),
        torch.stack(target_distributions).reshape(frame_count, height, width),
        torch.tensor(valid_frames, dtype=torch.bool, device=predicted_clean.device),
    )


def latent_correspondence_trajectory(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    motion_mask: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate image-plane centers while keeping the original public interface."""
    predicted, target, _, _, valid = latent_correspondence_motion(
        predicted_clean,
        target_clean,
        motion_mask,
        temperature=temperature,
    )
    return predicted, target, valid


def latent_object_center_loss(
    predicted_clean: list[torch.Tensor],
    target_clean: list[torch.Tensor],
    motion_masks: list[torch.Tensor],
    temperature: float = 0.07,
    beta: float = 0.05,
) -> torch.Tensor:
    """Keep one coarse object-location guard beside exact feature correspondence."""
    if not (
        len(predicted_clean) == len(target_clean) == len(motion_masks)
        and predicted_clean
    ):
        raise ValueError("motion supervision inputs have different or empty batches")
    if beta <= 0:
        raise ValueError("motion supervision Smooth-L1 beta must be positive")
    center_losses = []
    for predicted, target, mask in zip(
        predicted_clean, target_clean, motion_masks
    ):
        predicted_centers, target_centers, valid = latent_correspondence_trajectory(
            predicted, target, mask, temperature=temperature
        )
        if valid.any():
            center_losses.append(
                F.smooth_l1_loss(
                    predicted_centers[valid],
                    target_centers[valid],
                    beta=beta,
                    reduction="mean",
                )
            )
    zero = predicted_clean[0].sum() * 0.0
    return torch.stack(center_losses).mean() if center_losses else zero


def masked_flow_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> torch.Tensor:
    if not (len(predictions) == len(targets) == len(masks)):
        raise ValueError("prediction, target, and mask batch sizes differ")
    numerator = sum(
        ((prediction.float() - target.float()).square() * mask.float()).sum()
        for prediction, target, mask in zip(predictions, targets, masks)
    )
    denominator = sum(mask.float().sum() for mask in masks).clamp_min(1.0)
    return numerator / denominator


def balanced_motion_loss_mask(
    base_mask: torch.Tensor,
    motion_mask: torch.Tensor,
    foreground_share: float,
) -> torch.Tensor:
    """Allocate a fixed loss share to the controlled object's spacetime tube."""
    if not 0.0 < foreground_share < 1.0:
        raise ValueError("foreground_share must be between zero and one")
    if base_mask.ndim != 4 or motion_mask.ndim != 4:
        raise ValueError("loss and motion masks must have shape C x F x H x W")
    if motion_mask.shape[1:] != base_mask.shape[1:]:
        raise ValueError("loss and motion mask grids differ")
    if motion_mask.shape[0] == 1:
        motion_mask = motion_mask.expand(base_mask.shape[0], -1, -1, -1)
    elif motion_mask.shape[0] != base_mask.shape[0]:
        raise ValueError("motion mask channel count is not broadcastable")
    valid = base_mask.float()
    foreground = motion_mask.gt(0).float() * valid
    background = (1.0 - motion_mask.gt(0).float()) * valid
    foreground_count = foreground.sum()
    background_count = background.sum()
    if foreground_count.item() == 0 or background_count.item() == 0:
        return valid
    total = valid.sum()
    return (
        foreground * (foreground_share * total / foreground_count)
        + background * ((1.0 - foreground_share) * total / background_count)
    )


def source_target_motion_envelope(motion_mask: torch.Tensor) -> torch.Tensor:
    """Cover both source removal and per-frame target occupancy."""
    if motion_mask.ndim != 4 or motion_mask.shape[0] != 1:
        raise ValueError("motion mask must have shape 1 x F x H x W")
    source = motion_mask[:, :1].gt(0)
    if not bool(source.any()):
        raise ValueError("motion envelope requires a frame-zero object mask")
    source = source.expand(-1, motion_mask.shape[1], -1, -1)
    return torch.logical_or(source, motion_mask.gt(0)).to(motion_mask.dtype)


def masked_flow_response_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    minimum_target_energy: float = 1e-3,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match the predicted endpoint difference under common diffusion noise."""
    if len(predictions) != 2 or len(targets) != 2 or len(masks) != 2:
        raise ValueError("response loss requires exactly one low/high pair")
    if minimum_target_energy <= 0:
        raise ValueError("minimum target energy must be positive")
    valid_mask = masks[0].float() * masks[1].float()
    if response_mask is None:
        response_mask = valid_mask
    else:
        if response_mask.shape != valid_mask.shape:
            raise ValueError("response mask shape differs from the flow mask")
        response_mask = response_mask.float() * valid_mask
    predicted_delta = predictions[1].float() - predictions[0].float()
    target_delta = targets[1].float() - targets[0].float()
    denominator = response_mask.sum().clamp_min(1.0)
    error = ((predicted_delta - target_delta).square() * response_mask).sum()
    target_energy = (target_delta.square() * response_mask).sum() / denominator
    return (error / denominator) / target_energy.detach().clamp_min(
        minimum_target_energy
    )
