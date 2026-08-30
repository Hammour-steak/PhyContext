#!/usr/bin/env python3
"""Formal distributed PhyContext adapter training for Wan2.2 TI2V-5B."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

from cache_contract import (
    CANONICAL_CONDITION_FRAME_PROTOCOL,
    CURRENT_CACHE_SCHEMA,
    resolve_cache_artifact_root,
    resolve_cache_dataset_root,
    validate_cache_artifact,
    validate_cache_source_manifest,
)
from project_defaults import CACHE_MANIFEST, SCENE_TOKEN_COUNT
from project_defaults import INFERENCE_GUIDANCE_SCALE, INFERENCE_SAMPLING_STEPS
from conditioning_model import (
    PhyContextConditionEncoder,
    apply_condition_mode,
    collate_scene_conditions,
    compose_wan_context,
    controls_from_record,
    dynamics_from_record,
    initial_state_from_record,
    load_scene_condition,
)
from wan_training import (
    LoRALinear,
    balanced_motion_loss_mask,
    direct_modulation_parameters,
    enable_block_checkpointing,
    inject_cross_attention_lora,
    inject_direct_condition_modulation,
    inject_trajectory_conditioning,
    index_nominal_trajectory_records,
    latent_motion_supervision_losses,
    latent_temporal_consistency_loss,
    load_condition_checkpoint,
    lora_parameters,
    make_formal_training_batch,
    make_ti2v_flow_batch,
    masked_flow_loss,
    masked_flow_response_loss,
    motion_mask_from_point_track_map,
    recover_clean_latents,
    save_condition_checkpoint,
    select_sweep_endpoint_pairs,
    select_trajectory_input_records,
    set_direct_condition,
    set_trajectory_condition,
    shifted_uniform_sigmas,
    source_target_motion_envelope,
    canonical_trajectory_representation,
    trajectory_channel_names,
    validate_point_track_object_slots,
    trajectory_conditioner_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = CACHE_MANIFEST
DEFAULT_WAN_REPO = Path(os.environ.get("PHYCONTEXT_WAN_REPO", "external/Wan2.2"))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("PHYCONTEXT_WAN_CHECKPOINT", "checkpoints/Wan2.2-TI2V-5B")
)
DEFAULT_OUTPUT = Path("outputs/training/formal")
BASE_DYNAMICS_LAYOUT = [
    "inertia_xx",
    "inertia_yy",
    "inertia_zz",
    "inertia_xy",
    "inertia_xz",
    "inertia_yz",
    "rolling_friction",
    "spinning_friction",
    "linear_damping",
    "angular_damping",
]

TRAINING_CODE_PATHS = (
    Path("tools/phycontext/train_wan_formal.py"),
    Path("tools/phycontext/wan_training.py"),
    Path("tools/phycontext/conditioning_model.py"),
    Path("tools/phycontext/cache_contract.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wan-repo", type=Path, default=DEFAULT_WAN_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-adapter", type=Path)
    parser.add_argument(
        "--training-stage",
        choices=("joint", "trajectory_warmup"),
        default="joint",
        help=(
            "joint trains scene, physics, and optional trajectory conditions; "
            "trajectory_warmup keeps only scene plus target-matching trajectory "
            "conditions and freezes every physics-specific module"
        ),
    )
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument(
        "--ordinary-only",
        action="store_true",
        help=(
            "run exactly one ordinary update for a single-record integration "
            "smoke test; paired responses and validation are disabled"
        ),
    )
    parser.add_argument("--base-scene-count", type=int)
    parser.add_argument("--scene-tokens", type=int, default=SCENE_TOKEN_COUNT)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--modulation-rank", type=int, default=32)
    parser.add_argument("--modulation-alpha", type=float, default=32.0)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--direct-learning-rate", type=float, default=5e-5)
    parser.add_argument("--trajectory-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--encoder-weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=2.0)
    parser.add_argument("--lpips-loss-weight", type=float, default=0.1)
    parser.add_argument("--lpips-frame-count", type=int, default=2)
    parser.add_argument(
        "--lpips-temporal-window",
        type=int,
        default=3,
        help="number of contiguous latent frames decoded per LPIPS window",
    )
    parser.add_argument("--lpips-resolution", type=int, default=256)
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex"
    )
    parser.add_argument("--response-loss-weight", type=float, default=0.5)
    parser.add_argument("--motion-foreground-share", type=float, default=0.5)
    parser.add_argument("--minimum-response-energy", type=float, default=1e-3)
    parser.add_argument("--trajectory-center-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--trajectory-distribution-loss-weight", type=float, default=0.05
    )
    parser.add_argument("--trajectory-velocity-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--temporal-consistency-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--temporal-consistency-beta", type=float, default=0.05)
    parser.add_argument("--trajectory-temperature", type=float, default=0.03)
    parser.add_argument("--trajectory-beta", type=float, default=0.05)
    parser.add_argument(
        "--trajectory-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable target point-track conditioning; disable only for an ablation",
    )
    parser.add_argument(
        "--trajectory-input-source",
        choices=("target", "nominal_base"),
        default="target",
        help=(
            "target uses the trajectory cached for the same physical sample; "
            "nominal_base is an explicit fixed-trajectory response ablation"
        ),
    )
    parser.add_argument("--trajectory-condition-rank", type=int, default=32)
    parser.add_argument(
        "--trajectory-representation",
        choices=("das_3d_tracks", "dense_point_tracks"),
        default="das_3d_tracks",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=512)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=15,
        help=(
            "deterministic mixed validation microbatches per distributed rank; "
            "formal training requires a positive multiple of five"
        ),
    )
    parser.add_argument("--save-every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_code_sha256() -> dict[str, str]:
    """Fingerprint the local implementations that define training semantics."""
    return {
        relative.as_posix(): sha256(PROJECT_ROOT / relative)
        for relative in TRAINING_CODE_PATHS
    }


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal Wan training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def filter_base_scenes(records: list[dict], count: int | None) -> list[dict]:
    if count is None:
        return records
    if count <= 0:
        raise ValueError("base-scene-count must be positive")
    scene_ids = list(dict.fromkeys(item["base_scene_id"] for item in records))[:count]
    allowed = set(scene_ids)
    return [item for item in records if item["base_scene_id"] in allowed]


def dynamic_object_count(item: dict) -> int:
    return int(item["point_track"]["object_count"])


def record_dynamic_object_ids(item: dict) -> list[str]:
    physics = item["record"]["conditioning"]["physics"]
    objects = physics.get("objects")
    if objects is None:
        objects = [physics["object"]]
    elif isinstance(objects, dict):
        objects = [objects[key] for key in sorted(objects)]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 3:
        raise ValueError("training record must contain one to three dynamic objects")
    object_ids = [str(value["object_id"]) for value in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("training record dynamic object ids must be unique")
    return object_ids


def max_dynamic_object_count(records: list[dict]) -> int:
    if not records:
        return 1
    count = max(dynamic_object_count(item) for item in records)
    if count > 3:
        raise ValueError("at most three dynamic objects are supported")
    return count


def require_complete_cache(
    artifact_root: Path,
    dataset_root: Path,
    records: list[dict],
    split: str,
    trajectory_representation: str,
) -> None:
    if not records:
        raise ValueError(f"cache has no {split} records")
    canonical_trajectory_representation(trajectory_representation)
    missing = [item["sample_id"] for item in records if "point_track" not in item]
    if missing:
        raise ValueError(
            "point-track training requires cached point trajectories; "
            f"{len(missing)} missing in {split}"
        )
    verified_scenes: dict[Path, str] = {}
    for item in records:
        sample_id = item["sample_id"]
        for key, label in (
            ("latent", f"latent for {sample_id}"),
            ("text_context", f"text context for {sample_id}"),
            ("point_track", f"point track for {sample_id}"),
        ):
            validate_cache_artifact(artifact_root, item.get(key), label)
        latent = item["latent"]
        if (
            latent.get("condition_frame_protocol")
            != CANONICAL_CONDITION_FRAME_PROTOCOL
            or latent.get("condition_frame_sha256")
            != item.get("source_first_frame_sha256")
        ):
            raise ValueError(
                f"latent condition-frame binding is invalid for {sample_id}"
            )
        expected_object_ids = record_dynamic_object_ids(item)
        point_track = item["point_track"]
        if (
            point_track.get("object_ids") != expected_object_ids
            or int(point_track.get("object_count", -1)) != len(expected_object_ids)
        ):
            raise ValueError(f"point-track object binding is invalid for {sample_id}")
        scene_path = (
            dataset_root / item["record"]["conditioning"]["scene"]
        ).resolve()
        if not scene_path.is_relative_to(dataset_root):
            raise ValueError(f"scene condition escapes the dataset root: {sample_id}")
        expected_scene_hash = item.get("source_scene_sha256")
        previous_hash = verified_scenes.get(scene_path)
        if previous_hash is not None:
            if previous_hash != expected_scene_hash:
                raise ValueError(f"shared scene hash differs across records: {sample_id}")
        else:
            if (
                not expected_scene_hash
                or not scene_path.is_file()
                or sha256(scene_path) != expected_scene_hash
            ):
                raise ValueError(f"scene condition file/hash mismatch: {sample_id}")
            verified_scenes[scene_path] = expected_scene_hash


def split_condition_parameters(module: torch.nn.Module):
    decay = []
    no_decay = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        exclude = (
            parameter.ndim < 2
            or "norm" in lower
            or "embedding" in lower
            or lower.endswith("queries")
        )
        (no_decay if exclude else decay).append(parameter)
    return decay, no_decay


def training_condition_mode(stage: str) -> str:
    if stage == "joint":
        return "full"
    if stage == "trajectory_warmup":
        return "scene_only"
    raise ValueError(f"unsupported training stage: {stage}")


def configure_training_stage(
    condition_encoder: PhyContextConditionEncoder,
    model: torch.nn.Module,
    stage: str,
) -> None:
    """Freeze conditions that must remain invariant during trajectory warmup."""
    training_condition_mode(stage)
    physics_trainable = stage == "joint"
    for module in (
        condition_encoder.physics_encoder,
        condition_encoder.dynamics_encoder,
        condition_encoder.state_encoder,
    ):
        module.requires_grad_(physics_trainable)
    direct = getattr(model, "phycontext_direct_modulator", None)
    if direct is None:
        raise ValueError("formal training requires the direct condition module")
    direct.requires_grad_(physics_trainable)


def optimizer_groups(
    condition_encoder: torch.nn.Module,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> list[dict]:
    condition_decay, condition_no_decay = split_condition_parameters(
        condition_encoder
    )
    candidates = [
        {
            "name": "condition_decay",
            "params": condition_decay,
            "lr": args.encoder_learning_rate,
            "initial_lr": args.encoder_learning_rate,
            "weight_decay": args.encoder_weight_decay,
        },
        {
            "name": "condition_no_decay",
            "params": condition_no_decay,
            "lr": args.encoder_learning_rate,
            "initial_lr": args.encoder_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "wan_lora",
            "params": [
                parameter
                for parameter in lora_parameters(model)
                if parameter.requires_grad
            ],
            "lr": args.lora_learning_rate,
            "initial_lr": args.lora_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "direct_adaln",
            "params": [
                parameter
                for parameter in direct_modulation_parameters(model)
                if parameter.requires_grad
            ],
            "lr": args.direct_learning_rate,
            "initial_lr": args.direct_learning_rate,
            "weight_decay": 0.0,
        },
    ]
    if args.trajectory_input:
        candidates.append(
            {
                "name": "trajectory_conditioning",
                "params": [
                    parameter
                    for parameter in trajectory_conditioner_parameters(model)
                    if parameter.requires_grad
                ],
                "lr": args.trajectory_learning_rate,
                "initial_lr": args.trajectory_learning_rate,
                "weight_decay": 0.0,
            }
        )
    groups = [group for group in candidates if group["params"]]
    names = {group["name"] for group in groups}
    required = {"wan_lora"}
    if args.trajectory_input:
        required.add("trajectory_conditioning")
    if args.training_stage == "joint":
        required.add("direct_adaln")
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"missing optimizer parameter groups: {missing}")

    grouped_parameters = [
        parameter for group in groups for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("an optimizer parameter appears in more than one group")
    expected = {
        id(parameter): f"{prefix}.{name}"
        for prefix, module in (
            ("condition_encoder", condition_encoder),
            ("model", model),
        )
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    omitted = sorted(
        expected[parameter_id] for parameter_id in set(expected) - set(grouped_ids)
    )
    if omitted:
        raise ValueError(
            "trainable parameters are missing from the optimizer: "
            + ", ".join(omitted[:10])
        )
    return groups


def learning_rate_factor(
    step: int,
    total_steps: int,
    warmup_ratio: float,
    minimum_ratio: float,
) -> float:
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("learning-rate step is outside the training schedule")
    if not 0 <= warmup_ratio < 1 or not 0 < minimum_ratio <= 1:
        raise ValueError("learning-rate schedule ratios are invalid")
    warmup_steps = max(1, round(total_steps * warmup_ratio))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    decay_intervals = max(1, decay_steps - 1)
    progress = min(1.0, (step - warmup_steps) / decay_intervals)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def apply_learning_rate(optimizer, factor: float) -> dict[str, float]:
    values = {}
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * factor
        values[str(group["name"])] = float(group["lr"])
    return values


def load_microbatch(
    artifact_root: Path,
    dataset_root: Path,
    batch: list[dict],
    device: torch.device,
    trajectory_batch: list[dict] | None = None,
    trajectory_representation: str = "das_3d_tracks",
    scene_size_px: tuple[int, int] | None = None,
) -> dict:
    records = [item["record"] for item in batch]
    trajectory_representation = canonical_trajectory_representation(
        trajectory_representation
    )

    def load_point_map(item: dict) -> torch.Tensor:
        descriptor = item.get("point_track")
        if descriptor is None:
            raise ValueError(f"record has no point-track cache: {item['sample_id']}")
        point_map = load_file(
            str(artifact_root / descriptor["path"]), device="cpu"
        )[
            "point_track_map"
        ].to(device)
        return validate_point_track_object_slots(
            point_map,
            trajectory_representation,
            int(descriptor["object_count"]),
        )

    target_point_maps = [load_point_map(item) for item in batch]
    target_motion_masks = [
        motion_mask_from_point_track_map(point_map)
        for point_map in target_point_maps
    ]
    trajectory_batch = batch if trajectory_batch is None else trajectory_batch
    if len(trajectory_batch) != len(batch):
        raise ValueError("trajectory input and target batches must have equal length")
    if all(
        source["sample_id"] == target["sample_id"]
        for source, target in zip(trajectory_batch, batch)
    ):
        trajectory_point_maps = target_point_maps
    else:
        trajectory_point_maps = [load_point_map(item) for item in trajectory_batch]
    return {
        "records": records,
        "latents": [
            load_file(
                str(artifact_root / item["latent"]["path"]), device="cpu"
            )["latent"].to(device)
            for item in batch
        ],
        "text": [
            load_file(
                str(artifact_root / item["text_context"]["path"]), device="cpu"
            )
            ["context"].to(device)
            for item in batch
        ],
        "motion_masks": target_motion_masks,
        "trajectory_point_maps": trajectory_point_maps,
        "trajectory_sample_ids": [item["sample_id"] for item in trajectory_batch],
        "scene": collate_scene_conditions(
            [
                load_scene_condition(
                    dataset_root / record["conditioning"]["scene"],
                    device,
                    target_size_px=scene_size_px,
                )
                for record in records
            ]
        ),
        "controls": torch.stack(
            [controls_from_record(record, device) for record in records]
        ),
        "dynamics": torch.stack(
            [dynamics_from_record(record, device) for record in records]
        ),
        "initial_state": torch.stack(
            [initial_state_from_record(record, device) for record in records]
        ),
    }


def clean_latent_lpips_loss(
    predicted_clean: list[torch.Tensor],
    target_latents: list[torch.Tensor],
    vae,
    perceptual_model,
    window_count: int,
    resolution: int,
    temporal_window: int = 3,
) -> torch.Tensor:
    """Compare contiguous low-resolution temporal windows with LPIPS.

    Wan's VAE is causal and keeps temporal feature caches while decoding a
    sequence. Decoding one latent slice at a time therefore produces a
    different result from decoding a video. We keep the auxiliary graph small
    by decoding several contiguous generated latent frames per local window;
    each window deliberately starts a fresh causal cache and is an appearance
    objective, not a claim of exact full-sequence decoding.
    The clean TI2V condition frame is excluded because it is provided to the
    model rather than generated.
    """
    if len(predicted_clean) != len(target_latents):
        raise ValueError("LPIPS prediction and target batch sizes differ")
    if not predicted_clean:
        raise ValueError("LPIPS cannot run on an empty batch")
    if window_count <= 0:
        raise ValueError("LPIPS window count must be positive")
    if temporal_window <= 0:
        raise ValueError("LPIPS temporal window must be positive")
    losses = []
    for predicted, target in zip(predicted_clean, target_latents):
        if predicted.ndim != 4 or target.ndim != 4:
            raise ValueError("LPIPS latents must have shape C x F x H x W")
        if predicted.shape != target.shape:
            raise ValueError("LPIPS prediction and target latent shapes differ")
        latent_frame_count = int(predicted.shape[1])
        if latent_frame_count < 2:
            raise ValueError("LPIPS requires at least one generated latent frame")
        generated_frame_count = latent_frame_count - 1
        window_length = min(int(temporal_window), generated_frame_count)
        max_start = latent_frame_count - window_length
        window_starts = torch.linspace(
            1,
            max_start,
            steps=min(window_count, max_start),
            device=predicted.device,
        ).round().long().unique()
        # Decode no more spatial detail than LPIPS consumes.  Resizing after a
        # larger VAE decode wastes memory; at 832 x 480 it can exhaust a 48 GiB
        # card even though the actual perceptual comparison is only 256 px.
        maximum_latent_extent = max(8, resolution // 16)
        spatial_scale = min(
            1.0,
            maximum_latent_extent
            / float(max(predicted.shape[-2], predicted.shape[-1])),
        )
        decode_size = (
            max(8, round(predicted.shape[-2] * spatial_scale)),
            max(8, round(predicted.shape[-1] * spatial_scale)),
        )
        for window_start in window_starts.tolist():
            window_end = window_start + window_length
            predicted_slice = F.interpolate(
                predicted[:, window_start:window_end].float(),
                size=decode_size,
                mode="bilinear",
                align_corners=False,
            )
            target_slice = F.interpolate(
                target[:, window_start:window_end].float(),
                size=decode_size,
                mode="bilinear",
                align_corners=False,
            )
            decoded_prediction = vae.decode([predicted_slice])[0]
            with torch.no_grad():
                decoded_target = vae.decode([target_slice])[0]
            prediction_frames = decoded_prediction.permute(1, 0, 2, 3)
            target_frames = decoded_target.permute(1, 0, 2, 3)
            prediction_frames = F.interpolate(
                prediction_frames,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
            target_frames = F.interpolate(
                target_frames,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
            losses.append(
                perceptual_model(prediction_frames, target_frames).mean()
            )
    return torch.stack(losses).mean()


def forward_losses(
    model,
    condition_encoder,
    loaded: dict,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    response_enabled: bool,
    vae=None,
    perceptual_model=None,
    fixed_sigma: float | None = None,
) -> dict[str, torch.Tensor]:
    raw_model = unwrap(model)
    condition_tokens = condition_encoder(
        loaded["scene"],
        loaded["controls"],
        loaded["dynamics"],
        loaded["initial_state"],
    )
    condition_tokens = apply_condition_mode(
        condition_tokens,
        args.scene_tokens,
        training_condition_mode(args.training_stage),
    )
    context = compose_wan_context(
        loaded["text"],
        condition_tokens.to(loaded["text"][0].dtype),
        max_tokens=raw_model.text_len,
    )
    if not set_direct_condition(
        raw_model,
        loaded["controls"],
        loaded["initial_state"],
        enabled=args.training_stage == "joint",
    ):
        raise RuntimeError("formal training requires direct condition modulation")
    canonical_trajectory_representation(args.trajectory_representation)
    if args.trajectory_input and not set_trajectory_condition(
        raw_model,
        loaded["trajectory_point_maps"],
    ):
        raise RuntimeError("trajectory conditioning was requested but not injected")
    if fixed_sigma is None:
        sigma_count = 1 if response_enabled else len(loaded["latents"])
        sigmas = shifted_uniform_sigmas(
            sigma_count,
            device,
            shift=args.flow_shift,
            generator=generator,
        )
        if response_enabled:
            sigmas = sigmas.expand(len(loaded["latents"]))
    else:
        sigmas = torch.full(
            (len(loaded["latents"]),), fixed_sigma, device=device
        )
    flow = make_ti2v_flow_batch(
        loaded["latents"],
        sigmas,
        patch_size=tuple(raw_model.patch_size),
        generator=generator,
        shared_noise=response_enabled,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(
            flow["noisy_latents"],
            t=flow["timesteps"],
            context=context,
            seq_len=flow["seq_len"],
        )
        reconstruction_masks = [
            balanced_motion_loss_mask(
                base_mask,
                source_target_motion_envelope(motion_mask),
                args.motion_foreground_share,
            )
            for base_mask, motion_mask in zip(
                flow["loss_masks"], loaded["motion_masks"]
            )
        ]
        reconstruction = masked_flow_loss(
            predictions, flow["targets"], reconstruction_masks
        )
        response = reconstruction.new_zeros(())
        if response_enabled:
            response_region = torch.maximum(
                source_target_motion_envelope(loaded["motion_masks"][0]),
                source_target_motion_envelope(loaded["motion_masks"][1]),
            )
            response_mask = balanced_motion_loss_mask(
                flow["loss_masks"][0] * flow["loss_masks"][1],
                response_region,
                args.motion_foreground_share,
            )
            response = masked_flow_response_loss(
                predictions,
                flow["targets"],
                flow["loss_masks"],
                minimum_target_energy=args.minimum_response_energy,
                response_mask=response_mask,
            )
        predicted_clean = recover_clean_latents(
            flow["noisy_latents"], predictions, sigmas
        )
        motion = latent_motion_supervision_losses(
            predicted_clean,
            loaded["latents"],
            loaded["motion_masks"],
            temperature=args.trajectory_temperature,
            beta=args.trajectory_beta,
        )
        temporal_consistency = reconstruction.new_zeros(())
        if args.temporal_consistency_loss_weight > 0:
            temporal_consistency = latent_temporal_consistency_loss(
                predicted_clean,
                loaded["latents"],
                beta=args.temporal_consistency_beta,
            )
        lpips_loss = reconstruction.new_zeros(())
        if args.lpips_loss_weight > 0:
            if vae is None or perceptual_model is None:
                raise RuntimeError("LPIPS is enabled but its models are missing")
            with torch.autocast(device_type="cuda", enabled=False):
                lpips_loss = clean_latent_lpips_loss(
                    predicted_clean,
                    loaded["latents"],
                    vae,
                    perceptual_model,
                    args.lpips_frame_count,
                    args.lpips_resolution,
                    args.lpips_temporal_window,
                )
        total = (
            args.reconstruction_loss_weight * reconstruction
            + args.lpips_loss_weight * lpips_loss
            + args.temporal_consistency_loss_weight * temporal_consistency
            + args.response_loss_weight * response
            + args.trajectory_center_loss_weight * motion["center"]
            + args.trajectory_distribution_loss_weight * motion["distribution"]
            + args.trajectory_velocity_loss_weight * motion["velocity"]
        )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "response": response,
        "trajectory_center": motion["center"],
        "trajectory_distribution": motion["distribution"],
        "trajectory_velocity": motion["velocity"],
        "temporal_consistency": temporal_consistency,
        "lpips": lpips_loss,
        "sigma": sigmas[0],
    }


def reduce_metrics(
    values: dict[str, float], counts: dict[str, float], device, world_size: int
) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor(
        [item for key in keys for item in (values[key], counts[key])],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = {}
    for index, key in enumerate(keys):
        numerator = float(tensor[index * 2].cpu())
        denominator = float(tensor[index * 2 + 1].cpu())
        result[key] = numerator / max(denominator, 1.0)
    return result


@torch.no_grad()
def validate(
    model,
    condition_encoder,
    artifact_root: Path,
    dataset_root: Path,
    records: list[dict],
    pairs: list[dict],
    nominal_records: dict[str, dict],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    scene_size_px: tuple[int, int],
    vae=None,
    perceptual_model=None,
) -> dict[str, float]:
    """Evaluate the same 60/40 ordinary-response mixture used for training."""
    model.eval()
    condition_encoder.eval()
    totals = {
        key: 0.0
        for key in (
            "total",
            "reconstruction",
            "response",
            "trajectory_center",
            "trajectory_distribution",
            "trajectory_velocity",
            "temporal_consistency",
            "lpips",
        )
    }
    counts = {key: 0.0 for key in totals}
    mode_counts = {"ordinary": 0.0, "response": 0.0}
    for batch_index in range(args.validation_batches):
        targets, mode, _ = make_formal_training_batch(
            records,
            pairs,
            step=batch_index,
            accumulation_index=0,
            gradient_accumulation=1,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 9_000_000,
            response_updates=True,
        )
        trajectory_batch = (
            select_trajectory_input_records(
                targets, nominal_records, args.trajectory_input_source
            )
            if args.trajectory_input
            else targets
        )
        loaded = load_microbatch(
            artifact_root,
            dataset_root,
            targets,
            device,
            trajectory_batch=trajectory_batch,
            trajectory_representation=args.trajectory_representation,
            scene_size_px=scene_size_px,
        )
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 10_000_000 + batch_index * world_size + rank
        )
        response_enabled = mode == "response"
        losses = forward_losses(
            model,
            condition_encoder,
            loaded,
            args,
            device,
            generator,
            response_enabled=response_enabled,
            vae=vae,
            perceptual_model=perceptual_model,
            fixed_sigma=0.7,
        )
        mode_counts[mode] += 1.0
        for key in totals:
            if key == "response" and not response_enabled:
                continue
            totals[key] += float(losses[key].detach().cpu())
            counts[key] += 1.0
    metrics = reduce_metrics(totals, counts, device, world_size)
    mode_tensor = torch.tensor(
        [mode_counts["ordinary"], mode_counts["response"]],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(mode_tensor, op=dist.ReduceOp.SUM)
    metrics["ordinary_batches"] = float(mode_tensor[0].cpu())
    metrics["response_batches"] = float(mode_tensor[1].cpu())
    model.train()
    condition_encoder.train()
    return metrics


def adapter_metadata(
    args: argparse.Namespace,
    cache_path: Path,
    checkpoint: Path,
    model: torch.nn.Module,
    train_records: list[dict],
    world_size: int,
    completed_steps: int,
    history: list[dict],
    validation_history: list[dict],
    peak_memory: float,
    direct_object_slots: int,
) -> dict:
    return {
        "schema": "phycontext.wan_condition_adapter.v3",
        "base_model": str(checkpoint),
        "base_model_index_sha256": sha256(
            checkpoint / "diffusion_pytorch_model.safetensors.index.json"
        ),
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": sha256(cache_path),
        "steps": completed_steps,
        "planned_steps": args.steps,
        "training_mode": (
            "ordinary_only_integration_smoke"
            if args.ordinary_only
            else (
                "trajectory_warmup_scene_plus_target_trajectory"
                if args.training_stage == "trajectory_warmup"
                else (
                    "formal_mixed_trajectory_plus_physics"
                    if args.trajectory_input
                    else "formal_mixed_no_trajectory_input"
                )
            )
        ),
        "training_stage": args.training_stage,
        "condition_token_mode": training_condition_mode(args.training_stage),
        "initialized_from_adapter": (
            str(args.initialize_adapter) if args.initialize_adapter else None
        ),
        "optimizer_scope": (
            "scene_lora_trajectory"
            if args.training_stage == "trajectory_warmup"
            else "joint"
        ),
        "sample_count": len(train_records),
        "selected_base_scene_count": len(
            {item["base_scene_id"] for item in train_records}
        ),
        "scene_tokens": args.scene_tokens,
        "scene_point_conditioning": {
            "object_axis": "O",
            "object_point_layout": "[O, 2048, 3]",
            "max_objects": 3,
            "shared_point_encoder": True,
            "object_slot_embedding": "learned_slot_0_1_2",
        },
        "control_tokens": 3,
        "base_dynamics_tokens": 10,
        "base_dynamics_layout": BASE_DYNAMICS_LAYOUT,
        "state_tokens": 3,
        "physics_tokens": 16,
        "lora": {
            "target": "wan.blocks.*.cross_attn.{q,k,v,o}",
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "module_count": len(
                [module for module in model.modules() if isinstance(module, LoRALinear)]
            ),
        },
        "direct_modulation": {
            "enabled": True,
            "active_during_training": args.training_stage == "joint",
            "trainable_during_training": args.training_stage == "joint",
            "target": "wan.blocks.*.adaln",
            "source": "structured_controls_v2",
            "architecture": "per_layer_v2",
            "input_dim": 12 * direct_object_slots,
            "object_slots": direct_object_slots,
            "rank": args.modulation_rank,
            "alpha": args.modulation_alpha,
            "layer_count": len(model.blocks),
            "normalization": {
                "mass_kg": "log",
                "contact_friction": "log_clamped_1e-4",
                "contact_restitution": "linear_minus1_plus1",
                "linear_velocity_camera_m_s": "asinh",
                "angular_velocity_camera_rad_s": "asinh_div5",
                "gravity_camera_m_s2": "div9.81",
            },
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "encoder_learning_rate": args.encoder_learning_rate,
            "lora_learning_rate": args.lora_learning_rate,
            "direct_learning_rate": args.direct_learning_rate,
            "trajectory_learning_rate": args.trajectory_learning_rate,
            "encoder_weight_decay": args.encoder_weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
            "schedule": "linear_warmup_cosine_decay",
            "max_grad_norm": args.max_grad_norm,
        },
        "sampling": {
            "response_updates_enabled": not args.ordinary_only,
            "ordinary_share": 1.0 if args.ordinary_only else 0.6,
            "response_share": 0.0 if args.ordinary_only else 0.4,
            "response_axis_weights": (
                {}
                if args.ordinary_only
                else {
                    "contact_friction": 0.4,
                    "contact_restitution": 0.4,
                    "mass_kg": 0.2,
                }
            ),
            "response_pair": (
                None
                if args.ordinary_only
                else "same_base_same_axis_low_high_common_noise_sigma"
            ),
        },
        "validation": {
            "enabled": not args.ordinary_only,
            "batches_per_rank": 0 if args.ordinary_only else args.validation_batches,
            "selection": "deterministic_formal_training_schedule",
            "ordinary_share": 1.0 if args.ordinary_only else 0.6,
            "response_share": 0.0 if args.ordinary_only else 0.4,
            "response_metric_scope": "response_batches_only",
            "diffusion_sigma": 0.7,
            "clean_condition_frame_in_lpips": False,
            "lpips_decode_protocol": "local_generated_windows_fresh_causal_cache",
        },
        "motion_supervision": {
            "enabled": True,
            "mask_source": "trajectory_visibility_derived_transport_envelope",
            "foreground_share": args.motion_foreground_share,
            "reconstruction_region": "source_plus_per_frame_target_transport_envelope",
            "pair_region": "low_high_source_target_transport_union",
        },
        "trajectory_supervision": {
            "enabled": True,
            "provided_as_model_input": args.trajectory_input,
            "type": "clean_latent_correspondence_motion",
            "weights": {
                "center": args.trajectory_center_loss_weight,
                "distribution": args.trajectory_distribution_loss_weight,
                "velocity": args.trajectory_velocity_loss_weight,
            },
            "objectives": {
                "center": "normalized_image_xy_smooth_l1",
                "distribution": "spatial_probability_jensen_shannon",
                "velocity": "adjacent_normalized_image_xy_smooth_l1",
            },
            "temperature": args.trajectory_temperature,
            "beta": args.trajectory_beta,
            "coordinate_frame": "normalized_image_xy",
            "identity_reference": "current_target_frame_features",
        },
        "temporal_consistency": {
            "enabled": args.temporal_consistency_loss_weight > 0,
            "weight": args.temporal_consistency_loss_weight,
            "beta": args.temporal_consistency_beta,
            "space": "clean_latent_adjacent_delta",
            "target": "current_target_clean_latent_adjacent_delta",
        },
        "trajectory_conditioning": {
            "enabled": args.trajectory_input,
            "representation": args.trajectory_representation,
            "input_record": args.trajectory_input_source,
            "source": "target_sample_bound_simulation_3d_point_trajectory",
            "target_supervision_source": "cached_target_latent_with_trajectory_visibility_loss_mask",
            "future_appearance_or_silhouette": False,
            "source_identity_mask": (
                "first_frame_point_visibility_occupancy"
                if args.trajectory_representation == "das_3d_tracks"
                else "first_frame_source_point_occupancy"
            ),
            "future_target_geometry": (
                "first_frame_xyz_identity_rgb_with_full_resolution_dynamic_static_zbuffer_visibility"
                if args.trajectory_representation == "das_3d_tracks"
                else "all_2048_point_tracks_per_object_with_depth_and_validity"
            ),
            "channels": trajectory_channel_names(args.trajectory_representation),
            "input_channels": (
                getattr(model, "phycontext_trajectory_conditioner").input_channels
                if args.trajectory_input
                else 0
            ),
            "rank": args.trajectory_condition_rank,
            "architecture": (
                getattr(model, "phycontext_trajectory_conditioner").architecture
                if args.trajectory_input
                else None
            ),
            "patch_size": [int(value) for value in model.patch_size],
            "injection": "zero_initialized_additive_video_patch_residual",
            "spatial_visibility_resolution": (
                "full_preprocess_resolution_before_latent_reduction"
                if args.trajectory_representation == "das_3d_tracks"
                else "prealigned_latent_grid"
            ),
            "temporal_alignment": (
                "all_4n_plus_1_frames_then_learned_causal_stride4"
                if args.trajectory_representation == "das_3d_tracks"
                else "prealigned_latent_frames"
            ),
            "independently_switchable": True,
            "training_role": (
                "target_matching_object_motion_warmup"
                if args.training_stage == "trajectory_warmup"
                else (
                    "sample_bound_simulation_trajectory"
                    if args.trajectory_input_source == "target"
                    else "fixed_trajectory_physics_response_ablation"
                )
            ),
        },
        "response_loss_weight": args.response_loss_weight,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "lpips_loss_weight": args.lpips_loss_weight,
        "lpips_frame_count": args.lpips_frame_count,
        "lpips_temporal_window": args.lpips_temporal_window,
        "lpips_resolution": args.lpips_resolution,
        "lpips_net": args.lpips_net,
        "minimum_response_energy": args.minimum_response_energy,
        "flow_shift": args.flow_shift,
        "distributed": {
            "world_size": world_size,
            "microbatch_videos_per_rank": 2,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_global_video_batch": 2
            * world_size
            * args.gradient_accumulation,
        },
        "history": history,
        "validation_history": validation_history,
        "losses": [item["total"] for item in history],
        "reconstruction_losses": [item["reconstruction"] for item in history],
        "response_losses": [item["response"] for item in history],
        "lpips_losses": [item["lpips"] for item in history],
        "trajectory_center_losses": [
            item["trajectory_center"] for item in history
        ],
        "trajectory_distribution_losses": [
            item["trajectory_distribution"] for item in history
        ],
        "trajectory_velocity_losses": [
            item["trajectory_velocity"] for item in history
        ],
        "temporal_consistency_losses": [
            item.get("temporal_consistency", 0.0) for item in history
        ],
        "peak_cuda_memory_gib": peak_memory,
        "seed": args.seed,
    }


def write_input_contract(checkpoint_root: Path, metadata: dict) -> None:
    cache = json.loads(Path(metadata["cache_manifest"]).read_text(encoding="utf-8"))
    preprocess = cache["preprocess"]
    input_contract = {
        "schema": "phycontext.inference_input_contract.v5",
        "adapter_sha256": sha256(checkpoint_root / "adapter.safetensors"),
        "base_model_index_sha256": metadata["base_model_index_sha256"],
        "cache_manifest_sha256": metadata["cache_manifest_sha256"],
        "scene_tokens": metadata["scene_tokens"],
        "sampling": {
            "frames": int(preprocess["frames"]),
            "width": int(preprocess["width"]),
            "height": int(preprocess["height"]),
            "max_area": int(preprocess["width"]) * int(preprocess["height"]),
            "spatial_preprocess": "cover_then_center_crop",
            "condition_frame": preprocess["condition_frame"],
            "flow_shift": float(metadata["flow_shift"]),
            "guidance_scale": INFERENCE_GUIDANCE_SCALE,
            "steps": INFERENCE_SAMPLING_STEPS,
        },
        "scene": {
            "camera_intrinsics": "cover_then_center_crop_adjusted_and_target_normalized",
        },
        "trajectory": {
            "enabled": metadata["trajectory_conditioning"]["enabled"],
            "representation": metadata["trajectory_conditioning"]["representation"],
            "protocol": metadata["trajectory_conditioning"]["input_record"],
            "architecture": metadata["trajectory_conditioning"].get("architecture"),
            "condition_shape": (
                [
                    metadata["trajectory_conditioning"]["input_channels"],
                    (
                        int(preprocess["frames"])
                        if metadata["trajectory_conditioning"]["representation"]
                        == "das_3d_tracks"
                        else (int(preprocess["frames"]) - 1) // 4 + 1
                    ),
                    int(preprocess["height"]) // 16,
                    int(preprocess["width"]) // 16,
                ]
                if metadata["trajectory_conditioning"]["enabled"]
                else None
            ),
        },
    }
    (checkpoint_root / "input_contract.json").write_text(
        json.dumps(input_contract, indent=2, sort_keys=True), encoding="utf-8"
    )


def save_training_checkpoint(
    destination: Path,
    model,
    condition_encoder,
    metadata: dict,
    optimizer,
    completed_steps: int,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_condition_checkpoint(
        temporary, unwrap(model), unwrap(condition_encoder), metadata
    )
    write_input_contract(temporary, metadata)
    torch.save(
        {
            "completed_steps": completed_steps,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        temporary / "training_state.pt",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def write_run_contract(
    output: Path,
    cache_path: Path,
    cache: dict,
    args: argparse.Namespace,
    train_records: list[dict],
    validation_records: list[dict],
) -> None:
    payload = {
        "schema": "phycontext.formal_training_run.v2",
        "training_code_sha256": training_code_sha256(),
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": sha256(cache_path),
        "source_manifest": cache["source_manifest"],
        "source_manifest_sha256": cache["source_manifest_sha256"],
        "train_sample_ids": [item["sample_id"] for item in train_records],
        "validation_sample_ids": [item["sample_id"] for item in validation_records],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "run_contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


RESUME_IMMUTABLE_ARGUMENTS = (
    "checkpoint",
    "wan_repo",
    "training_stage",
    "steps",
    "ordinary_only",
    "base_scene_count",
    "scene_tokens",
    "lora_rank",
    "lora_alpha",
    "modulation_rank",
    "modulation_alpha",
    "encoder_learning_rate",
    "lora_learning_rate",
    "direct_learning_rate",
    "trajectory_learning_rate",
    "warmup_ratio",
    "minimum_learning_rate_ratio",
    "encoder_weight_decay",
    "gradient_accumulation",
    "flow_shift",
    "reconstruction_loss_weight",
    "lpips_loss_weight",
    "lpips_frame_count",
    "lpips_temporal_window",
    "lpips_resolution",
    "lpips_net",
    "response_loss_weight",
    "motion_foreground_share",
    "minimum_response_energy",
    "trajectory_center_loss_weight",
    "trajectory_distribution_loss_weight",
    "trajectory_velocity_loss_weight",
    "temporal_consistency_loss_weight",
    "temporal_consistency_beta",
    "trajectory_temperature",
    "trajectory_beta",
    "trajectory_input",
    "trajectory_input_source",
    "trajectory_condition_rank",
    "trajectory_representation",
    "max_grad_norm",
    "validation_every",
    "validation_batches",
    "save_every",
    "no_gradient_checkpointing",
    "seed",
)


def validate_resume_contract(
    root: Path, resume: Path, cache_path: Path, args: argparse.Namespace
) -> None:
    source = (root / resume).resolve()
    contract_path = source.parent / "run_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"resume requires its original run contract: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "phycontext.formal_training_run.v2":
        raise ValueError(
            "resume requires a v2 run contract with training code fingerprints"
        )
    previous_code = contract.get("training_code_sha256")
    current_code = training_code_sha256()
    if previous_code != current_code:
        raise ValueError(
            "resume training code differs from the original run; initialize a "
            "new adapter run instead"
        )
    if contract.get("cache_manifest_sha256") != sha256(cache_path):
        raise ValueError("resume cache manifest differs from the original run")
    previous = contract.get("arguments", {})
    current = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    changed = {
        key: {"previous": previous.get(key), "current": current.get(key)}
        for key in RESUME_IMMUTABLE_ARGUMENTS
        if previous.get(key) != current.get(key)
    }
    if changed:
        raise ValueError(
            "resume cannot change the formal run contract: "
            + json.dumps(changed, sort_keys=True)
        )


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.initialize_adapter is not None:
        raise ValueError("resume and initialize-adapter are mutually exclusive")
    if args.training_stage == "trajectory_warmup" and (
        not args.trajectory_input or args.trajectory_input_source != "target"
    ):
        raise ValueError(
            "trajectory_warmup requires --trajectory-input and "
            "--trajectory-input-source target"
        )
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient accumulation must be positive")
    if args.scene_tokens <= 0:
        raise ValueError("scene token count must be positive")
    if min(
        args.lora_rank,
        args.modulation_rank,
        args.trajectory_condition_rank,
    ) <= 0:
        raise ValueError("adapter ranks must be positive")
    if args.lora_alpha <= 0 or args.modulation_alpha <= 0:
        raise ValueError("adapter alpha values must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup ratio must lie in [0, 1)")
    if not 0 < args.minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must lie in (0, 1]")
    if args.encoder_weight_decay < 0:
        raise ValueError("encoder weight decay must be nonnegative")
    if args.flow_shift <= 0 or args.max_grad_norm <= 0:
        raise ValueError("flow shift and maximum gradient norm must be positive")
    if args.response_loss_weight < 0:
        raise ValueError("response loss weight must be nonnegative")
    if args.minimum_response_energy <= 0:
        raise ValueError("minimum response energy must be positive")
    if args.ordinary_only and args.steps != 1:
        raise ValueError("ordinary-only integration smoke requires exactly one step")
    if (
        args.validation_every <= 0
        or args.validation_batches <= 0
        or args.save_every <= 0
    ):
        raise ValueError("validation and save intervals must be positive")
    if not args.ordinary_only and (
        args.validation_batches < 5 or args.validation_batches % 5 != 0
    ):
        raise ValueError(
            "formal validation batches must be a positive multiple of five "
            "to preserve the 60/40 mixture"
        )
    if not 0 < args.motion_foreground_share < 1:
        raise ValueError("motion foreground share must be between zero and one")
    if args.reconstruction_loss_weight < 0:
        raise ValueError("reconstruction loss weight must be nonnegative")
    if args.lpips_loss_weight < 0 or args.lpips_frame_count <= 0:
        raise ValueError("LPIPS weight and frame count must be nonnegative/positive")
    if args.lpips_temporal_window <= 0:
        raise ValueError("LPIPS temporal window must be positive")
    if args.temporal_consistency_loss_weight < 0:
        raise ValueError("temporal consistency loss weight must be nonnegative")
    if args.temporal_consistency_beta <= 0:
        raise ValueError("temporal consistency Smooth-L1 beta must be positive")
    if args.lpips_loss_weight > 0 and args.lpips_resolution < 32:
        raise ValueError("LPIPS resolution must be at least 32 when enabled")
    if min(
        args.encoder_learning_rate,
        args.lora_learning_rate,
        args.direct_learning_rate,
        args.trajectory_learning_rate,
        args.trajectory_temperature,
        args.trajectory_beta,
        float(args.trajectory_condition_rank),
    ) <= 0:
        raise ValueError("formal optimization settings must be positive")
    trajectory_weights = (
        args.trajectory_center_loss_weight,
        args.trajectory_distribution_loss_weight,
        args.trajectory_velocity_loss_weight,
    )
    if min(trajectory_weights) < 0 or not any(trajectory_weights):
        raise ValueError("trajectory weights must be nonnegative and not all zero")

    rank, local_rank, world_size, device = setup_distributed()
    root = args.project_root.resolve()
    cache_path = (root / args.cache_manifest).resolve()
    output = (root / args.output).resolve()
    checkpoint = (root / args.checkpoint).resolve()
    wan_repo = (root / args.wan_repo).resolve()
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_representation = cache.get("point_track_preprocess", {}).get(
            "representation", "dense_point_tracks"
        )
        if cache_representation != args.trajectory_representation:
            raise ValueError(
                "cache trajectory representation differs from the training request"
            )
        if (
            args.trajectory_representation == "das_3d_tracks"
            and cache.get("schema") != CURRENT_CACHE_SCHEMA
        ):
            raise ValueError(
                "full-rate das_3d_tracks training requires the current Wan cache "
                f"schema {CURRENT_CACHE_SCHEMA}"
            )
        validate_cache_source_manifest(root, cache)
        dataset_root = resolve_cache_dataset_root(root, cache)
        artifact_root = resolve_cache_artifact_root(root, cache)
        scene_size_px = (
            int(cache["preprocess"]["width"]),
            int(cache["preprocess"]["height"]),
        )
        train_records = filter_base_scenes(
            [item for item in cache["records"] if item["record"]["split"] == "train"],
            args.base_scene_count,
        )
        validation_records = filter_base_scenes(
            [
                item
                for item in cache["records"]
                if item["record"]["split"] == "validation"
            ],
            args.base_scene_count,
        )
        if not validation_records and not args.ordinary_only:
            raise ValueError(
                "formal response training requires a non-empty validation split"
            )
        require_complete_cache(
            artifact_root,
            dataset_root,
            train_records,
            "train",
            args.trajectory_representation,
        )
        if validation_records:
            require_complete_cache(
                artifact_root,
                dataset_root,
                validation_records,
                "validation",
                args.trajectory_representation,
            )
        max_dynamic_object_count(train_records)
        max_dynamic_object_count(validation_records)
        if args.resume is not None:
            validate_resume_contract(root, args.resume, cache_path, args)
        direct_object_slots = 3
        train_base_count = len({item["base_scene_id"] for item in train_records})
        train_pairs = (
            []
            if args.ordinary_only
            else select_sweep_endpoint_pairs(train_records, train_base_count)
        )
        train_nominal_records = index_nominal_trajectory_records(train_records)
        validation_pairs = []
        validation_nominal_records = {}
        if validation_records and not args.ordinary_only:
            validation_pairs = select_sweep_endpoint_pairs(
                validation_records,
                len({item["base_scene_id"] for item in validation_records}),
            )
            validation_nominal_records = index_nominal_trajectory_records(
                validation_records
            )

        if output.exists() and any(output.iterdir()) and args.resume is None:
            raise FileExistsError(f"formal training output is not empty: {output}")
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not (output / "run_contract.json").is_file():
                write_run_contract(
                    output,
                    cache_path,
                    cache,
                    args,
                    train_records,
                    validation_records,
                )
        barrier(world_size)

        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        sys.path.insert(0, str(wan_repo))
        from wan.modules.model import WanModel

        model = WanModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        if model.model_type != "ti2v" or model.in_dim != 48 or model.text_dim != 4096:
            raise ValueError("checkpoint is not Wan2.2 TI2V-5B")
        model.requires_grad_(False)
        condition_encoder = PhyContextConditionEncoder(
            scene_token_count=args.scene_tokens
        )
        completed_steps = 0
        history = []
        validation_history = []
        adapter_source = args.resume or args.initialize_adapter
        previous = None
        if adapter_source is not None:
            source = (root / adapter_source).resolve()
            previous = load_condition_checkpoint(source, model, condition_encoder)
            if (
                int(previous["scene_tokens"]) != args.scene_tokens
                or int(previous["lora"]["rank"]) != args.lora_rank
                or float(previous["lora"]["alpha"]) != args.lora_alpha
            ):
                raise ValueError("source adapter architecture differs from arguments")
            previous_direct = previous.get("direct_modulation", {})
            if (
                not previous_direct.get("enabled", False)
                or int(previous_direct.get("rank", 0)) != args.modulation_rank
                or float(previous_direct.get("alpha", 0.0))
                != args.modulation_alpha
            ):
                raise ValueError(
                    "source adapter direct-modulation architecture differs "
                    "from arguments"
                )
            previous_slots = int(
                previous_direct.get(
                    "object_slots", int(previous_direct.get("input_dim", 12)) // 12
                )
            )
            if previous_slots != direct_object_slots:
                raise ValueError(
                    "source adapter uses a different object-slot width; "
                    "start a new three-slot adapter"
                )
            if args.resume is not None:
                history = list(previous.get("history", []))
                validation_history = list(previous.get("validation_history", []))
        else:
            inject_cross_attention_lora(
                model, rank=args.lora_rank, alpha=args.lora_alpha
            )
            inject_direct_condition_modulation(
                model,
                rank=args.modulation_rank,
                alpha=args.modulation_alpha,
                input_dim=12 * direct_object_slots,
                object_slots=direct_object_slots,
            )

        existing_trajectory = getattr(
            model, "phycontext_trajectory_conditioner", None
        )
        if args.resume is not None:
            previous_trajectory = bool(
                previous.get("trajectory_conditioning", {}).get("enabled", False)
            )
            if previous_trajectory != args.trajectory_input:
                raise ValueError(
                    "resume cannot add or remove trajectory conditioning; "
                    "use --initialize-adapter for a new run"
                )
        if args.trajectory_input:
            expected_trajectory_architecture = (
                "full_frame_causal_patch_v2"
                if args.trajectory_representation == "das_3d_tracks"
                else "framewise_patch"
            )
            if existing_trajectory is None:
                existing_trajectory = inject_trajectory_conditioning(
                    model,
                    rank=args.trajectory_condition_rank,
                    representation=args.trajectory_representation,
                )
            elif existing_trajectory.rank != args.trajectory_condition_rank:
                raise ValueError("trajectory conditioner rank differs from arguments")
            elif existing_trajectory.representation != args.trajectory_representation:
                raise ValueError(
                    "trajectory conditioner representation differs from arguments"
                )
            elif (
                existing_trajectory.architecture
                != expected_trajectory_architecture
            ):
                raise ValueError(
                    "trajectory conditioner architecture differs from the current "
                    "cache protocol; start a new adapter"
                )
        elif existing_trajectory is not None:
            raise ValueError("source adapter contains trajectory conditioning")

        configure_training_stage(condition_encoder, model, args.training_stage)
        checkpointed_blocks = 0
        if not args.no_gradient_checkpointing:
            checkpointed_blocks = enable_block_checkpointing(model)
        model.to(device).train()
        condition_encoder.to(device).train()
        vae = None
        perceptual_model = None
        if args.lpips_loss_weight > 0:
            import lpips
            from wan.modules.vae2_2 import Wan2_2_VAE

            vae = Wan2_2_VAE(
                vae_pth=str(checkpoint / "Wan2.2_VAE.pth"),
                dtype=torch.bfloat16,
                device=device,
            )
            vae.model.to(dtype=torch.bfloat16).requires_grad_(False).eval()
            perceptual_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
            for parameter in perceptual_model.parameters():
                parameter.requires_grad_(False)
        groups = optimizer_groups(condition_encoder, model, args)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        if args.resume is not None:
            state = torch.load(
                (root / args.resume / "training_state.pt").resolve(),
                map_location=device,
                weights_only=False,
            )
            optimizer.load_state_dict(state["optimizer"])
            completed_steps = int(state["completed_steps"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])
        if completed_steps >= args.steps:
            raise ValueError("resume checkpoint already reached the requested steps")

        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if any(parameter.requires_grad for parameter in condition_encoder.parameters()):
                condition_encoder = DistributedDataParallel(
                    condition_encoder,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
        torch.cuda.reset_peak_memory_stats(device)
        progress_path = output / "progress.jsonl"
        best_score = min(
            (item["total"] for item in validation_history), default=float("inf")
        )

        for step in range(completed_steps, args.steps):
            factor = learning_rate_factor(
                step,
                args.steps,
                args.warmup_ratio,
                args.minimum_learning_rate_ratio,
            )
            learning_rates = apply_learning_rate(optimizer, factor)
            optimizer.zero_grad(set_to_none=True)
            local_sums = {
                key: 0.0
                for key in (
                    "total",
                    "reconstruction",
                    "response",
                    "trajectory_center",
                    "trajectory_distribution",
                    "trajectory_velocity",
                    "temporal_consistency",
                    "lpips",
                    "sigma",
                )
            }
            local_counts = {key: 0.0 for key in local_sums}
            step_mode = None
            step_axis = None
            sample_ids = []
            trajectory_sample_ids = []
            for accumulation_index in range(args.gradient_accumulation):
                batch, mode, axis = make_formal_training_batch(
                    train_records,
                    train_pairs,
                    step,
                    accumulation_index,
                    args.gradient_accumulation,
                    rank,
                    world_size,
                    args.seed,
                    response_updates=not args.ordinary_only,
                )
                step_mode = mode
                step_axis = axis
                sample_ids.extend(item["sample_id"] for item in batch)
                trajectory_batch = (
                    select_trajectory_input_records(
                        batch, train_nominal_records, args.trajectory_input_source
                    )
                    if args.trajectory_input
                    else batch
                )
                loaded = load_microbatch(
                    artifact_root,
                    dataset_root,
                    batch,
                    device,
                    trajectory_batch=trajectory_batch,
                    trajectory_representation=args.trajectory_representation,
                    scene_size_px=scene_size_px,
                )
                trajectory_sample_ids.extend(loaded["trajectory_sample_ids"])
                generator = torch.Generator(device=device).manual_seed(
                    args.seed
                    + step * 100_000
                    + accumulation_index * 1_000
                    + rank
                )
                synchronize = accumulation_index + 1 == args.gradient_accumulation
                with contextlib.ExitStack() as stack:
                    if not synchronize and world_size > 1:
                        stack.enter_context(model.no_sync())
                        stack.enter_context(condition_encoder.no_sync())
                    losses = forward_losses(
                        model,
                        condition_encoder,
                        loaded,
                        args,
                        device,
                        generator,
                        response_enabled=mode == "response",
                        vae=vae,
                        perceptual_model=perceptual_model,
                    )
                    if not torch.isfinite(losses["total"]):
                        raise RuntimeError("formal training loss is non-finite")
                    (losses["total"] / args.gradient_accumulation).backward()
                for key in local_sums:
                    local_sums[key] += float(losses[key].detach().cpu())
                    local_counts[key] += float(key != "response" or mode == "response")

            raw_model = unwrap(model)
            raw_condition = unwrap(condition_encoder)
            trainable = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            completed_steps = step + 1
            metrics = reduce_metrics(local_sums, local_counts, device, world_size)
            report = {
                "step": completed_steps,
                "mode": step_mode,
                "axis": step_axis,
                "sample_ids_rank0": sample_ids if rank == 0 else [],
                "trajectory_sample_ids_rank0": (
                    trajectory_sample_ids if rank == 0 else []
                ),
                **metrics,
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rates": learning_rates,
            }
            if rank == 0:
                history.append(report)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report, sort_keys=True) + "\n")
                print(json.dumps(report), flush=True)

            validation_due = bool(
                validation_pairs
                and (
                    completed_steps % args.validation_every == 0
                    or completed_steps == args.steps
                )
            )
            if validation_due:
                validation = validate(
                    model,
                    condition_encoder,
                    artifact_root,
                    dataset_root,
                    validation_records,
                    validation_pairs,
                    validation_nominal_records,
                    args,
                    device,
                    rank,
                    world_size,
                    scene_size_px,
                    vae=vae,
                    perceptual_model=perceptual_model,
                )
                validation["step"] = completed_steps
                if rank == 0:
                    validation_history.append(validation)
                    print(json.dumps({"validation": validation}), flush=True)

            improved = False
            if validation_due:
                improved = validation["total"] < best_score
                if improved:
                    best_score = validation["total"]

            save_due = bool(
                completed_steps % args.save_every == 0
                or completed_steps == args.steps
                or validation_due
            )
            if save_due:
                barrier(world_size)
                if rank == 0:
                    peak = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
                    metadata = adapter_metadata(
                        args,
                        cache_path,
                        checkpoint,
                        raw_model,
                        train_records,
                        world_size,
                        completed_steps,
                        history,
                        validation_history,
                        peak,
                        direct_object_slots,
                    )
                    save_training_checkpoint(
                        output / "latest",
                        raw_model,
                        raw_condition,
                        metadata,
                        optimizer,
                        completed_steps,
                    )
                    if validation_due and improved:
                        best_tmp = output / "best.tmp"
                        if best_tmp.exists():
                            shutil.rmtree(best_tmp)
                        shutil.copytree(output / "latest", best_tmp)
                        if (output / "best").exists():
                            shutil.rmtree(output / "best")
                        best_tmp.replace(output / "best")
                barrier(world_size)

        barrier(world_size)
        if rank == 0:
            final = output / "final"
            if final.exists():
                shutil.rmtree(final)
            shutil.copytree(output / "latest", final)
            print(
                json.dumps(
                    {
                        "passed": True,
                        "output": str(output),
                        "completed_steps": completed_steps,
                        "checkpointed_blocks": checkpointed_blocks,
                        "effective_global_video_batch": 2
                        * world_size
                        * args.gradient_accumulation,
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
