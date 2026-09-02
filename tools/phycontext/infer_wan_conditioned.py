#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from conditioning_model import (
    PhyContextConditionEncoder,
    apply_condition_mode,
    compose_wan_context,
    controls_from_record,
    dynamics_from_record,
    initial_state_from_record,
    load_scene_condition,
)
from cache_contract import (
    CANONICAL_CONDITION_FRAME_PROTOCOL,
    CURRENT_CACHE_SCHEMA,
    resolve_cache_artifact_root,
    resolve_cache_dataset_root,
    validate_cache_artifact,
    validate_cache_source_manifest,
)
from project_defaults import (
    CACHE_MANIFEST,
    INFERENCE_FLOW_SHIFT,
    INFERENCE_GUIDANCE_SCALE,
    INFERENCE_SAMPLING_STEPS,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from video_preprocess import cover_center_crop_frames
from wan_training import (
    index_nominal_trajectory_records,
    load_condition_checkpoint,
    set_direct_condition,
    set_trajectory_condition,
    select_trajectory_input_records,
    canonical_trajectory_representation,
    validate_point_track_object_slots,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = Path("outputs/training/formal/final")
DEFAULT_CACHE = CACHE_MANIFEST
DEFAULT_WAN_REPO = Path(os.environ.get("PHYCONTEXT_WAN_REPO", "external/Wan2.2"))
CONDITION_MODES = (
    "full",
    "no_trajectory",
    "trajectory_only",
    "scene_only",
    "physics_only",
    "adapter_only",
    "pretrained",
)


def condition_token_mode(inference_mode: str) -> str:
    """Map trajectory ablations onto the trained scene/physics token slots."""
    if inference_mode == "no_trajectory":
        return "full"
    if inference_mode == "trajectory_only":
        return "scene_only"
    return inference_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run physics-conditioned Wan2.2 TI2V inference"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wan-repo", type=Path, default=DEFAULT_WAN_REPO)
    parser.add_argument("--sample-id")
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--scene-condition", type=Path)
    parser.add_argument(
        "--trajectory-condition",
        type=Path,
        help=(
            "camera-view trajectory cache produced by the current scene's "
            "physics simulation; dataset mode resolves this automatically"
        ),
    )
    parser.add_argument(
        "--trajectory-protocol",
        help="required for external trajectory inputs when the adapter has an input contract",
    )
    parser.add_argument("--text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=CONDITION_MODES, default="full")
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--contact-friction", type=float)
    parser.add_argument("--restitution", type=float)
    parser.add_argument("--inertia-tensor-camera-kg-m2", type=float, nargs=9)
    parser.add_argument("--rolling-friction", type=float)
    parser.add_argument("--spinning-friction", type=float)
    parser.add_argument("--linear-damping", type=float)
    parser.add_argument("--angular-damping", type=float)
    parser.add_argument("--linear-velocity-camera", type=float, nargs=3)
    parser.add_argument("--angular-velocity-camera", type=float, nargs=3)
    parser.add_argument("--gravity-camera", type=float, nargs=3)
    parser.add_argument("--frames", type=int, default=VIDEO_FRAMES)
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument(
        "--sampling-steps", type=int, default=INFERENCE_SAMPLING_STEPS
    )
    parser.add_argument("--guide-scale", type=float, default=INFERENCE_GUIDANCE_SCALE)
    parser.add_argument("--flow-shift", type=float, default=INFERENCE_FLOW_SHIFT)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_and_validate_input_contract(
    adapter_root: Path,
    adapter_metadata: dict,
    args: argparse.Namespace,
) -> dict:
    contract_path = adapter_root / "input_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"formal adapter is missing its input contract: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") not in {
        "phycontext.inference_input_contract.v5",
        "phycontext.inference_input_contract.v6",
    }:
        raise ValueError("adapter input contract schema is unsupported")
    contract_trajectory = contract.get("trajectory", {})
    metadata_trajectory = adapter_metadata.get("trajectory_conditioning", {})
    for key in ("enabled", "representation", "architecture"):
        expected = contract_trajectory.get(key)
        actual = metadata_trajectory.get(key)
        if expected is not None and expected != actual:
            raise ValueError(
                f"adapter input contract trajectory {key} mismatch: "
                f"expected {expected}, got {actual}"
            )
    checks = {
        "adapter_sha256": sha256(adapter_root / "adapter.safetensors"),
        "base_model_index_sha256": adapter_metadata.get("base_model_index_sha256"),
        "cache_manifest_sha256": adapter_metadata.get("cache_manifest_sha256"),
    }
    for key, actual in checks.items():
        expected = contract.get(key)
        if expected and actual != expected:
            raise ValueError(
                f"adapter input contract mismatch for {key}: expected {expected}, got {actual}"
            )
    expected_sampling = contract.get("sampling", {})
    requested_sampling = {
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "max_area": args.width * args.height,
        "spatial_preprocess": "cover_then_center_crop",
        "condition_frame": CANONICAL_CONDITION_FRAME_PROTOCOL,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guide_scale,
        "steps": args.sampling_steps,
    }
    mismatches = [
        f"{key}: expected {expected_sampling[key]}, got {actual}"
        for key, actual in requested_sampling.items()
        if key in expected_sampling and actual != expected_sampling[key]
    ]
    if mismatches:
        raise ValueError("sampling settings violate adapter input contract: " + "; ".join(mismatches))
    expected_intrinsics = contract.get("scene", {}).get("camera_intrinsics")
    actual_intrinsics = "cover_then_center_crop_adjusted_and_target_normalized"
    if expected_intrinsics != actual_intrinsics:
        raise ValueError(
            "adapter input contract has an unsupported camera-intrinsics protocol"
        )
    return contract


def validate_external_inputs(args: argparse.Namespace) -> bool:
    external = args.first_frame is not None
    companion_values = (
        args.scene_condition,
        getattr(args, "trajectory_condition", None),
        getattr(args, "trajectory_protocol", None),
        args.text,
        args.linear_velocity_camera,
        args.angular_velocity_camera,
        args.gravity_camera,
        args.inertia_tensor_camera_kg_m2,
        args.rolling_friction,
        args.spinning_friction,
        args.linear_damping,
        args.angular_damping,
    )
    if not external and any(value is not None for value in companion_values):
        raise ValueError("external scene arguments require --first-frame")
    if not external:
        return False
    if args.sample_id is not None:
        raise ValueError("--sample-id cannot be combined with --first-frame")
    required = {
        "scene-condition": args.scene_condition,
        "text": args.text,
        "mass-kg": args.mass_kg,
        "contact-friction": args.contact_friction,
        "restitution": args.restitution,
        "inertia-tensor-camera-kg-m2": args.inertia_tensor_camera_kg_m2,
        "rolling-friction": args.rolling_friction,
        "spinning-friction": args.spinning_friction,
        "linear-damping": args.linear_damping,
        "angular-damping": args.angular_damping,
        "linear-velocity-camera": args.linear_velocity_camera,
        "angular-velocity-camera": args.angular_velocity_camera,
        "gravity-camera": args.gravity_camera,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"external inference is missing: {', '.join(missing)}")
    if not args.text.strip():
        raise ValueError("external inference text must not be empty")
    validate_external_physics(args)
    return True


def validate_external_physics(args: argparse.Namespace) -> None:
    """Validate external physical inputs before loading the 5B model."""
    inertia = np.asarray(
        args.inertia_tensor_camera_kg_m2, dtype=np.float64
    ).reshape(3, 3)
    vectors = np.asarray(
        [
            *args.linear_velocity_camera,
            *args.angular_velocity_camera,
            *args.gravity_camera,
        ],
        dtype=np.float64,
    )
    scalars = np.asarray(
        [
            args.mass_kg,
            args.contact_friction,
            args.restitution,
            args.rolling_friction,
            args.spinning_friction,
            args.linear_damping,
            args.angular_damping,
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(inertia).all()
        or not np.isfinite(vectors).all()
        or not np.isfinite(scalars).all()
    ):
        raise ValueError("external physical inputs must be finite")
    if (
        args.mass_kg <= 0
        or args.contact_friction < 0
        or not 0 <= args.restitution <= 1
    ):
        raise ValueError(
            "controls require mass > 0, friction >= 0, restitution in [0, 1]"
        )
    dissipative = (
        args.rolling_friction,
        args.spinning_friction,
        args.linear_damping,
        args.angular_damping,
    )
    if any(value < 0 for value in dissipative):
        raise ValueError("friction and damping terms must be nonnegative")
    scale = max(float(np.abs(inertia).max()), 1.0e-15)
    if not np.allclose(inertia, inertia.T, rtol=0.0, atol=scale * 1.0e-7):
        raise ValueError("inertia tensor must be symmetric")
    if float(np.linalg.eigvalsh(inertia).min()) <= 0.0:
        raise ValueError("inertia tensor must be positive definite")


def validate_parameter_trajectory_consistency(
    args: argparse.Namespace, *, external: bool, trajectory_requested: bool
) -> None:
    overrides = (args.mass_kg, args.contact_friction, args.restitution)
    if not external and trajectory_requested and any(
        value is not None for value in overrides
    ):
        raise ValueError(
            "dataset parameter overrides require a trajectory recomputed for the "
            "new physical state; use external inference with matching scene and "
            "--trajectory-condition, or disable trajectory input for an ablation"
        )


class ConditionedTextEncoder:
    def __init__(self, base, condition_tokens: torch.Tensor, max_tokens: int):
        self.base = base
        self.model = base.model
        self.condition_tokens = condition_tokens
        self.max_tokens = max_tokens

    def __call__(self, prompts, device):
        text = self.base(prompts, device)
        tokens = self.condition_tokens.to(device=device, dtype=text[0].dtype)
        if len(text) != len(tokens):
            if len(tokens) != 1:
                raise ValueError("condition token batch does not match prompt batch")
            tokens = tokens.expand(len(text), -1, -1)
        return compose_wan_context(text, tokens, max_tokens=self.max_tokens)


def select_cache_record(
    cache: dict,
    adapter_root: Path,
    root: Path,
    sample_id: str | None,
) -> dict:
    adapter_metadata = json.loads(
        (adapter_root / "adapter.json").read_text(encoding="utf-8")
    )
    inputs_path = None
    if "training_inputs" in adapter_metadata:
        inputs_path = (root / adapter_metadata["training_inputs"]).resolve()
    allowed = None
    if inputs_path is not None:
        snapshot = json.loads(inputs_path.read_text(encoding="utf-8"))
        allowed = [record["sample_id"] for record in snapshot["records"]]
    if sample_id is None:
        if not allowed:
            raise ValueError("sample-id is required for an adapter without inputs.json")
        sample_id = allowed[0]
    if allowed is not None and sample_id not in set(allowed):
        raise ValueError("sample-id was not part of this adapter's training inputs")
    by_id = {record["sample_id"]: record for record in cache["records"]}
    if sample_id not in by_id:
        raise ValueError(f"sample is not available in the Wan cache: {sample_id}")
    return by_id[sample_id]


def main() -> None:
    args = parse_args()
    external = validate_external_inputs(args)
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError("frames must have the form 4n+1")
    if args.width <= 0 or args.height <= 0 or args.sampling_steps <= 0:
        raise ValueError("width, height, and sampling-steps must be positive")
    if args.width % 32 or args.height % 32:
        raise ValueError("Wan2.2 TI2V-5B inference width and height must divide by 32")
    root = args.project_root.resolve()
    adapter_root = (root / args.adapter).resolve()
    adapter_metadata = json.loads(
        (adapter_root / "adapter.json").read_text(encoding="utf-8")
    )
    input_contract = load_and_validate_input_contract(
        adapter_root, adapter_metadata, args
    )
    trajectory_enabled = bool(
        adapter_metadata.get("trajectory_conditioning", {}).get("enabled", False)
    )
    trajectory_representation = canonical_trajectory_representation(
        adapter_metadata.get("trajectory_conditioning", {}).get(
            "representation", "dense_point_tracks"
        )
    )
    if external:
        record = None
        sample_id = args.first_frame.stem
        first_frame_path = (root / args.first_frame).resolve()
        scene_path = (root / args.scene_condition).resolve()
        trajectory_path = (
            (root / args.trajectory_condition).resolve()
            if args.trajectory_condition is not None
            else None
        )
        trajectory_descriptor = None
        trajectory_sample_id = None
        prompt = args.text
        input_source = "external_scene"
        default_output = root / "outputs" / "inference" / f"{sample_id}__{args.mode}.mp4"
    else:
        cache_path = (root / args.cache_manifest).resolve()
        expected_cache_hash = adapter_metadata.get("cache_manifest_sha256")
        actual_cache_hash = sha256(cache_path)
        if expected_cache_hash and actual_cache_hash != expected_cache_hash:
            raise ValueError(
                "cache manifest does not match the adapter training input: "
                f"expected {expected_cache_hash}, got {actual_cache_hash}"
            )
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_representation = cache.get("point_track_preprocess", {}).get(
            "representation", "dense_point_tracks"
        )
        if cache_representation != trajectory_representation:
            raise ValueError(
                "inference cache trajectory representation differs from the adapter"
            )
        if (
            adapter_metadata.get("schema")
            in {
                "phycontext.wan_condition_adapter.v4",
                "phycontext.wan_condition_adapter.v5",
                "phycontext.wan_condition_adapter.v6",
            }
            and cache.get("schema") != CURRENT_CACHE_SCHEMA
        ):
            raise ValueError(
                "Track4Gen adapter inference requires the current Wan cache schema "
                f"{CURRENT_CACHE_SCHEMA}"
            )
        validate_cache_source_manifest(root, cache)
        dataset_root = resolve_cache_dataset_root(root, cache)
        artifact_root = resolve_cache_artifact_root(root, cache)
        item = select_cache_record(cache, adapter_root, root, args.sample_id)
        record = item["record"]
        sample_id = item["sample_id"]
        first_frame_path = (
            dataset_root / record["conditioning"]["first_frame"]
        ).resolve()
        scene_path = (dataset_root / record["conditioning"]["scene"]).resolve()
        if cache.get("schema") == CURRENT_CACHE_SCHEMA:
            source_files = (
                (first_frame_path, item.get("source_first_frame_sha256"), "first frame"),
                (scene_path, item.get("source_scene_sha256"), "scene condition"),
            )
            for source_path, expected_hash, label in source_files:
                if (
                    not expected_hash
                    or not source_path.is_file()
                    or sha256(source_path) != expected_hash
                ):
                    raise ValueError(f"{label} file no longer matches the Wan cache")
        trajectory_item = item
        if trajectory_enabled:
            trajectory_source = adapter_metadata.get(
                "trajectory_conditioning", {}
            ).get("input_record", "target")
            trajectory_item = select_trajectory_input_records(
                [item],
                index_nominal_trajectory_records(cache["records"]),
                trajectory_source,
            )[0]
        trajectory_descriptor = trajectory_item.get("point_track")
        if trajectory_enabled and trajectory_descriptor is None:
            raise ValueError("cache record has no point_track descriptor")
        trajectory_path = (
            validate_cache_artifact(
                artifact_root,
                trajectory_descriptor,
                f"point track for {trajectory_item['sample_id']}",
            )
            if trajectory_enabled
            else None
        )
        trajectory_sample_id = (
            trajectory_item["sample_id"] if trajectory_enabled else None
        )
        prompt = record["conditioning"]["text"]
        prompt_descriptor = item.get("text_context", {})
        expected_prompt_hash = prompt_descriptor.get("prompt_sha256")
        actual_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (
            cache.get("schema") == CURRENT_CACHE_SCHEMA
            and not expected_prompt_hash
        ):
            raise ValueError("Wan cache is missing its prompt hash binding")
        if expected_prompt_hash and actual_prompt_hash != expected_prompt_hash:
            raise ValueError("record prompt does not match its cached text context")
        input_source = "training_manifest"
        default_output = adapter_root / "inference" / f"{sample_id}__{args.mode}.mp4"
    if not first_frame_path.is_file():
        raise FileNotFoundError(f"missing first frame: {first_frame_path}")
    if not scene_path.is_file():
        raise FileNotFoundError(f"missing scene condition: {scene_path}")
    trajectory_requested = trajectory_enabled and args.mode in {
        "full",
        "trajectory_only",
    }
    validate_parameter_trajectory_consistency(
        args, external=external, trajectory_requested=trajectory_requested
    )
    if trajectory_requested:
        if trajectory_path is None:
            raise ValueError(
                "trajectory-conditioned external inference requires a trajectory "
                "cache generated from the current scene simulation; pass "
                "--trajectory-condition"
            )
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"missing trajectory condition: {trajectory_path}"
            )
        if external:
            expected_protocol = input_contract["trajectory"]["protocol"]
            if args.trajectory_protocol != expected_protocol:
                raise ValueError(
                    "external trajectory protocol does not match the adapter: "
                    f"expected --trajectory-protocol {expected_protocol}"
                )
    elif trajectory_path is not None and not trajectory_enabled:
        raise ValueError("adapter does not support trajectory conditioning")
    output = (
        (root / args.output).resolve()
        if args.output
        else default_output
    )
    if output.exists():
        raise FileExistsError(f"inference output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    config = WAN_CONFIGS["ti2v-5B"]
    checkpoint = Path(adapter_metadata["base_model"]).resolve()
    model_index = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    expected_model_hash = adapter_metadata.get("base_model_index_sha256")
    if expected_model_hash:
        if not model_index.is_file():
            raise FileNotFoundError(f"missing base-model index: {model_index}")
        if sha256(model_index) != expected_model_hash:
            raise ValueError("base model does not match the adapter checkpoint")
    pipeline = wan.WanTI2V(
        config=config,
        checkpoint_dir=str(checkpoint),
        device_id=args.device_id,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    device = pipeline.device
    overrides = (args.mass_kg, args.contact_friction, args.restitution)
    if external:
        controls = torch.tensor(overrides, dtype=torch.float32, device=device)
        inertia = np.asarray(
            args.inertia_tensor_camera_kg_m2, dtype=np.float64
        ).reshape(3, 3)
        dynamics = torch.tensor(
            [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
                args.rolling_friction,
                args.spinning_friction,
                args.linear_damping,
                args.angular_damping,
            ],
            dtype=torch.float32,
            device=device,
        )
        initial_state = torch.tensor(
            [
                *args.linear_velocity_camera,
                *args.angular_velocity_camera,
                *args.gravity_camera,
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        controls = controls_from_record(record, device)
        dynamics = dynamics_from_record(record, device)
        initial_state = initial_state_from_record(record, device)
    if args.mode == "pretrained" and not external and any(
        value is not None for value in overrides
    ):
        raise ValueError("control overrides have no effect in pretrained mode")
    if not external:
        if controls.ndim == 1:
            original_mass = controls[0].clone()
            for index, value in enumerate(overrides):
                if value is not None:
                    controls[index] = value
            if args.mass_kg is not None:
                dynamics[:6] *= controls[0] / original_mass
        elif any(value is not None for value in overrides):
            raise ValueError(
                "CLI physical overrides are single-object only; provide "
                "multi-object values in the dataset record"
            )
    if (
        not torch.isfinite(controls).all()
        or not torch.isfinite(dynamics).all()
        or not torch.isfinite(initial_state).all()
    ):
        raise ValueError("controls, base dynamics, and initial state must be finite")
    if controls.ndim == 1:
        valid_controls = (
            bool(controls[0] > 0)
            and bool(controls[1] >= 0)
            and bool(0 <= controls[2] <= 1)
        )
    else:
        valid_controls = bool(
            (controls[:, 0] > 0).all()
            and (controls[:, 1] >= 0).all()
            and ((controls[:, 2] >= 0) & (controls[:, 2] <= 1)).all()
        )
    if not valid_controls:
        raise ValueError(
            "controls require mass > 0, friction >= 0, restitution in [0, 1]"
        )
    if args.mode != "pretrained":
        scene_token_count = int(adapter_metadata["scene_tokens"])
        condition_encoder = PhyContextConditionEncoder(
            scene_token_count=scene_token_count
        )
        load_condition_checkpoint(adapter_root, pipeline.model, condition_encoder)
        direct_controls = controls.unsqueeze(0) if controls.ndim == 2 else controls
        direct_initial_state = (
            initial_state.unsqueeze(0) if initial_state.ndim == 2 else initial_state
        )
        set_direct_condition(
            pipeline.model,
            direct_controls,
            direct_initial_state,
            enabled=args.mode in {"full", "no_trajectory", "physics_only"},
        )
        if trajectory_enabled:
            point_track_maps = None
            if trajectory_requested:
                point_track_map = load_file(
                    str(trajectory_path), device="cpu"
                )["point_track_map"]
                condition_shape = input_contract.get("trajectory", {}).get(
                    "condition_shape"
                )
                if condition_shape is not None:
                    expected_condition_shape = tuple(
                        int(value) for value in condition_shape
                    )
                    if tuple(point_track_map.shape) != expected_condition_shape:
                        raise ValueError(
                            "trajectory condition shape violates the adapter "
                            f"input contract: {tuple(point_track_map.shape)} != "
                            f"{expected_condition_shape}"
                        )
                object_count = 1 if controls.ndim == 1 else int(controls.shape[0])
                point_track_maps = [
                    validate_point_track_object_slots(
                        point_track_map,
                        trajectory_representation,
                        object_count,
                    )
                ]
            set_trajectory_condition(
                pipeline.model,
                point_track_maps or [],
                enabled=trajectory_requested,
            )
        condition_encoder.to(device).eval()
        scene = load_scene_condition(
            scene_path,
            device,
            target_size_px=(args.width, args.height),
        )
        with torch.inference_mode():
            condition_tokens = condition_encoder(
                scene, controls, dynamics, initial_state
            ).detach()
            token_mode = condition_token_mode(args.mode)
            condition_tokens = apply_condition_mode(
                condition_tokens, scene_token_count, token_mode
            )
        pipeline.text_encoder = ConditionedTextEncoder(
            pipeline.text_encoder, condition_tokens, max_tokens=pipeline.model.text_len
        )

    with Image.open(first_frame_path) as source_first_frame:
        source_rgb = np.asarray(source_first_frame.convert("RGB"))[None]
    first_frame = Image.fromarray(
        cover_center_crop_frames(source_rgb, args.width, args.height)[0],
        mode="RGB",
    )
    video = pipeline.generate(
        prompt,
        img=first_frame,
        max_area=args.width * args.height,
        frame_num=args.frames,
        shift=args.flow_shift,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=True,
    )
    save_video(
        tensor=video[None],
        save_file=str(output),
        fps=config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    capture = cv2.VideoCapture(str(output))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if (
        frame_count != args.frames
        or width != args.width
        or height != args.height
    ):
        raise RuntimeError(
            "saved inference video does not match the exact training resolution"
        )
    if controls.ndim == 1:
        controls_report = {
            "mass_kg": float(controls[0].cpu()),
            "contact_friction": float(controls[1].cpu()),
            "restitution": float(controls[2].cpu()),
        }
        initial_state_report = {
            "linear_velocity_camera_m_s": [
                float(value) for value in initial_state[:3].cpu()
            ],
            "angular_velocity_camera_rad_s": [
                float(value) for value in initial_state[3:6].cpu()
            ],
            "gravity_camera_m_s2": [
                float(value) for value in initial_state[6:].cpu()
            ],
        }
    else:
        controls_report = {
            "object_count": int(controls.shape[0]),
            "objects": [
                {
                    "mass_kg": float(value[0].cpu()),
                    "contact_friction": float(value[1].cpu()),
                    "restitution": float(value[2].cpu()),
                }
                for value in controls
            ],
        }
        initial_state_report = {
            "object_count": int(initial_state.shape[0]),
            "objects": [
                {
                    "linear_velocity_camera_m_s": [
                        float(value) for value in state[:3].cpu()
                    ],
                    "angular_velocity_camera_rad_s": [
                        float(value) for value in state[3:6].cpu()
                    ],
                    "gravity_camera_m_s2": [
                        float(value) for value in state[6:].cpu()
                    ],
                }
                for state in initial_state
            ],
        }
    report = {
        "schema": "phycontext.conditioned_inference.v3",
        "sample_id": sample_id,
        "input_source": input_source,
        "first_frame": report_path(first_frame_path, root),
        "adapter": report_path(adapter_root, root),
        "adapter_sha256": sha256(adapter_root / "adapter.safetensors"),
        "input_contract": input_contract.get("schema"),
        "trajectory_protocol": input_contract.get("trajectory", {}).get("protocol"),
        "mode": args.mode,
        "trajectory_conditioning": {
            "available": trajectory_enabled,
            "enabled": trajectory_enabled
            and args.mode in {"full", "trajectory_only"},
            "path": (
                report_path(trajectory_path, root)
                if trajectory_path is not None
                else None
            ),
            "sample_id": trajectory_sample_id,
            "source": (
                "external_current_scene_simulation_cache"
                if external
                else "sample_bound_simulation_cache"
            ),
            "input_record": adapter_metadata.get(
                "trajectory_conditioning", {}
            ).get("input_record", "target"),
            "representation": adapter_metadata.get(
                "trajectory_conditioning", {}
            ).get("representation"),
        },
        "scene": report_path(scene_path, root),
        "controls": controls_report,
        "initial_state": initial_state_report,
        "frames": frame_count,
        "width": width,
        "height": height,
        "spatial_preprocess": "cover_then_center_crop",
        "sampling_steps": args.sampling_steps,
        "seed": args.seed,
        "video": report_path(output, root),
        "video_sha256": sha256(output),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, **report}, indent=2))


if __name__ == "__main__":
    main()
