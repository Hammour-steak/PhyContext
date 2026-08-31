#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = Path("outputs/training/formal/final")
TRAJECTORY_COMPONENTS = ("center", "distribution", "velocity")
DECODED_TEMPORAL_COMPONENTS = ("object", "background")
TRAJECTORY_REPRESENTATION_CONTRACTS = {
    "dense_point_tracks": {
        "input_channels": 18,
        "channels": [
            f"object_{slot}_{channel}"
            for slot in range(3)
            for channel in (
                "source_occupancy",
                "current_occupancy",
                "delta_x",
                "delta_y",
                "depth",
                "validity",
            )
        ],
    },
    "das_3d_tracks": {
        "input_channels": 12,
        "channels": [
            f"object_{slot}_{channel}"
            for slot in range(3)
            for channel in (
                "identity_r",
                "identity_g",
                "identity_b",
                "visibility",
            )
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PhyContext Wan adapter")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    args = parser.parse_args()
    root = args.project_root.resolve()
    adapter_root = (root / args.adapter).resolve()
    tensor_path = adapter_root / "adapter.safetensors"
    metadata_path = adapter_root / "adapter.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensor_path), device="cpu")
    errors = []
    if metadata.get("schema") not in {
        "phycontext.wan_condition_adapter.v2",
        "phycontext.wan_condition_adapter.v3",
    }:
        errors.append("adapter conditioning schema mismatch")
    lora_keys = sorted(key for key in tensors if key.startswith("wan_lora."))
    condition_keys = sorted(
        key for key in tensors if key.startswith("condition_encoder.")
    )
    direct_keys = sorted(
        key for key in tensors if key.startswith("direct_modulation.")
    )
    trajectory_conditioning_keys = sorted(
        key for key in tensors if key.startswith("trajectory_conditioning.")
    )
    expected_modules = int(metadata["lora"]["module_count"])
    if len(lora_keys) != expected_modules * 2:
        errors.append("LoRA tensor count does not match module count")
    if not condition_keys:
        errors.append("condition encoder state is missing")
    direct_enabled = bool(
        metadata.get("direct_modulation", {}).get("enabled", False)
    )
    if direct_enabled and not direct_keys:
        errors.append("direct modulation state is missing")
    if not direct_enabled and direct_keys:
        errors.append("unexpected direct modulation state")
    direct_config = metadata.get("direct_modulation", {})
    if direct_enabled and direct_config.get("architecture") != "per_layer_v2":
        errors.append("direct modulation architecture is unsupported")
    deprecated_direct_keys = [
        key
        for key in direct_keys
        if key.startswith("direct_modulation.calibrator.")
        or key.startswith("direct_modulation.control_warp.")
    ]
    if deprecated_direct_keys:
        errors.append("adapter contains deprecated direct modulation state")
    if any(
        key in direct_config
        for key in (
            "endpoint_preserving_calibration",
            "endpoint_preserving_control_warp",
        )
    ):
        errors.append("adapter contains deprecated direct modulation metadata")
    if "direct_endpoint_anchor" in metadata or "endpoint_anchor_losses" in metadata:
        errors.append("adapter contains deprecated endpoint anchor metadata")
    trajectory_conditioning_config = metadata.get(
        "trajectory_conditioning", {}
    )
    trajectory_conditioning_enabled = bool(
        trajectory_conditioning_config.get("enabled", False)
    )
    if trajectory_conditioning_enabled:
        if not trajectory_conditioning_keys:
            errors.append("trajectory conditioning state is missing")
        representation = trajectory_conditioning_config.get("representation")
        contract = TRAJECTORY_REPRESENTATION_CONTRACTS.get(representation)
        if contract is None:
            errors.append("trajectory conditioning representation is invalid")
        if trajectory_conditioning_config.get("future_appearance_or_silhouette"):
            errors.append("trajectory conditioning leaks future appearance")
        if contract is not None:
            architecture = trajectory_conditioning_config.get("architecture")
            supported_architectures = (
                {"framewise_patch", "full_frame_causal_patch_v2"}
                if representation == "das_3d_tracks"
                else {"framewise_patch"}
            )
            if architecture not in supported_architectures:
                errors.append("trajectory conditioning architecture is unsupported")
            expected_channels = int(contract["input_channels"])
            if int(trajectory_conditioning_config.get("input_channels", 0)) != (
                expected_channels
            ):
                errors.append(
                    "trajectory conditioning input channels do not match representation"
                )
            metadata_channels = trajectory_conditioning_config.get("channels")
            if metadata_channels is not None and metadata_channels != contract["channels"]:
                errors.append(
                    "trajectory conditioning channel semantics do not match representation"
                )
            projection_key = "trajectory_conditioning.patch_projection.weight"
            projection = tensors.get(projection_key)
            if projection is None:
                errors.append("trajectory conditioning patch projection is missing")
            elif projection.ndim != 5:
                errors.append("trajectory conditioning patch projection rank is invalid")
            else:
                if int(projection.shape[1]) != expected_channels:
                    errors.append(
                        "trajectory conditioning tensor channels do not match representation"
                    )
                if int(projection.shape[0]) != int(
                    trajectory_conditioning_config.get("rank", 0)
                ):
                    errors.append("trajectory conditioning tensor rank is invalid")
                patch_size = tuple(
                    int(value)
                    for value in trajectory_conditioning_config.get("patch_size", [])
                )
                if patch_size and tuple(projection.shape[2:]) != patch_size:
                    errors.append("trajectory conditioning patch size is invalid")
            temporal_key = "trajectory_conditioning.temporal_projection.weight"
            temporal_projection = tensors.get(temporal_key)
            if architecture == "full_frame_causal_patch_v2":
                expected_temporal_shape = (
                    expected_channels,
                    expected_channels,
                    4,
                    1,
                    1,
                )
                if temporal_projection is None or tuple(
                    temporal_projection.shape
                ) != expected_temporal_shape:
                    errors.append(
                        "trajectory temporal projection is missing or has invalid shape"
                    )
            elif temporal_projection is not None:
                errors.append("legacy trajectory adapter has unexpected temporal state")
        if int(trajectory_conditioning_config.get("rank", 0)) <= 0:
            errors.append("trajectory conditioning rank is invalid")
    elif trajectory_conditioning_keys:
        errors.append("unexpected trajectory conditioning state")
    for key, tensor in tensors.items():
        if not torch.isfinite(tensor.float()).all():
            errors.append(f"non-finite adapter tensor: {key}")
    cache_manifest = Path(metadata["cache_manifest"])
    if sha256(cache_manifest) != metadata["cache_manifest_sha256"]:
        errors.append("cache manifest hash mismatch")
    model_index = (
        Path(metadata["base_model"])
        / "diffusion_pytorch_model.safetensors.index.json"
    )
    if sha256(model_index) != metadata["base_model_index_sha256"]:
        errors.append("base model index hash mismatch")
    losses = [float(value) for value in metadata["losses"]]
    if not losses or not all(math.isfinite(value) for value in losses):
        errors.append("loss history is empty or non-finite")
    trajectory_config = metadata.get("trajectory_supervision", {})
    trajectory_weights = trajectory_config.get("weights", {})
    trajectory_histories = {
        name: [
            float(value)
            for value in metadata.get(f"trajectory_{name}_losses", [])
        ]
        for name in TRAJECTORY_COMPONENTS
    }
    if trajectory_config.get("enabled"):
        if bool(trajectory_config.get("provided_as_model_input")) != (
            trajectory_conditioning_enabled
        ):
            errors.append(
                "trajectory supervision/input declarations are inconsistent"
            )
        if not metadata.get("motion_supervision", {}).get("enabled"):
            errors.append("trajectory supervision requires motion masks")
        if trajectory_config.get("type") != "clean_latent_correspondence_motion":
            errors.append("trajectory supervision type is invalid")
        parsed_weights = []
        for name in TRAJECTORY_COMPONENTS:
            weight = trajectory_weights.get(name)
            if weight is None or not math.isfinite(float(weight)) or float(weight) < 0:
                errors.append(f"trajectory {name} weight is invalid")
            else:
                parsed_weights.append(float(weight))
            history = trajectory_histories[name]
            if len(history) != int(metadata["steps"]):
                errors.append(f"trajectory {name} history length mismatch")
            elif not all(math.isfinite(value) and value >= 0 for value in history):
                errors.append(f"trajectory {name} history is invalid")
        if len(parsed_weights) == len(TRAJECTORY_COMPONENTS) and not any(
            parsed_weights
        ):
            errors.append("trajectory weights are all zero")
    decoded_temporal_config = metadata.get("decoded_temporal_supervision")
    decoded_temporal_histories = {
        name: [
            float(value)
            for value in metadata.get(f"{name}_temporal_losses", [])
        ]
        for name in DECODED_TEMPORAL_COMPONENTS
    }
    if decoded_temporal_config is not None:
        decoded_temporal_enabled = bool(
            decoded_temporal_config.get("enabled")
        )
        decode_config = decoded_temporal_config.get("decode", {})
        if (
            decode_config.get("time_mapping")
            != "latent_k_to_source_frames_4k_minus_3_through_4k"
            or decode_config.get("window")
            != "two_adjacent_latents_eight_source_frames"
            or not decode_config.get("cross_chunk_transitions_supervised")
            or not decode_config.get("condition_frame_excluded")
        ):
            errors.append("decoded temporal time/window contract is invalid")
        if (
            int(decode_config.get("long_edge", 0)) < 16
            or int(decode_config.get("long_edge", 0)) % 16
            or decode_config.get("spatial_resize")
            != "aspect_preserving_integer_vae_latent_grid"
        ):
            errors.append("decoded temporal spatial contract is invalid")
        decoded_temporal_weights = []
        for name in DECODED_TEMPORAL_COMPONENTS:
            component = decoded_temporal_config.get(name, {})
            weight = component.get("weight")
            beta = component.get("beta")
            if (
                weight is None
                or not math.isfinite(float(weight))
                or float(weight) < 0
                or beta is None
                or not math.isfinite(float(beta))
                or float(beta) <= 0
            ):
                errors.append(f"decoded temporal {name} weight/beta is invalid")
            else:
                decoded_temporal_weights.append(float(weight))
            history = decoded_temporal_histories[name]
            if len(history) != int(metadata["steps"]):
                errors.append(f"decoded temporal {name} history length mismatch")
            elif not all(math.isfinite(value) and value >= 0 for value in history):
                errors.append(f"decoded temporal {name} history is invalid")
        for step in metadata.get("history", []):
            for key in ("object_temporal_pairs", "background_temporal_pixels"):
                value = step.get(key)
                if value is None or not math.isfinite(float(value)) or float(value) < 0:
                    errors.append(f"decoded temporal {key} history is invalid")
                    break
        if len(decoded_temporal_weights) == len(DECODED_TEMPORAL_COMPONENTS):
            if decoded_temporal_enabled != any(decoded_temporal_weights):
                errors.append("decoded temporal enabled flag disagrees with weights")
    report = {
        "passed": not errors,
        "adapter": str(adapter_root),
        "adapter_sha256": sha256(tensor_path),
        "adapter_size_mib": round(tensor_path.stat().st_size / 1024**2, 3),
        "lora_module_count": expected_modules,
        "lora_tensor_count": len(lora_keys),
        "condition_tensor_count": len(condition_keys),
        "direct_modulation_tensor_count": len(direct_keys),
        "trajectory_conditioning": {
            "enabled": trajectory_conditioning_enabled,
            "tensor_count": len(trajectory_conditioning_keys),
            "representation": trajectory_conditioning_config.get(
                "representation"
            ),
            "rank": trajectory_conditioning_config.get("rank"),
        },
        "steps": metadata["steps"],
        "training_input_count": int(metadata["sample_count"]),
        "losses": losses,
        "trajectory_supervision": {
            "enabled": bool(trajectory_config.get("enabled")),
            "provided_as_model_input": trajectory_config.get(
                "provided_as_model_input"
            ),
            "type": trajectory_config.get("type"),
            "weights": trajectory_weights,
            "components": {
                name: {
                    "final_loss": history[-1] if history else None,
                    "max_loss": max(history) if history else None,
                }
                for name, history in trajectory_histories.items()
            },
        },
        "decoded_temporal_supervision": {
            "enabled": bool(
                decoded_temporal_config
                and decoded_temporal_config.get("enabled")
            ),
            "components": {
                name: {
                    "weight": (
                        decoded_temporal_config.get(name, {}).get("weight")
                        if decoded_temporal_config
                        else None
                    ),
                    "final_loss": history[-1] if history else None,
                    "max_loss": max(history) if history else None,
                }
                for name, history in decoded_temporal_histories.items()
            },
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
