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
TRACK_CORRESPONDENCE_ARCHITECTURE = "track4gen_swept_latent_v2"
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
    schema = metadata.get("schema")
    if schema not in {
        "phycontext.wan_condition_adapter.v2",
        "phycontext.wan_condition_adapter.v3",
        "phycontext.wan_condition_adapter.v4",
        "phycontext.wan_condition_adapter.v5",
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
    track_correspondence_keys = sorted(
        key for key in tensors if key.startswith("track_correspondence.")
    )
    cross_lora_keys = [key for key in lora_keys if ".cross_attn." in key]
    self_lora_keys = [key for key in lora_keys if ".self_attn." in key]
    expected_modules = int(metadata["lora"]["module_count"])
    if len(cross_lora_keys) != expected_modules * 2:
        errors.append("LoRA tensor count does not match module count")
    self_lora_config = metadata.get("self_attention_lora", {})
    if schema == "phycontext.wan_condition_adapter.v5":
        expected_self_modules = int(self_lora_config.get("module_count", 0))
        if not self_lora_config.get("enabled", False) or expected_self_modules <= 0:
            errors.append("v5 self-attention LoRA is disabled or invalid")
        if len(self_lora_keys) != expected_self_modules * 2:
            errors.append("self-attention LoRA tensor count does not match metadata")
        if self_lora_config.get("target") != "wan.blocks.*.self_attn.{q,k,v,o}":
            errors.append("self-attention LoRA target is invalid")
    elif self_lora_keys or self_lora_config.get("enabled", False):
        errors.append("legacy adapter unexpectedly contains self-attention LoRA")
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
    trajectory_histories = {}
    decoded_temporal_config = metadata.get("decoded_temporal_supervision")
    decoded_temporal_report_components = {}
    track_config = metadata.get("track_correspondence", {})
    track_report = {"enabled": False, "tensor_count": len(track_correspondence_keys)}
    if schema in {
        "phycontext.wan_condition_adapter.v2",
        "phycontext.wan_condition_adapter.v3",
    }:
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
                if (
                    weight is None
                    or not math.isfinite(float(weight))
                    or float(weight) < 0
                ):
                    errors.append(f"trajectory {name} weight is invalid")
                else:
                    parsed_weights.append(float(weight))
                history = trajectory_histories[name]
                if len(history) != int(metadata["steps"]):
                    errors.append(f"trajectory {name} history length mismatch")
                elif not all(
                    math.isfinite(value) and value >= 0 for value in history
                ):
                    errors.append(f"trajectory {name} history is invalid")
            if len(parsed_weights) == len(TRAJECTORY_COMPONENTS) and not any(
                parsed_weights
            ):
                errors.append("trajectory weights are all zero")
    else:
        trajectory_histories = {
            "center": [
                float(value)
                for value in metadata.get("trajectory_center_losses", [])
            ]
        }
        if trajectory_config.get("type") != "clean_latent_object_center_guard":
            errors.append("v4 trajectory supervision must be the center guard")
        if bool(trajectory_config.get("provided_as_model_input")) != (
            trajectory_conditioning_enabled
        ):
            errors.append(
                "v4 trajectory supervision/input declarations are inconsistent"
            )
        center_weight = trajectory_config.get("weight")
        if (
            not trajectory_config.get("enabled")
            or center_weight is None
            or not math.isfinite(float(center_weight))
            or float(center_weight) <= 0
        ):
            errors.append("v4 trajectory center weight is invalid")
        center_history = trajectory_histories["center"]
        if len(center_history) != int(metadata["steps"]):
            errors.append("trajectory center history length mismatch")
        elif not all(
            math.isfinite(value) and value >= 0 for value in center_history
        ):
            errors.append("trajectory center history is invalid")
        removed_fields = (
            "trajectory_distribution_losses",
            "trajectory_velocity_losses",
            "flow_temporal_losses",
            "flow_object_diagnostics",
            "flow_background_diagnostics",
            "object_temporal_losses",
            "background_temporal_losses",
            "decoded_temporal_supervision",
        )
        for field in removed_fields:
            if field in metadata:
                errors.append(f"v4 adapter contains removed objective metadata: {field}")
        if not track_config.get("enabled"):
            errors.append("v4 track correspondence is disabled")
        expected_track_architecture = (
            TRACK_CORRESPONDENCE_ARCHITECTURE
            if schema == "phycontext.wan_condition_adapter.v5"
            else "track4gen_swept_latent_v1"
        )
        if track_config.get("architecture") != expected_track_architecture:
            errors.append("track correspondence architecture is invalid")
        expected_track_semantics = {
            "query": "clean_first_frame_visible_material_point",
            "target": "visible_same_point_swept_over_each_four_rgb_frame_latent_window",
            "similarity": "global_target_frame_cosine_cost_volume",
            "visibility": "cached_global_dynamic_static_z_buffer",
            "feedback": "zero_initialized_linear_of_stop_gradient_refined_features",
        }
        expected_track_semantics.update(
            (
                {
                    "objective": "soft_argmax_feature_coordinate_huber",
                    "sampling": "equal_foreground_background_then_equal_slow_medium_fast_bins",
                    "conditioning_during_correspondence": "text_only_no_scene_physics_or_trajectory",
                }
                if schema == "phycontext.wan_condition_adapter.v5"
                else {
                    "objective": "soft_target_cross_entropy_plus_expected_coordinate_huber",
                    "sampling": "equal_slow_medium_fast_feature_displacement_bins",
                }
            )
        )
        for field, expected in expected_track_semantics.items():
            if track_config.get(field) != expected:
                errors.append(f"track correspondence {field} semantics are invalid")
        if (
            track_config.get("feature_objective_normalization")
            != "equal_video_weight_then_ddp_rank_mean"
        ):
            errors.append("track correspondence per-video normalization is invalid")
        expected_shortcut_mitigation = (
            "trajectory_branch_fully_disabled"
            if schema == "phycontext.wan_condition_adapter.v5"
            else "training_only_rgb_identity_dropout_with_occupancy_preserved"
        )
        if track_config.get("identity_shortcut_mitigation") != expected_shortcut_mitigation:
            errors.append("track correspondence identity-shortcut mitigation is invalid")
        identity_dropout = track_config.get("identity_dropout_probability")
        if (
            identity_dropout is None
            or not math.isfinite(float(identity_dropout))
            or not 0.0 < float(identity_dropout) < 1.0
        ):
            errors.append("track correspondence identity dropout is invalid")
        positive_track_fields = (
            "feature_dim",
            "refiner_blocks",
            "loss_weight",
            "maximum_pairs_per_video",
            "temperature",
            "gaussian_sigma_feature_cells",
        )
        for field in positive_track_fields:
            value = track_config.get(field)
            if (
                value is None
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                errors.append(f"track correspondence {field} is invalid")
        block_index = track_config.get("block_index")
        if block_index is None or int(block_index) < 0:
            errors.append("track correspondence block index is invalid")
        if schema == "phycontext.wan_condition_adapter.v5":
            if int(track_config.get("feature_grid_upsample_factor", 0)) != 2:
                errors.append("v5 track feature upsample factor is invalid")
            sigma = track_config.get("correspondence_sigma")
            if sigma is None or not 0 < float(sigma) < 1:
                errors.append("v5 correspondence sigma is invalid")
        else:
            coordinate_weight = track_config.get("coordinate_weight")
            if (
                coordinate_weight is None
                or not math.isfinite(float(coordinate_weight))
                or float(coordinate_weight) < 0
            ):
                errors.append("track correspondence coordinate weight is invalid")
        if not track_correspondence_keys:
            errors.append("track correspondence state is missing")
        feature_dim = int(track_config.get("feature_dim", 0))
        expected_track_shapes = {
            "track_correspondence.input_norm.weight": (3072,),
            "track_correspondence.input_norm.bias": (3072,),
            "track_correspondence.input_projection.weight": (feature_dim, 3072),
            "track_correspondence.input_projection.bias": (feature_dim,),
            "track_correspondence.feedback.weight": (3072, feature_dim),
        }
        for key, expected_shape in expected_track_shapes.items():
            tensor = tensors.get(key)
            if tensor is None or tuple(tensor.shape) != expected_shape:
                errors.append(f"track correspondence tensor shape mismatch: {key}")
        refiner_prefixes = {
            key.split(".")[2]
            for key in track_correspondence_keys
            if key.startswith("track_correspondence.refiner.")
            and len(key.split(".")) > 3
        }
        if len(refiner_prefixes) != int(track_config.get("refiner_blocks", 0)):
            errors.append("track correspondence refiner block count mismatch")
        track_losses = [
            float(value)
            for value in metadata.get("track_correspondence_losses", [])
        ]
        track_displacements = [
            float(value)
            for value in metadata.get(
                "track_correspondence_mean_displacement_tokens", []
            )
        ]
        track_kl = [
            float(value) for value in metadata.get("track_correspondence_kl", [])
        ]
        track_epe = [
            float(value)
            for value in metadata.get("track_correspondence_epe_tokens", [])
        ]
        track_pck = [
            float(value)
            for value in metadata.get("track_correspondence_pck_1", [])
        ]
        track_fast_epe = [
            float(value)
            for value in metadata.get("track_correspondence_fast_epe_tokens", [])
        ]
        track_fast_pck = [
            float(value)
            for value in metadata.get("track_correspondence_fast_pck_1", [])
        ]
        track_foreground_pck = [
            float(value)
            for value in metadata.get(
                "track_correspondence_foreground_pck_1", []
            )
        ]
        track_background_pck = [
            float(value)
            for value in metadata.get(
                "track_correspondence_background_pck_1", []
            )
        ]
        for label, values in (
            ("loss", track_losses),
            ("displacement", track_displacements),
            ("kl", track_kl),
            ("epe", track_epe),
            ("pck@1", track_pck),
            ("fast epe", track_fast_epe),
            ("fast pck@1", track_fast_pck),
            *((
                ("foreground pck@1", track_foreground_pck),
                ("background pck@1", track_background_pck),
            ) if schema == "phycontext.wan_condition_adapter.v5" else ()),
        ):
            if len(values) != int(metadata["steps"]):
                errors.append(f"track correspondence {label} history length mismatch")
            elif not all(math.isfinite(value) and value >= 0 for value in values):
                errors.append(f"track correspondence {label} history is invalid")
        if (track_pck and not all(value <= 1.0 for value in track_pck)) or (
            track_fast_pck
            and not all(value <= 1.0 for value in track_fast_pck)
        ):
            errors.append("track correspondence pck@1 history is outside [0, 1]")
        dropout_history = [
            float(value)
            for value in metadata.get("trajectory_identity_dropout_fractions", [])
        ]
        if len(dropout_history) != int(metadata["steps"]) or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in dropout_history
        ):
            errors.append("trajectory identity dropout history is invalid")
        for step in metadata.get("history", []):
            for key in (
                "track_correspondence_pairs",
                "track_correspondence_fast_pairs",
                *(
                    (
                        "track_correspondence_foreground_pairs",
                        "track_correspondence_background_pairs",
                    )
                    if schema == "phycontext.wan_condition_adapter.v5"
                    else ()
                ),
            ):
                value = step.get(key)
                if (
                    value is None
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    errors.append(f"{key} history is invalid")
                    break
        track_report = {
            "enabled": bool(track_config.get("enabled")),
            "architecture": track_config.get("architecture"),
            "block_index": block_index,
            "feature_dim": track_config.get("feature_dim"),
            "tensor_count": len(track_correspondence_keys),
            "final_loss": track_losses[-1] if track_losses else None,
            "mean_displacement_tokens": (
                track_displacements[-1] if track_displacements else None
            ),
            "final_kl": track_kl[-1] if track_kl else None,
            "final_epe_tokens": track_epe[-1] if track_epe else None,
            "final_pck_1": track_pck[-1] if track_pck else None,
        }
        optimizer = metadata.get("optimizer", {})
        feature_lr = optimizer.get("track_correspondence_learning_rate")
        feedback_lr = optimizer.get("track_correspondence_feedback_learning_rate")
        if (
            feature_lr is None
            or feedback_lr is None
            or not 0.0 < float(feedback_lr) < float(feature_lr)
        ):
            errors.append("track feedback learning rate is not safely separated")
        response = metadata.get("response_supervision", {})
        response_sigma = response.get("sigma")
        if (
            response.get("protocol")
            != "isolated_structured_physics_counterfactual_v1"
            or response_sigma is None
            or not math.isfinite(float(response_sigma))
            or float(response_sigma) != 1.0
        ):
            errors.append("physics response supervision is not causally isolated")
    if decoded_temporal_config is not None and schema != "phycontext.wan_condition_adapter.v4":
        decoded_temporal_enabled = bool(
            decoded_temporal_config.get("enabled")
        )
        decode_config = decoded_temporal_config.get("decode", {})
        if (
            int(decode_config.get("long_edge", 0)) < 16
            or int(decode_config.get("long_edge", 0)) % 16
            or decode_config.get("spatial_resize")
            != "aspect_preserving_integer_vae_latent_grid"
        ):
            errors.append("decoded temporal spatial contract is invalid")
        flow_objective = decoded_temporal_config.get("objective")
        if isinstance(flow_objective, dict):
            expected_window = (
                "two_adjacent_generated_latents_eight_source_frames_with_"
                "condition_frame_prefix_for_first_window"
            )
            if (
                flow_objective.get("type")
                != "flow_aligned_rgb_temporal_residual"
                or decode_config.get("time_mapping")
                != "latent_k_to_source_frames_4k_minus_3_through_4k"
                or decode_config.get("window") != expected_window
                or not decode_config.get("cross_chunk_transitions_supervised")
                or not decode_config.get(
                    "condition_to_first_generated_transition_supervised"
                )
            ):
                errors.append("decoded flow time/window contract is invalid")
            weight = flow_objective.get("weight")
            beta = flow_objective.get("beta")
            foreground_share = flow_objective.get("foreground_share")
            boundary_share = flow_objective.get(
                "condition_boundary_window_share"
            )
            if (
                weight is None
                or not math.isfinite(float(weight))
                or float(weight) < 0
                or beta is None
                or not math.isfinite(float(beta))
                or float(beta) <= 0
                or foreground_share is None
                or not math.isfinite(float(foreground_share))
                or not 0 < float(foreground_share) < 1
                or boundary_share != 0.25
            ):
                errors.append("decoded flow weight/beta/share is invalid")
            flow_history = [
                float(value)
                for value in metadata.get("flow_temporal_losses", [])
            ]
            object_diagnostics = [
                float(value)
                for value in metadata.get("flow_object_diagnostics", [])
            ]
            background_diagnostics = [
                float(value)
                for value in metadata.get("flow_background_diagnostics", [])
            ]
            for label, history in (
                ("flow", flow_history),
                ("object diagnostic", object_diagnostics),
                ("background diagnostic", background_diagnostics),
            ):
                if len(history) != int(metadata["steps"]):
                    errors.append(f"decoded {label} history length mismatch")
                elif not all(
                    math.isfinite(value) and value >= 0 for value in history
                ):
                    errors.append(f"decoded {label} history is invalid")
            for population in ("object_population", "background_population"):
                if not isinstance(decoded_temporal_config.get(population), dict):
                    errors.append(f"decoded flow {population} is missing")
            for step in metadata.get("history", []):
                for key in ("flow_object_pairs", "flow_background_pixels"):
                    value = step.get(key)
                    if (
                        value is None
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        errors.append(f"decoded flow {key} history is invalid")
                        break
            if weight is not None and math.isfinite(float(weight)):
                if decoded_temporal_enabled != (float(weight) > 0):
                    errors.append(
                        "decoded flow enabled flag disagrees with its weight"
                    )
            decoded_temporal_report_components["flow"] = {
                "weight": weight,
                "foreground_share": foreground_share,
                "final_loss": flow_history[-1] if flow_history else None,
                "max_loss": max(flow_history) if flow_history else None,
                "object_final": (
                    object_diagnostics[-1] if object_diagnostics else None
                ),
                "background_final": (
                    background_diagnostics[-1]
                    if background_diagnostics
                    else None
                ),
            }
        else:
            if (
                decode_config.get("time_mapping")
                != "latent_k_to_source_frames_4k_minus_3_through_4k"
                or decode_config.get("window")
                != "two_adjacent_latents_eight_source_frames"
                or not decode_config.get("cross_chunk_transitions_supervised")
                or not decode_config.get("condition_frame_excluded")
            ):
                errors.append("decoded temporal time/window contract is invalid")
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
                    errors.append(
                        f"decoded temporal {name} weight/beta is invalid"
                    )
                else:
                    decoded_temporal_weights.append(float(weight))
                history = [
                    float(value)
                    for value in metadata.get(f"{name}_temporal_losses", [])
                ]
                decoded_temporal_report_components[name] = {
                    "weight": weight,
                    "final_loss": history[-1] if history else None,
                    "max_loss": max(history) if history else None,
                }
                if len(history) != int(metadata["steps"]):
                    errors.append(
                        f"decoded temporal {name} history length mismatch"
                    )
                elif not all(
                    math.isfinite(value) and value >= 0 for value in history
                ):
                    errors.append(f"decoded temporal {name} history is invalid")
            for step in metadata.get("history", []):
                for key in (
                    "object_temporal_pairs",
                    "background_temporal_pixels",
                ):
                    value = step.get(key)
                    if (
                        value is None
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        errors.append(f"decoded temporal {key} history is invalid")
                        break
            if len(decoded_temporal_weights) == len(DECODED_TEMPORAL_COMPONENTS):
                if decoded_temporal_enabled != any(decoded_temporal_weights):
                    errors.append(
                        "decoded temporal enabled flag disagrees with weights"
                    )
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
        "track_correspondence": track_report,
        "steps": metadata["steps"],
        "training_input_count": int(metadata["sample_count"]),
        "losses": losses,
        "trajectory_supervision": {
            "enabled": bool(trajectory_config.get("enabled")),
            "provided_as_model_input": trajectory_config.get(
                "provided_as_model_input"
            ),
            "type": trajectory_config.get("type"),
            "weights": (
                trajectory_weights
                if schema
                not in {
                    "phycontext.wan_condition_adapter.v4",
                    "phycontext.wan_condition_adapter.v5",
                }
                else {"center": trajectory_config.get("weight")}
            ),
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
            "components": decoded_temporal_report_components,
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
