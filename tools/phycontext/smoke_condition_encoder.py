#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from conditioning_model import (
    PhyContextConditionEncoder,
    compose_wan_context,
    controls_from_record,
    dynamics_from_record,
    initial_state_from_record,
    load_scene_condition,
)
from schema import iter_jsonl, validate_training_record
from project_defaults import DATASET_MANIFEST, DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scene/physics encoder on one real manifest sample"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        required=DATASET_ROOT is None,
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scene-tokens", type=int, default=128)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--wan-text-length", type=int, default=512)
    return parser.parse_args()


def select_record(manifest: Path, sample_index: int) -> dict:
    if sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    for index, record in enumerate(iter_jsonl(manifest)):
        if index == sample_index:
            return record
    raise IndexError(f"manifest has no sample at index {sample_index}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    manifest = dataset_root / DATASET_MANIFEST
    record = select_record(manifest, args.sample_index)
    validate_training_record(record, dataset_root, check_files=True)

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")

    scene_path = dataset_root / record["conditioning"]["scene"]
    scene = load_scene_condition(scene_path, device=device)
    controls = controls_from_record(record, device=device)
    dynamics = dynamics_from_record(record, device=device)
    initial_state = initial_state_from_record(record, device=device)
    encoder = PhyContextConditionEncoder(
        scene_token_count=args.scene_tokens,
    ).to(device)
    encoder.train()

    condition_tokens = encoder(scene, controls, dynamics, initial_state)
    text_context = torch.randn(
        args.text_tokens,
        condition_tokens.shape[-1],
        dtype=condition_tokens.dtype,
        device=device,
    )
    wan_context = compose_wan_context(
        text_context,
        condition_tokens,
        max_tokens=args.wan_text_length,
    )
    loss = condition_tokens.square().mean()
    loss.backward()
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    finite_gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
    )

    report = {
        "passed": bool(finite_gradients),
        "sample_id": record["sample_id"],
        "base_scene_id": record["base_scene_id"],
        "device": str(device),
        "scene_path": record["conditioning"]["scene"],
        "scene_point_count": int(
            scene["object_xyz_camera_m"].shape[0]
            + scene["environment_xyz_camera_m"].shape[0]
        ),
        "controls": {
            "mass_kg": float(controls[0].item()),
            "contact_friction": float(controls[1].item()),
            "contact_restitution": float(controls[2].item()),
        },
        "condition_shape": list(condition_tokens.shape),
        "scene_token_count": args.scene_tokens,
        "control_token_count": 3,
        "base_dynamics_token_count": 10,
        "state_token_count": 3,
        "physics_token_count": 16,
        "wan_context_shapes": [list(item.shape) for item in wan_context],
        "wan_text_length_limit": args.wan_text_length,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "finite_gradients": bool(finite_gradients),
    }
    print(json.dumps(report, indent=2))
    if not finite_gradients:
        raise RuntimeError("condition encoder produced missing or non-finite gradients")


if __name__ == "__main__":
    main()
