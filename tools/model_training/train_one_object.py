#!/usr/bin/env python3
"""Validate a published dataset/cache pair and launch formal Wan training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHYCONTEXT_TOOLS = PROJECT_ROOT / "tools" / "phycontext"
if str(PHYCONTEXT_TOOLS) not in sys.path:
    sys.path.insert(0, str(PHYCONTEXT_TOOLS))

from cache_contract import (  # noqa: E402
    CENTER_PIXEL_TRACK_CORRESPONDENCE_CACHE_SCHEMAS,
    CURRENT_CACHE_SCHEMA,
    resolve_cache_dataset_root,
    validate_cache_source_manifest,
)


DEFAULT_CONFIG = Path("configs/training/one_object.json")
FORBIDDEN_KEYS = {
    "sampling_matrix",
    "physics_workers",
    "render_workers",
    "render_samples",
    "sweep_config",
}


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "phycontext_training_run_v1":
        raise ValueError("unsupported training config")
    forbidden = sorted(_walk_keys(config) & FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"dataset-generation keys in training config: {', '.join(forbidden)}")
    return config


def _path(root: Path, value: str, env_name: str | None = None) -> Path:
    configured = os.environ.get(env_name, value) if env_name else value
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def preflight(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    dataset_root_value = os.environ.get("PHYCONTEXT_DATASET_ROOT") or config.get(
        "dataset_root"
    )
    if not dataset_root_value:
        raise ValueError(
            "PhysSweep root is required via PHYCONTEXT_DATASET_ROOT or dataset_root"
        )
    dataset_root = _path(root, str(dataset_root_value))
    dataset_manifest = dataset_root / config.get("dataset_manifest", "manifest.jsonl")
    cache_manifest = _path(root, config["cache_manifest"], "PHYCONTEXT_CACHE_MANIFEST")
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"published dataset manifest is missing: {dataset_manifest}")
    if cache_manifest.is_relative_to(dataset_root):
        raise ValueError("model cache must not live inside the published dataset")
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"Wan cache manifest is missing: {cache_manifest}")
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    expected_representation = config.get("training", {}).get(
        "trajectory_representation"
    )
    if expected_representation is not None:
        cache_representation = cache.get("point_track_preprocess", {}).get(
            "representation", "dense_point_tracks"
        )
        if cache_representation != expected_representation:
            raise ValueError(
                "Wan cache trajectory representation differs from training config"
            )
        if (
            cache.get("schema")
            not in CENTER_PIXEL_TRACK_CORRESPONDENCE_CACHE_SCHEMAS
        ):
            raise ValueError(
                "Track4Gen correspondence training requires the current cache "
                f"schema {CURRENT_CACHE_SCHEMA}"
            )
    cache_dataset_root = resolve_cache_dataset_root(root, cache)
    if cache_dataset_root != dataset_root:
        raise ValueError(
            f"Wan cache belongs to a different dataset root: {cache_dataset_root}"
        )
    source = validate_cache_source_manifest(root, cache)
    if source != dataset_manifest:
        raise ValueError(f"cache belongs to a different dataset: {source}")
    if cache.get("source_manifest_sha256") != _sha256(dataset_manifest):
        raise ValueError("Wan cache is stale for the selected dataset")
    return dataset_manifest, cache_manifest


def build_command(root: Path, config: dict[str, Any], cache_manifest: Path) -> list[str]:
    train = config["training"]
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(config['gpus'])}",
        "tools/phycontext/train_wan_formal.py",
        "--project-root", str(root),
        "--cache-manifest", str(cache_manifest),
        "--wan-repo", str(_path(root, config["wan_repo"], "PHYCONTEXT_WAN_REPO")),
        "--checkpoint", str(_path(root, config["checkpoint"], "PHYCONTEXT_WAN_CHECKPOINT")),
        "--output", str(_path(root, config["output_root"], "PHYCONTEXT_TRAINING_OUTPUT")),
    ]
    boolean_flags = {"trajectory_input": "--trajectory-input"}
    for key, flag in boolean_flags.items():
        command.append(flag if train.get(key, True) else f"--no-{flag[2:]}")
    for key, value in train.items():
        if key in boolean_flags:
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    dataset_manifest, cache_manifest = preflight(root, config)
    command = build_command(root, config, cache_manifest)
    print(f"dataset_manifest: {dataset_manifest}")
    print(f"cache_manifest: {cache_manifest}")
    print("command: " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
