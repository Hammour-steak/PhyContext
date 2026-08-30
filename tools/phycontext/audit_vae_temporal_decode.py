#!/usr/bin/env python3
"""Compare Wan VAE full-sequence decoding with isolated latent-slice decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from cache_contract import resolve_cache_artifact_root, resolve_cache_dataset_root
from cache_wan_inputs import load_canonical_first_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 12, 24])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cache_manifest = args.cache_manifest.resolve()
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    artifact_root = resolve_cache_artifact_root(project_root, cache)
    dataset_root = resolve_cache_dataset_root(project_root, cache)
    records = cache["records"]
    item = next(
        (
            record
            for record in records
            if args.sample_id is None or record["sample_id"] == args.sample_id
        ),
        None,
    )
    if item is None:
        raise ValueError(f"sample is not present in cache: {args.sample_id}")
    latent_path = artifact_root / item["latent"]["path"]
    latent = load_file(str(latent_path), device="cpu")["latent"]
    if latent.ndim != 4:
        raise ValueError(f"latent must have shape C x F x H x W: {latent.shape}")
    if any(index < 0 or index >= latent.shape[1] for index in args.indices):
        raise ValueError("slice index is outside the latent temporal axis")

    sys.path.insert(0, str(args.wan_repo.resolve()))
    from wan.modules.vae2_2 import Wan2_2_VAE

    device = torch.device(args.device)
    vae = Wan2_2_VAE(
        vae_pth=str(args.checkpoint.resolve() / "Wan2.2_VAE.pth"),
        dtype=torch.bfloat16,
        device=device,
    )
    vae.model.to(dtype=torch.bfloat16)
    latent = latent.to(device)
    with torch.inference_mode():
        full = vae.decode([latent])[0].float().cpu()
        if full.ndim != 4 or not torch.isfinite(full).all():
            raise ValueError("full VAE decode must be a finite C x F x H x W tensor")
        preprocess = cache["preprocess"]
        canonical = load_canonical_first_frame(
            dataset_root / item["record"]["conditioning"]["first_frame"],
            int(preprocess["width"]),
            int(preprocess["height"]),
        ).float()
        if full[:, :1].shape != canonical.shape:
            raise ValueError("decoded and canonical first-frame shapes differ")
        condition_difference = (full[:, :1] - canonical).abs()
        comparisons = []
        for index in args.indices:
            isolated = vae.decode([latent[:, index : index + 1]])[0].float().cpu()
            start = 0 if index == 0 else 1 + 4 * (index - 1)
            stop = start + isolated.shape[1]
            entry = {
                "latent_index": int(index),
                "full_shape": list(full.shape),
                "isolated_shape": list(isolated.shape),
                "full_frame_start": int(start),
            }
            if stop <= full.shape[1] and isolated.shape[1] == full[:, start:stop].shape[1]:
                difference = (isolated - full[:, start:stop]).abs()
                entry.update(
                    {
                        "mean_abs_difference": float(difference.mean()),
                        "p95_abs_difference": float(torch.quantile(difference.flatten(), 0.95)),
                        "max_abs_difference": float(difference.max()),
                    }
                )
            else:
                entry["comparison"] = "temporal_shapes_do_not_align"
            comparisons.append(entry)
    result = {
        "schema": "phycontext.vae_temporal_decode_audit.v1",
        "sample_id": item["sample_id"],
        "latent": str(latent_path),
        "decoded_shape": list(full.shape),
        "decoded_condition_frame_mean_abs_error": float(
            condition_difference.mean()
        ),
        "decoded_condition_frame_max_abs_error": float(
            condition_difference.max()
        ),
        "comparisons": comparisons,
    }
    output = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
