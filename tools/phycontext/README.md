# PhyContext Core

This directory owns dataset readers, condition encoders, Wan cache construction,
training, evaluation, and inference. Dataset files are external, immutable
inputs. Defaults from `project_defaults.py` can be overridden by the project
entry point or environment variables:

```text
dataset       datasets/physweep_training
cache         cache/wan/physweep_training/das_3d_tracks_track4gen_v7_832x480x97
resolution    832 x 480 x 97 frames
scene tokens  128
```

Training videos and inference first frames use the same shared
`cover_then_center_crop` transform. Before the full causal VAE encode, every
target video's frame zero is replaced by its group's published canonical first
frame, so the clean latent condition exactly matches inference. Raw camera
intrinsics are transformed through the same resize/crop operation before scene
encoding. Checkpoint input contracts bind both protocols and the exact
`832 x 480 x 97` shape; inference rejects incompatible inputs.

The maintained chain is: adapt the immutable release when needed, validate the
model-ready manifest, cache Wan inputs, merge and audit cache shards, train with
`train_wan_formal.py`, then infer under the saved run contract. Dataset sampling,
simulation, rendering, and publication belong to the separate dataset project.

`audit_wan_cache.py` verifies the complete merged cache and all hash bindings.
`audit_wan_cache_samples.py` then independently rerasterizes and re-encodes a
stratified sample (40 by default), requiring tensor-exact point maps, exact
material-point correspondence artifacts, and latent reconstruction under the
canonical first-frame protocol. The complementary
`audit_vae_temporal_decode.py` verifies that cached latents decode to finite,
correctly aligned video tensors and reports first-frame reconstruction error.
`track_correspondence.py` implements the formal Track4Gen-style feature
objective. It supervises exact first-frame-visible simulator material points
against their swept, z-buffer-visible positions in every four-RGB-frame Wan
latent window. The trainer averages correspondence per video, uses a lower
learning rate for the generation feedback bridge, and drops only RGB point IDs
on 10% of samples while preserving occupancy trajectories. It reports KL, EPE,
PCK@1 and fast-motion EPE/PCK@1. Paired response uses a separate sigma-1
forward with common noise, text, first frame, and trajectory so only structured
physics conditions vary. `audit_wan_adapter.py` fails closed on the selected
block, refiner and feedback tensors, objective settings, per-step loss history,
and pair counts. Decoded RGB flow, trajectory-distribution, and velocity objectives
are legacy checkpoint metadata only and are not part of the maintained trainer.

## PhysSweep release adapter

`adapt_physweep_release.py` converts `outputs/one_object` without modifying it.
For every release group it writes one canonical base first frame, one shared
scene condition, 13 sample records, and 13 fixed-material-point trajectories.
It also verifies release hashes, the one-factor sweep invariant, camera
handedness, rigid-pose and camera inverse transforms, scene/trajectory frame-zero
alignment, and the collision-proxy projection against the published instance
mask. The default derived output is `datasets/physweep_training`.

```bash
python tools/phycontext/adapt_physweep_release.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --release-root outputs/one_object
```

Use a separate output for a non-training smoke build:

```bash
python tools/phycontext/adapt_physweep_release.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --release-root outputs/one_object \
  --output-root datasets/adapter_smoke \
  --limit-groups 1
```

The dynamic object surface is the simulator's collision proxy because the
release does not publish its rendered visual mesh. Static points come from the
released collision fixture, including bound static-prop placement transforms.
For `asset_proxy_v3`, they also include the simulator's authoritative `z=0`
environment floor: the release fixture preserves its contact material but not
the implicit PyBullet plane, so the adapter reconstructs only its in-frame
camera-frustum footprint. Both choices are recorded in output provenance; mask
overlap is diagnostic, not a false claim of visual-mesh equivalence.

The adapter refuses non-empty output directories by default. `--overwrite`
accepts only a completed output or an interrupted output carrying this
adapter's ownership marker; it never removes an unrelated directory. The marker
is deleted after a successful build.

`train_wan_formal.py` is the only training entry point. Unsupported checkpoint
contracts are rejected explicitly during audit and inference.
