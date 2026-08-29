# PhyContext Core

This directory owns dataset readers, condition encoders, Wan cache construction,
training, evaluation, and inference. Dataset files are external, immutable
inputs. Defaults from `project_defaults.py` can be overridden by the project
entry point or environment variables:

```text
dataset       datasets/physweep_training
cache         cache/wan/physweep_training/das_3d_tracks_fullres_v4_832x480x97
resolution    832 x 480 x 97 frames
scene tokens  128
```

The maintained chain is: adapt the immutable release when needed, validate the
model-ready manifest, cache Wan inputs, merge and audit cache shards, train with
`train_wan_formal.py`, then infer under the saved run contract. Dataset sampling,
simulation, rendering, and publication belong to the separate dataset project.

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
Both choices are recorded in the output provenance; mask overlap is diagnostic,
not a false claim of visual-mesh equivalence.

`train_wan_formal.py` is the only training entry point. Unsupported checkpoint
contracts are rejected explicitly during audit and inference.
