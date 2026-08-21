# PhyContext Core

This directory owns dataset readers, condition encoders, Wan cache construction,
training, evaluation, and inference. Dataset files are external, immutable
inputs. Defaults from `project_defaults.py` can be overridden by the project
entry point or environment variables:

```text
dataset       datasets/physweep_training
cache         cache/wan/physweep_training/dense_point_tracks_832x480x97
resolution    832 x 480 x 97 frames
scene tokens  128
```

The maintained chain is: validate the published manifest, cache Wan inputs,
merge and audit cache shards, train with `train_wan_formal.py`, then infer under
the saved run contract. Dataset sampling, simulation, rendering, and trajectory
export belong to the separate dataset project.

`train_wan_formal.py` is the only training entry point. Unsupported checkpoint
contracts are rejected explicitly during audit and inference.
