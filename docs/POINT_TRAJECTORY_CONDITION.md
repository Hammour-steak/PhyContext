# Dense Point-Trajectory Condition

PhyContext uses a simulator-derived dense point-trajectory condition:

1. The simulator is the source of truth for each object's pose.
2. Each object keeps a fixed set of 2048 surface points, so point identity is
   stable over time.
3. The cache stores world and camera coordinates for every frame as
   `[T, O, 2048, 3]`, plus projected pixels, depth, and projection validity.
4. The video model receives projected track maps, not a raw variable-length
   point cloud. Each of three object slots has six channels: source occupancy,
   current occupancy, source-anchored x/y displacement, depth, and validity.

The fixed-width condition is `[18, T_latent, H_latent, W_latent]`. Empty object
slots are zero-filled, so the encoder interface can represent one, two, or three
objects. The current published dataset schema and formal training configuration
remain one-object; multi-object publication is not claimed yet.
The trajectory must be generated for the same physical sample as the target
video; a nominal base trajectory is retained only as an explicit ablation.

The scene encoder adds a learned object-slot embedding (`0`, `1`, or `2`) to
each object's point block. The point encoder weights are shared across objects,
but the model can distinguish object identity and align it with the matching
per-object physics tokens and track-map channels.

The current point cache uses projection and clipping validity. It does not claim
to be a z-buffer visibility mask. Occlusion-aware visibility can be added later
without changing the point identity or object-slot contract.

The rasterized depth channel uses one robust percentile range per sample across
all frames and object slots, so its scale does not change from frame to frame.

The motion mask is not a model input in this representation. Point-track maps
are the trajectory condition; when auxiliary motion losses are enabled, the
trainer derives a temporary source/current occupancy envelope from those maps.
This keeps the point-track cache self-contained. Centroid-only and
source-bound mask representations are not part of the maintained pipeline.

The direct physical modulation branch uses the same fixed object slots. Each
slot contributes three physical controls plus one initial state, so its input
is `12 * O_max` values; absent slots are zero-padded.

## Pipeline

Point trajectories must be exported and published by the dataset project from
the same simulation record used to render each target video. This method
project treats them as immutable dataset inputs.

Build Wan cache entries with the point-trajectory manifest. The cache records
its hash, so a trajectory from another sweep sample cannot be silently reused:

```bash
python tools/phycontext/cache_wan_inputs.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --point-trajectory-manifest datasets/physweep_training/point_trajectories/manifest.json
```

Train with `--trajectory-input --trajectory-representation
dense_point_tracks`. The default trajectory source is the target sample;
`nominal_base` remains an explicit fixed-trajectory ablation only.
