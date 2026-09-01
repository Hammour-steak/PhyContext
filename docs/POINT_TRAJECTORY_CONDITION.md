# DaS-Style 3D Point-Trajectory Condition

PhyContext uses simulator-derived dense point trajectories. The simulator is the
source of truth for object motion, and every dynamic object keeps 2048 stable
surface-point identities through the sequence. The published cache contains
world and camera coordinates with shape `[T, O, 2048, 3]`, projected pixels,
depth, and projection validity.

## Default representation: `das_3d_tracks`

The default Wan condition is a DaS-inspired representation adapted to Wan2.2.
It uses DaS point identities but a lightweight Wan patch conditioner rather
than DaS's copied multi-block condition DiT:

1. Assign every finite positive-depth point a fixed RGB identity from its
   first-frame camera coordinate `(x, y, 1/z)`. Frame-zero image bounds do not
   suppress a point that enters view later.
2. Normalize camera x/y with sample-wide min/max and inverse z with the robust
   2nd/98th percentiles across all object slots.
3. Select the same 97 evenly spaced source frames used by video preprocessing,
   require the source video and trajectory frame counts to agree, and project
   the same points using the simulator-provided camera projection.
4. At 832 x 480, resolve dynamic points and static-environment points with one
   global nearest-positive-depth z-buffer.
5. Only after visibility is resolved, aggregate visible identities and binary
   occupancy onto the 52 x 30 Wan VAE spatial grid.
6. Keep the RGB identity unchanged while the projected position moves.

The cover-resize coordinate transform uses OpenCV's half-pixel center
convention, matching the RGB preprocessing rather than approximating it with a
corner-aligned scale.

After source-array decoding, rasterization geometry stays in float64 through
projection, resize, and crop, and is converted to integer pixels only at the
z-buffer; this prevents additional precision loss at a half-pixel rounding
boundary. Cache schemas v4-v9 record this geometry contract together with
static-point camera clipping; v5 adds the canonical TI2V first-frame binding,
v6 adds exact material-point correspondence targets, and v7 preserves each
float64 visible raster decision when its coordinate is serialized as float32.
Version 8 gives loss-only correspondence a separate zero-radius point-center
z-buffer. This avoids treating a visible neighboring splat pixel as evidence
that the point center is visible, while also preventing 3 x 3 control-map
dilation from falsely hiding an adjacent material point.
Version 9 additionally stores up to 256 spatially balanced, frame-zero-visible
static scene identities and their per-frame point-center visibility. Training
balances foreground and background queries so static walls and floors receive
the same correspondence objective as moving objects.
Older point maps remain
readable for legacy adapters but cannot train the current correspondence path.

`unproject_physweep_tracks` provides the metric-depth inverse of the published
projection. `tools/phycontext/audit_das_roundtrip.py` uses it to verify a real
sample in both directions: raw rigid poses, world/camera transforms, pixels and
depth, GT scene identity, DaS RGB identity, an independent reference z-buffer,
rendered masks, and the 97-to-25 temporal window. The audit fails closed when a
metric exceeds its declared numerical or alignment tolerance. Temporal window
averaging and per-cell RGB aggregation are intentionally many-to-one, so their
conservation semantics can be verified but individual source frames or points
cannot be reconstructed from the compressed condition alone.

Each object slot contributes four channels:

```text
identity_r, identity_g, identity_b, visibility
```

With three fixed object slots, the cached model input is `[12, 97, 30, 52]`.
The trajectory conditioner preserves frame zero and applies a learned causal
four-frame window to frames 1-96, producing the Wan-aligned temporal grid
`[12, 25, 30, 52]` before patch projection. Empty slots are zero-filled. The explicit
visibility channel avoids confusing black background with a valid point whose
normalized coordinate color is near zero. Visibility also supplies the latent
motion envelope used by reconstruction and the coarse center guard; it is not
a second model condition.

The v9 cache writes a separate loss-only correspondence artifact with
`track_xy_px [97,O,2048,2]`, `track_depth_m [97,O,2048]`, and
`track_visible [97,O,2048]`, plus `background_track_xy_px [97,S,2]`, depth,
and visibility for `1 <= S <= 256`. `background_point_indices [S]` binds those
trajectories back to the immutable scene-condition point cloud. The condition
map and this artifact are emitted by the same float64 projection geometry. The
condition map uses 3 x 3 splats for
coverage, while correspondence uses zero-radius dynamic/static point centers.
Thus splat dilation cannot create or suppress an exact material-point feature
pair. Points that project to the same pixel still compete by metric depth in a
global z-buffer across all object slots and the static scene.
Unlike the many-to-one 30 x 52 RGB condition, the loss artifact retains the
exact object slot and material-point index. A first-frame feature query retrieves
the same point from the global intermediate-Wan feature map for each swept
four-RGB-frame target window.

Static occlusion uses the 8,192 environment points in the scene condition,
transformed through the published camera transform or camera sequence. A single
static `[4,4]` PhysSweep camera is broadcast over all frames. Static projection
uses the same PhysSweep right/up/forward camera convention as dynamic tracks,
including its left-handed camera basis and the image-row sign flip
`v = cy - fy*y/z`. It is a point-cloud
approximation; continuous mesh/depth-buffer visibility would require the dataset
to publish an environment depth video or renderable scene mesh.

## Legacy ablation: `dense_point_tracks`

The previous representation remains supported as an explicit ablation. Each of
three object slots has six channels: source occupancy, current occupancy,
source-anchored x/y displacement, normalized depth, and projection validity. Its
fixed shape is `[18, T_latent, H_latent, W_latent]`.

Old caches and checkpoints are not overwritten. DaS-style caching uses:

```text
cache/wan/physweep_training/das_3d_tracks_track4gen_v9_bg_balanced_832x480x97/
```

The legacy cache remains under `dense_point_tracks_832x480x97/`.

## Pipeline

Point trajectories must come from the same simulation record used to render each
target video. For the immutable one-object release,
`tools/phycontext/adapt_physweep_release.py` constructs them directly from the
published rigid poses and a deterministic 2,048-point collision-proxy surface.
It applies every frame's pose, projects through the published camera, checks the
inverse rigid and camera transforms, and requires exact frame-zero alignment with
the shared scene condition. The source release remains immutable and every
derived trajectory is hash-bound in its own manifest.

The release does not contain rendered dynamic-object meshes. Consequently these
points are simulation-grounded material identities, not visual-surface truth.
The adapter reports their first-frame mask coverage as a diagnostic and labels
the limitation in scene and sample provenance.

Build the default cache with:

```bash
python tools/phycontext/cache_wan_inputs.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --point-trajectory-manifest \
    datasets/physweep_training/point_trajectories/manifest.json \
  --trajectory-representation das_3d_tracks
```

Train with:

```bash
python tools/phycontext/train_wan_formal.py \
  --trajectory-input \
  --trajectory-representation das_3d_tracks
```

Correspondence supervision is a separate low-noise forward pass. That pass
uses only text and video latents: the target trajectory branch, scene/physics
tokens, and direct physics modulation are all disabled. The resulting
intermediate Wan feature grid is refined at 2x spatial resolution, then trained
with Track4Gen's soft-argmax coordinate Huber objective. The target trajectory
is therefore a label only and cannot be read as an input shortcut. Self-attention
LoRA is trained across all Wan blocks; the zero-initialized feedback bridge is
trained only by generation losses.

For a one-record, one-step integration smoke test, add `--ordinary-only`. This
disables paired physics-response updates and validation and records zero response
share in the adapter metadata. The trainer rejects more than one step in this
mode. Do not use the flag for an ablation or full PhysSweep training run: the
default remains the 60/40 ordinary/paired-response schedule and requires complete
low/base/high sweep groups.

For the current 41,600-sample release, the formal one-object run uses 9,000
optimizer steps. With four ranks, this visits every training sample and every
mass endpoint pair at least once. Validation uses 25 microbatches per rank: this
is one complete schedule window, so it preserves both the 60/40
ordinary/response split and the 2:2:1 friction/restitution/mass response-axis
weights. Other formal validation sizes must be positive multiples of 25.

The default trajectory source is the target sample. `nominal_base` remains an
explicit fixed-trajectory physics-response ablation. The scene encoder and direct
physical modulation branches retain their three object slots, so point-track
conditions stay aligned with the corresponding material and initial-state tokens.
