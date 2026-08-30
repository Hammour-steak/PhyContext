# Inference Input Contract

This contract applies to the PhyContext formal adapter.

## Sample Identity

The first frame, prompt, scene points, object parameters, initial state, camera,
gravity, and trajectory must all identify the same target sample. Sweep members
must use their own PyBullet trajectory. Reusing the nominal base trajectory is
allowed only in an explicitly named ablation.

## Geometry And Physics

- 8,192 static-environment points with normals, friction, and restitution
- 2,048 ordered surface points per dynamic object
- up to three object slots, zero-padded when absent
- mass, friction, restitution, inertia, damping, linear/angular velocity, gravity
- 2,048 projected point tracks per object over 97 frames

All geometry, dynamics, and trajectories use the coordinate frames declared in
the sample metadata. Inputs with missing hashes, mismatched sample IDs, or mixed
coordinate/projection protocols must fail before inference.

Physical parameter overrides cannot reuse the cached trajectory of the original
sample. A changed mass, friction, or restitution value requires a trajectory
recomputed from that same modified state. Fixed-trajectory parameter changes are
separate ablations and must disable trajectory input explicitly.

## Video Contract

- Resolution: 832 x 480
- Frames: 97
- First-frame protocol: the group-shared canonical condition replaces target
  frame zero before the full causal VAE encode
- Camera intrinsics: transformed through the exact cover-resize and center-crop,
  then normalized by 832 x 480
- Flow shift: 5.0
- Guidance scale: 5.0
- Sampling steps: 30
- Seed: explicit and recorded
- Scene tokens: 128
- Trajectory representation: `das_3d_tracks`
- Trajectory channels: three object slots × `(identity_r, identity_g,
  identity_b, visibility)` = 12
- Point identity: fixed first-frame camera `(x, y, 1/z)` RGB
- Cached trajectory shape: `[12, 97, 30, 52]`
- Visibility: full-resolution nearest-positive-depth z-buffer across dynamic
  objects and static scene points, followed by spatial aggregation
- Temporal alignment: frame zero plus learned causal four-frame windows,
  matching Wan's 97-to-25 temporal layout
- Trajectory source: `target`

## Adapter Identity

The next formal output is `outputs/training/formal/final`. Training writes the
optimization contract to `outputs/training/formal/run_contract.json` and embeds
an `input_contract.json` in every saved checkpoint. There is currently no valid adapter;
inference must not silently fall back to a deleted or legacy checkpoint.
