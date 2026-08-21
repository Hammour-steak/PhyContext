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
- Flow shift: 5.0
- Guidance scale: 5.0
- Sampling steps: 30
- Seed: explicit and recorded
- Scene tokens: 128
- Trajectory representation: `dense_point_tracks`
- Trajectory source: `target`

## Adapter Identity

The next formal output is `outputs/training/formal/final`. Training writes the
optimization contract to `outputs/training/formal/run_contract.json` and embeds
an `input_contract.json` in every saved checkpoint. There is currently no valid adapter;
inference must not silently fall back to a deleted or legacy checkpoint.
