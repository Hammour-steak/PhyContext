# PhyContext Method

## Task

PhyContext generates a video from five aligned conditions:

1. an RGB first frame and optional semantic prompt;
2. a structured 3D scene in the camera coordinate frame;
3. physical properties bound to environment surfaces and controlled objects;
4. object initial state and global gravity;
5. the dense object point trajectory produced by that same physical setup.

The structured scene explains the physical causes. The trajectory is their
spatiotemporal realization. Neither is treated as an unrelated control signal.

## Representation

For one controlled object, the scene contains:

- 8,192 static-environment points with position, normal, friction, and restitution;
- 2,048 complete object-surface points;
- object mass, friction, restitution, inertia, and damping; the initial pose is
  already represented by the complete object points;
- initial linear velocity, angular velocity, and gravity;
- 2,048 object points tracked through all 97 target frames.

Every quantity uses one documented camera coordinate frame. The same explicit
`object_id` binds the scene record, physical state, and point trajectory.

## Model

- Wan2.2 TI2V supplies first-frame and text conditioning and remains frozen.
- `PhyContextConditionEncoder` maps scene geometry, physics, and initial state to
  128 context tokens.
- A direct zero-initialized modulation branch injects compact physical controls
  into Wan transformer blocks.
- A zero-initialized trajectory conditioner projects dense target point tracks
  onto the Wan spatiotemporal latent grid.
- Rank-16 cross-attention LoRA learns how Wan uses the added context.

The trajectory branch specifies gross motion. Structured scene and physical
conditions provide contact geometry, material context, and interpretable cause.

## Training

Training reads only a published PhysSweep manifest, resolved against the
external PhysSweep project/data root, and its matching audited Wan
cache. The production path always uses each target sample's own trajectory.

The maintained objective combines flow matching, controlled-region
reconstruction, trajectory alignment, and optional LPIPS supervision. The
formal configuration is the single source of truth for weights and schedules.

Required ablations are:

- pretrained Wan;
- trajectory only;
- structured scene and physics without trajectory;
- complete PhyContext;
- mismatched trajectory as a diagnostic, never as production input.

## Inference

Dataset evaluation uses the published structured scene and cached simulator
trajectory. External-image inference reconstructs a scene, binds user-specified
physical values and initial state, runs a physics solver, then supplies the
resulting matching trajectory to the same model interface.

A monocular image cannot uniquely determine absolute scale, hidden geometry,
material values, or initial velocity. Those values must be calibrated or
provided explicitly; PhyContext does not silently invent them.
