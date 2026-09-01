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
Raw camera intrinsics are transformed through the exact cover-resize and
center-crop applied to RGB before they are normalized for the scene encoder.

## Model

- Wan2.2 TI2V supplies first-frame and text conditioning and remains frozen.
- `PhyContextConditionEncoder` maps scene geometry, physics, and initial state to
  128 context tokens.
- A direct zero-initialized modulation branch injects compact physical controls
  into Wan transformer blocks.
- A zero-initialized trajectory conditioner projects DaS-style fixed RGB point
  identities plus visibility. It resolves dynamic/static occlusion at full
  preprocessing resolution, retains all 97 condition frames, and learns the
  causal 97-to-25 temporal projection used by Wan.
- A Track4Gen-style 128-D spatial refiner reads an intermediate Wan block.
  First-frame query features must retrieve the same simulator material points
  in later latent windows through a global cosine cost volume. A zero-initialized
  bridge feeds detached refined features back to Wan.
- Rank-16 cross-attention LoRA learns how Wan uses the added context.

The trajectory branch specifies gross motion. Structured scene and physical
conditions provide contact geometry, material context, and interpretable cause.

## Training

Training reads only a published PhysSweep manifest, resolved against the
external PhysSweep project/data root, and its matching audited Wan
cache. Before encoding a target with Wan's causal VAE, frame zero is replaced
by the group's canonical first-frame condition; all later target frames are
left unchanged. The production path always uses each target sample's own
trajectory.

The maintained objective combines flow matching, controlled-region
reconstruction, paired physical-response separation, LPIPS, a low-weight clean
latent object-center guard, and feature correspondence. The correspondence
cache preserves exact object/point axes, projected pixels, metric depth, and
center-pixel winners from the shared dynamic/static z-buffer for all 97 frames.
Winning only a neighboring control-map splat pixel does not mark the point's
center coordinate visible. Wan latent zero is
the condition frame; latent `k >= 1` maps to source frames `4k-3 .. 4k`. Each
target is therefore a visibility-weighted swept heatmap of the same material
point over those four RGB frames. Candidate queries are balanced across slow,
medium, and fast feature-grid displacement.

The feature objective uses soft-target cross entropy over the global target
feature map plus a small expected-coordinate Huber term. It directly trains the
refiner and upstream LoRA features. Videos, rather than visible point counts,
receive equal loss weight across DDP ranks. The diffusion objective trains the
zero-initialized feedback bridge at one tenth of the refiner learning rate,
while stop-gradient prevents that bridge from turning the refiner into an
unconstrained shortcut. On 10% of training samples only the DaS RGB identities
are removed while every per-object occupancy trajectory remains; this forces
the feature objective to retain visual evidence without changing the model's
trajectory-conditioned task. Monitoring reports KL above soft-target entropy,
expected-coordinate EPE, PCK@1, and the fast-motion EPE/PCK@1 subset rather than
interpreting raw cross entropy alone.

Paired physical response uses a separate sigma-1 counterfactual forward. The
future noise, first-frame latent, text, and trajectory are identical for the
low/high endpoints; only structured physics tokens and direct physics
modulation differ. The response loss therefore cannot be satisfied by reading
two different target trajectories. LPIPS is evaluated on ordinary updates only;
response updates reserve memory for the counterfactual forward and keep the
same reconstruction, center, and exact-correspondence supervision.

Decoded RGB optical-flow,
trajectory-distribution, and adjacent-latent velocity losses are not part of the
formal path; they were removed because they duplicate or conflict with exact
material correspondence, especially at occlusion and high speed.

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
