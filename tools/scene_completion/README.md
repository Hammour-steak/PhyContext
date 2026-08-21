# Single-Frame Image-to-Scene

This tool implements the route defined in `../../docs/IMAGE_TO_SCENE_PIPELINE.md`:

1. frozen VGGT reconstructs visible geometry and camera parameters;
2. frozen SAM2 selects the controllable foreground object;
3. Big-LaMa removes the selected object in RGB, then frozen official DepthLab completes the corresponding depth hole while conditioning on the original VGGT depth;
4. official InstantMesh reconstructs a complete textured object mesh;
5. candidate-view matching, PnP, VGGT scene scale, and bounded hard-raster silhouette/depth search place the visual mesh back in the scene;
6. local support planes bind a separate rigid collision proxy without moving the image-aligned visual mesh;
7. the exporter keeps environment geometry, object geometry, camera, transform, and physical parameters separately addressable.

No reconstruction model is trained in this first version. PhysSweep ground truth is reserved for evaluation.

## Run

From the PhyContext project root:

```bash
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=tools/scene_completion \
envs/phycontext-scene-completion/bin/python \
  tools/scene_completion/infer_first_frame.py \
  --image /path/to/frame_0001.png \
  --output-dir outputs/scene_reconstruction/basketball_ramp \
  --device cuda:7 \
  --instantmesh-device 6
```

`HF_ENDPOINT` is only needed while official InstantMesh weights are missing from the local cache. Pass `--gt-mask` only for evaluation; it never changes inference.

## Outputs

```text
scene/
  environment_observed.ply
  environment_completed.ply
  environment_dense.ply
  environment_inpainted.png
  environment_inpainting_mask.png
  environment_depthlab_rgb.png
  environment_depthlab_known_depth.npy
  environment_depthlab_mask.png
  environment_depthlab_raw_depth.npy
  depthlab_manifest.json
  environment_completion_mask.png
  environment_completion_depth.npy
  object_visible.ply
  object_mesh.obj
  object_mesh.mtl
  object_mesh.png
  object_mesh_manifest.json
  object_mesh_aligned.ply
  object_collision_proxy.obj
  aligned_object_mask.png
  aligned_object_depth.npy
  object_transform.json
  camera.json
  physics.json
  scene.npz
  scene_manifest.json
  report.json
  viewer.html
```

`scene.npz` uses source labels `0=observed_environment`, `1=completed_environment`, and `2=reconstructed_object`. The original finite environment is immutable: DepthLab can add points only where the object or its mixed boundary removed geometry. Its raw prediction is retained for audit, while every known VGGT depth is restored exactly before fusion. The archive also stores the VGGT point map, depth/confidence, dense/completion masks, aligned object depth, camera, and object transforms. `scene_manifest.json` hashes the scene package so stale or mismatched artifacts are detectable.

`scene.npz` is an audited reconstruction intermediate. It is not accepted
directly by the formal PhyContext conditioner, which requires the published
`physweep.model_scene_condition.v17` contract.

`physics.json` binds `object_collision_proxy.obj` and deliberately leaves mass, contact friction, and restitution unset so the downstream PhyContext controller can assign them without rerunning reconstruction. The textured visual mesh remains fixed by image alignment; only the local collision proxy is translated rigidly when hidden InstantMesh geometry crosses the inferred support plane. Coordinates are in consistent `vggt_scene_units`; monocular RGB requires an external calibration before those units can be called meters.

## Environments

- main geometry and registration: `envs/phycontext-scene-completion`
- isolated official DepthLab runtime: `envs/phycontext-depthlab`
- isolated official InstantMesh runtime: `envs/phycontext-instantmesh`
- pinned external repositories: `external/scene_completion`
- model cache: `checkpoints/scene_completion`

Recreate the native dependencies with
`tools/scene_completion/install_external_dependencies.sh`. The installer
creates all three environments, checks out pinned external revisions, installs
VGGT and SAM2 into the main scene environment, and downloads the required
VGGT, SAM2, and Big-LaMa checkpoints with SHA-256 verification. PyTorch3D and
nvdiffrast are compiled against CUDA 12.1 for RTX 4090
(`TORCH_CUDA_ARCH_LIST=8.9`).

## Validation

```bash
PYTHONPATH=tools/scene_completion \
envs/phycontext-scene-completion/bin/python -m unittest \
  tools/scene_completion/test_geometry_pipeline.py -v

PYTHONPATH=tools/scene_completion \
envs/phycontext-scene-completion/bin/python \
  tools/scene_completion/audit_scene_package.py \
  outputs/scene_reconstruction/basketball_ramp
```

The automated checks cover object/environment separation, VGGT padding removal, DepthLab input construction, exact known-depth locking, immutable observed geometry, proportional LaMa masking, scale recovery, GT-mask preflight validation, support analysis, collision-proxy geometry, and mesh provenance. End-to-end runs additionally gate DepthLab hole coverage and known-depth scale consistency, the completed/observed depth-seam 90th percentile, camera-frustum coverage, mask overlap, relative aligned-mesh depth error, support/proxy status, point counts, timings, checkpoint provenance, and artifact hashes. The package auditor then verifies every hash and all required structured-array contracts.
