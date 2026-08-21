# Image-to-Scene Pipeline

## Goal

Convert one RGB first frame into a structured 3D scene containing:

- completed environment geometry;
- a complete foreground object mesh;
- the object's pose and scale in the scene;
- camera information and physical attributes.

## Difference from PhysChoreo

PhysChoreo reconstructs each object with InstantMesh and registers it into the visible scene reconstructed by DUSt3R.

Our pipeline explicitly separates the controllable object from the environment. It removes the object's visible points, completes the environment region hidden by that object, reconstructs the object independently, and then places the complete object back into the completed environment.

## Pipeline

1. **Estimate visible geometry**
   - Use VGGT to predict depth, camera parameters, and the visible scene point cloud.
   - Use Grounded-SAM2 or SAM2 to obtain the controllable object's image mask.

2. **Separate object and environment**
   - Project the image mask onto the VGGT point cloud.
   - Save the selected points as the visible object point cloud.
   - Remove them from the scene to obtain the visible environment point cloud.

3. **Complete the environment**
   - Remove the selected object from RGB with frozen Big-LaMa using a mask margin proportional to object size.
   - Remove VGGT padding and send the object-free RGB, original known depth, and explicit hole mask to frozen official DepthLab.
   - Treat the expanded RGB-inpainting region and any invalid original depth as unknown; all other VGGT depth is the scale anchor.
   - Restore every known VGGT depth exactly and add DepthLab points only inside the newly hidden region.
   - Record observed/completed provenance and reject incomplete holes or inconsistent boundary depth.

4. **Reconstruct the complete object**
   - Crop the masked object from the input image.
   - Use InstantMesh to generate a complete textured object mesh.

5. **Place the object back into the scene**
   - Render candidate views of the object mesh and match them to the input object.
   - Use PnP to initialize orientation and translation.
   - Use VGGT depth and the observed silhouette to place the mesh in the same scene scale.
   - Refine scale, rotation, and translation with a bounded hard-raster silhouette/depth search.
   - Keep this textured visual mesh fixed after image alignment.
   - Infer the principal support plane and rigidly align a separate collision proxy for non-penetrating simulation.

6. **Export the structured scene**
   - Merge the completed environment and aligned object.
   - Keep the environment, object mesh, object transform, and physical attributes separately addressable.

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
  object_mesh_manifest.json
  object_collision_proxy.obj
  object_transform.json
  camera.json
  physics.json
  scene.npz
  scene_manifest.json
  report.json
  viewer.html
```

`scene.npz` retains source/body labels for every point and pixel-level fields linking the input image to VGGT geometry and the reconstructed object. `scene_manifest.json` binds the camera, geometry layers, visual object transform, collision proxy, physics placeholders, and artifact hashes into one scene package. Visual alignment and physical contact are intentionally separate: uncertain hidden geometry cannot move an object away from its observed silhouette.

This scene package is a reconstruction intermediate, not a
`physweep.model_scene_condition.v17` model input. The current formal model path
accepts only published, audited v17 scene conditions; an external-image adapter
must emit and validate that exact schema before it can be enabled.

## Scale Boundary

A single uncalibrated RGB image does not determine absolute SI scale. The exported coordinates therefore use one consistent `vggt_scene_units` frame. Metric PhysSweep ground truth or another known-size reference can later supply meters per scene unit without changing object/environment alignment.

## First Version

The first version uses frozen pretrained models and geometric optimization, so it does not require reconstruction training. It reconstructs a dense camera-visible environment surface after removing the object; it does not claim to recover exact geometry behind walls or outside the input camera frustum. PhysSweep ground truth is used only for evaluation and optional scale calibration, never as an inference input.
