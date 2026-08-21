# Three.js Viewer Runtime

Vendored from npm package `three@0.185.1` for the self-contained PhysSweep
scene-input review artifact.

- `three.module.min.js`: Three.js ES module build.
- `three.core.min.js`: core module imported by the Three.js module build.
- `OrbitControls.js`: official Three.js orbit controls.
- `GLTFLoader.js`: official glTF/GLB loader.
- `BufferGeometryUtils.js`: GLTFLoader geometry dependency.
- `SkeletonUtils.js`: GLTFLoader scene dependency.
- `viewer_bundle_entry.js`: reproducible offline viewer bundle entry.
- `three-viewer.iife.min.js`: prebuilt classic browser bundle used by generated HTML.
- `LICENSE.txt`: upstream MIT license.

These files are embedded into generated HTML; the viewer makes no network
requests.

The two utility imports in the vendored GLTFLoader are normalized to this flat
directory before building the classic offline bundle.

The checked-in bundle is generated with `esbuild@0.25.8` from
`viewer_bundle_entry.js`; generated review pages embed only this classic IIFE
and do not create JavaScript modules at runtime.
