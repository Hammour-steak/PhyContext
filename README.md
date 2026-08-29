# PhyContext

PhyContext grounds trajectory-controlled video generation in structured 3D
physical context. It consumes a published PhysSweep dataset build and conditions
Wan2.2 on the first frame, scene geometry, surface and object physics, initial
state, and the matching dense object point trajectory.

PhyContext does not contain asset ingestion, sampling matrices, PyBullet dataset
simulation, Blender rendering, or dataset publication code. It does include a
consumer-side adapter for the immutable PhysSweep one-object release.

## Input Contract

The configured dataset directory is immutable input. It must provide:

- `manifest.jsonl`;
- first frames and target videos;
- 8,192 static-environment points with normals and surface physics;
- 2,048 points for each controlled object;
- object physical parameters and initial state;
- the matching 2,048-point trajectory for every target sample.

The trajectory is the physical realization of the same structured scene and
parameter assignment. Reusing another sample's trajectory is allowed only in an
explicitly named ablation.

If the external root contains the released `outputs/one_object` layout rather
than this model-ready contract, run `tools/phycontext/adapt_physweep_release.py`.
It writes a separate `datasets/physweep_training` tree and never writes below the
release directory. Dynamic surfaces are labeled as simulator collision proxies;
the adapter does not claim that unpublished rendered object meshes are available.
Static geometry includes released fixture proxies and any backend-authored
collision surface that the fixture references but does not serialize, currently
the authoritative `z=0` asset-proxy environment floor.

## Maintained Pipeline

```text
PhysSweep project/data root + immutable one-object release
        -> adapt release to the model input contract (once)
        -> validate dataset contract
        -> build and audit Wan cache
        -> train PhyContext adapter
        -> validate or run inference
```

The dataset is read only. Wan latents, text embeddings, full-rate DaS-style
point-track maps,
checkpoints, and inference results stay under this repository's `cache/` and
`outputs/` directories.

## Quick Start

```bash
conda create --override-channels --channel conda-forge \
  --prefix .venv python=3.10 pip -y
conda activate "$PWD/.venv"
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121
# flash-attn builds against the active PyTorch/CUDA toolkit. Point CUDA_HOME at
# a toolkit supported by the installed PyTorch build (11.7 or newer here).
CUDA_HOME=/path/to/cuda-toolkit \
  pip install --no-build-isolation -r requirements.txt

export PHYCONTEXT_DATASET_ROOT=/path/to/PhysSweep
export PHYCONTEXT_WAN_REPO=/path/to/Wan2.2
export PHYCONTEXT_WAN_CHECKPOINT=/path/to/Wan2.2-TI2V-5B

python tools/phycontext/adapt_physweep_release.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --release-root outputs/one_object

python tools/phycontext/cache_wan_inputs.py \
  --dataset-root "$PHYCONTEXT_DATASET_ROOT" \
  --manifest datasets/physweep_training/manifest.jsonl
python tools/phycontext/audit_wan_cache.py

python tools/model_training/train_one_object.py \
  --config configs/training/one_object.json \
  --dry-run
```

The dataset root is the PhysSweep project/data root, not a directory inside
PhyContext. Raw release paths remain read only; all Wan cache files are written
under this repository. The release adapter writes only its explicitly selected,
disjoint derived-data directory under the external root. Add
`--limit-groups 1 --output-root datasets/adapter_smoke` to validate one complete
13-sample group without caching or training. Remove `--dry-run` only after the
dataset and cache audits pass.

## Layout

- `configs/training/`: cache, model, optimization, and validation settings.
- `tools/model_training/`: configuration-driven entry points.
- `tools/phycontext/`: data readers, encoders, cache, training, evaluation, and inference.
- `tools/scene_completion/`: optional first-frame-to-scene preprocessing.
- `docs/`: method, trajectory, and inference contracts.
- `tests/`: boundary, conditioning, and training tests.

## Formal Defaults

```text
backbone              Wan2.2 TI2V-5B, frozen
resolution            832 x 480
frames                97 at 24 fps
scene tokens          128
trajectory            DaS-style 3D tracks, `[12, 97, 30, 52]`
cross-attention LoRA  rank 16
direct modulation     rank 32
trajectory branch     rank 32
sampling steps        30
```

## Validation

```bash
python -m compileall -q tools tests
python -m unittest discover -s tests -p "test_*.py"
```

See `docs/PHYCONTEXT_METHOD.md`, `docs/POINT_TRAJECTORY_CONDITION.md`, and
`docs/INFERENCE_INPUT_CONTRACT.md` for the maintained contracts.
