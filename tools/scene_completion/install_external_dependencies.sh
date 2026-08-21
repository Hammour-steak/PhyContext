#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCENE_ENV="${PROJECT_ROOT}/envs/phycontext-scene-completion"
INSTANT_ENV="${PROJECT_ROOT}/envs/phycontext-instantmesh"
DEPTHLAB_ENV="${PROJECT_ROOT}/envs/phycontext-depthlab"
EXTERNAL_ROOT="${PROJECT_ROOT}/external/scene_completion"
SCENE_CHECKPOINTS="${PROJECT_ROOT}/checkpoints/scene_completion"
DEPTHLAB_CHECKPOINTS="${PROJECT_ROOT}/checkpoints/scene_completion/depthlab"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"

download_checked() {
  local url="$1"
  local output="$2"
  local expected_sha256="$3"
  local temporary="${output}.tmp"

  mkdir -p "$(dirname "${output}")"
  if [[ -f "${output}" ]] && echo "${expected_sha256}  ${output}" | sha256sum --check --status; then
    return
  fi
  rm -f "${temporary}"
  curl --fail --location --retry 3 --output "${temporary}" "${url}"
  if ! echo "${expected_sha256}  ${temporary}" | sha256sum --check --status; then
    rm -f "${temporary}"
    echo "checkpoint checksum mismatch: ${output}" >&2
    return 1
  fi
  mv "${temporary}" "${output}"
}

mkdir -p "${EXTERNAL_ROOT}"
if [[ ! -d "${EXTERNAL_ROOT}/vggt/.git" ]]; then
  git clone https://github.com/facebookresearch/vggt.git "${EXTERNAL_ROOT}/vggt"
fi
git -C "${EXTERNAL_ROOT}/vggt" checkout a288dd0f14786c93483e45524328726ab7b1b4ce

if [[ ! -d "${EXTERNAL_ROOT}/sam2/.git" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "${EXTERNAL_ROOT}/sam2"
fi
git -C "${EXTERNAL_ROOT}/sam2" checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4

if [[ ! -d "${EXTERNAL_ROOT}/InstantMesh/.git" ]]; then
  git clone https://github.com/TencentARC/InstantMesh.git "${EXTERNAL_ROOT}/InstantMesh"
fi
git -C "${EXTERNAL_ROOT}/InstantMesh" checkout 08822c52fdc399b93ea00e4fa9e596344ed52ccc

if [[ ! -d "${EXTERNAL_ROOT}/DepthLab/.git" ]]; then
  git clone https://github.com/ant-research/DepthLab.git "${EXTERNAL_ROOT}/DepthLab"
fi
git -C "${EXTERNAL_ROOT}/DepthLab" checkout 33500d7caae2eecd06c472c31f85e752b890de89

if [[ ! -d "${EXTERNAL_ROOT}/pytorch3d/.git" ]]; then
  git clone https://github.com/facebookresearch/pytorch3d.git "${EXTERNAL_ROOT}/pytorch3d"
fi
git -C "${EXTERNAL_ROOT}/pytorch3d" checkout 75ebeeaea0908c5527e7b1e305fbc7681382db47

if [[ ! -d "${EXTERNAL_ROOT}/nvdiffrast/.git" ]]; then
  git clone https://github.com/NVlabs/nvdiffrast.git "${EXTERNAL_ROOT}/nvdiffrast"
fi
git -C "${EXTERNAL_ROOT}/nvdiffrast" checkout 253ac4fcea7de5f396371124af597e6cc957bfae

if [[ ! -x "${SCENE_ENV}/bin/python" ]]; then
  "${CONDA_BIN}" env create \
    --prefix "${SCENE_ENV}" \
    --file "${SCRIPT_DIR}/environment.yml"
fi

export CUDA_HOME="${SCENE_ENV}"
export PATH="${CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="${MAX_JOBS:-16}"
export CC=gcc
export CXX=g++
"${SCENE_ENV}/bin/pip" install --no-build-isolation \
  "${EXTERNAL_ROOT}/pytorch3d"
"${SCENE_ENV}/bin/pip" install --no-deps --editable "${EXTERNAL_ROOT}/vggt"
SAM2_BUILD_CUDA=0 "${SCENE_ENV}/bin/pip" install --no-deps --editable "${EXTERNAL_ROOT}/sam2"

download_checked \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  "${SCENE_CHECKPOINTS}/vggt/model.pt" \
  d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
download_checked \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  "${SCENE_CHECKPOINTS}/sam2/sam2.1_hiera_small.pt" \
  6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38
download_checked \
  https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt \
  "${SCENE_CHECKPOINTS}/lama/big-lama.pt" \
  344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9

if [[ ! -x "${INSTANT_ENV}/bin/python" ]]; then
  "${SCENE_ENV}/bin/python" -m venv "${INSTANT_ENV}"
fi
"${INSTANT_ENV}/bin/pip" install -U pip wheel setuptools==68.2.2
"${INSTANT_ENV}/bin/pip" install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
"${INSTANT_ENV}/bin/pip" install xformers==0.0.22.post7 triton==2.1.0
"${INSTANT_ENV}/bin/pip" install \
  pytorch-lightning==2.1.2 huggingface-hub==0.17.3 \
  transformers==4.34.1 diffusers==0.20.2 accelerate==0.23.0 \
  einops omegaconf torchmetrics==1.2.1 webdataset tensorboard \
  PyMCubes trimesh==4.6.9 rembg==2.0.57 onnxruntime \
  imageio imageio-ffmpeg xatlas plyfile

CUDA_HOME="${SCENE_ENV}" \
PATH="${SCENE_ENV}/bin:/usr/local/bin:/usr/bin:/bin" \
TORCH_CUDA_ARCH_LIST="8.9" \
CC=gcc CXX=g++ MAX_JOBS="${MAX_JOBS}" \
"${INSTANT_ENV}/bin/pip" install --no-build-isolation \
  "${EXTERNAL_ROOT}/nvdiffrast"

if [[ ! -x "${DEPTHLAB_ENV}/bin/python" ]]; then
  "${CONDA_BIN}" env create \
    --prefix "${DEPTHLAB_ENV}" \
    --file "${SCRIPT_DIR}/depthlab_environment.yml"
fi

HF_CLI="${DEPTHLAB_ENV}/bin/huggingface-cli"
mkdir -p \
  "${DEPTHLAB_CHECKPOINTS}/DepthLab" \
  "${DEPTHLAB_CHECKPOINTS}/marigold-depth-v1-0" \
  "${DEPTHLAB_CHECKPOINTS}/CLIP-ViT-H-14-laion2B-s32B-b79K"

if [[ ! -f "${DEPTHLAB_CHECKPOINTS}/DepthLab/denoising_unet.pth" ]]; then
  "${HF_CLI}" download Johanan0528/DepthLab \
    denoising_unet.pth reference_unet.pth mapping_layer.pth README.md \
    --revision ff9bba42b9ec458ac25acade326cf3007627f46d \
    --local-dir "${DEPTHLAB_CHECKPOINTS}/DepthLab"
fi
if [[ ! -f "${DEPTHLAB_CHECKPOINTS}/marigold-depth-v1-0/unet/diffusion_pytorch_model.safetensors" ]]; then
  "${HF_CLI}" download prs-eth/marigold-depth-v1-0 \
    model_index.json LICENSE.txt README.md \
    scheduler/scheduler_config.json \
    tokenizer/merges.txt tokenizer/special_tokens_map.json \
    tokenizer/tokenizer_config.json tokenizer/vocab.json \
    text_encoder/config.json text_encoder/model.safetensors \
    unet/config.json unet/diffusion_pytorch_model.safetensors \
    vae/config.json vae/diffusion_pytorch_model.safetensors \
    --revision f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c \
    --local-dir "${DEPTHLAB_CHECKPOINTS}/marigold-depth-v1-0"
fi
if [[ ! -f "${DEPTHLAB_CHECKPOINTS}/CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors" ]]; then
  "${HF_CLI}" download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    config.json model.safetensors preprocessor_config.json README.md \
    --revision 1c2b8495b28150b8a4922ee1c8edee224c284c0c \
    --local-dir "${DEPTHLAB_CHECKPOINTS}/CLIP-ViT-H-14-laion2B-s32B-b79K"
fi
