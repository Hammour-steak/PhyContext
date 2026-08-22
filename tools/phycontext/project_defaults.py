"""Model-side defaults for the current PhysSweep training dataset build.

Dataset production owns ``datasets/``. Wan preprocessing and training own
``cache/`` and ``outputs/training/`` and must never write derived model data
inside the external PhysSweep project/data root.
"""

import os
from pathlib import Path


_DATASET_ROOT_VALUE = os.environ.get("PHYCONTEXT_DATASET_ROOT")
DATASET_ROOT = Path(_DATASET_ROOT_VALUE).expanduser() if _DATASET_ROOT_VALUE else None
DATASET_MANIFEST = Path("datasets/physweep_training/manifest.jsonl")
CACHE_ROOT = Path(
    "cache/wan/physweep_training/das_3d_tracks_fullres_v4_832x480x97"
)
CACHE_MANIFEST = CACHE_ROOT / "manifest.json"
POINT_TRAJECTORY_MANIFEST = Path(
    "datasets/physweep_training/point_trajectories/manifest.json"
)

VIDEO_WIDTH = 832
VIDEO_HEIGHT = 480
VIDEO_FRAMES = 97
VIDEO_MAX_AREA = VIDEO_WIDTH * VIDEO_HEIGHT

SCENE_TOKEN_COUNT = 128

INFERENCE_SAMPLING_STEPS = 30
INFERENCE_GUIDANCE_SCALE = 5.0
INFERENCE_FLOW_SHIFT = 5.0
