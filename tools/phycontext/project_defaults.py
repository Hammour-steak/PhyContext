"""Model-side defaults for the current PhysSweep training dataset build.

PhysSweep owns the immutable raw release. The release adapter owns the derived
``datasets/physweep_training`` interface under the external data root. Wan
preprocessing and training own only this repository's ``cache/`` and
``outputs/training/`` trees.
"""

import os
from pathlib import Path


_DATASET_ROOT_VALUE = os.environ.get("PHYCONTEXT_DATASET_ROOT")
DATASET_ROOT = Path(_DATASET_ROOT_VALUE).expanduser() if _DATASET_ROOT_VALUE else None
DATASET_MANIFEST = Path("datasets/physweep_training/manifest.jsonl")
CACHE_ROOT = Path(
    "cache/wan/physweep_training/das_3d_tracks_track4gen_v8_center_visibility_832x480x97"
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
