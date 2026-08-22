import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from point_trajectory import rasterize_das_3d_tracks  # noqa: E402
from wan_training import (  # noqa: E402
    inject_trajectory_conditioning,
    motion_mask_from_point_track_map,
    set_trajectory_condition,
)


class TinyWan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 8
        self.patch_size = (1, 2, 2)
        self.patch_embedding = nn.Conv3d(
            4, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, videos):
        return [self.patch_embedding(video.unsqueeze(0)) for video in videos]


def trajectory_payload() -> dict[str, np.ndarray]:
    frame_count = 9
    point_count = 2048
    points = np.zeros((frame_count, 1, point_count, 3), np.float32)
    points[..., 2] = 2.0
    tracks = np.zeros((frame_count, 1, point_count, 2), np.float32)
    valid = np.zeros((frame_count, 1, point_count), bool)
    valid[:, 0, :2] = True
    for frame in range(frame_count):
        tracks[frame, 0, 0] = (2.0 + frame, 3.0)
        tracks[frame, 0, 1] = (12.0 + frame, 12.0)
    points[:, 0, 0] = (-1.0, 0.0, 2.0)
    points[:, 0, 1] = (1.0, 0.5, 4.0)
    depth = points[..., 2].copy()
    return {
        "time_s": np.arange(frame_count, dtype=np.float32),
        "object_ids": np.asarray(["object_0"]),
        "points_world_m": points.copy(),
        "points_camera_m": points,
        "tracks_xy_px": tracks,
        "depth_m": depth,
        "valid": valid,
        "initial_points_camera_m": points[0].copy(),
        "camera_from_world": np.repeat(np.eye(4)[None], frame_count, axis=0),
        "camera_intrinsics": np.eye(3),
        "image_size_px": np.asarray([32, 32]),
        "metadata_json": np.asarray("{}"),
    }


class DasWanIntegrationTest(unittest.TestCase):
    def test_full_rate_render_reaches_wan_patch_grid_and_gradients(self) -> None:
        rendered = rasterize_das_3d_tracks(
            trajectory_payload(),
            (4, 4),
            preprocess_size_px=(32, 32),
            frame_indices=list(range(9)),
        )
        self.assertEqual(rendered.shape, (12, 9, 4, 4))
        point_map = torch.from_numpy(rendered)
        motion_mask = motion_mask_from_point_track_map(
            point_map, representation="das_3d_tracks"
        )
        self.assertEqual(motion_mask.shape, (1, 3, 4, 4))

        model = TinyWan()
        video = torch.zeros(4, 3, 4, 4)
        baseline = model([video])[0].detach().clone()
        conditioner = inject_trajectory_conditioning(
            model, rank=4, representation="das_3d_tracks"
        )
        set_trajectory_condition(model, [point_map])
        identity_output = model([video])[0]
        torch.testing.assert_close(identity_output, baseline)

        with torch.no_grad():
            conditioner.output_projection.weight.fill_(1.0)
        conditioned = model([video])[0]
        self.assertEqual(conditioned.shape, (1, 8, 3, 2, 2))
        self.assertGreater(float((conditioned - baseline).abs().sum()), 0.0)
        conditioned.sum().backward()
        self.assertIsNotNone(conditioner.temporal_projection.weight.grad)
        self.assertGreater(
            float(conditioner.temporal_projection.weight.grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
