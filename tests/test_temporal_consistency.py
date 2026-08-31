import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from train_wan_formal import (  # noqa: E402
    clean_latent_lpips_loss,
    scheduled_temporal_window,
)
from temporal_supervision import (  # noqa: E402
    background_temporal_residual_loss,
    decode_wan_causal_window,
    dynamic_zbuffer_visibility_from_masks,
    object_track_temporal_residual_loss,
    source_frames_for_latent_window,
    temporal_decoded_size,
)


class FakeVAE:
    def __init__(self):
        self.temporal_lengths = []

    def decode(self, values):
        value = values[0]
        self.temporal_lengths.append(int(value.shape[1]))
        return [value]


class FakePerceptual(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean(dim=(1, 2, 3))


def pixel_grid(x: int, y: int, width: int, height: int) -> torch.Tensor:
    return torch.tensor(
        [2.0 * (x + 0.5) / width - 1.0, 2.0 * (y + 0.5) / height - 1.0]
    )


class FakePointwise(torch.nn.Module):
    kernel_size = (1, 1, 1)

    def forward(self, value):
        return value


class FakeCausalModel:
    def __init__(self):
        self.conv2 = FakePointwise()

    def clear_cache(self):
        self._feat_map = []

    def decoder(self, value, *, feat_cache, feat_idx, first_chunk):
        del feat_cache, feat_idx
        frames = 1 if first_chunk else 4
        value = torch.nn.functional.interpolate(
            value.squeeze(2), size=(8, 8), mode="nearest"
        ).unsqueeze(2)
        return value.repeat(1, 1, frames, 1, 1)


class FakeCausalVAE:
    def __init__(self):
        self.model = FakeCausalModel()
        self.scale = [torch.zeros(12), torch.ones(12)]
        self.dtype = torch.float32


class TemporalConsistencyTest(unittest.TestCase):
    def test_source_frame_mapping_excludes_condition_frame(self):
        self.assertEqual(source_frames_for_latent_window(1), list(range(1, 9)))
        self.assertEqual(source_frames_for_latent_window(23), list(range(89, 97)))
        with self.assertRaises(ValueError):
            source_frames_for_latent_window(0)

    def test_distributed_schedule_covers_every_two_chunk_window(self):
        selected = {
            scheduled_temporal_window(step, rank, 4, 25)
            for step in range(23)
            for rank in range(4)
        }
        self.assertEqual(selected, set(range(1, 24)))

    def test_temporal_decode_preserves_video_aspect_ratio_on_latent_grid(self):
        self.assertEqual(temporal_decoded_size((832, 480), 128), (128, 80))

    def test_causal_decode_backpropagates_only_selected_chunk(self):
        latent = (0.1 * torch.randn(12, 5, 1, 1)).requires_grad_(True)
        decoded = decode_wan_causal_window(
            FakeCausalVAE(), latent, latent_window_start=3, decoded_resolution=16
        )
        self.assertEqual(tuple(decoded.shape), (3, 8, 16, 16))
        decoded.square().mean().backward()
        self.assertEqual(float(latent.grad[:, :3].abs().sum()), 0.0)
        self.assertGreater(float(latent.grad[:, 3].abs().sum()), 0.0)
        self.assertGreater(float(latent.grad[:, 4].abs().sum()), 0.0)

    def test_dynamic_zbuffer_keeps_only_nearest_masked_point(self):
        tracks = np.asarray([[[[2.0, 2.0], [2.0, 2.0]]]])
        depth = np.asarray([[[1.0, 2.0]]])
        valid = np.ones((1, 1, 2), dtype=bool)
        masks = np.ones((1, 1, 5, 5), dtype=bool)
        visible = dynamic_zbuffer_visibility_from_masks(
            tracks, depth, valid, masks, point_radius_px=0
        )
        self.assertEqual(visible.tolist(), [[[True, False]]])

    def test_object_loss_allows_large_motion_and_target_appearance_change(self):
        height = width = 7
        target = torch.zeros(3, 4, height, width)
        positions = [(0, 1), (6, 1), (0, 5), (6, 5)]
        for frame, (x, y) in enumerate(positions):
            target[:, frame, y, x] = float(frame) / 3.0
        prediction = target.clone().requires_grad_(True)
        grid = torch.stack(
            [pixel_grid(x, y, width, height) for x, y in positions]
        ).view(4, 1, 1, 2)
        visible = torch.ones(4, 1, 1, dtype=torch.bool)
        loss, pairs = object_track_temporal_residual_loss(
            prediction, target, grid, visible, beta=0.02
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(int(pairs), 3)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_object_loss_detects_flicker_but_ignores_occluded_pairs(self):
        target = torch.zeros(3, 4, 5, 5)
        prediction = target.clone()
        prediction[:, 2, 2, 2] = 1.0
        prediction.requires_grad_(True)
        grid = pixel_grid(2, 2, 5, 5).view(1, 1, 1, 2).repeat(4, 1, 1, 1)
        visible = torch.ones(4, 1, 1, dtype=torch.bool)
        loss, _ = object_track_temporal_residual_loss(
            prediction, target, grid, visible, beta=0.02
        )
        self.assertGreater(float(loss), 0.0)
        occluded = visible.clone()
        occluded[2] = False
        ignored, pairs = object_track_temporal_residual_loss(
            prediction, target, grid, occluded, beta=0.02
        )
        self.assertEqual(float(ignored), 0.0)
        self.assertEqual(int(pairs), 1)

    def test_object_loss_covers_the_four_frame_chunk_boundary(self):
        target = torch.zeros(3, 8, 5, 5)
        prediction = target.clone()
        prediction[:, 4, 2, 2] = 1.0
        grid = pixel_grid(2, 2, 5, 5).view(1, 1, 1, 2).repeat(8, 1, 1, 1)
        visible = torch.ones(8, 1, 1, dtype=torch.bool)
        loss, pairs = object_track_temporal_residual_loss(
            prediction, target, grid, visible, beta=0.02
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(int(pairs), 7)

    def test_background_loss_excludes_three_frame_object_sweep(self):
        target = torch.zeros(3, 4, 7, 7)
        prediction = target.clone()
        prediction[:, 1, 3, 3] = 1.0
        prediction.requires_grad_(True)
        masks = torch.zeros(4, 1, 7, 7, dtype=torch.bool)
        masks[:, 0, 3, 3] = True
        loss, pixels = background_temporal_residual_loss(
            prediction,
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertGreater(int(pixels), 0)

    def test_background_loss_matches_target_change_and_detects_extra_flicker(self):
        target = torch.zeros(3, 4, 5, 5)
        target[:, :, 0, 0] = torch.tensor([0.0, 0.2, 0.7, 1.0])
        matching = target.clone().requires_grad_(True)
        masks = torch.zeros(4, 1, 5, 5, dtype=torch.bool)
        exact, _ = background_temporal_residual_loss(
            matching,
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertEqual(float(exact), 0.0)
        extra = target.clone()
        extra[:, 2, 4, 4] = 1.0
        extra.requires_grad_(True)
        loss, _ = background_temporal_residual_loss(
            extra,
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(extra.grad).all())

    def test_background_loss_covers_the_four_frame_chunk_boundary(self):
        target = torch.zeros(3, 8, 5, 5)
        prediction = target.clone()
        prediction[:, 4, 0, 0] = 1.0
        masks = torch.zeros(8, 1, 5, 5, dtype=torch.bool)
        loss, _ = background_temporal_residual_loss(
            prediction,
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertGreater(float(loss), 0.0)

    def test_background_loss_matches_small_target_residual_instead_of_zero(self):
        target = torch.zeros(3, 4, 5, 5)
        target[:, :, 0, 0] = torch.tensor([0.0, 0.01, 0.03, 0.04])
        masks = torch.zeros(4, 1, 5, 5, dtype=torch.bool)
        exact, _ = background_temporal_residual_loss(
            target,
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        stationary, _ = background_temporal_residual_loss(
            torch.zeros_like(target),
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(stationary), 0.0)

    def test_background_loss_excludes_target_dynamic_shadow_or_reflection(self):
        target = torch.zeros(3, 4, 5, 5)
        target[:, 2:, 0, 0] = 1.0
        masks = torch.zeros(4, 1, 5, 5, dtype=torch.bool)
        loss, _ = background_temporal_residual_loss(
            torch.zeros_like(target),
            target,
            masks,
            dilation_px=0,
            stability_threshold=0.05,
            beta=0.02,
        )
        self.assertEqual(float(loss), 0.0)

    def test_lpips_decodes_contiguous_temporal_windows(self):
        predicted = torch.randn(3, 7, 8, 8, requires_grad=True)
        target = torch.randn(3, 7, 8, 8)
        vae = FakeVAE()
        loss = clean_latent_lpips_loss(
            [predicted],
            [target],
            vae,
            FakePerceptual(),
            window_count=2,
            resolution=16,
            temporal_window=3,
        )
        loss.backward()
        self.assertEqual(vae.temporal_lengths, [3, 3, 3, 3])
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_lpips_excludes_the_clean_condition_frame(self):
        predicted = torch.zeros(3, 7, 8, 8, requires_grad=True)
        with torch.no_grad():
            predicted[:, 0].fill_(10.0)
        target = torch.zeros_like(predicted)
        loss = clean_latent_lpips_loss(
            [predicted],
            [target],
            FakeVAE(),
            FakePerceptual(),
            window_count=2,
            resolution=16,
            temporal_window=3,
        )
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(float(predicted.grad[:, 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
