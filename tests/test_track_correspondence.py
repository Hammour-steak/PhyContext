import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from track_correspondence import (  # noqa: E402
    TrackCorrespondenceAdapter,
    latent_rgb_windows,
    track4gen_correspondence_loss,
    validate_track_correspondence,
)


def synthetic_correspondence() -> dict[str, torch.Tensor]:
    frames, objects, points = 9, 1, 2048
    xy = torch.zeros(frames, objects, points, 2)
    depth = torch.ones(frames, objects, points)
    visible = torch.zeros(frames, objects, points, dtype=torch.bool)
    path_x = torch.tensor([1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.float32)
    xy[:, 0, 0, 0] = path_x
    xy[:, 0, 0, 1] = 1
    visible[:, 0, 0] = True
    background_xy = torch.zeros(frames, 2, 2)
    background_xy[:, 0] = torch.tensor([0.0, 3.0])
    background_xy[:, 1] = torch.tensor([5.0, 3.0])
    background_depth = torch.ones(frames, 2)
    background_visible = torch.ones(frames, 2, dtype=torch.bool)
    return {
        "track_xy_px": xy,
        "track_depth_m": depth,
        "track_visible": visible,
        "background_track_xy_px": background_xy,
        "background_track_depth_m": background_depth,
        "background_track_visible": background_visible,
        "source_frame_indices": torch.arange(frames),
    }


class TrackCorrespondenceTest(unittest.TestCase):
    def test_wan_causal_windows_cover_each_rgb_frame_once(self) -> None:
        windows = latent_rgb_windows(9, 3)
        self.assertEqual(windows, [(0, 1), (1, 5), (5, 9)])
        covered = [index for start, end in windows for index in range(start, end)]
        self.assertEqual(covered, list(range(9)))

    def test_contract_rejects_visible_out_of_frame_points(self) -> None:
        value = synthetic_correspondence()
        validate_track_correspondence(
            value, preprocess_size_px=(6, 4), expected_frames=9
        )
        value["track_xy_px"][0, 0, 0, 0] = 6
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_track_correspondence(
                value, preprocess_size_px=(6, 4), expected_frames=9
            )

    def test_contract_accepts_subpixel_centers_on_boundary_pixels(self) -> None:
        value = synthetic_correspondence()
        value["track_xy_px"][0, 0, 0] = torch.tensor([-0.49, 3.49])
        validate_track_correspondence(
            value, preprocess_size_px=(6, 4), expected_frames=9
        )
        value["track_xy_px"][0, 0, 0, 0] = -0.51
        with self.assertRaisesRegex(ValueError, "rasterizes outside"):
            validate_track_correspondence(
                value, preprocess_size_px=(6, 4), expected_frames=9
            )

    def test_contract_rejects_float32_half_tie_beyond_last_pixel(self) -> None:
        value = synthetic_correspondence()
        value["track_xy_px"][0, 0, 0, 0] = 5.5
        with self.assertRaisesRegex(ValueError, "rasterizes outside"):
            validate_track_correspondence(
                value, preprocess_size_px=(6, 4), expected_frames=9
            )

    def test_training_rejects_nonconsecutive_source_frames(self) -> None:
        features = torch.randn(8, 3, 4, 6)
        correspondence = synthetic_correspondence()
        correspondence["source_frame_indices"] += 1
        with self.assertRaisesRegex(ValueError, "consecutive source frames"):
            track4gen_correspondence_loss(
                [features],
                [correspondence],
                preprocess_size_px=(6, 4),
                maximum_pairs=16,
                temperature=0.05,
                gaussian_sigma=0.2,
                generator=torch.Generator().manual_seed(7),
            )

    def test_zero_bridge_preserves_wan_output_and_separates_gradients(self) -> None:
        adapter = TrackCorrespondenceAdapter(
            hidden_dim=8, feature_dim=4, refiner_blocks=1
        )
        tokens = torch.randn(1, 8, 8, requires_grad=True)
        grid = torch.tensor([[2, 2, 2]])
        output = adapter(tokens, grid)
        torch.testing.assert_close(output, tokens)

        output.sum().backward()
        self.assertGreater(float(adapter.feedback.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(adapter.input_projection.weight.grad)

        adapter.zero_grad(set_to_none=True)
        tokens.grad = None
        adapter(tokens, grid)
        features = adapter.consume_features()[0]
        features.square().mean().backward()
        self.assertGreater(
            float(adapter.input_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNone(adapter.feedback.weight.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)

    def test_aligned_features_have_lower_loss_and_receive_gradients(self) -> None:
        height, width, feature_dim = 4, 6, 25
        aligned = torch.zeros(feature_dim, 3, height, width)
        anchor_cell = (1, 1)
        target_cells = ((1, 3), (1, 5))
        identity = torch.zeros(feature_dim)
        identity[-1] = 8.0
        aligned[:, 0, anchor_cell[0], anchor_cell[1]] = identity
        aligned[:, 1, target_cells[0][0], target_cells[0][1]] = identity
        aligned[:, 2, target_cells[1][0], target_cells[1][1]] = identity
        # Give every other cell an orthogonal, nonzero feature so normalization
        # cannot turn empty features into accidental matches.
        for frame in range(3):
            for y in range(height):
                for x in range(width):
                    if aligned[:, frame, y, x].abs().sum() == 0:
                        aligned[y * width + x, frame, y, x] = 1.0
        aligned.requires_grad_(True)
        wrong = aligned.detach().clone()
        wrong[:, 2, 1, 5] = 0
        wrong[:, 2, 0, 0] = identity
        wrong.requires_grad_(True)
        correspondence = synthetic_correspondence()
        kwargs = {
            "preprocess_size_px": (width, height),
            "maximum_pairs": 16,
            "temperature": 0.05,
            "gaussian_sigma": 0.2,
        }
        aligned_result = track4gen_correspondence_loss(
            [aligned],
            [correspondence],
            generator=torch.Generator().manual_seed(7),
            **kwargs,
        )
        wrong_result = track4gen_correspondence_loss(
            [wrong],
            [correspondence],
            generator=torch.Generator().manual_seed(7),
            **kwargs,
        )
        self.assertEqual(float(aligned_result["pairs"]), 4.0)
        self.assertEqual(float(aligned_result["foreground_pairs"]), 2.0)
        self.assertEqual(float(aligned_result["background_pairs"]), 2.0)
        self.assertGreaterEqual(float(aligned_result["fast_pairs"]), 1.0)
        self.assertLess(
            float(aligned_result["loss"]), float(wrong_result["loss"])
        )
        self.assertLess(float(aligned_result["kl"]), float(wrong_result["kl"]))
        self.assertLess(
            float(aligned_result["epe_tokens"]),
            float(wrong_result["epe_tokens"]),
        )
        self.assertGreater(
            float(aligned_result["pck_1"]), float(wrong_result["pck_1"])
        )
        aligned_result["loss"].backward()
        self.assertGreater(float(aligned.grad.abs().sum()), 0.0)

    def test_training_objective_weights_videos_not_visible_pair_counts(self) -> None:
        first_features = torch.randn(8, 3, 4, 6)
        second_features = torch.randn(8, 3, 4, 6)
        first = synthetic_correspondence()
        second = synthetic_correspondence()
        # Give the second video four times as many independently visible points.
        for point in range(4):
            second["track_xy_px"][:, 0, point] = second["track_xy_px"][:, 0, 0]
            second["track_xy_px"][:, 0, point, 1] = float(point)
            second["track_visible"][:, 0, point] = True
        kwargs = {
            "preprocess_size_px": (6, 4),
            "maximum_pairs": 64,
            "temperature": 0.05,
            "gaussian_sigma": 0.2,
        }
        first_loss = track4gen_correspondence_loss(
            [first_features],
            [first],
            generator=torch.Generator().manual_seed(11),
            **kwargs,
        )["loss"]
        second_loss = track4gen_correspondence_loss(
            [second_features],
            [second],
            generator=torch.Generator().manual_seed(11),
            **kwargs,
        )["loss"]
        batch_loss = track4gen_correspondence_loss(
            [first_features, second_features],
            [first, second],
            generator=torch.Generator().manual_seed(11),
            **kwargs,
        )["loss"]
        torch.testing.assert_close(batch_loss, (first_loss + second_loss) / 2)


if __name__ == "__main__":
    unittest.main()
