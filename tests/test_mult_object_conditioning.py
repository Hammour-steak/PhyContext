import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from conditioning_model import PhyContextConditionEncoder  # noqa: E402
from wan_training import (  # noqa: E402
    TrajectoryPatchConditioner,
    latent_correspondence_motion,
    motion_mask_from_point_track_map,
    source_target_motion_envelope,
    validate_point_track_object_slots,
    structured_direct_condition,
)


class MultiObjectConditioningTest(unittest.TestCase):
    def _scene(self, object_count: int = 2) -> dict[str, torch.Tensor]:
        object_xyz = torch.zeros(object_count, 8, 3)
        object_xyz[..., 2] = 2.0
        object_normal = torch.zeros_like(object_xyz)
        object_normal[..., 2] = 1.0
        environment_xyz = torch.zeros(12, 3)
        environment_xyz[:, 2] = 3.0
        environment_normal = torch.zeros_like(environment_xyz)
        environment_normal[:, 2] = 1.0
        return {
            "object_xyz_camera_m": object_xyz,
            "object_normal_camera": object_normal,
            "environment_xyz_camera_m": environment_xyz,
            "environment_normal_camera": environment_normal,
            "environment_friction": torch.full((12, 1), 0.4),
            "environment_restitution": torch.full((12, 1), 0.1),
            "camera_intrinsics_normalized": torch.tensor([1.0, 1.0, 0.5, 0.5]),
        }

    def test_multi_object_condition_tokens_keep_object_axes(self):
        encoder = PhyContextConditionEncoder(
            hidden_dim=16,
            context_dim=32,
            scene_token_count=4,
            num_heads=4,
            num_layers=1,
        )
        controls = torch.tensor(
            [[0.5, 0.2, 0.1], [1.0, 0.8, 0.4]], dtype=torch.float32
        )
        dynamics = torch.zeros(2, 10)
        state = torch.zeros(2, 9)
        tokens = encoder(self._scene(), controls, dynamics, state)
        self.assertEqual(tokens.shape, (1, 4 + 2 * 3 + 2 * 10 + 2 * 3, 32))
        self.assertEqual(
            tuple(encoder.scene_encoder.object_slot_embedding.weight.shape), (3, 16)
        )

    def test_scene_encoder_rejects_more_than_three_objects(self):
        encoder = PhyContextConditionEncoder(
            hidden_dim=16,
            context_dim=32,
            scene_token_count=4,
            num_heads=4,
            num_layers=1,
        )
        with self.assertRaisesRegex(ValueError, "maximum is 3"):
            encoder(
                self._scene(object_count=4),
                torch.zeros(4, 3),
                torch.zeros(4, 10),
                torch.zeros(4, 9),
            )

    def test_single_object_shape_remains_compatible(self):
        encoder = PhyContextConditionEncoder(
            hidden_dim=16,
            context_dim=32,
            scene_token_count=4,
            num_heads=4,
            num_layers=1,
        )
        tokens = encoder(
            self._scene(object_count=1),
            torch.tensor([0.5, 0.2, 0.1]),
            torch.zeros(10),
            torch.zeros(9),
        )
        self.assertEqual(tokens.shape, (1, 4 + 3 + 10 + 3, 32))

    def test_direct_modulation_packs_multi_object_slots(self):
        condition = structured_direct_condition(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 9),
            object_slots=3,
        )
        self.assertEqual(condition.shape, (1, 36))
        self.assertTrue(torch.equal(condition[:, 24:], torch.zeros(1, 12)))

    def test_point_track_maps_enter_the_video_patch_conditioner(self):
        conditioner = TrajectoryPatchConditioner(
            hidden_dim=8,
            patch_size=(1, 2, 2),
            rank=4,
            representation="das_3d_tracks",
        )
        conditioner.set_condition([torch.zeros(12, 13, 4, 4)])
        conditioner.begin_forward(1)
        video = torch.zeros(1, 3, 4, 4, 4)
        patch_features = torch.zeros(1, 8, 4, 2, 2)
        output = conditioner.add_residual(video, patch_features)
        conditioner.end_forward()
        self.assertEqual(output.shape, patch_features.shape)

    def test_trajectory_conditioner_keeps_two_samples_isolated(self):
        conditioner = TrajectoryPatchConditioner(
            hidden_dim=8,
            patch_size=(1, 2, 2),
            rank=4,
            representation="das_3d_tracks",
        )
        with torch.no_grad():
            conditioner.patch_projection.weight.fill_(1.0)
            conditioner.output_projection.weight.fill_(1.0)
        zero_map = torch.zeros(12, 5, 4, 4)
        one_map = torch.ones_like(zero_map)
        video = torch.zeros(1, 3, 2, 4, 4)
        patch_features = torch.zeros(1, 8, 2, 2, 2)
        conditioner.set_condition([zero_map, one_map])
        conditioner.begin_forward(2)
        first = conditioner.add_residual(video, patch_features)
        second = conditioner.add_residual(video, patch_features)
        conditioner.end_forward()
        self.assertAlmostEqual(float(first.abs().sum().detach()), 0.0, places=6)
        self.assertGreater(float(second.abs().sum().detach()), 0.0)

    def test_point_tracks_supply_current_occupancy_before_envelope_expansion(self):
        point_track_map = torch.zeros(12, 5, 3, 4)
        point_track_map[3, 0, 1, 2] = 1.0
        point_track_map[3, 2, 2, 3] = 1.0
        mask = motion_mask_from_point_track_map(point_track_map)
        self.assertEqual(mask.shape, (1, 2, 3, 4))
        self.assertEqual(float(mask[0, 0, 1, 2]), 1.0)
        self.assertEqual(float(mask[0, 1, 1, 2]), 0.0)
        self.assertEqual(float(mask[0, 1, 2, 3]), 1.0)
        self.assertEqual(float(mask.sum()), 2.0)

        envelope = source_target_motion_envelope(mask)
        self.assertEqual(float(envelope[0, 1, 1, 2]), 1.0)
        self.assertEqual(float(envelope.sum()), 3.0)

        _, target_centers, _, _, valid = latent_correspondence_motion(
            torch.zeros(2, 2, 3, 4),
            torch.zeros(2, 2, 3, 4),
            mask,
        )
        self.assertTrue(bool(valid[1]))
        self.assertTrue(
            torch.allclose(target_centers[1], torch.tensor([1.0, 1.0]))
        )

    def test_das_condition_rejects_nonbinary_visibility_and_background_rgb(self):
        nonbinary = torch.zeros(12, 5, 3, 4)
        nonbinary[3, 0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "visibility must be binary"):
            motion_mask_from_point_track_map(nonbinary, "das_3d_tracks")

        background_rgb = torch.zeros(12, 5, 3, 4)
        background_rgb[0, 0, 0, 0] = 0.25
        with self.assertRaisesRegex(ValueError, "zero outside visible"):
            motion_mask_from_point_track_map(background_rgb, "das_3d_tracks")

    def test_dense_ablation_also_keeps_current_and_source_regions_separate(self):
        point_track_map = torch.zeros(18, 2, 2, 4)
        point_track_map[0, :, 0, 1] = 1.0
        point_track_map[1, 0, 0, 1] = 1.0
        point_track_map[1, 1, 1, 3] = 1.0
        mask = motion_mask_from_point_track_map(
            point_track_map, "dense_point_tracks"
        )
        self.assertEqual(float(mask[0, 1, 0, 1]), 0.0)
        self.assertEqual(float(mask[0, 1, 1, 3]), 1.0)
        envelope = source_target_motion_envelope(mask)
        self.assertEqual(float(envelope[0, 1, 0, 1]), 1.0)
        self.assertEqual(float(envelope[0, 1, 1, 3]), 1.0)

    def test_point_track_padding_must_match_object_count(self):
        point_map = torch.zeros(12, 5, 3, 4)
        point_map[7, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "unused.*slots"):
            validate_point_track_object_slots(
                point_map, "das_3d_tracks", object_count=1
            )
        validate_point_track_object_slots(
            point_map, "das_3d_tracks", object_count=2
        )


if __name__ == "__main__":
    unittest.main()
