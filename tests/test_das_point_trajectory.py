import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from point_trajectory import (  # noqa: E402
    _serialize_raster_consistent_track_coordinates,
    build_das_track_correspondence,
    rasterize_das_3d_tracks,
    validate_point_trajectory,
    cover_center_crop_coordinates,
    project_camera_points,
    unproject_physweep_tracks,
)


def point_payload(object_count: int = 1) -> dict[str, np.ndarray]:
    frame_count = 2
    point_count = 2048
    points = np.zeros((frame_count, object_count, point_count, 3), np.float32)
    points[..., 2] = 2.0
    tracks = np.zeros((frame_count, object_count, point_count, 2), np.float32)
    depth = points[..., 2].copy()
    valid = np.zeros((frame_count, object_count, point_count), bool)
    return {
        "time_s": np.arange(frame_count, dtype=np.float32),
        "object_ids": np.asarray([f"object_{index}" for index in range(object_count)]),
        "points_world_m": points.copy(),
        "points_camera_m": points,
        "tracks_xy_px": tracks,
        "depth_m": depth,
        "valid": valid,
        "initial_points_camera_m": points[0].copy(),
        "camera_from_world": np.repeat(np.eye(4)[None], frame_count, axis=0),
        "camera_intrinsics": np.eye(3),
        "image_size_px": np.asarray([8, 8]),
        "metadata_json": np.asarray("{}"),
    }


class DasPointTrajectoryTest(unittest.TestCase):
    def test_float32_correspondence_preserves_visible_raster_pixel(self) -> None:
        coordinates = np.asarray(
            [
                [
                    [
                        [np.nextafter(831.5, -np.inf), 184.4762420654297],
                        [20.25, np.nextafter(479.5, -np.inf)],
                    ]
                ]
            ],
            dtype=np.float64,
        )
        visible = np.ones(coordinates.shape[:-1], dtype=bool)

        serialized = _serialize_raster_consistent_track_coordinates(
            coordinates, visible
        )

        self.assertEqual(serialized.dtype, np.float32)
        self.assertLess(float(serialized[0, 0, 0, 0]), 831.5)
        self.assertLess(float(serialized[0, 0, 1, 1]), 479.5)
        np.testing.assert_array_equal(
            np.rint(serialized.astype(np.float64)),
            np.rint(coordinates),
        )

    def test_invisible_coordinates_are_not_semantically_adjusted(self) -> None:
        coordinates = np.asarray(
            [[[[np.nextafter(831.5, -np.inf), 2.0]]]], dtype=np.float64
        )
        visible = np.zeros(coordinates.shape[:-1], dtype=bool)

        serialized = _serialize_raster_consistent_track_coordinates(
            coordinates, visible
        )

        self.assertEqual(float(serialized[0, 0, 0, 0]), 831.5)

    def test_physweep_projection_round_trips_through_metric_depth(self) -> None:
        intrinsics = np.asarray(
            [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]],
            np.float32,
        )
        camera_points = np.asarray(
            [[[[1.0, 0.5, 2.0], [-0.4, -0.2, 4.0]]]], np.float32
        )
        tracks = np.stack(
            (
                intrinsics[0, 0] * camera_points[..., 0] / camera_points[..., 2]
                + intrinsics[0, 2],
                intrinsics[1, 2]
                - intrinsics[1, 1]
                * camera_points[..., 1]
                / camera_points[..., 2],
            ),
            axis=-1,
        )

        recovered = unproject_physweep_tracks(
            tracks, camera_points[..., 2], intrinsics
        )

        np.testing.assert_allclose(recovered, camera_points, rtol=1.0e-6, atol=1.0e-6)

    def test_serialized_projection_owns_the_image_boundary_contract(self) -> None:
        point_before_serialization = np.asarray(
            [[[[np.nextafter(1280.0, -np.inf), 0.0, 1.0]]]],
            dtype=np.float64,
        )
        self.assertLess(float(point_before_serialization[..., 0].item()), 1280.0)
        serialized_point = point_before_serialization.astype(np.float32)
        self.assertEqual(float(serialized_point[..., 0].item()), 1280.0)

        tracks, valid = project_camera_points(
            serialized_point,
            np.eye(3, dtype=np.float32),
            (1280, 720),
            0.03,
            100.0,
        )

        self.assertEqual(float(tracks[..., 0].item()), 1280.0)
        self.assertFalse(bool(valid.item()))

    def test_published_payload_geometry_and_projection_are_cross_checked(self) -> None:
        payload = point_payload()
        payload["points_world_m"][..., 2] = 2.0
        payload["points_camera_m"][..., 2] = 2.0
        payload["initial_points_camera_m"] = payload["points_camera_m"][0].copy()
        payload["depth_m"] = payload["points_camera_m"][..., 2].copy()
        payload["camera_from_world"][:, 1, 1] = -1.0
        payload["camera_intrinsics"] = np.asarray(
            [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
            np.float32,
        )
        payload["tracks_xy_px"][...] = (4.0, 4.0)
        payload["valid"][...] = True
        payload["metadata_json"] = np.asarray(
            '{"schema":"physweep.point_trajectories.v1",'
            '"point_count":2048,"object_count":1,"object_ids":["object_0"],'
            '"coordinate_frame_world":"pybullet_world_xyz",'
            '"coordinate_frame_camera":"camera_right_up_forward",'
            '"track_definition":"perspective_projection_of_fixed_material_points",'
            '"visibility_definition":"in_frame_and_clip_validity;_not_a_z_buffer",'
            '"clip_start_m":0.03,"clip_end_m":100.0}'
        )
        validate_point_trajectory(payload)

        corrupted = dict(payload)
        corrupted["tracks_xy_px"] = payload["tracks_xy_px"].copy()
        corrupted["tracks_xy_px"][0, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "camera projection"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["points_world_m"] = payload["points_world_m"].copy()
        corrupted["points_world_m"][0, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "camera_from_world"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["valid"] = payload["valid"].copy()
        corrupted["valid"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "in-frame and clip validity"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["metadata_json"] = np.asarray(
            str(payload["metadata_json"].item()).replace(
                '"object_count":1', '"object_count":2'
            )
        )
        with self.assertRaisesRegex(ValueError, "object count"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["camera_from_world"] = np.repeat(
            np.eye(4, dtype=np.float32)[None], 2, axis=0
        )
        with self.assertRaisesRegex(ValueError, "handedness"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["camera_from_world"] = payload["camera_from_world"].copy()
        corrupted["camera_from_world"][0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["camera_intrinsics"] = payload["camera_intrinsics"].copy()
        corrupted["camera_intrinsics"][0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive focal lengths"):
            validate_point_trajectory(corrupted)

        corrupted = dict(payload)
        corrupted["valid"] = payload["valid"].astype(np.uint8)
        corrupted["valid"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary values"):
            validate_point_trajectory(corrupted)

    def test_static_camera_matrix_is_broadcast_across_frames(self) -> None:
        payload = point_payload()
        payload["camera_from_world"] = np.eye(4)
        payload["initial_points_camera_m"][0, 0] = (2.0, 2.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (2.0, 2.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (1.0, 1.0)
        rendered = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray([[1.0, -1.0, 1.0]], np.float32),
        )
        self.assertEqual(float(rendered[3].sum()), 0.0)

    def test_static_projection_uses_physweep_right_up_forward_convention(self) -> None:
        payload = point_payload()
        payload["camera_intrinsics"] = np.asarray(
            [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
            np.float32,
        )
        payload["initial_points_camera_m"][0, 0] = (2.0, 2.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (2.0, 2.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (5.0, 3.0)

        rendered = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray([[1.0, 1.0, 1.0]], np.float32),
        )

        self.assertEqual(float(rendered[3].sum()), 0.0)

    def test_resize_coordinates_use_opencv_half_pixel_centers(self) -> None:
        transformed = cover_center_crop_coordinates(
            np.asarray([[0.0, 0.0], [3.0, 3.0]], np.float32),
            (4, 4),
            (8, 8),
        )
        np.testing.assert_allclose(
            transformed,
            np.asarray([[0.5, 0.5], [6.5, 6.5]], np.float32),
        )

    def test_resize_keeps_subpixel_side_of_half_integer(self) -> None:
        transformed = cover_center_crop_coordinates(
            np.asarray([[205.0803, 0.0], [908.85516, 0.0]], np.float32),
            (1280, 720),
            (832, 480),
        )
        self.assertEqual(transformed.dtype, np.float64)
        self.assertEqual(int(np.rint(transformed[0, 0])), 127)
        self.assertEqual(int(np.rint(transformed[1, 0])), 595)

    def test_validation_rejects_depth_and_metadata_semantic_mismatches(self) -> None:
        payload = point_payload()
        payload["depth_m"][0, 0, 0] = 9.0
        with self.assertRaisesRegex(ValueError, "camera-space z"):
            validate_point_trajectory(payload)

        payload = point_payload()
        payload["metadata_json"] = np.asarray(
            '{"schema":"physweep.point_trajectories.v0"}'
        )
        with self.assertRaisesRegex(ValueError, "unsupported trajectory schema"):
            validate_point_trajectory(payload)

        payload = point_payload()
        payload["metadata_json"] = np.asarray(
            '{"coordinate_frame_camera":"camera_right_down_forward"}'
        )
        with self.assertRaisesRegex(ValueError, "unsupported camera coordinate frame"):
            validate_point_trajectory(payload)

    def test_point_identity_color_is_fixed_while_projection_moves(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 2.0)
        payload["initial_points_camera_m"][0, 1] = (1.0, 0.5, 4.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 2.0)
        payload["points_camera_m"][:, 0, 1] = (1.0, 0.5, 4.0)
        payload["depth_m"][:, 0, :2] = (2.0, 4.0)
        payload["valid"][:, 0, :2] = True
        payload["tracks_xy_px"][0, 0, :2] = ((2.0, 3.0), (5.0, 5.0))
        payload["tracks_xy_px"][1, 0, :2] = ((3.0, 3.0), (6.0, 5.0))

        rendered = rasterize_das_3d_tracks(
            payload, (8, 8), preprocess_size_px=(8, 8)
        )

        self.assertEqual(rendered.shape, (12, 2, 8, 8))
        np.testing.assert_allclose(rendered[:3, 0, 3, 2], rendered[:3, 1, 3, 3])
        np.testing.assert_allclose(rendered[:3, 0, 5, 5], rendered[:3, 1, 5, 6])
        self.assertEqual(float(rendered[3, 1, 3, 3]), 1.0)

    def test_positive_depth_point_can_enter_view_after_frame_zero(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["tracks_xy_px"][0, 0, 0] = (-2.0, 3.0)
        payload["tracks_xy_px"][1, 0, 0] = (3.0, 3.0)
        payload["valid"][1, 0, 0] = True

        rendered = rasterize_das_3d_tracks(
            payload, (8, 8), preprocess_size_px=(8, 8)
        )

        self.assertEqual(float(rendered[3, 0].sum()), 0.0)
        self.assertGreater(float(rendered[3, 1].sum()), 0.0)

    def test_global_zbuffer_keeps_only_nearest_object_visible(self) -> None:
        payload = point_payload(object_count=2)
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 3.0)
        payload["initial_points_camera_m"][1, 0] = (1.0, 0.0, 1.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 3.0)
        payload["points_camera_m"][:, 1, 0] = (1.0, 0.0, 1.0)
        payload["depth_m"][:, 0, 0] = 3.0
        payload["depth_m"][:, 1, 0] = 1.0
        payload["valid"][:, :, 0] = True
        payload["tracks_xy_px"][:, :, 0] = (4.0, 4.0)

        rendered = rasterize_das_3d_tracks(
            payload, (8, 8), preprocess_size_px=(8, 8)
        )

        self.assertEqual(float(rendered[3, :, 4, 4].sum()), 0.0)
        self.assertEqual(rendered[7, :, 4, 4].tolist(), [1.0, 1.0])

    def test_condition_and_correspondence_agree_on_same_pixel_depth(self) -> None:
        payload = point_payload(object_count=2)
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 3.0)
        payload["initial_points_camera_m"][1, 0] = (1.0, 0.0, 1.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 3.0)
        payload["points_camera_m"][:, 1, 0] = (1.0, 0.0, 1.0)
        payload["depth_m"][:, 0, 0] = 3.0
        payload["depth_m"][:, 1, 0] = 1.0
        payload["valid"][:, :, 0] = True
        payload["tracks_xy_px"][:, :, 0] = (4.0, 4.0)

        rendered, correspondence = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            return_correspondence=True,
        )

        self.assertFalse(bool(correspondence["track_visible"][:, 0, 0].any()))
        self.assertTrue(bool(correspondence["track_visible"][:, 1, 0].all()))
        self.assertEqual(float(rendered[3].sum()), 0.0)
        self.assertGreater(float(rendered[7].sum()), 0.0)

    def test_control_splat_does_not_hide_adjacent_correspondence(self) -> None:
        payload = point_payload(object_count=2)
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 3.0)
        payload["initial_points_camera_m"][1, 0] = (1.0, 0.0, 1.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 3.0)
        payload["points_camera_m"][:, 1, 0] = (1.0, 0.0, 1.0)
        payload["depth_m"][:, 0, 0] = 3.0
        payload["depth_m"][:, 1, 0] = 1.0
        payload["valid"][:, :, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (4.0, 4.0)
        payload["tracks_xy_px"][:, 1, 0] = (4.0, 3.0)

        rendered, correspondence = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            return_correspondence=True,
        )
        built = build_das_track_correspondence(
            payload,
            preprocess_size_px=(8, 8),
        )

        # The near point's 3x3 splat wins at the far point's adjacent center in
        # the coverage map, but the zero-radius correspondence z-buffer keeps
        # both material points because they occupy distinct projected pixels.
        self.assertGreater(float(rendered[3].sum()), 0.0)
        self.assertEqual(float(rendered[3, :, 4, 4].sum()), 0.0)
        self.assertTrue(bool(correspondence["track_visible"][:, 0, 0].all()))
        self.assertTrue(bool(correspondence["track_visible"][:, 1, 0].all()))
        np.testing.assert_array_equal(
            built["track_visible"], correspondence["track_visible"]
        )

    def test_full_resolution_visibility_does_not_merge_distinct_pixels(self) -> None:
        payload = point_payload(object_count=2)
        payload["image_size_px"] = np.asarray([16, 16])
        payload["initial_points_camera_m"][0, 0] = (-1.0, 0.0, 1.0)
        payload["initial_points_camera_m"][1, 0] = (1.0, 0.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (-1.0, 0.0, 1.0)
        payload["points_camera_m"][:, 1, 0] = (1.0, 0.0, 2.0)
        payload["depth_m"][:, 0, 0] = 1.0
        payload["depth_m"][:, 1, 0] = 2.0
        payload["valid"][:, :, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (1.0, 1.0)
        payload["tracks_xy_px"][:, 1, 0] = (7.0, 7.0)

        rendered = rasterize_das_3d_tracks(
            payload, (1, 1), preprocess_size_px=(16, 16)
        )

        self.assertEqual(float(rendered[3, :, 0, 0].sum()), 2.0)
        self.assertEqual(float(rendered[7, :, 0, 0].sum()), 2.0)

    def test_static_scene_points_occlude_dynamic_tracks(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (8.0, 8.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (8.0, 8.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (4.0, 4.0)

        rendered, correspondence = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray([[4.0, -4.0, 1.0]], np.float32),
            return_correspondence=True,
        )

        self.assertEqual(float(rendered[3].sum()), 0.0)
        self.assertFalse(bool(correspondence["track_visible"][:, 0, 0].any()))

    def test_static_control_splat_does_not_hide_adjacent_correspondence(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (8.0, 8.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (8.0, 8.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (4.0, 4.0)

        rendered, correspondence = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray([[4.0, -3.0, 1.0]], np.float32),
            return_correspondence=True,
        )

        self.assertEqual(float(rendered[3, :, 4, 4].sum()), 0.0)
        self.assertTrue(bool(correspondence["track_visible"][:, 0, 0].all()))

    def test_static_scene_points_respect_camera_clip_range(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (8.0, 8.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (8.0, 8.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][:, 0, 0] = (4.0, 4.0)

        rendered = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray(
                [[800.0, -800.0, 200.0]], np.float32
            ),
        )

        self.assertGreater(float(rendered[3].sum()), 0.0)

    def test_static_scene_occlusion_follows_camera_motion(self) -> None:
        payload = point_payload()
        payload["initial_points_camera_m"][0, 0] = (2.0, 2.0, 2.0)
        payload["points_camera_m"][:, 0, 0] = (2.0, 2.0, 2.0)
        payload["depth_m"][:, 0, 0] = 2.0
        payload["valid"][:, 0, 0] = True
        payload["tracks_xy_px"][0, 0, 0] = (1.0, 1.0)
        payload["tracks_xy_px"][1, 0, 0] = (3.0, 4.0)
        payload["camera_from_world"][1, 0, 3] = -1.0

        rendered = rasterize_das_3d_tracks(
            payload,
            (8, 8),
            preprocess_size_px=(8, 8),
            static_points_camera0_m=np.asarray([[4.0, -4.0, 1.0]], np.float32),
        )

        self.assertGreater(float(rendered[3, 0].sum()), 0.0)
        self.assertEqual(float(rendered[3, 1].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
