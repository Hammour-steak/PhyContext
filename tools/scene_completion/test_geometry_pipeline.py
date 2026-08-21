from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from decomposition import decompose_scene
from depthlab_frontend import merge_depthlab_prediction, prepare_depthlab_input
from environment_completion import _depth_seam_diagnostics, complete_environment
from infer_first_frame import _load_binary_mask, _mask_iou
from instantmesh_frontend import run_instantmesh
from lama_frontend import build_inpainting_mask, run_lama_inpainting
from registration import (
    _build_collision_proxy,
    _resolve_support_relation,
    _estimate_metric_scale,
    _initialize_metric_pose,
)
from segmentation import segmentation_from_mask
from vggt_frontend import preprocess_binary_mask_for_vggt_pad


class GeometryPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 96
        self.width = 128
        self.intrinsic = np.asarray(
            [[110.0, 0.0, 63.5], [0.0, 110.0, 47.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.extrinsic = np.hstack(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]
        )
        pixels_x, pixels_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        rays = np.stack(
            [
                (pixels_x - self.intrinsic[0, 2]) / self.intrinsic[0, 0],
                (pixels_y - self.intrinsic[1, 2]) / self.intrinsic[1, 1],
                np.ones_like(pixels_x),
            ],
            axis=-1,
        ).astype(np.float32)
        self.object_mask = np.zeros((self.height, self.width), dtype=bool)
        self.object_mask[34:68, 50:82] = True
        self.depth = np.full((self.height, self.width), 4.0, dtype=np.float32)
        self.depth[self.object_mask] = 2.5
        self.points = rays * self.depth[..., None]
        self.colors = np.full((self.height, self.width, 3), [135, 145, 155], dtype=np.uint8)
        self.colors[self.object_mask] = [210, 55, 65]
        self.confidence = np.ones((self.height, self.width), dtype=np.float32)

    def test_object_is_removed_before_environment_completion(self) -> None:
        decomposition = decompose_scene(
            self.points,
            self.colors,
            self.depth,
            self.confidence,
            self.object_mask,
            confidence_quantile=0.0,
            content_mask=np.ones_like(self.object_mask),
        )
        self.assertGreater(len(decomposition.object_points), 500)
        self.assertGreater(len(decomposition.environment_points), 5000)
        self.assertTrue(np.all(decomposition.object_points[:, 2] < 3.0))
        self.assertTrue(np.all(decomposition.environment_points[:, 2] > 3.5))

        inpainted_rgb = self.colors.copy()
        inpainted_rgb[self.object_mask] = [135, 145, 155]
        completion = complete_environment(
            self.points,
            inpainted_rgb,
            self.depth,
            decomposition.environment_mask,
            decomposition.reliable_environment_mask,
            self.object_mask,
            np.ones_like(self.object_mask),
            np.full_like(self.depth, 4.0),
            np.ones_like(self.object_mask),
            self.intrinsic,
            self.extrinsic,
        )
        self.assertGreater(len(completion.points), 700)
        self.assertGreater(float(completion.pixel_mask[self.object_mask].mean()), 0.75)
        np.testing.assert_allclose(np.median(completion.points[:, 2]), 4.0, atol=0.03)
        self.assertEqual(
            completion.diagnostics["observed_geometry_policy"],
            "immutable_original_vggt_points",
        )
        self.assertGreater(completion.diagnostics["content_surface_coverage"], 0.99)
        repeated = complete_environment(
            self.points,
            inpainted_rgb,
            self.depth,
            decomposition.environment_mask,
            decomposition.reliable_environment_mask,
            self.object_mask,
            np.ones_like(self.object_mask),
            np.full_like(self.depth, 4.0),
            np.ones_like(self.object_mask),
            self.intrinsic,
            self.extrinsic,
        )
        np.testing.assert_allclose(completion.points, repeated.points)
        self.assertEqual(completion.planes, repeated.planes)

    def test_vggt_padding_is_excluded_from_scene_geometry(self) -> None:
        content_mask = np.zeros((self.height, self.width), dtype=bool)
        content_mask[12:84] = True
        decomposition = decompose_scene(
            self.points,
            self.colors,
            self.depth,
            self.confidence,
            self.object_mask,
            confidence_quantile=0.0,
            content_mask=content_mask,
        )
        self.assertFalse(decomposition.valid_mask[:12].any())
        self.assertFalse(decomposition.valid_mask[84:].any())
        self.assertFalse(decomposition.environment_mask[~content_mask].any())

    def test_depthlab_input_does_not_include_vggt_padding(self) -> None:
        content_mask = np.zeros((self.height, self.width), dtype=bool)
        content_mask[12:84] = True
        prepared = prepare_depthlab_input(
            self.colors,
            self.depth,
            self.object_mask,
            content_mask,
        )
        self.assertEqual(prepared.rgb.shape, (72, self.width, 3))
        self.assertEqual(prepared.known_depth.shape, (72, self.width))
        self.assertEqual(prepared.bounds_yxyx, (12, 84, 0, self.width))
        np.testing.assert_array_equal(prepared.rgb, self.colors[12:84])
        self.assertTrue(np.all(prepared.known_depth[prepared.completion_mask] == 0))

    def test_depthlab_merge_locks_every_known_depth_exactly(self) -> None:
        depth = self.depth.copy()
        depth[20, 20] = np.nan
        prepared = prepare_depthlab_input(
            self.colors,
            depth,
            self.object_mask,
            np.ones_like(self.object_mask),
        )
        self.assertTrue(prepared.completion_mask[20, 20])
        prediction = np.full_like(prepared.known_depth, 4.25)
        fused, completion_mask = merge_depthlab_prediction(
            depth,
            prediction,
            prepared.completion_mask,
            prepared.bounds_yxyx,
        )
        np.testing.assert_array_equal(fused[~completion_mask], depth[~completion_mask])
        self.assertTrue(np.all(fused[completion_mask] == 4.25))

    def test_object_uses_its_own_confidence_distribution(self) -> None:
        confidence = np.ones_like(self.confidence)
        confidence[self.object_mask] = 0.01
        decomposition = decompose_scene(
            self.points,
            self.colors,
            self.depth,
            confidence,
            self.object_mask,
            confidence_quantile=0.25,
            content_mask=np.ones_like(self.object_mask),
        )
        self.assertGreater(len(decomposition.object_points), 500)
        self.assertAlmostEqual(decomposition.confidence_threshold, 1.0)
        self.assertAlmostEqual(decomposition.object_confidence_threshold, 0.01)

    def test_environment_completion_never_replaces_observed_pixels(self) -> None:
        content_mask = np.ones_like(self.object_mask)
        decomposition = decompose_scene(
            self.points,
            self.colors,
            self.depth,
            self.confidence,
            self.object_mask,
            confidence_quantile=0.0,
            content_mask=content_mask,
        )
        completion = complete_environment(
            self.points,
            self.colors,
            self.depth,
            decomposition.environment_mask,
            decomposition.reliable_environment_mask,
            self.object_mask,
            content_mask,
            np.full_like(self.depth, 4.0),
            np.ones_like(self.object_mask),
            self.intrinsic,
            self.extrinsic,
        )
        self.assertFalse(
            (completion.pixel_mask & decomposition.environment_mask).any()
        )
        np.testing.assert_array_equal(
            completion.dense_mask,
            completion.pixel_mask | decomposition.environment_mask,
        )

    def test_environment_completion_preserves_depthlab_metric_surfaces(self) -> None:
        reference_depth = self.depth.copy()
        reference_depth[:, self.width // 2 :] *= 1.5
        points = self.points.copy()
        points[:, self.width // 2 :] *= 1.5
        decomposition = decompose_scene(
            points,
            self.colors,
            reference_depth,
            self.confidence,
            self.object_mask,
            confidence_quantile=0.0,
            content_mask=np.ones_like(self.object_mask),
        )
        columns = np.indices(self.object_mask.shape)[1]
        completion_depth = np.where(
            columns < self.width // 2,
            4.0,
            6.0,
        ).astype(np.float32)
        completion = complete_environment(
            points,
            self.colors,
            reference_depth,
            decomposition.environment_mask,
            decomposition.reliable_environment_mask,
            self.object_mask,
            np.ones_like(self.object_mask),
            completion_depth,
            np.ones_like(self.object_mask),
            self.intrinsic,
            self.extrinsic,
        )
        left = completion.pixel_mask & (
            columns < self.width // 2
        )
        right = completion.pixel_mask & ~left
        self.assertAlmostEqual(float(np.median(completion.depth[left])), 4.0, delta=0.05)
        self.assertAlmostEqual(float(np.median(completion.depth[right])), 6.0, delta=0.05)
        self.assertLess(completion.diagnostics["depth_seam"]["p90_relative_error"], 0.05)

    def test_depth_seam_check_handles_foreground_occlusion_edges(self) -> None:
        completed_mask = np.zeros_like(self.object_mask)
        completed_mask[38:54, 52:68] = True
        observed = ~completed_mask
        reference_depth = np.full_like(self.depth, 6.0)
        reference_depth[54:58, 52:68] = 4.0
        completed_depth = np.zeros_like(self.depth)
        completed_depth[completed_mask] = 6.0
        diagnostics = _depth_seam_diagnostics(
            completed_depth,
            reference_depth,
            completed_mask,
            observed,
        )
        self.assertEqual(
            diagnostics["method"],
            "minimum_relative_error_over_adjacent_observed_surfaces",
        )
        self.assertLess(diagnostics["p90_relative_error"], 0.01)

    def test_provided_object_mask_is_preserved_exactly(self) -> None:
        result = segmentation_from_mask(self.colors, self.object_mask)
        np.testing.assert_array_equal(result.mask, self.object_mask)
        self.assertEqual(result.selected_index, -1)
        self.assertEqual(_mask_iou(result.mask, self.object_mask), 1.0)

    def test_lama_mask_expansion_is_object_relative_and_inside_content(self) -> None:
        content_mask = np.zeros_like(self.object_mask)
        content_mask[8:88] = True
        expanded, pixels = build_inpainting_mask(
            self.object_mask,
            content_mask,
            expansion_ratio=0.15,
        )
        self.assertEqual(pixels, 5)
        self.assertTrue(np.all(expanded <= content_mask))
        self.assertGreater(expanded.sum(), self.object_mask.sum())

    def test_lama_frontend_preserves_every_unmasked_pixel(self) -> None:
        class ConstantFill(torch.nn.Module):
            def forward(self, image, mask):
                return image * (1.0 - mask) + 0.5 * mask

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dummy_lama.pt"
            example_image = torch.zeros((1, 3, 96, 128), dtype=torch.float32)
            example_mask = torch.zeros((1, 1, 96, 128), dtype=torch.float32)
            torch.jit.trace(ConstantFill(), (example_image, example_mask)).save(
                str(checkpoint)
            )
            result = run_lama_inpainting(
                self.colors,
                self.object_mask,
                np.ones_like(self.object_mask),
                checkpoint,
                torch.device("cpu"),
                expansion_ratio=0.0,
            )
            np.testing.assert_array_equal(
                result.rgb[~result.inpaint_mask],
                self.colors[~result.inpaint_mask],
            )
            self.assertTrue(np.all(result.rgb[result.inpaint_mask] == 127))

    def test_vggt_pad_transform_preserves_widescreen_mask_geometry(self) -> None:
        source = np.zeros((720, 1280), dtype=np.uint8)
        cv2.circle(source, (960, 360), 90, 1, thickness=-1)
        transformed = preprocess_binary_mask_for_vggt_pad(
            source,
            source.shape,
            (518, 518),
        )
        rows, columns = np.nonzero(transformed)
        width = int(columns.max() - columns.min() + 1)
        height = int(rows.max() - rows.min() + 1)
        self.assertAlmostEqual(width / height, 1.0, delta=0.04)
        self.assertAlmostEqual(float(columns.mean()), 388.5, delta=1.0)
        self.assertAlmostEqual(float(rows.mean()), 259.0, delta=1.0)
        self.assertFalse(transformed[:112].any())
        self.assertFalse(transformed[406:].any())

    def test_mask_iou_rejects_mismatched_preprocessing(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask shapes must match"):
            _mask_iou(np.zeros((32, 32), dtype=bool), np.zeros((18, 32), dtype=bool))

    def test_vggt_depth_recovers_metric_pnp_scale(self) -> None:
        object_points = np.asarray(
            [
                [-0.4, -0.3, -0.2],
                [0.4, -0.3, -0.2],
                [-0.4, 0.3, -0.2],
                [0.4, 0.3, -0.2],
                [-0.3, -0.2, 0.2],
                [0.3, -0.2, 0.2],
                [-0.3, 0.2, 0.2],
                [0.3, 0.2, 0.2],
            ],
            dtype=np.float32,
        )
        rotation, _ = cv2.Rodrigues(np.asarray([0.12, -0.18, 0.04], dtype=np.float32))
        translation = np.asarray([0.05, -0.03, 2.1], dtype=np.float32)
        expected_scale = 1.7
        camera_unscaled = object_points @ rotation.T + translation
        camera_metric = camera_unscaled * expected_scale
        image_points = np.column_stack(
            [
                self.intrinsic[0, 0] * camera_metric[:, 0] / camera_metric[:, 2]
                + self.intrinsic[0, 2],
                self.intrinsic[1, 1] * camera_metric[:, 1] / camera_metric[:, 2]
                + self.intrinsic[1, 2],
            ]
        ).astype(np.float32)
        point_map = np.full((self.height, self.width, 3), np.nan, dtype=np.float32)
        mask = np.zeros((self.height, self.width), dtype=bool)
        rounded = np.rint(image_points).astype(np.int64)
        for pixel, point in zip(rounded, camera_metric):
            point_map[pixel[1], pixel[0]] = point
            mask[pixel[1], pixel[0]] = True
        scale, count = _estimate_metric_scale(
            object_points,
            image_points,
            rotation,
            translation,
            point_map,
            self.extrinsic,
            mask,
        )
        self.assertGreaterEqual(count, 4)
        self.assertAlmostEqual(scale, expected_scale, places=4)

    def test_metric_pose_matches_depth_and_silhouette_extent(self) -> None:
        vertices = np.asarray(
            [[-1.0, -0.8, 0.0], [1.0, -0.8, 0.0], [1.0, 0.8, 0.0], [-1.0, 0.8, 0.0]],
            dtype=np.float32,
        )
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[36:60, 49:79] = True
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        depth[mask] = 4.0
        scale, translation, _ = _initialize_metric_pose(
            vertices,
            np.eye(3, dtype=np.float32),
            mask,
            depth,
            self.intrinsic,
        )
        transformed = scale * vertices + translation
        projected = np.column_stack(
            [
                self.intrinsic[0, 0] * transformed[:, 0] / transformed[:, 2]
                + self.intrinsic[0, 2],
                self.intrinsic[1, 1] * transformed[:, 1] / transformed[:, 2]
                + self.intrinsic[1, 2],
            ]
        )
        self.assertAlmostEqual(float(np.ptp(projected[:, 0])), 29.0, delta=1.5)
        self.assertAlmostEqual(float(np.ptp(projected[:, 1])), 23.0, delta=1.5)
        self.assertAlmostEqual(float(np.mean(projected[:, 0])), 63.5, delta=0.75)
        self.assertAlmostEqual(float(np.mean(projected[:, 1])), 47.5, delta=0.75)
        self.assertAlmostEqual(float(np.median(transformed[:, 2])), 4.0, places=4)

    def test_reused_mesh_requires_recorded_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "object_mesh.obj"
            mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "without provenance"):
                run_instantmesh(
                    crop_path=root / "missing_crop.png",
                    output_dir=root,
                    repository=root / "missing_repository",
                    python_executable=root / "missing_python",
                    checkpoint_cache=root / "cache",
                    device_index=0,
                    seed=17,
                    diffusion_steps=99,
                    reuse_existing=True,
                )

    def test_rgb_image_is_rejected_as_gt_mask_during_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_a_mask.png"
            rgba = np.zeros((12, 16, 4), dtype=np.uint8)
            rgba[..., 0] = np.arange(16, dtype=np.uint8)
            rgba[..., 3] = 255
            cv2.imwrite(str(path), rgba)
            with self.assertRaisesRegex(ValueError, "RGB image"):
                _load_binary_mask(path)

    def test_visual_pose_is_not_changed_by_support_analysis(self) -> None:
        vertices = np.asarray(
            [
                [x, y, z]
                for x in (-0.5, 0.5)
                for y in (-0.5, 0.5)
                for z in (-0.5, 0.5)
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[2, 3] = 0.45
        original = transform.copy()
        diagnostics = _resolve_support_relation(
            vertices,
            transform,
            [{"index": 0, "equation": [0.0, 0.0, 1.0, 0.0]}],
        )
        self.assertTrue(diagnostics["resolved"])
        self.assertFalse(diagnostics["visual_transform_modified"])
        np.testing.assert_array_equal(transform, original)

    def test_plane_through_object_bulk_is_not_silently_corrected(self) -> None:
        vertices = np.asarray(
            [
                [x, y, z]
                for x in (-0.5, 0.5)
                for y in (-0.5, 0.5)
                for z in (-0.5, 0.5)
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)
        diagnostics = _resolve_support_relation(
            vertices,
            transform,
            [{"index": 0, "equation": [0.0, 0.0, 1.0, 0.0]}],
        )
        self.assertFalse(diagnostics["resolved"])
        self.assertEqual(diagnostics["status"], "support_plane_intersects_object_bulk")

    def test_collision_proxy_is_rigidly_shifted_to_support(self) -> None:
        vertices = np.zeros((100, 3), dtype=np.float32)
        vertices[:, 0] = np.linspace(-0.5, 0.5, 100)
        vertices[:, 2] = 0.1
        vertices[:5, 2] = -0.04
        proxy, diagnostics = _build_collision_proxy(
            vertices,
            np.eye(4, dtype=np.float32),
            {
                "oriented_plane_equation": [0.0, 0.0, 1.0, 0.0],
                "tolerance": 0.002,
                "plane_index": 0,
            },
        )
        self.assertTrue(diagnostics["safe"])
        self.assertEqual(diagnostics["status"], "support_translated")
        self.assertGreaterEqual(float(proxy[:, 2].min()), 0.002 - 1e-6)
        np.testing.assert_allclose(np.ptp(proxy, axis=0), np.ptp(vertices, axis=0))


if __name__ == "__main__":
    unittest.main()
