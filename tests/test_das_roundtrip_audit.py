import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from audit_das_roundtrip import (  # noqa: E402
    load_cached_das_map,
    load_nontrivial_alpha_mask,
)
from audit_point_track_condition import summarize_object  # noqa: E402


class DasRoundtripAuditTest(unittest.TestCase):
    def test_cached_map_loader_accepts_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_track.safetensors"
            expected = torch.zeros(12, 97, 30, 52, dtype=torch.float32)
            save_file({"point_track_map": expected}, path)

            actual = load_cached_das_map(path)

            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.dtype, np.float32)

    def test_cached_map_loader_rejects_extra_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_track.safetensors"
            save_file(
                {
                    "point_track_map": torch.zeros(1, dtype=torch.float32),
                    "unexpected": torch.zeros(1, dtype=torch.float32),
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "only point_track_map"):
                load_cached_das_map(path)

    def test_mask_loader_accepts_nontrivial_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            rgba[2:6, 3:7, 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(path)

            alpha = load_nontrivial_alpha_mask(path)

            self.assertEqual(alpha.getextrema(), (0, 255))

    def test_mask_loader_rejects_opaque_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(path)

            with self.assertRaisesRegex(ValueError, "nonempty and non-full"):
                load_nontrivial_alpha_mask(path)

    def test_mask_loader_rejects_rgb_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(
                path
            )

            with self.assertRaisesRegex(ValueError, "no alpha channel"):
                load_nontrivial_alpha_mask(path)

    def test_das_audit_does_not_report_identity_rgb_as_displacement(self) -> None:
        point_map = np.zeros((12, 2, 2, 2), dtype=np.float32)
        point_map[0:3, 0, 0, 0] = np.asarray([0.2, 0.4, 0.6])
        point_map[3, 0, 0, 0] = 1.0
        point_map[0:3, 1, 0, 1] = np.asarray([0.2, 0.4, 0.6])
        point_map[3, 1, 0, 1] = 1.0

        result = summarize_object(point_map, 0, "das_3d_tracks")

        self.assertTrue(result["slot_active"])
        self.assertNotIn("displacement_step_abs_mean", result)
        self.assertAlmostEqual(result["identity_rgb_min"], 0.2, places=6)
        self.assertAlmostEqual(result["identity_rgb_max"], 0.6, places=6)

    def test_empty_das_slot_is_stable_and_inactive(self) -> None:
        result = summarize_object(
            np.zeros((12, 3, 2, 2), dtype=np.float32),
            1,
            "das_3d_tracks",
        )

        self.assertFalse(result["slot_active"])
        self.assertEqual(result["target_iou_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
