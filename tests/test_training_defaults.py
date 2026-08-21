from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

import cache_wan_inputs  # noqa: E402
import infer_wan_conditioned  # noqa: E402
import train_wan_formal  # noqa: E402
from project_defaults import (  # noqa: E402
    CACHE_MANIFEST,
    DATASET_MANIFEST,
    POINT_TRAJECTORY_MANIFEST,
    SCENE_TOKEN_COUNT,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


class TrainingDefaultTests(unittest.TestCase):
    def test_cache_defaults_are_formal_high_resolution_inputs(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["cache_wan_inputs.py", "--dataset-root", "/tmp/physweep"],
        ):
            args = cache_wan_inputs.parse_args()
        self.assertEqual((args.width, args.height, args.frames), (832, 480, 97))
        self.assertEqual(args.manifest, DATASET_MANIFEST)
        self.assertEqual(args.point_trajectory_manifest, POINT_TRAJECTORY_MANIFEST)

    def test_cache_requires_an_external_dataset_root(self) -> None:
        with patch.object(sys, "argv", ["cache_wan_inputs.py"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cache_wan_inputs.parse_args()

    def test_formal_training_defaults_cannot_drop_point_tracks_silently(self) -> None:
        with patch.object(sys, "argv", ["train_wan_formal.py"]):
            args = train_wan_formal.parse_args()
        self.assertEqual(args.cache_manifest, CACHE_MANIFEST)
        self.assertEqual(args.scene_tokens, SCENE_TOKEN_COUNT)
        self.assertTrue(args.trajectory_input)
        self.assertEqual(args.trajectory_input_source, "target")
        self.assertEqual(args.trajectory_representation, "dense_point_tracks")
        self.assertEqual(
            (VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMES), (832, 480, 97)
        )

    def test_resume_rejects_changed_optimization_contract(self) -> None:
        with patch.object(sys, "argv", ["train_wan_formal.py"]):
            args = train_wan_formal.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.json"
            cache.write_text("{}", encoding="utf-8")
            checkpoint = root / "run" / "latest"
            checkpoint.mkdir(parents=True)
            arguments = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            (checkpoint.parent / "run_contract.json").write_text(
                json.dumps(
                    {
                        "cache_manifest_sha256": train_wan_formal.sha256(cache),
                        "arguments": arguments,
                    }
                ),
                encoding="utf-8",
            )
            train_wan_formal.validate_resume_contract(
                root, Path("run/latest"), cache, args
            )
            args.reconstruction_loss_weight += 1.0
            with self.assertRaisesRegex(ValueError, "run contract"):
                train_wan_formal.validate_resume_contract(
                    root, Path("run/latest"), cache, args
                )

    def test_dataset_override_cannot_reuse_the_original_trajectory(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["infer_wan_conditioned.py", "--contact-friction", "0.4"],
        ):
            args = infer_wan_conditioned.parse_args()
        with self.assertRaisesRegex(ValueError, "trajectory recomputed"):
            infer_wan_conditioned.validate_parameter_trajectory_consistency(
                args, external=False, trajectory_requested=True
            )
        infer_wan_conditioned.validate_parameter_trajectory_consistency(
            args, external=False, trajectory_requested=False
        )

    def test_checkpoint_input_contract_records_high_resolution_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
            cache = root / "cache.json"
            cache.write_text(
                json.dumps(
                    {"preprocess": {"width": 832, "height": 480, "frames": 97}}
                ),
                encoding="utf-8",
            )
            metadata = {
                "cache_manifest": str(cache),
                "cache_manifest_sha256": "cache-hash",
                "base_model_index_sha256": "model-hash",
                "scene_tokens": 128,
                "flow_shift": 5.0,
                "trajectory_conditioning": {
                    "enabled": True,
                    "representation": "dense_point_tracks",
                    "input_record": "target",
                },
            }
            train_wan_formal.write_input_contract(checkpoint, metadata)
            contract = json.loads(
                (checkpoint / "input_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["sampling"]["max_area"], 832 * 480)
            self.assertEqual(contract["trajectory"]["protocol"], "target")


if __name__ == "__main__":
    unittest.main()
