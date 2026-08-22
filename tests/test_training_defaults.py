from __future__ import annotations

import io
import hashlib
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
        self.assertEqual(args.trajectory_representation, "das_3d_tracks")

    def test_das_cache_keeps_all_frames_until_the_conditioner(self) -> None:
        indices = cache_wan_inputs.trajectory_frame_indices(
            "das_3d_tracks", 97, 97, 25
        )
        self.assertEqual(indices, list(range(97)))
        self.assertEqual(
            cache_wan_inputs.expected_point_track_shape(
                "das_3d_tracks", 97, (48, 25, 30, 52)
            ),
            (12, 97, 30, 52),
        )

    def test_dense_ablation_retains_legacy_latent_rate(self) -> None:
        indices = cache_wan_inputs.trajectory_frame_indices(
            "dense_point_tracks", 97, 97, 25
        )
        self.assertEqual(indices, list(range(0, 97, 4)))
        self.assertEqual(
            cache_wan_inputs.expected_point_track_shape(
                "dense_point_tracks", 97, (48, 25, 30, 52)
            ),
            (18, 25, 30, 52),
        )

    def test_trajectory_sampling_matches_video_sampling_for_long_sources(self) -> None:
        video_indices = cache_wan_inputs.evenly_spaced_frame_indices(193, 97)
        trajectory_indices = cache_wan_inputs.trajectory_frame_indices(
            "das_3d_tracks", 193, 97, 25
        )
        self.assertEqual(trajectory_indices, video_indices)
        self.assertEqual(trajectory_indices[::24], [0, 48, 96, 144, 192])

    def test_frame_sampling_rejects_short_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer frames"):
            cache_wan_inputs.evenly_spaced_frame_indices(96, 97)

    def test_cache_artifact_requires_a_matching_manifest_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "cache" / "sample.safetensors"
            artifact.parent.mkdir()
            artifact.write_bytes(b"complete-atomic-artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            descriptor = {
                "path": "cache/sample.safetensors",
                "sha256": digest,
            }
            self.assertTrue(
                cache_wan_inputs.descriptor_matches_file(
                    descriptor, artifact, root
                )
            )
            self.assertFalse(
                cache_wan_inputs.descriptor_matches_file(None, artifact, root)
            )
            descriptor["sha256"] = "0" * 64
            self.assertFalse(
                cache_wan_inputs.descriptor_matches_file(
                    descriptor, artifact, root
                )
            )

    def test_untrusted_local_artifact_cannot_shadow_verified_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "orphan.safetensors"
            self.assertFalse(
                cache_wan_inputs.should_build_local_artifact(
                    artifact,
                    current_matches=False,
                    reusable_matches=True,
                )
            )
            artifact.write_bytes(b"untrusted-local-file")
            self.assertTrue(
                cache_wan_inputs.should_build_local_artifact(
                    artifact,
                    current_matches=False,
                    reusable_matches=True,
                )
            )
            self.assertFalse(
                cache_wan_inputs.should_build_local_artifact(
                    artifact,
                    current_matches=True,
                )
            )

    def test_cache_preserves_multi_object_slot_order(self) -> None:
        record = {
            "conditioning": {
                "physics": {
                    "objects": {
                        "slot_b": {"object_id": "object_b"},
                        "slot_a": {"object_id": "object_a"},
                    }
                }
            }
        }
        self.assertEqual(
            cache_wan_inputs.record_dynamic_object_ids(record),
            ["object_a", "object_b"],
        )

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
        self.assertEqual(args.trajectory_representation, "das_3d_tracks")
        self.assertFalse(args.ordinary_only)
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
                    "representation": "das_3d_tracks",
                    "input_record": "target",
                    "input_channels": 12,
                    "architecture": "full_frame_causal_patch_v2",
                },
            }
            train_wan_formal.write_input_contract(checkpoint, metadata)
            contract = json.loads(
                (checkpoint / "input_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["sampling"]["max_area"], 832 * 480)
            self.assertEqual(contract["trajectory"]["protocol"], "target")
            self.assertEqual(
                contract["trajectory"]["condition_shape"], [12, 97, 30, 52]
            )
            self.assertEqual(
                contract["trajectory"]["architecture"],
                "full_frame_causal_patch_v2",
            )

            metadata["trajectory_conditioning"] = {
                "enabled": False,
                "representation": "das_3d_tracks",
                "input_record": "target",
                "input_channels": 0,
                "architecture": None,
            }
            train_wan_formal.write_input_contract(checkpoint, metadata)
            disabled_contract = json.loads(
                (checkpoint / "input_contract.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(
                disabled_contract["trajectory"]["condition_shape"]
            )


if __name__ == "__main__":
    unittest.main()
