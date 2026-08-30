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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

import cache_wan_inputs  # noqa: E402
import conditioning_model  # noqa: E402
import infer_wan_conditioned  # noqa: E402
import merge_wan_cache_manifests  # noqa: E402
import train_wan_formal  # noqa: E402
from cache_contract import CANONICAL_CONDITION_FRAME_PROTOCOL  # noqa: E402
from video_preprocess import (  # noqa: E402
    cover_center_crop_frames,
    cover_center_crop_intrinsics,
)
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
    def test_inference_uses_the_exact_training_resolution(self) -> None:
        with patch.object(sys, "argv", ["infer_wan_conditioned.py"]):
            args = infer_wan_conditioned.parse_args()
        self.assertEqual((args.width, args.height, args.frames), (832, 480, 97))
        self.assertEqual(args.width % 32, 0)
        self.assertEqual(args.height % 32, 0)

    def test_first_frame_and_video_share_cover_center_crop(self) -> None:
        frame = np.zeros((1, 720, 1280, 3), dtype=np.uint8)
        frame[:, :, :, 0] = np.arange(1280, dtype=np.uint16) % 256
        processed = cover_center_crop_frames(frame, 832, 480)
        self.assertEqual(processed.shape, (1, 480, 832, 3))
        self.assertTrue(processed.flags.c_contiguous)

    def test_canonical_first_frame_replaces_only_video_frame_zero(self) -> None:
        video = cache_wan_inputs.torch.zeros((3, 4, 2, 3))
        video[:, 1:] = 0.25
        canonical = cache_wan_inputs.torch.full((3, 1, 2, 3), -0.75)
        result = cache_wan_inputs.bind_canonical_condition_frame(video, canonical)
        self.assertIs(result, video)
        self.assertTrue(cache_wan_inputs.torch.equal(video[:, :1], canonical))
        self.assertTrue(
            cache_wan_inputs.torch.equal(
                video[:, 1:], cache_wan_inputs.torch.full((3, 3, 2, 3), 0.25)
            )
        )

    def test_intrinsics_follow_exact_cover_center_crop_geometry(self) -> None:
        source_size = (1280, 720)
        target_size = (832, 480)
        intrinsics = np.asarray(
            [[1760.0, 0.0, 640.0], [0.0, 1760.0, 360.0], [0.0, 0.0, 1.0]]
        )
        transformed = cover_center_crop_intrinsics(
            intrinsics, source_size, target_size
        )
        self.assertTrue(
            np.allclose(
                transformed,
                np.asarray(
                    [
                        [1172.875, 0.0, 416.333203125],
                        [0.0, 1173.3333333333333, 239.83333333333331],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                atol=1.0e-10,
            )
        )
        point = np.asarray([0.12, -0.08, 1.7])
        source_pixel = (intrinsics @ point)[:2] / point[2]
        resize_x = 853 / 1280
        resize_y = 480 / 720
        expected_pixel = np.asarray(
            [
                (source_pixel[0] + 0.5) * resize_x - 0.5 - 10,
                (source_pixel[1] + 0.5) * resize_y - 0.5,
            ]
        )
        transformed_pixel = (transformed @ point)[:2] / point[2]
        self.assertTrue(np.allclose(transformed_pixel, expected_pixel, atol=1.0e-10))

    def test_scene_loader_normalizes_transformed_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npz"
            np.savez_compressed(
                path,
                object_xyz_camera_m=np.zeros((1, 3), dtype=np.float32),
                object_normal_camera=np.asarray(
                    [[0.0, 1.0, 0.0]], dtype=np.float32
                ),
                environment_xyz_camera_m=np.zeros((2, 3), dtype=np.float32),
                environment_normal_camera=np.asarray(
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
                ),
                environment_friction=np.full(2, 0.5, dtype=np.float32),
                environment_restitution=np.full(2, 0.1, dtype=np.float32),
                camera_intrinsics=np.asarray(
                    [[1760.0, 0.0, 640.0], [0.0, 1760.0, 360.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                image_size_px=np.asarray([1280, 720], dtype=np.int32),
                camera_intrinsics_normalized=np.asarray(
                    [1.375, 2.4444444, 0.5, 0.5], dtype=np.float32
                ),
            )
            loaded = conditioning_model.load_scene_condition(
                path, target_size_px=(832, 480)
            )
            expected = np.asarray(
                [1172.875 / 832, 1173.3333333333333 / 480, 416.333203125 / 832, 239.83333333333331 / 480],
                dtype=np.float32,
            )
            self.assertTrue(
                np.allclose(
                    loaded["camera_intrinsics_normalized"].numpy(), expected
                )
            )

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

    def test_cache_shards_merge_into_source_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            source_manifest = dataset_root / "manifest.jsonl"
            source_manifest.write_text(
                '{"sample_id":"sample_a"}\n{"sample_id":"sample_b"}\n',
                encoding="utf-8",
            )
            common = {
                "schema": cache_wan_inputs.CACHE_SCHEMA,
                "dataset_root": str(dataset_root),
                "source_manifest": "manifest.jsonl",
            }
            shard_paths = []
            for index, sample_id in enumerate(("sample_b", "sample_a")):
                path = root / f"shard-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            **common,
                            "records": [
                                {
                                    "sample_id": sample_id,
                                    "point_track": {"path": f"{sample_id}.safetensors"},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                shard_paths.append(path)
            output = root / "manifest.json"
            argv = [
                "merge_wan_cache_manifests.py",
                "--project-root",
                str(root),
                "--shard",
                str(shard_paths[0]),
                "--shard",
                str(shard_paths[1]),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                merge_wan_cache_manifests.main()
            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["sample_id"] for item in merged["records"]],
                ["sample_a", "sample_b"],
            )

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

    def test_external_reused_text_is_materialized_in_the_new_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_root = root / "old"
            new_root = root / "new"
            prompt_hash = "a" * 64
            source = old_root / "text" / f"{prompt_hash}.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"verified-context")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            should_build = cache_wan_inputs.prepare_text_artifact(
                prompt_hash,
                new_root / "text",
                new_root,
                old_root,
                {},
                {
                    prompt_hash: {
                        "path": f"text/{prompt_hash}.safetensors",
                        "sha256": digest,
                    }
                },
                False,
            )
            target = new_root / "text" / f"{prompt_hash}.safetensors"
            self.assertFalse(should_build)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_v4_migration_reuses_geometry_but_not_video_latents(self) -> None:
        current = {
            "width": 832,
            "height": 480,
            "frames": 97,
            "resize": "cover_then_center_crop",
            "condition_frame": CANONICAL_CONDITION_FRAME_PROTOCOL,
        }
        legacy = {
            key: value for key, value in current.items() if key != "condition_frame"
        }
        self.assertFalse(
            cache_wan_inputs.reusable_latent_protocol_matches(
                "phycontext.wan_ti2v_cache.v4", legacy, current
            )
        )
        self.assertTrue(
            cache_wan_inputs.reusable_latent_protocol_matches(
                "phycontext.wan_ti2v_cache.v5", current, current
            )
        )
        with self.assertRaisesRegex(ValueError, "video preprocessing"):
            cache_wan_inputs.reusable_latent_protocol_matches(
                "phycontext.wan_ti2v_cache.v3", legacy, current
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
        self.assertEqual(args.validation_batches, 15)
        self.assertEqual(args.validation_batches % 5, 0)
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
                        "schema": "phycontext.formal_training_run.v2",
                        "training_code_sha256": (
                            train_wan_formal.training_code_sha256()
                        ),
                        "cache_manifest_sha256": train_wan_formal.sha256(cache),
                        "arguments": arguments,
                    }
                ),
                encoding="utf-8",
            )
            train_wan_formal.validate_resume_contract(
                root, Path("run/latest"), cache, args
            )
            contract_path = checkpoint.parent / "run_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["training_code_sha256"][
                "tools/phycontext/train_wan_formal.py"
            ] = "changed"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training code differs"):
                train_wan_formal.validate_resume_contract(
                    root, Path("run/latest"), cache, args
                )
            contract["training_code_sha256"] = (
                train_wan_formal.training_code_sha256()
            )
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
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
                    {
                        "preprocess": {
                            "width": 832,
                            "height": 480,
                            "frames": 97,
                            "condition_frame": CANONICAL_CONDITION_FRAME_PROTOCOL,
                        }
                    }
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
            self.assertEqual(contract["schema"], "phycontext.inference_input_contract.v5")
            self.assertEqual(
                (contract["sampling"]["width"], contract["sampling"]["height"]),
                (832, 480),
            )
            self.assertEqual(contract["sampling"]["max_area"], 832 * 480)
            self.assertEqual(
                contract["sampling"]["spatial_preprocess"],
                "cover_then_center_crop",
            )
            self.assertEqual(
                contract["sampling"]["condition_frame"],
                CANONICAL_CONDITION_FRAME_PROTOCOL,
            )
            self.assertEqual(
                contract["scene"]["camera_intrinsics"],
                "cover_then_center_crop_adjusted_and_target_normalized",
            )
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
