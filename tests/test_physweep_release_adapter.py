from __future__ import annotations

from copy import deepcopy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from adapt_physweep_release import (  # noqa: E402
    IMAGE_SIZE_PX,
    POINT_COUNT,
    _artifact_paths,
    _group_sample_descriptors,
    _load_fixture,
    _physics_condition,
    _prepare_output_root,
    _project_camera,
    _sample_descriptor,
    _validate_release_group_sample,
    build_point_trajectory_payload,
    camera_contract,
    fixture_components,
    implicit_environment_components,
    quaternion_matrix_wxyz,
    resolve_roots,
    sample_dynamic_proxy,
    sha256_file,
)


class PhysSweepReleaseAdapterTest(unittest.TestCase):
    def test_release_artifacts_require_a_fully_decoded_97_frame_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary)
            artifacts = {
                "video": sample / "video.mp4",
                "trajectory": sample / "trajectory.npz",
                "mask_manifest": sample / "mask_manifest.json",
            }
            for name, path in artifacts.items():
                path.write_bytes(name.encode("utf-8"))
            (sample / "metadata.json").write_text("{}", encoding="utf-8")
            metadata = {
                "artifacts": {
                    "video": {"sha256": sha256_file(artifacts["video"])},
                    "trajectory": {"sha256": sha256_file(artifacts["trajectory"])},
                    "masks": {
                        "manifest_sha256": sha256_file(artifacts["mask_manifest"])
                    },
                }
            }
            with patch(
                "adapt_physweep_release.decoded_video_frame_count",
                return_value=70,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "expected=97 observed=70",
                ):
                    _artifact_paths(sample, metadata, "ffprobe")
            with patch(
                "adapt_physweep_release.decoded_video_frame_count",
                return_value=97,
            ):
                self.assertEqual(
                    _artifact_paths(sample, metadata, "ffprobe")["video"],
                    artifacts["video"],
                )

    def test_sweep_descriptor_is_explicit_and_does_not_mutate_inputs(self) -> None:
        physics = {"object": {"mass_kg": 1.25}}
        sweep = {
            "parameter": "contact_friction",
            "level_index": 4,
            "value": 0.8,
        }
        original = deepcopy(sweep)
        base_descriptor = _sample_descriptor(physics, None)
        sweep_descriptor = _sample_descriptor(physics, sweep)
        self.assertEqual(base_descriptor["mode"], "base")
        self.assertEqual(base_descriptor["source_value"], 1.25)
        self.assertEqual(sweep_descriptor["mode"], "one_factor")
        self.assertEqual(sweep_descriptor["axis"], "contact_friction")
        self.assertEqual(sweep, original)

    def test_group_descriptor_validation_rejects_duplicate_sweep_cells(self) -> None:
        sweeps = [
            {"parameter": axis, "level_index": level}
            for axis in ("mass_kg", "contact_friction", "contact_restitution")
            for level in (0, 1, 3, 4)
        ]
        group = {"group_id": "group_a", "base": {"scene_id": "base"}, "sweeps": sweeps}
        self.assertEqual(len(_group_sample_descriptors(group)), 13)
        group["sweeps"][-1] = deepcopy(group["sweeps"][0])
        with self.assertRaisesRegex(ValueError, "required 12 sweeps"):
            _group_sample_descriptors(group)

    def test_fixture_hash_is_validated_before_path_construction(self) -> None:
        metadata = {"physics": {"fixture": {"sha256": "../outside"}}}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "lowercase hexadecimal digest"):
                _load_fixture(Path(temporary), metadata, {}, {})

    def test_interrupted_adapter_output_is_owned_and_safely_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "derived"
            marker = _prepare_output_root(output, overwrite=False)
            (output / "partial.npz").write_bytes(b"incomplete")
            self.assertTrue(marker.is_file())
            with self.assertRaises(FileExistsError):
                _prepare_output_root(output, overwrite=False)
            replacement_marker = _prepare_output_root(output, overwrite=True)
            self.assertTrue(replacement_marker.is_file())
            self.assertFalse((output / "partial.npz").exists())

    def test_unrelated_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unrelated"
            output.mkdir()
            (output / "user_file.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not owned"):
                _prepare_output_root(output, overwrite=True)
            self.assertEqual(
                (output / "user_file.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_camera_contract_projects_target_to_principal_point(self) -> None:
        camera = {
            "position_m": [2.0, -4.0, 3.0],
            "target_m": [0.0, 0.0, 1.0],
            "focal_length_mm": 50.0,
            "sensor_width_mm": 36.0,
        }
        camera_from_world, intrinsics = camera_contract(camera)
        target_camera = (
            camera_from_world[:3, :3] @ np.asarray(camera["target_m"])
            + camera_from_world[:3, 3]
        )
        projected = _project_camera(target_camera[None], intrinsics)[0]
        np.testing.assert_allclose(projected, np.asarray(IMAGE_SIZE_PX) / 2.0)
        np.testing.assert_allclose(
            camera_from_world[:3, :3] @ camera_from_world[:3, :3].T,
            np.eye(3),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(float(np.linalg.det(camera_from_world[:3, :3])), -1.0)

    def test_every_release_dynamic_proxy_family_produces_fixed_material_points(self) -> None:
        proxies = [
            {"type": "sphere", "radius_m": 0.1},
            {"type": "cuboid", "size_m": [0.1, 0.2, 0.3]},
            {"type": "cylinder", "size_m": [0.1, 0.1, 0.3]},
            {
                "type": "compound",
                "colliders": [
                    {
                        "shape": "box",
                        "size_m": [0.1, 0.2, 0.3],
                        "position_m": [0.1, 0.0, 0.0],
                        "rotation_euler_degrees": [0.0, 0.0, 20.0],
                    },
                    {
                        "shape": "sphere",
                        "size_m": [0.1, 0.1, 0.1],
                        "position_m": [-0.1, 0.0, 0.0],
                    },
                ],
            },
        ]
        for index, proxy in enumerate(proxies):
            points, normals, report = sample_dynamic_proxy(
                proxy, np.random.default_rng(index)
            )
            self.assertEqual(points.shape, (POINT_COUNT, 3))
            self.assertEqual(normals.shape, (POINT_COUNT, 3))
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-6)
            self.assertEqual(sum(report["component_point_counts"]), POINT_COUNT)
            if proxy["type"] == "sphere":
                np.testing.assert_allclose(
                    np.linalg.norm(points, axis=1),
                    0.1,
                    rtol=0.0,
                    atol=1.0e-12,
                )

    def test_compound_sphere_requires_isotropic_diameter(self) -> None:
        proxy = {
            "type": "compound",
            "colliders": [
                {
                    "shape": "sphere",
                    "size_m": [0.1, 0.1, 0.2],
                    "position_m": [0.0, 0.0, 0.0],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "isotropic"):
            sample_dynamic_proxy(proxy, np.random.default_rng(0))

    def test_rigid_track_export_and_inverse_audits_preserve_point_identity(self) -> None:
        camera = {
            "position_m": [0.0, -4.0, 2.0],
            "target_m": [0.0, 0.0, 1.0],
            "focal_length_mm": 48.0,
            "sensor_width_mm": 36.0,
            "clip_start_m": 0.03,
            "clip_end_m": 100.0,
        }
        camera_from_world, intrinsics = camera_contract(camera)
        rng = np.random.default_rng(4)
        local = rng.normal(scale=0.05, size=(1, POINT_COUNT, 3))
        half_angle = np.deg2rad(45.0)
        trajectory = {
            "time_s": np.asarray([0.0, 1.0]),
            "object_ids": np.asarray(["object_a"]),
            "position_m": np.asarray([[[0.0, 0.0, 1.0]], [[0.2, 0.1, 1.1]]]),
            "quaternion_wxyz": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]], [[np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]]]
            ),
        }
        payload, report = build_point_trajectory_payload(
            trajectory, local, camera_from_world, intrinsics, camera
        )
        self.assertLess(report["rigid_roundtrip_max_abs_error_m"], 1.0e-10)
        self.assertLess(report["camera_roundtrip_max_abs_error_m"], 1.0e-10)
        second_rotation = quaternion_matrix_wxyz(trajectory["quaternion_wxyz"][1, 0])
        recovered = np.einsum(
            "ij,nj->ni",
            second_rotation.T,
            payload["points_world_m"][1, 0] - trajectory["position_m"][1, 0],
        )
        np.testing.assert_allclose(recovered, local[0], atol=2.0e-7)

    def test_axial_velocity_uses_reflection_sign_and_inertia_stays_positive(self) -> None:
        camera = {
            "position_m": [0.0, -4.0, 2.0],
            "target_m": [0.0, 0.0, 1.0],
            "focal_length_mm": 48.0,
            "sensor_width_mm": 36.0,
        }
        camera_from_world, _ = camera_contract(camera)
        metadata = {
            "physics": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "inertia_diagonal_kg_m2": [0.1, 0.2, 0.3],
                        "initial_state": {
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "linear_velocity_m_s": [1.0, 2.0, 3.0],
                            "angular_velocity_rad_s": [0.5, -0.2, 0.7],
                        },
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                            "rolling_friction": 0.01,
                            "spinning_friction": 0.02,
                            "linear_damping": 0.03,
                            "angular_damping": 0.04,
                        },
                    }
                ],
                "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
            }
        }
        condition = _physics_condition(metadata, "object_a", camera_from_world)
        rotation = camera_from_world[:3, :3]
        expected_angular = np.linalg.det(rotation) * rotation @ np.asarray([0.5, -0.2, 0.7])
        np.testing.assert_allclose(
            condition["object"]["initial_state"]["angular_velocity_camera_rad_s"],
            expected_angular,
        )
        self.assertTrue(
            np.all(np.linalg.eigvalsh(condition["object"]["inertia_tensor_camera_kg_m2"]) > 0.0)
        )

    def test_fixture_parser_uses_collision_geometry_and_ignores_hidden_primitives(self) -> None:
        fixture = {
            "physical": {
                "support": {
                    "dynamics": {"lateral_friction": 0.6, "restitution": 0.1},
                    "colliders": [
                        {
                            "id": "visible",
                            "primitive": "box",
                            "size_m": [2.0, 2.0, 0.1],
                            "position_m": [0.0, 0.0, -0.05],
                            "collision_enabled": True,
                            "visible": True,
                        },
                        {
                            "id": "hidden",
                            "primitive": "box",
                            "size_m": [1.0, 1.0, 1.0],
                            "visible": False,
                        },
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            components = fixture_components(fixture, Path(temporary))
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].friction, 0.6)
        self.assertEqual(components[0].restitution, 0.1)

    def test_static_prop_local_proxy_is_composed_with_its_world_placement(self) -> None:
        fixture = {
            "physical": {
                "static_dynamics": {
                    "static_prop": {"lateral_friction": 0.58, "restitution": 0.06}
                },
                "static_prop_binding": {
                    "position_m": [2.0, 3.0, 4.0],
                    "yaw_degrees": 90.0,
                },
                "static_prop_record": {
                    "proxy": {
                        "colliders": [
                            {
                                "id": "offset_box",
                                "shape": "box",
                                "size_m": [0.2, 0.4, 0.6],
                                "position_m": [1.0, 0.0, 0.5],
                                "rotation_euler_degrees": [0.0, 0.0, 0.0],
                            }
                        ]
                    }
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            components = fixture_components(fixture, Path(temporary))
        self.assertEqual(len(components), 1)
        np.testing.assert_allclose(
            components[0].vertices_world_m.mean(axis=0),
            np.asarray([2.0, 4.0, 4.5]),
            atol=1.0e-12,
        )
        self.assertEqual(components[0].friction, 0.58)
        self.assertEqual(components[0].restitution, 0.06)

    def test_asset_backend_recovers_authoritative_implicit_ground(self) -> None:
        camera = {
            "position_m": [0.0, -4.0, 2.0],
            "target_m": [0.0, 0.0, 0.8],
            "focal_length_mm": 48.0,
            "sensor_width_mm": 36.0,
        }
        camera_from_world, intrinsics = camera_contract(camera)
        metadata = {"physics": {"backend": {"adapter_id": "asset_proxy_v3"}}}
        fixture = {
            "physical": {
                "static_dynamics": {
                    "ground": {"lateral_friction": 0.75, "restitution": 0.08}
                }
            }
        }
        components = implicit_environment_components(
            metadata, fixture, camera_from_world, intrinsics
        )
        self.assertEqual(len(components), 1)
        component = components[0]
        self.assertEqual(component.component_id, "implicit/environment_floor")
        self.assertEqual(component.friction, 0.75)
        self.assertEqual(component.restitution, 0.08)
        np.testing.assert_allclose(component.vertices_world_m[:, 2], 0.0)
        normals = np.cross(
            component.vertices_world_m[component.faces][:, 1]
            - component.vertices_world_m[component.faces][:, 0],
            component.vertices_world_m[component.faces][:, 2]
            - component.vertices_world_m[component.faces][:, 0],
        )
        self.assertTrue(np.all(normals[:, 2] > 0.0))
        projected = _project_camera(
            (
                np.einsum(
                    "ij,nj->ni",
                    camera_from_world[:3, :3],
                    component.vertices_world_m,
                )
                + camera_from_world[:3, 3]
            ),
            intrinsics,
        )
        np.testing.assert_allclose(
            projected,
            [[0.0, 0.0], [1279.0, 0.0], [1279.0, 719.0], [0.0, 719.0]],
            atol=1.0e-9,
        )

    def test_other_backends_do_not_invent_an_implicit_ground(self) -> None:
        metadata = {"physics": {"backend": {"adapter_id": "generic_rigid_v1"}}}
        self.assertEqual(
            implicit_environment_components(metadata, {}, np.eye(4), np.eye(3)),
            [],
        )

    def test_source_and_output_roots_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            release = dataset / "outputs" / "one_object"
            release.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                resolve_roots(dataset, Path("outputs/one_object"), Path("outputs"))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                resolve_roots(dataset, Path("outputs/one_object"), Path("outputs/one_object/derived"))
            resolved = resolve_roots(dataset, Path("outputs/one_object"), Path("datasets/derived"))
            self.assertEqual(resolved[1], release.resolve())

    def test_one_factor_validator_rejects_a_second_material_change(self) -> None:
        base = {
            "scene_id": "sample_base",
            "group_id": "group_a",
            "family": "generic",
            "sample_kind": "base",
            "seed": 7,
            "visual": {"camera": "fixed"},
            "text": {"caption": "fixed"},
            "semantics": {"object": "fixed"},
            "physics": {
                "backend": {"id": "fixed"},
                "fixture": {"id": "fixed"},
                "solver": {"id": "fixed"},
                "time": {"duration_s": 4.0, "output_fps": 24, "simulation_hz": 240},
                "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
                "objects": [
                    {
                        "object_id": "object_a",
                        "collision_proxy": {"type": "sphere", "radius_m": 0.1},
                        "initial_state": {"position_m": [0.0, 0.0, 1.0]},
                        "inertia_diagonal_kg_m2": [0.1, 0.2, 0.3],
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                    }
                ],
            },
        }
        group = {
            "group_id": "group_a",
            "family": "generic",
            "target_object_id": "object_a",
        }
        base_descriptor = {"scene_id": "sample_base"}
        _validate_release_group_sample(
            base, base, base_descriptor, group, "object_a", True
        )
        extra_object = deepcopy(base)
        extra_object["physics"]["objects"].append(
            deepcopy(extra_object["physics"]["objects"][0])
        )
        extra_object["physics"]["objects"][1]["object_id"] = "object_b"
        with self.assertRaisesRegex(ValueError, "exactly one physics object"):
            _validate_release_group_sample(
                extra_object,
                extra_object,
                base_descriptor,
                group,
                "object_a",
                True,
            )
        invalid_rate = deepcopy(base)
        invalid_rate["physics"]["time"]["simulation_hz"] = 239
        with self.assertRaisesRegex(ValueError, "closed-endpoint protocol"):
            _validate_release_group_sample(
                invalid_rate,
                invalid_rate,
                base_descriptor,
                group,
                "object_a",
                True,
            )
        sweep = deepcopy(base)
        sweep.update({"scene_id": "sample_sweep", "sample_kind": "sweep"})
        sweep["physics"]["objects"][0]["material"]["contact_friction"] = 0.8
        sweep["sweep"] = {
            "parameter": "contact_friction",
            "level_index": 4,
            "target_object_id": "object_a",
            "value": 0.8,
        }
        descriptor = {
            "scene_id": "sample_sweep",
            "parameter": "contact_friction",
            "level_index": 4,
        }
        _validate_release_group_sample(
            base, sweep, descriptor, group, "object_a", False
        )
        sweep["physics"]["objects"][0]["material"]["contact_restitution"] = 0.3
        with self.assertRaisesRegex(ValueError, "unexpectedly changes"):
            _validate_release_group_sample(
                base, sweep, descriptor, group, "object_a", False
            )


if __name__ == "__main__":
    unittest.main()
