from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectBoundaryTest(unittest.TestCase):
    def test_dataset_generator_is_not_part_of_method_repository(self) -> None:
        self.assertFalse((ROOT / "assets").exists())
        self.assertFalse((ROOT / "tools/dataset_generation").exists())
        self.assertFalse((ROOT / "tools/training_data").exists())
        self.assertFalse((ROOT / "configs/datasets").exists())
        self.assertFalse((ROOT / "tools/phycontext/gt_scene_input.py").exists())
        self.assertFalse((ROOT / "tools/phycontext/prompt_contract.py").exists())
        self.assertFalse(
            (ROOT / "tools/phycontext/build_training_policy_manifest.py").exists()
        )

    def test_obsolete_scene_condition_bridge_is_absent(self) -> None:
        self.assertFalse((ROOT / "tools/phycontext/scene_condition.py").exists())
        self.assertFalse((ROOT / "tools/phycontext/prepare_inference_scene.py").exists())
        self.assertFalse((ROOT / "tools/phycontext/audit_scene_condition_samples.py").exists())

    def test_training_config_contains_no_generation_settings(self) -> None:
        config = json.loads(
            (ROOT / "configs/training/one_object.json").read_text(encoding="utf-8")
        )
        serialized = json.dumps(config, sort_keys=True)
        for forbidden in ("sampling_matrix", "pybullet", "blender", "render_workers"):
            self.assertNotIn(forbidden, serialized.lower())
        self.assertNotIn("dataset_root", config)

    def test_training_entry_does_not_import_generation_backends(self) -> None:
        source = (ROOT / "tools/model_training/train_one_object.py").read_text(
            encoding="utf-8"
        )
        imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
        }
        for forbidden in ("pybullet", "bpy", "blender"):
            self.assertNotIn(forbidden, imports)

    def test_legacy_method_names_are_absent(self) -> None:
        forbidden = (
            "PhysBind",
            "physbind",
            "PHYSWEEP_WAN_",
            "PHYSWEEP_CACHE_",
            "PHYSWEEP_TRAINING_",
            "physctrl_point_tracks",
            "source_bound_displacement",
            "centroid_heatmap",
            "temporal_mixing_v1",
            "trajectory_condition_sigma",
            "sigma_latent_cells",
            "trajectory_condition_maps_from_masks",
            "PhysSweepThreeBundle",
            "shared_v1",
            "endpoint_flow_blend",
            "legacy_trajectory_losses",
            "physweep.scene_condition.v1",
            "phycontext.wan_ti2v_cache.v1",
            "phycontext.wan_ti2v_cache.v2",
            "phycontext.wan_training_inputs.v1",
            "source_mask_and_centroid_path",
            "source_motion_manifest",
            "motion-mask-manifest",
            "motion_preprocess",
        )
        for path in ROOT.rglob("*"):
            if path == Path(__file__).resolve():
                continue
            if not path.is_file() or any(
                part in {".git", "__pycache__"} for part in path.parts
            ):
                continue
            if path.suffix not in {".py", ".md", ".json", ".yml", ".yaml", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"legacy token {token!r} in {path}")

    def test_obsolete_cache_merger_is_absent(self) -> None:
        self.assertFalse(
            (ROOT / "tools" / "phycontext" / "merge_wan_cache_shards.py").exists()
        )
        self.assertFalse(
            (ROOT / "tools" / "phycontext" / "audit_latent_trajectory_loss.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
