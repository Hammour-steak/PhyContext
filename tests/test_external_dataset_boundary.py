from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHYCONTEXT_TOOLS = ROOT / "tools" / "phycontext"
if str(PHYCONTEXT_TOOLS) not in sys.path:
    sys.path.insert(0, str(PHYCONTEXT_TOOLS))

from cache_contract import validate_cache_source_manifest  # noqa: E402


def load_training_entry():
    path = ROOT / "tools" / "model_training" / "train_one_object.py"
    spec = importlib.util.spec_from_file_location("train_one_object_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalDatasetBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary.name)
        self.method_root = temporary_root / "method"
        self.dataset_root = temporary_root / "dataset"
        self.method_root.mkdir()
        release = self.dataset_root / "datasets" / "physweep_training"
        point_root = release / "point_trajectories"
        point_root.mkdir(parents=True)
        self.manifest = release / "manifest.jsonl"
        self.manifest.write_text('{"sample_id":"sample_a"}\n', encoding="utf-8")
        self.summary = release / "summary.json"
        self.summary.write_text(
            json.dumps(
                {
                    "path_base": "physweep_project_root",
                    "manifest": "datasets/physweep_training/manifest.jsonl",
                    "manifest_sha256": sha256(self.manifest),
                }
            ),
            encoding="utf-8",
        )
        self.point_manifest = point_root / "manifest.json"
        self.point_manifest.write_text('{"records":[]}\n', encoding="utf-8")
        self.cache = {
            "schema": "phycontext.wan_ti2v_cache.v3",
            "dataset_root": str(self.dataset_root),
            "source_manifest": "datasets/physweep_training/manifest.jsonl",
            "source_manifest_sha256": sha256(self.manifest),
            "source_dataset_summary": "datasets/physweep_training/summary.json",
            "source_dataset_summary_sha256": sha256(self.summary),
            "source_point_trajectory_manifest": (
                "datasets/physweep_training/point_trajectories/manifest.json"
            ),
            "source_point_trajectory_manifest_sha256": sha256(self.point_manifest),
            "records": [],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cache_contract_resolves_an_external_dataset(self) -> None:
        resolved = validate_cache_source_manifest(self.method_root, self.cache)
        self.assertEqual(resolved, self.manifest.resolve())

    def test_cache_contract_rejects_changed_point_trajectories(self) -> None:
        self.point_manifest.write_text('{"records":[1]}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "point-trajectory manifest changed"):
            validate_cache_source_manifest(self.method_root, self.cache)

    def test_cache_contract_rejects_changed_dataset_summary(self) -> None:
        self.summary.write_text('{"path_base":"wrong"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset summary changed"):
            validate_cache_source_manifest(self.method_root, self.cache)

    def test_training_preflight_keeps_dataset_and_cache_roots_separate(self) -> None:
        cache_path = self.method_root / "cache" / "manifest.json"
        cache_path.parent.mkdir()
        cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        config = {
            "dataset_root": str(self.dataset_root),
            "dataset_manifest": "datasets/physweep_training/manifest.jsonl",
            "cache_manifest": "cache/manifest.json",
        }
        entry = load_training_entry()
        manifest, cache = entry.preflight(self.method_root, config)
        self.assertEqual(manifest, self.manifest.resolve())
        self.assertEqual(cache, cache_path.resolve())

    def test_training_preflight_requires_an_external_dataset_root(self) -> None:
        entry = load_training_entry()
        with self.assertRaisesRegex(ValueError, "PhysSweep root is required"):
            entry.preflight(self.method_root, {"cache_manifest": "cache/manifest.json"})

    def test_training_preflight_rejects_a_cache_from_another_dataset(self) -> None:
        cache_path = self.method_root / "cache" / "manifest.json"
        cache_path.parent.mkdir()
        cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        other_root = self.dataset_root.parent / "other_dataset"
        other_manifest = other_root / "datasets" / "physweep_training" / "manifest.jsonl"
        other_manifest.parent.mkdir(parents=True)
        other_manifest.write_text(self.manifest.read_text(encoding="utf-8"), encoding="utf-8")
        config = {
            "dataset_root": str(other_root),
            "dataset_manifest": "datasets/physweep_training/manifest.jsonl",
            "cache_manifest": "cache/manifest.json",
        }
        entry = load_training_entry()
        with self.assertRaisesRegex(ValueError, "different dataset root"):
            entry.preflight(self.method_root, config)


if __name__ == "__main__":
    unittest.main()
