import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from audit_temporal_consistency import main  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class TemporalAuditTest(unittest.TestCase):
    def test_main_resolves_point_track_from_external_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_root = root / "project"
            dataset_root = root / "dataset"
            artifact_root = root / "cache"
            evaluation_root = root / "evaluation" / "full"
            for path in (
                project_root,
                dataset_root,
                artifact_root / "point_tracks",
                evaluation_root,
            ):
                path.mkdir(parents=True, exist_ok=True)

            point_path = artifact_root / "point_tracks" / "sample.safetensors"
            point_map = torch.zeros(12, 3, 4, 4)
            point_map[3, :, 1, 1] = 1.0
            save_file({"point_track_map": point_map}, str(point_path))

            video_path = evaluation_root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (4, 4),
            )
            self.assertTrue(writer.isOpened())
            for value in (10, 20, 30):
                writer.write(np.full((4, 4, 3), value, dtype=np.uint8))
            writer.release()

            report = {
                "sample_id": "sample",
                "frames": 3,
                "height": 4,
                "width": 4,
                "video_sha256": sha256(video_path),
            }
            video_path.with_suffix(".json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            cache_manifest = root / "cache_manifest.json"
            cache_manifest.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "artifact_path_base": "cache_root",
                        "artifact_root": str(artifact_root),
                        "records": [
                            {
                                "sample_id": "sample",
                                "point_track": {
                                    "path": "point_tracks/sample.safetensors",
                                    "sha256": sha256(point_path),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "audit.json"

            argv = [
                "audit_temporal_consistency.py",
                "--project-root",
                str(project_root),
                "--evaluation-root",
                str(evaluation_root.parent),
                "--cache-manifest",
                str(cache_manifest),
                "--output",
                str(output_path),
                "--dilation-px",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "sys.stdout", new_callable=io.StringIO
            ):
                main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["sample_id"], "sample")


if __name__ == "__main__":
    unittest.main()
