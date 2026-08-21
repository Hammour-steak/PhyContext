#!/usr/bin/env python3
"""Audit whether every contacted static collider is present in the t=0 scene input."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from schema import iter_jsonl
from project_defaults import DATASET_MANIFEST, DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("outputs/audits/contact_surface_coverage.json")


def _aliases(collider_id: str, collider: dict) -> dict[str, str]:
    aliases = {collider_id: "direct_collider_id"}
    if collider.get("render_replaced_by_solid_wedge"):
        aliases["solid_ramp_wedge"] = "solid_wedge_visual_replacement"
    if collider_id == "support":
        aliases["support_visual_mesh"] = "reviewed_support_mesh_replacement"
    prefix = "environment_mesh_"
    if collider_id.startswith(prefix):
        aliases[
            "scene_mesh_" + collider_id[len(prefix) :]
        ] = "reviewed_environment_mesh_replacement"
    return aliases


def _represented_parts(scene_path: Path) -> set[str]:
    with np.load(scene_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    counts = metadata["environment_material_binding"][
        "resolved_scene_part_point_counts"
    ]
    return {str(part_id) for part_id, count in counts.items() if int(count) > 0}


def _contacted_colliders(metadata: dict, trajectory_path: Path) -> list[str]:
    objects = metadata["simulation"]["objects"]
    if len(objects) != 1:
        raise ValueError("the one-object coverage audit requires one simulated object")
    prefix = f"{objects[0]['object_id']}__collider_contact_count__"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        return sorted(
            key[len(prefix) :]
            for key in archive.files
            if key.startswith(prefix) and np.any(archive[key] > 0)
        )


def audit(dataset_root: Path) -> dict:
    dataset = dataset_root.resolve()
    records = list(iter_jsonl(dataset / DATASET_MANIFEST))
    parts_by_scene = {
        path.as_posix(): _represented_parts(dataset / path)
        for path in sorted(
            {Path(record["conditioning"]["scene"]) for record in records}
        )
    }
    status_counts = Counter()
    missing_collider_counts = Counter()
    affected_samples: list[dict] = []
    affected_bases = set()
    mapped_alias_counts = Counter()

    for record in records:
        target_metadata_path = dataset / record["target"]["metadata"]
        metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
        trajectory_path = dataset / metadata["trajectory"]["path"]
        represented = parts_by_scene[record["conditioning"]["scene"]]
        colliders = {
            str(collider["id"]): collider
            for collider in metadata["simulation"]["support"].get("colliders", [])
        }
        colliders.update(
            {
                str(collider["id"]): collider
                for collider in metadata.get("environment_binding", {}).get(
                    "colliders", []
                )
            }
        )
        missing = []
        for collider_id in _contacted_colliders(metadata, trajectory_path):
            aliases = _aliases(collider_id, colliders.get(collider_id, {}))
            matches = sorted(set(aliases) & represented)
            if collider_id in matches:
                status_counts["direct"] += 1
            elif matches:
                status_counts["visual_alias"] += 1
                mapped_alias_counts[aliases[matches[0]]] += 1
            else:
                status_counts["not_represented_at_t0"] += 1
                missing_collider_counts[collider_id] += 1
                missing.append(collider_id)
        if missing:
            affected_bases.add(record["base_scene_id"])
            affected_samples.append(
                {
                    "sample_id": record["sample_id"],
                    "base_scene_id": record["base_scene_id"],
                    "missing_contacted_colliders": missing,
                    "represented_scene_parts": sorted(represented),
                }
            )

    return {
        "schema": "physweep.contact_surface_coverage_audit.v1",
        "scene_policy": "complete_ground_plus_camera_first_hit_visible_non_ground_at_t0",
        "complete_contact_surface_coverage": not affected_samples,
        "sample_count": len(records),
        "base_scene_count": len({record["base_scene_id"] for record in records}),
        "affected_sample_count": len(affected_samples),
        "affected_base_scene_count": len(affected_bases),
        "contact_status_counts": dict(sorted(status_counts.items())),
        "visual_alias_counts": dict(sorted(mapped_alias_counts.items())),
        "missing_collider_counts": dict(
            sorted(missing_collider_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "affected_base_scene_ids": sorted(affected_bases),
        "examples": affected_samples[:50],
        "interpretation": (
            "Unrepresented contacts are future contacts with surfaces absent from the "
            "initial observable scene. They are reported rather than completed from "
            "future trajectory information."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        required=DATASET_ROOT is None,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset = (
        args.dataset_root.resolve()
        if args.dataset_root.is_absolute()
        else (root / args.dataset_root).resolve()
    )
    report = audit(dataset)
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    if args.require_complete and not report["complete_contact_surface_coverage"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
