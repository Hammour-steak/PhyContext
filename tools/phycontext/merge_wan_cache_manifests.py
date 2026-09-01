#!/usr/bin/env python3
"""Merge compatible Wan cache shards into one training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_contract import (
    SUPPORTED_CACHE_SCHEMAS,
    TRACK_CORRESPONDENCE_CACHE_SCHEMAS,
    validate_cache_record_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def validated_shard_selections(payloads: list[dict]) -> list[dict] | None:
    selections = [payload.get("selection") for payload in payloads]
    if all(selection is None for selection in selections):
        return None
    if any(not isinstance(selection, dict) for selection in selections):
        raise ValueError("cache shards must all record their selection settings")
    counts = {int(selection["shard_count"]) for selection in selections}
    if counts != {len(payloads)}:
        raise ValueError("cache shard_count does not match the supplied shard set")
    indices = [int(selection["shard_index"]) for selection in selections]
    if sorted(indices) != list(range(len(payloads))):
        raise ValueError("cache shards must contain every shard index exactly once")
    common = {
        key: value for key, value in selections[0].items() if key != "shard_index"
    }
    for selection in selections[1:]:
        candidate = {
            key: value for key, value in selection.items() if key != "shard_index"
        }
        if candidate != common:
            raise ValueError("cache shards use different source selection filters")
    return sorted(selections, key=lambda item: int(item["shard_index"]))


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    shards = [resolve(root, value) for value in args.shard]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    if not payloads:
        raise ValueError("at least one cache shard is required")
    required_schema = payloads[0].get("schema")
    if required_schema not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError("first cache shard uses an unsupported schema")
    for path, payload in zip(shards, payloads):
        if payload.get("schema") != required_schema:
            raise ValueError(f"{path} does not use the same cache schema")

    invariant_keys = (
        "artifact_path_base",
        "artifact_root",
        "dataset_root",
        "source_manifest",
        "source_manifest_sha256",
        "source_dataset_summary",
        "source_dataset_summary_sha256",
        "checkpoint",
        "preprocess",
        "source_point_trajectory_manifest",
        "source_point_trajectory_manifest_sha256",
        "point_track_preprocess",
        "point_correspondence_preprocess",
    )
    first = payloads[0]
    for key in invariant_keys:
        expected = first.get(key)
        for path, payload in zip(shards[1:], payloads[1:]):
            if payload.get(key) != expected:
                raise ValueError(f"cache shard invariant mismatch for {key}: {path}")
    shard_selections = validated_shard_selections(payloads)
    records = {}
    for path, payload in zip(shards, payloads):
        for record in payload.get("records", []):
            sample_id = record.get("sample_id")
            if not sample_id:
                raise ValueError(f"record without sample_id: {path}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id across shards: {sample_id}")
            required_descriptors = ["record", "latent", "text_context", "point_track"]
            if required_schema in TRACK_CORRESPONDENCE_CACHE_SCHEMAS:
                required_descriptors.append("track_correspondence")
            missing_descriptors = [
                key for key in required_descriptors if key not in record
            ]
            if missing_descriptors:
                raise ValueError(
                    f"incomplete cache record {sample_id}: "
                    f"missing {missing_descriptors}"
                )
            records[sample_id] = record

    dataset_root = Path(first["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    source_path = dataset_root.resolve() / first["source_manifest"]
    source_records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_order = {record["sample_id"]: index for index, record in enumerate(source_records)}
    unknown = sorted(set(records) - set(source_order))
    if unknown:
        raise ValueError(f"cache contains samples absent from source manifest: {unknown[:5]}")
    if shard_selections is not None:
        for path, payload in zip(shards, payloads):
            validate_cache_record_coverage(
                payload, source_records, label=f"cache shard {path}"
            )

    merged = dict(first)
    merged["selection"] = {
        "mode": "merged_shards",
        "shard_count": len(shards),
        "shards": [portable_path(path, root) for path in shards],
    }
    if shard_selections is not None:
        merged["selection"]["shard_selections"] = shard_selections
    merged["records"] = sorted(records.values(), key=lambda item: source_order[item["sample_id"]])
    validate_cache_record_coverage(merged, source_records, label="merged Wan cache")
    merged["selection"]["expected_sample_count"] = len(merged["records"])
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"manifest": output.as_posix(), "records": len(merged["records"]), "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
