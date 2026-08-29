#!/usr/bin/env python3
"""Merge compatible Wan cache shards into one training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_contract import SUPPORTED_CACHE_SCHEMAS


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
    )
    first = payloads[0]
    for key in invariant_keys:
        expected = first.get(key)
        for path, payload in zip(shards[1:], payloads[1:]):
            if payload.get(key) != expected:
                raise ValueError(f"cache shard invariant mismatch for {key}: {path}")
    records = {}
    for path, payload in zip(shards, payloads):
        for record in payload.get("records", []):
            sample_id = record.get("sample_id")
            if not sample_id:
                raise ValueError(f"record without sample_id: {path}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id across shards: {sample_id}")
            if "point_track" not in record:
                raise ValueError(f"incomplete point-track record: {sample_id}")
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

    merged = dict(first)
    merged["selection"] = {
        "mode": "merged_shards",
        "shard_count": len(shards),
        "shards": [path.relative_to(root).as_posix() for path in shards],
    }
    merged["records"] = sorted(records.values(), key=lambda item: source_order[item["sample_id"]])
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"manifest": output.as_posix(), "records": len(merged["records"]), "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
