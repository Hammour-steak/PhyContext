#!/usr/bin/env python3
"""Measure temporal variation on the ground-truth target videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_temporal_consistency import load_point_mask, read_gray, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilation-px", type=int, default=8)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    manifest = json.loads(args.cache_manifest.resolve().read_text(encoding="utf-8"))
    records = {item["sample_id"]: item for item in manifest["records"]}
    results = []
    for sample_id in args.sample_id:
        item = records[sample_id]
        target_video = project_root / item["record"]["target"]["video"]
        video = read_gray(target_video)
        mask = load_point_mask(
            project_root,
            item,
            video.shape[0],
            video.shape[1],
            video.shape[2],
            args.dilation_px,
        )
        results.append(
            {
                "sample_id": sample_id,
                "video": str(target_video),
                **score(video, mask),
            }
        )
    payload = {
        "schema": "phycontext.gt_temporal_consistency_audit.v1",
        "cache_manifest": str(args.cache_manifest.resolve()),
        "dilation_px": args.dilation_px,
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
