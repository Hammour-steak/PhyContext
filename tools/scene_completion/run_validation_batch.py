#!/usr/bin/env python3
"""Run reproducible first-frame reconstruction validation across several GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CORE_TOOLS = PROJECT_ROOT / "tools" / "phycontext"
if str(CORE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CORE_TOOLS))

from schema import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--pose-refinement-iterations", type=int, default=48)
    parser.add_argument(
        "--use-exact-object-mask",
        action="store_true",
        help="use the validation mask as controlled-object selection input",
    )
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume an incomplete scene directory and reuse its generated object mesh",
    )
    return parser.parse_args()


def resolve_inside(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must remain inside project root: {resolved}")
    return resolved


def load_jobs(root: Path, manifest: Path, mask_manifest: Path, scene_ids: list[str]) -> list[dict]:
    base_records = {
        record["base_scene_id"]: record
        for record in iter_jsonl(manifest)
        if record["sweep"]["mode"] == "base"
    }
    mask_payload = json.loads(mask_manifest.read_text(encoding="utf-8"))
    masks = {record["sample_id"]: record for record in mask_payload["records"]}
    jobs = []
    for index, scene_id in enumerate(scene_ids, 1):
        if scene_id not in base_records:
            raise ValueError(f"unknown base scene: {scene_id}")
        record = base_records[scene_id]
        sample_id = record["sample_id"]
        if sample_id not in masks:
            raise ValueError(f"exact first-frame mask is missing: {sample_id}")
        image = resolve_inside(root, Path(record["conditioning"]["first_frame"]))
        mask = resolve_inside(
            root,
            Path(masks[sample_id]["mask"]["directory"]) / "frame_0001.png",
        )
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(image if not image.is_file() else mask)
        jobs.append(
            {
                "index": index,
                "scene_id": scene_id,
                "sample_id": sample_id,
                "image": image,
                "mask": mask,
            }
        )
    return jobs


def run_job(
    root: Path,
    output_root: Path,
    job: dict,
    gpu: int,
    diffusion_steps: int,
    pose_iterations: int,
    reuse_complete: bool,
    resume_existing: bool,
    use_exact_object_mask: bool,
) -> dict:
    output_dir = output_root / f"{job['index']:02d}_{job['scene_id']}"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "scene_manifest.json"
    if reuse_complete and report_path.is_file() and manifest_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            **job,
            "gpu": gpu,
            "status": "reused" if report["quality_gate"]["passed"] else "quality_rejected",
            "output_dir": str(output_dir.relative_to(root)),
            "quality_gate": report.get("quality_gate"),
            "gt_mask_iou": report.get("gt_mask_iou"),
            "timings": report.get("timings"),
        }
    if output_dir.exists() and not resume_existing:
        raise FileExistsError(f"incomplete output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "infer_first_frame.py"),
        "--image",
        str(job["image"]),
        "--gt-mask",
        str(job["mask"]),
        "--output-dir",
        str(output_dir),
        "--device",
        f"cuda:{gpu}",
        "--instantmesh-device",
        str(gpu),
        "--instantmesh-diffusion-steps",
        str(diffusion_steps),
        "--pose-refinement-iterations",
        str(pose_iterations),
    ]
    if resume_existing and (output_dir / "object_mesh.obj").is_file():
        command.append("--reuse-object-mesh")
    if use_exact_object_mask:
        command.extend(["--object-mask", str(job["mask"])])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SCRIPT_DIR)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode:
        return {
            **job,
            "gpu": gpu,
            "status": "failed",
            "return_code": result.returncode,
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir.relative_to(root)),
            "log": str(log_path.relative_to(root)),
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        **job,
        "gpu": gpu,
        "status": "passed" if report["quality_gate"]["passed"] else "quality_rejected",
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir.relative_to(root)),
        "quality_gate": report["quality_gate"],
        "gt_mask_iou": report.get("gt_mask_iou"),
        "timings": report["timings"],
    }


def serializable(result: dict, root: Path) -> dict:
    return {
        key: str(value.relative_to(root)) if isinstance(value, Path) else value
        for key, value in result.items()
    }


def run_gpu_queue(
    root: Path,
    output_root: Path,
    jobs: list[dict],
    gpu: int,
    diffusion_steps: int,
    pose_iterations: int,
    reuse_complete: bool,
    resume_existing: bool,
    use_exact_object_mask: bool,
) -> list[dict]:
    results = []
    for job in jobs:
        result = run_job(
            root,
            output_root,
            job,
            gpu,
            diffusion_steps,
            pose_iterations,
            reuse_complete,
            resume_existing,
            use_exact_object_mask,
        )
        results.append(result)
        print(f"scene {result['status']} {result['scene_id']} gpu={gpu}", flush=True)
    return results


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest = resolve_inside(root, args.manifest)
    mask_manifest = resolve_inside(root, args.mask_manifest)
    output_root = resolve_inside(root, args.output_root)
    if output_root.exists() and not (args.reuse_complete or args.resume_existing):
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(root, manifest, mask_manifest, args.scene_id)
    results = []
    queues = [jobs[index :: len(args.gpu)] for index in range(len(args.gpu))]
    with ThreadPoolExecutor(max_workers=len(args.gpu)) as executor:
        futures = [
            executor.submit(
                run_gpu_queue,
                root,
                output_root,
                queue,
                gpu,
                args.diffusion_steps,
                args.pose_refinement_iterations,
                args.reuse_complete,
                args.resume_existing,
                args.use_exact_object_mask,
            )
            for gpu, queue in zip(args.gpu, queues)
        ]
        for future in as_completed(futures):
            results.extend(serializable(result, root) for result in future.result())
    results.sort(key=lambda result: result["index"])
    batch = {
        "schema": "phycontext.scene_reconstruction_validation_batch.v1",
        "source_manifest": str(manifest.relative_to(root)),
        "mask_manifest": str(mask_manifest.relative_to(root)),
        "gpus": args.gpu,
        "object_selection": (
            "exact_validation_mask" if args.use_exact_object_mask else "automatic_sam2"
        ),
        "scene_count": len(results),
        "process_pass_count": sum(result["status"] != "failed" for result in results),
        "quality_pass_count": sum(result["status"] in {"passed", "reused"} for result in results),
        "results": results,
    }
    report_path = output_root / "batch_report.json"
    report_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in batch.items() if key != "results"}, indent=2))
    raise SystemExit(0 if batch["process_pass_count"] == len(results) else 1)


if __name__ == "__main__":
    main()
