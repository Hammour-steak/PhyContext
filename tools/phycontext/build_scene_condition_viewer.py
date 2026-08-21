#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from schema import iter_jsonl
from project_defaults import DATASET_MANIFEST, DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("outputs/training_data_review/scene_conditions.html")
BODY_NAMES = {
    0: "Environment",
    1: "Controlled object",
}
BODY_COLORS = {
    0: "#5c86bd",
    1: "#e84b57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline viewer for fixed PhysSweep scene conditions"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        required=DATASET_ROOT is None,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--scene-id", action="append", default=[])
    return parser.parse_args()


def _image_data(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _round(values: np.ndarray) -> list[float]:
    return np.round(values.astype(np.float64), 5).tolist()


def _css(colors: np.ndarray) -> list[str]:
    values = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in values.tolist()]


def _ranges(points: np.ndarray) -> list[list[float]]:
    low = np.quantile(points, 0.005, axis=0)
    high = np.quantile(points, 0.995, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.58, 1e-4)
    return [[float(value - radius), float(value + radius)] for value in center]


def _base_records(manifest: Path) -> list[dict]:
    return [record for record in iter_jsonl(manifest) if record["sweep"]["mode"] == "base"]


def _select_records(records: list[dict], scene_ids: list[str], count: int, seed: int) -> list[dict]:
    by_id = {record["base_scene_id"]: record for record in records}
    if scene_ids:
        missing = [scene_id for scene_id in scene_ids if scene_id not in by_id]
        if missing:
            raise ValueError(f"unknown base scene ids: {missing}")
        return [by_id[scene_id] for scene_id in scene_ids]
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=min(count, len(records)), replace=False))
    return [records[int(index)] for index in indices]


def _build_scene(record: dict, root: Path) -> dict:
    scene_path = root / record["conditioning"]["scene"]
    frame_path = root / record["conditioning"]["first_frame"]
    with np.load(scene_path, allow_pickle=False) as archive:
        xyz = archive["xyz_normalized"].astype(np.float32)
        rgb = archive["rgb"].astype(np.float32)
        rgb_valid = archive["rgb_valid"].astype(bool)
        body_id = archive["body_id"].astype(np.int16)
        radius = float(archive["context_radius"])
    oriented = np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])
    traces = []
    for body in sorted(int(value) for value in np.unique(body_id)):
        mask = body_id == body
        points = oriented[mask]
        colors = rgb[mask]
        color_valid = rgb_valid[mask]
        rgb_colors = _css(colors)
        if not color_valid.all():
            fallback = BODY_COLORS.get(body, "#9aa0a6")
            rgb_colors = [color if is_valid else fallback for color, is_valid in zip(rgb_colors, color_valid)]
        traces.append(
            {
                "name": BODY_NAMES.get(body, f"Body {body}"),
                "x": _round(points[:, 0]),
                "y": _round(points[:, 1]),
                "z": _round(points[:, 2]),
                "rgb": rgb_colors,
                "bodyColor": BODY_COLORS.get(body, "#9aa0a6"),
                "size": 3.4 if body == 1 else 2.5,
            }
        )
    object_mask = body_id == 1
    return {
        "key": record["base_scene_id"],
        "label": record["base_scene_id"],
        "frame": _image_data(frame_path),
        "traces": traces,
        "ranges": _ranges(oriented),
        "pointCount": int(len(body_id)),
        "objectPointCount": int(object_mask.sum()),
        "contextRadius": radius,
    }


def _html(scenes: list[dict]) -> str:
    payload = json.dumps(scenes, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysSweep Training Scene Conditions</title>
<style>
:root {{ color-scheme:dark; --bg:#17191c; --panel:#22252a; --line:#3b4048; --text:#f2f3f5; --muted:#abb1ba; --accent:#f0b849; }}
* {{ box-sizing:border-box; }}
html,body {{ width:100%;height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:14px/1.4 Inter,Segoe UI,Arial,sans-serif;letter-spacing:0; }}
body {{ display:grid;grid-template-columns:340px minmax(0,1fr); }}
aside {{ padding:18px;background:var(--panel);border-right:1px solid var(--line);overflow:auto; }}
h1 {{ margin:0 0 14px;font-size:18px; }}
#frame {{ width:100%;aspect-ratio:16/9;object-fit:cover;border:1px solid var(--line);border-radius:6px;background:#0d0f11; }}
#label {{ margin:12px 0 3px;font-weight:650;overflow-wrap:anywhere; }}
#meta {{ color:var(--muted);font-variant-numeric:tabular-nums; }}
#list {{ display:grid;gap:7px;margin-top:16px; }}
.scene {{ min-height:38px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:#2a2e34;color:var(--text);text-align:left;cursor:pointer;overflow-wrap:anywhere; }}
.scene.active {{ border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent); }}
main {{ min-width:0;min-height:0;display:grid;grid-template-rows:58px minmax(0,1fr); }}
header {{ display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 18px;border-bottom:1px solid var(--line); }}
#title {{ min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600; }}
.segmented {{ display:flex;padding:3px;border:1px solid var(--line);border-radius:6px;background:#111316; }}
.mode {{ min-width:72px;height:30px;border:0;border-radius:4px;background:transparent;color:var(--muted);cursor:pointer; }}
.mode.active {{ background:#363b43;color:var(--text); }}
#plot {{ width:100%;height:100%;min-width:0;min-height:0; }}
@media(max-width:760px) {{ html,body {{ overflow:auto; }} body {{ grid-template-columns:1fr;grid-template-rows:auto 68vh; }} aside {{ border-right:0;border-bottom:1px solid var(--line); }} }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<aside><h1>Training Scene Conditions</h1><img id="frame" alt="First frame"><div id="label"></div><div id="meta"></div><div id="list"></div></aside>
<main><header><div id="title"></div><div class="segmented"><button class="mode active" data-mode="body">Body</button><button class="mode" data-mode="rgb">RGB</button></div></header><div id="plot"></div></main>
<script>
const scenes={payload}; let selected=0; let mode="body";
const plot=document.getElementById("plot");
function traces(scene) {{ return scene.traces.map(trace=>({{type:"scatter3d",mode:"markers",name:trace.name,x:trace.x,y:trace.y,z:trace.z,hoverinfo:"name",marker:{{size:trace.size,opacity:0.92,color:mode==="rgb"?trace.rgb:trace.bodyColor,line:{{width:0}}}}}})); }}
function layout(scene) {{ const axis=range=>({{visible:false,range}}); return {{margin:{{l:0,r:0,t:0,b:0}},paper_bgcolor:"#17191c",plot_bgcolor:"#17191c",legend:{{x:0.01,y:0.99,bgcolor:"rgba(23,25,28,.75)"}},scene:{{bgcolor:"#17191c",aspectmode:"cube",xaxis:axis(scene.ranges[0]),yaxis:axis(scene.ranges[1]),zaxis:axis(scene.ranges[2]),camera:{{eye:{{x:1.15,y:-1.8,z:.9}},up:{{x:0,y:0,z:1}}}},dragmode:"orbit"}},uirevision:scene.key}}; }}
function render(index) {{ selected=index; const scene=scenes[index]; document.getElementById("frame").src=scene.frame; document.getElementById("label").textContent=scene.label; document.getElementById("title").textContent=scene.label; document.getElementById("meta").textContent=`${{scene.pointCount.toLocaleString()}} points / ${{scene.objectPointCount.toLocaleString()}} object / radius ${{scene.contextRadius.toFixed(3)}} m`; document.querySelectorAll(".scene").forEach((item,i)=>item.classList.toggle("active",i===index)); Plotly.react(plot,traces(scene),layout(scene),{{responsive:true,scrollZoom:true,displaylogo:false}}); }}
scenes.forEach((scene,index)=>{{const button=document.createElement("button");button.className="scene";button.textContent=`${{String(index+1).padStart(2,"0")}}  ${{scene.label}}`;button.onclick=()=>render(index);document.getElementById("list").appendChild(button);}});
document.querySelectorAll(".mode").forEach(button=>button.onclick=()=>{{mode=button.dataset.mode;document.querySelectorAll(".mode").forEach(item=>item.classList.toggle("active",item===button));render(selected);}});
render(0);
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    records = _base_records(dataset_root / DATASET_MANIFEST)
    selected = _select_records(records, args.scene_id, args.count, args.seed)
    scenes = [_build_scene(record, dataset_root) for record in selected]
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html(scenes), encoding="utf-8")
    print(json.dumps({"output": str(output), "scene_ids": [item["key"] for item in scenes]}, indent=2))


if __name__ == "__main__":
    main()
