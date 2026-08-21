#!/usr/bin/env python3
"""Build a self-contained GT-vs-recovered scene and video comparison viewer."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-scene", type=Path, required=True)
    parser.add_argument("--recovered-scene", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--gt-video", type=Path, required=True)
    parser.add_argument("--recovered-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_scene(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        object_xyz = archive["object_xyz_camera_m"].astype(np.float32)
        environment_xyz = archive["environment_xyz_camera_m"].astype(np.float32)
        intrinsics = archive["camera_intrinsics_normalized"].astype(np.float32)
        friction = archive["environment_friction"].astype(np.float32).reshape(-1)
        restitution = archive["environment_restitution"].astype(np.float32).reshape(-1)

    # Camera coordinates are displayed as x-right, z-up, -y-forward.
    def orient(points: np.ndarray) -> np.ndarray:
        return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])

    object_points = orient(object_xyz)
    environment_points = orient(environment_xyz)
    all_points = np.concatenate([object_points, environment_points], axis=0)
    return {
        "object": np.round(object_points, 5).tolist(),
        "environment": np.round(environment_points, 5).tolist(),
        "intrinsics": np.round(intrinsics, 6).tolist(),
        "object_count": int(len(object_points)),
        "environment_count": int(len(environment_points)),
        "friction": {
            "min": float(friction.min()),
            "max": float(friction.max()),
            "mean": float(friction.mean()),
        },
        "restitution": {
            "min": float(restitution.min()),
            "max": float(restitution.max()),
            "mean": float(restitution.mean()),
        },
        "bounds": {
            "min": np.round(all_points.min(axis=0), 5).tolist(),
            "max": np.round(all_points.max(axis=0), 5).tolist(),
        },
    }


def image_data(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _video_src(path: Path, output: Path) -> str:
    try:
        return path.resolve().relative_to(output.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def build_html(args: argparse.Namespace) -> str:
    gt = load_scene(args.gt_scene)
    recovered = load_scene(args.recovered_scene)
    payload = {
        "gt": gt,
        "recovered": recovered,
        "frame": image_data(args.first_frame),
        "gt_video": _video_src(args.gt_video, args.output),
        "recovered_video": _video_src(args.recovered_video, args.output),
    }
    all_points = np.concatenate(
        [
            np.asarray(gt["object"], dtype=np.float32),
            np.asarray(gt["environment"], dtype=np.float32),
            np.asarray(recovered["object"], dtype=np.float32),
            np.asarray(recovered["environment"], dtype=np.float32),
        ],
        axis=0,
    )
    low = np.quantile(all_points, 0.005, axis=0)
    high = np.quantile(all_points, 0.995, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.58, 1e-4)
    payload["range"] = {
        "x": [float(center[0] - radius), float(center[0] + radius)],
        "y": [float(center[1] - radius), float(center[1] + radius)],
        "z": [float(center[2] - radius), float(center[2] + radius)],
    }
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    plotly_js = get_plotlyjs()
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysSweep GT vs Recovered Scene</title>
<style>
:root {{ color-scheme:dark; --bg:#15171a; --panel:#20242a; --line:#3b424c; --text:#f1f3f5; --muted:#aeb6c0; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; min-height:100%; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif; }}
body {{ padding:18px; }}
main {{ max-width:1500px; margin:auto; }}
h1 {{ margin:0 0 5px; font-size:22px; }}
p {{ margin:5px 0 14px; color:var(--muted); }}
.videos {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
figure {{ margin:0; padding:10px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }}
figcaption {{ margin-bottom:8px; font-weight:650; }}
video {{ display:block; width:100%; background:#000; }}
.stats {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:14px 0; }}
.stat {{ padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:6px; color:var(--muted); white-space:pre-wrap; }}
.stat strong {{ color:var(--text); }}
.clouds {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.cloud {{ min-width:0; height:620px; background:#111316; border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.cloud-title {{ padding:10px 12px; height:42px; background:rgba(32,36,42,.96); font-weight:650; }}
.plot {{ height:578px; }}
img {{ display:block; max-width:420px; width:100%; margin:14px 0; border:1px solid var(--line); }}
@media(max-width:850px) {{ .videos,.stats,.clouds {{ grid-template-columns:1fr; }} .cloud {{ height:520px; }} .plot {{ height:478px; }} }}
</style>
<script>{plotly_js}</script>
</head>
<body>
<main>
<h1>PhysSweep: GT Scene vs Image-Recovered Scene</h1>
<p>Same first frame, trajectory condition, physical values, adapter, and seed. Only the scene condition is changed. Drag either 3D panel to orbit.</p>
<img src="{payload["frame"]}" alt="Input first frame">
<section class="videos">
  <figure><figcaption>GT scene condition · formal 30 steps</figcaption><video controls preload="metadata" src="{payload["gt_video"]}"></video></figure>
  <figure><figcaption>Image-recovered scene condition · formal 30 steps</figcaption><video controls preload="metadata" src="{payload["recovered_video"]}"></video></figure>
</section>
<section class="stats">
  <div class="stat" id="gtStats"></div>
  <div class="stat" id="recoveredStats"></div>
</section>
<section class="clouds">
  <section class="cloud"><div class="cloud-title">GT model input: 2,048 object + 8,192 environment points</div><div id="gtPlot" class="plot"></div></section>
  <section class="cloud"><div class="cloud-title">Recovered model input: 2,048 object + 8,192 environment points</div><div id="recoveredPlot" class="plot"></div></section>
</section>
</main>
<script>
const data = {payload_json};
const commonRange = data.range;
function traces(scene) {{
  const make = (points, name, color, size) => ({{
    type:'scatter3d', mode:'markers', name, x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2]),
    marker:{{size, color, opacity:.86, line:{{width:0}}}}, hoverinfo:'name+x+y+z'
  }});
  return [make(scene.environment,'Environment','#5c86bd',2.5), make(scene.object,'Controlled object','#f28e2b',4.2)];
}}
function layout() {{
  const axis = title => ({{title, range:commonRange[title[0]], showbackground:false, showgrid:false, zeroline:false, showticklabels:false}});
  return {{margin:{{l:0,r:0,t:0,b:0}}, paper_bgcolor:'#111316', plot_bgcolor:'#111316', showlegend:true,
    legend:{{x:.01,y:.99,bgcolor:'rgba(17,19,22,.75)'}},
    scene:{{aspectmode:'cube', xaxis:axis('x'), yaxis:axis('y'), zaxis:axis('z'), camera:{{eye:{{x:1.35,y:-1.7,z:1.0}},up:{{x:0,y:0,z:1}}}}}}
  }};
}}
function stats(scene, label) {{
  const fmt = value => JSON.stringify(value);
  return `<strong>${{label}}</strong>\npoints: ${{scene.object_count + scene.environment_count}} (object ${{scene.object_count}}, environment ${{scene.environment_count}})\n`+
    `camera intrinsics [fx, fy, cx, cy]: ${{fmt(scene.intrinsics)}}\n`+
    `friction min/mean/max: ${{scene.friction.min.toFixed(4)}} / ${{scene.friction.mean.toFixed(4)}} / ${{scene.friction.max.toFixed(4)}}\n`+
    `restitution min/mean/max: ${{scene.restitution.min.toFixed(4)}} / ${{scene.restitution.mean.toFixed(4)}} / ${{scene.restitution.max.toFixed(4)}}\n`+
    `display bounds min: ${{fmt(scene.bounds.min)}}\n`+
    `display bounds max: ${{fmt(scene.bounds.max)}}`;
}}
document.getElementById('gtStats').innerHTML=stats(data.gt,'GT scene condition');
document.getElementById('recoveredStats').innerHTML=stats(data.recovered,'Image-recovered scene condition');
Plotly.newPlot('gtPlot',traces(data.gt),layout(),{{responsive:true,displaylogo:false,scrollZoom:true}});
Plotly.newPlot('recoveredPlot',traces(data.recovered),layout(),{{responsive:true,displaylogo:false,scrollZoom:true}});
</script>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(args), encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
