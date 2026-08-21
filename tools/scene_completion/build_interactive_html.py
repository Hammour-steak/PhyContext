from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


SOURCE_GROUPS = (
    ("visible_environment", "Visible environment", 0, 0, "#4a90e2", 1.25),
    ("completed_environment", "Completed environment", 1, 0, "#4fbe7d", 1.5),
    ("reconstructed_object", "Aligned InstantMesh object", 2, 1, "#ff4a5c", 1.9),
)

POINT_LIMITS = {
    "visible_environment": 30000,
    "completed_environment": 10000,
    "reconstructed_object": 16384,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline multi-scene point-cloud viewer")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        nargs=2,
        action="append",
        metavar=("DIRECTORY", "LABEL"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def encode_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sample_indices(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return np.sort(rng.choice(indices, size=limit, replace=False))


def rgb_css(colors: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in colors.tolist()]


def rounded(values: np.ndarray) -> list[float]:
    return np.round(values.astype(np.float64), 5).tolist()


def equal_ranges(points: np.ndarray, padding: float) -> list[list[float]]:
    low = np.quantile(points, 0.002, axis=0)
    high = np.quantile(points, 0.998, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * padding, 1e-5)
    return [[float(value - radius), float(value + radius)] for value in center]


def build_scene(scene_dir: Path, label: str, seed: int) -> dict:
    archive_path = scene_dir / "scene.npz"
    report_path = scene_dir / "report.json"
    for path in (archive_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    reported_image = Path(report.get("input_image", ""))
    image_path = reported_image if reported_image.is_file() else scene_dir / "input_preprocessed.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    with np.load(archive_path, allow_pickle=False) as archive:
        xyz = archive["xyz"].astype(np.float32)
        rgb = archive["rgb"].astype(np.uint8)
        source = archive["source"].astype(np.uint8)
        body_id = archive["body_id"].astype(np.int16)

    rng = np.random.default_rng(seed)
    traces = []
    displayed = []

    for key, name, source_id, body, semantic_color, size in SOURCE_GROUPS:
        indices = np.flatnonzero((source == source_id) & (body_id == body))
        indices = sample_indices(indices, POINT_LIMITS[key], rng)
        points = xyz[indices]
        colors = rgb[indices]
        displayed.append(len(indices))
        traces.append(
            {
                "key": key,
                "name": name,
                "x": rounded(points[:, 0]),
                "y": rounded(points[:, 2]),
                "z": rounded(-points[:, 1]),
                "rgb": rgb_css(colors),
                "sourceColor": semantic_color,
                "size": size,
            }
        )

    scene_indices = np.concatenate(
        [
            np.flatnonzero((source == source_id) & (body_id == body))
            for _, _, source_id, body, _, _ in SOURCE_GROUPS
        ]
    )
    object_indices = np.flatnonzero(body_id == 1)
    scene_oriented = np.column_stack(
        [xyz[scene_indices, 0], xyz[scene_indices, 2], -xyz[scene_indices, 1]]
    )
    object_oriented = np.column_stack(
        [xyz[object_indices, 0], xyz[object_indices, 2], -xyz[object_indices, 1]]
    )

    return {
        "key": scene_dir.name,
        "label": label,
        "qualityPassed": bool(report["quality_gate"]["passed"]),
        "image": encode_image(image_path),
        "traces": traces,
        "ranges": {
            "scene": equal_ranges(scene_oriented, 0.54),
            "object": equal_ranges(object_oriented, 0.62),
        },
        "counts": report["point_counts"],
        "displayed": int(sum(displayed)),
        "timings": report.get("timings", {}),
    }


def build_html(scenes: list[dict]) -> str:
    scene_json = json.dumps(scenes, ensure_ascii=True, separators=(",", ":"))
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhyContext Scene Completion</title>
<style>
:root {{ color-scheme: dark; --bg:#15171a; --panel:#202328; --line:#373b42; --text:#f2f3f5; --muted:#a8adb5; --accent:#f6c344; }}
* {{ box-sizing:border-box; }}
html, body {{ width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font:14px/1.4 Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }}
body {{ display:grid; grid-template-columns:320px minmax(0,1fr); }}
aside {{ min-width:0; padding:18px; background:var(--panel); border-right:1px solid var(--line); overflow:auto; }}
h1 {{ margin:0 0 14px; font-size:18px; font-weight:650; letter-spacing:0; }}
#frame {{ display:block; width:100%; aspect-ratio:16/9; object-fit:cover; border:1px solid var(--line); border-radius:6px; background:#0d0f11; }}
#scene-label {{ margin:12px 0 2px; font-size:16px; font-weight:650; }}
#scene-meta {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
.scene-list {{ display:grid; gap:7px; margin-top:18px; }}
.scene-button {{ width:100%; min-height:38px; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:#292d33; color:var(--text); text-align:left; cursor:pointer; }}
.scene-button:hover {{ border-color:#737a84; }}
.scene-button.rejected {{ color:#ffb3a7; }}
.scene-button.active {{ border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }}
main {{ min-width:0; min-height:0; display:grid; grid-template-rows:58px minmax(0,1fr); }}
header {{ min-width:0; display:flex; align-items:center; justify-content:space-between; gap:14px; padding:0 18px; border-bottom:1px solid var(--line); }}
#plot-title {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:15px; font-weight:600; }}
.toolbar {{ flex:0 0 auto; display:flex; align-items:center; gap:8px; }}
.segmented {{ flex:0 0 auto; display:flex; padding:3px; border:1px solid var(--line); border-radius:6px; background:#111316; }}
.mode-button {{ min-width:72px; height:30px; border:0; border-radius:4px; background:transparent; color:var(--muted); cursor:pointer; }}
.mode-button.active {{ background:#343941; color:var(--text); }}
#plot {{ width:100%; height:100%; min-width:0; min-height:0; }}
#loading {{ position:fixed; inset:58px 0 0 320px; display:grid; place-items:center; pointer-events:none; color:var(--muted); background:var(--bg); z-index:5; }}
@media (max-width:760px) {{
  html,body {{ overflow:auto; }}
  body {{ grid-template-columns:1fr; grid-template-rows:auto 68vh; }}
  aside {{ border-right:0; border-bottom:1px solid var(--line); }}
  main {{ min-height:68vh; }}
  #loading {{ inset:auto 0 0; height:68vh; }}
}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<aside>
<h1>PhyContext Scene Completion</h1>
  <img id="frame" alt="Selected first frame">
  <div id="scene-label"></div>
  <div id="scene-meta"></div>
  <div id="scene-list" class="scene-list"></div>
</aside>
<main>
  <header>
    <div id="plot-title"></div>
    <div class="toolbar">
      <div class="segmented" aria-label="Point-cloud framing">
        <button class="mode-button view-button active" data-view="scene">Scene</button>
        <button class="mode-button view-button" data-view="object">Object</button>
      </div>
      <div class="segmented" aria-label="Point color mode">
        <button class="mode-button color-button" data-mode="source">Sources</button>
        <button class="mode-button color-button active" data-mode="rgb">RGB</button>
      </div>
    </div>
  </header>
  <div id="plot"></div>
</main>
<div id="loading">Loading point cloud...</div>
<script>
const scenes={scene_json};
let currentScene=0;
let colorMode="rgb";
let viewMode="scene";
const plot=document.getElementById("plot");
const frame=document.getElementById("frame");
const sceneLabel=document.getElementById("scene-label");
const sceneMeta=document.getElementById("scene-meta");
const plotTitle=document.getElementById("plot-title");
const loading=document.getElementById("loading");

function tracesFor(scene) {{
  const selected=scene.traces.filter(trace => viewMode === "scene"
    ? true
    : trace.key === "reconstructed_object");
  let x=[];
  let y=[];
  let z=[];
  let colors=[];
  selected.forEach(trace => {{
    x=x.concat(trace.x);
    y=y.concat(trace.y);
    z=z.concat(trace.z);
    colors=colors.concat(colorMode === "rgb"
      ? trace.rgb
      : Array(trace.x.length).fill(trace.sourceColor));
  }});
  return [{{
    type:"scatter3d",
    mode:"markers",
    name:viewMode === "scene" ? "Completed scene" : "Object comparison",
    x,
    y,
    z,
    hoverinfo:"skip",
    marker:{{
      size:viewMode === "scene" ? 2.15 : 2.8,
      opacity:1,
      color:colors,
      line:{{width:0}}
    }}
  }}];
}}

function layoutFor(scene) {{
  const axis=range => ({{visible:false,showgrid:false,zeroline:false,range}});
  return {{
    margin:{{l:0,r:0,t:0,b:0}},
    paper_bgcolor:"#15171a",
    plot_bgcolor:"#15171a",
    showlegend:false,
    scene:{{
      bgcolor:"#15171a",
      aspectmode:"cube",
      xaxis:axis(scene.ranges[viewMode][0]),
      yaxis:axis(scene.ranges[viewMode][1]),
      zaxis:axis(scene.ranges[viewMode][2]),
      camera:{{eye:{{x:1.05,y:-1.9,z:0.88}},up:{{x:0,y:0,z:1}}}},
      dragmode:"orbit"
    }},
    uirevision:scene.key + "-" + viewMode
  }};
}}

function updateDetails(scene) {{
  const status=scene.qualityPassed ? "PASS" : "REJECT";
  frame.src=scene.image;
  sceneLabel.textContent=status + " / " + scene.label;
  plotTitle.textContent=status + " / " + scene.label + " / completed 3D scene";
  const counts=scene.counts;
  const seconds=Object.values(scene.timings).reduce((total,value)=>total+Number(value||0),0);
  sceneMeta.textContent=`${{counts.scene.toLocaleString()}} total points / ${{scene.displayed.toLocaleString()}} displayed / ${{seconds.toFixed(1)}} s`;
  document.querySelectorAll(".scene-button").forEach((button,index)=>button.classList.toggle("active",index===currentScene));
}}

async function renderScene(index) {{
  currentScene=index;
  const scene=scenes[index];
  updateDetails(scene);
  await Plotly.react(plot,tracesFor(scene),layoutFor(scene),{{responsive:true,scrollZoom:true,displaylogo:false}});
  loading.style.display="none";
}}

scenes.forEach((scene,index)=>{{
  const button=document.createElement("button");
  button.className="scene-button" + (scene.qualityPassed ? "" : " rejected");
  const status=scene.qualityPassed ? "PASS" : "REJECT";
  button.textContent=`${{String(index+1).padStart(2,"0")}}  [${{status}}]  ${{scene.label}}`;
  button.addEventListener("click",()=>renderScene(index));
  document.getElementById("scene-list").appendChild(button);
}});

document.querySelectorAll(".color-button").forEach(button=>{{
  button.addEventListener("click",()=>{{
    colorMode=button.dataset.mode;
    document.querySelectorAll(".color-button").forEach(item=>item.classList.toggle("active",item===button));
    renderScene(currentScene);
  }});
}});

document.querySelectorAll(".view-button").forEach(button=>{{
  button.addEventListener("click",()=>{{
    viewMode=button.dataset.view;
    document.querySelectorAll(".view-button").forEach(item=>item.classList.toggle("active",item===button));
    renderScene(currentScene);
  }});
}});

renderScene(0);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    scenes = [
        build_scene(args.root / directory, label, args.seed + index)
        for index, (directory, label) in enumerate(args.scene)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(scenes), encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
