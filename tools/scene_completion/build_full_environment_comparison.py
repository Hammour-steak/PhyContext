#!/usr/bin/env python3
"""Build an offline, full-density before/after environment point-cloud viewer."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def _encode_float32(values: np.ndarray) -> str:
    packed = np.asarray(values, dtype="<f4").reshape(-1)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _equal_ranges(points: np.ndarray, padding: float = 0.54) -> list[list[float]]:
    low = np.quantile(points, 0.002, axis=0)
    high = np.quantile(points, 0.998, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * padding, 1e-5)
    return [
        [round(float(value - radius), 5), round(float(value + radius), 5)]
        for value in center
    ]


def _pack_points(points: np.ndarray) -> dict:
    oriented = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    return {
        "count": int(len(oriented)),
        "x": _encode_float32(oriented[:, 0]),
        "y": _encode_float32(oriented[:, 1]),
        "z": _encode_float32(oriented[:, 2]),
    }


def build_scene(scene_dir: Path) -> dict:
    archive_path = scene_dir / "scene.npz"
    report_path = scene_dir / "report.json"
    if not archive_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"incomplete scene package: {scene_dir}")

    with np.load(archive_path, allow_pickle=False) as archive:
        xyz = archive["xyz"].astype(np.float32)
        source = archive["source"].astype(np.uint8)
        body_id = archive["body_id"].astype(np.int16)

    observed = xyz[(source == 0) & (body_id == 0)]
    completed = xyz[(source == 1) & (body_id == 0)]
    if not len(observed) or not len(completed):
        raise ValueError(f"scene lacks observed or completed environment: {scene_dir}")
    all_environment = np.concatenate([observed, completed], axis=0)
    oriented_environment = np.column_stack(
        [all_environment[:, 0], all_environment[:, 2], -all_environment[:, 1]]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    measurements = report["quality_gate"]["measurements"]
    return {
        "label": scene_dir.name,
        "observed": _pack_points(observed),
        "completed": _pack_points(completed),
        "ranges": _equal_ranges(oriented_environment),
        "coverage": float(measurements["environment_completion_fraction"]),
        "seamP90": float(measurements["environment_depth_seam_p90_error"]),
    }


def build_html(scenes: list[dict]) -> str:
    scene_json = json.dumps(scenes, ensure_ascii=True, separators=(",", ":"))
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhyContext Full Environment Completion Comparison</title>
<style>
:root {{ color-scheme:dark; --bg:#15171a; --panel:#202328; --line:#3b4048; --text:#f2f3f5; --muted:#a8adb5; --blue:#4a90e2; --green:#4fbe7d; }}
* {{ box-sizing:border-box; }}
html,body {{ width:100%; min-height:100%; margin:0; background:var(--bg); color:var(--text); font:14px/1.4 Inter,"Segoe UI",Arial,sans-serif; letter-spacing:0; }}
body {{ padding:16px; }}
header {{ display:flex; align-items:end; justify-content:space-between; gap:18px; margin-bottom:10px; }}
h1 {{ margin:0 0 4px; font-size:20px; font-weight:650; letter-spacing:0; }}
#metrics {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
.controls {{ display:flex; align-items:center; gap:8px; min-width:0; }}
label {{ color:var(--muted); font-size:12px; }}
select {{ width:min(520px,48vw); height:36px; padding:0 30px 0 10px; border:1px solid var(--line); border-radius:5px; background:var(--panel); color:var(--text); font:inherit; }}
.legend {{ display:flex; gap:18px; margin-bottom:10px; color:var(--muted); font-size:12px; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.swatch {{ width:10px; height:10px; border-radius:2px; }}
.observed {{ background:var(--blue); }}
.completed {{ background:var(--green); }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
section {{ min-width:0; border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
h2 {{ margin:10px 12px 1px; font-size:15px; font-weight:650; letter-spacing:0; }}
.subtitle {{ margin:0 12px; color:var(--muted); font-size:12px; }}
.plot {{ width:100%; height:calc(100vh - 150px); min-height:520px; }}
#loading {{ position:fixed; inset:0; display:grid; place-items:center; background:rgba(21,23,26,.88); color:var(--muted); z-index:10; pointer-events:none; }}
@media (max-width:820px) {{
  header {{ align-items:stretch; flex-direction:column; }}
  .controls {{ flex-wrap:wrap; }}
  select {{ flex:1 1 260px; width:auto; }}
  .grid {{ grid-template-columns:1fr; }}
  .plot {{ height:62vh; min-height:380px; }}
}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<header>
  <div>
    <h1>环境点云补全：全密度对比</h1>
    <div id="metrics"></div>
  </div>
  <div class="controls">
    <label for="scene-select">场景</label>
    <select id="scene-select"></select>
  </div>
</header>
<div class="legend">
  <span><i class="swatch observed"></i>原始可见环境</span>
  <span><i class="swatch completed"></i>DepthLab 新增区域</span>
</div>
<div class="grid">
  <section>
    <h2>补全前</h2>
    <div class="subtitle">全部 VGGT 环境点</div>
    <div id="before" class="plot"></div>
  </section>
  <section>
    <h2>补全后</h2>
    <div class="subtitle">全部原始点与全部补全点</div>
    <div id="after" class="plot"></div>
  </section>
</div>
<div id="loading">正在加载完整点云...</div>
<script>
const scenes={scene_json};
const decoded=new Map();
const before=document.getElementById("before");
const after=document.getElementById("after");
const selector=document.getElementById("scene-select");
const metrics=document.getElementById("metrics");
const loading=document.getElementById("loading");
let syncing=false;
const initialCamera={{eye:{{x:1.05,y:-1.9,z:.88}},up:{{x:0,y:0,z:1}}}};

function decodeFloat32(encoded) {{
  const raw=atob(encoded);
  const bytes=new Uint8Array(raw.length);
  for(let index=0;index<raw.length;index+=1) bytes[index]=raw.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}}

function decodePoints(record) {{
  return {{x:decodeFloat32(record.x),y:decodeFloat32(record.y),z:decodeFloat32(record.z),count:record.count}};
}}

function decodedScene(index) {{
  if(!decoded.has(index)) {{
    const scene=scenes[index];
    decoded.set(index,{{...scene,observed:decodePoints(scene.observed),completed:decodePoints(scene.completed)}});
  }}
  return decoded.get(index);
}}

function trace(points,color,name,size) {{
  return {{type:"scatter3d",mode:"markers",name,x:points.x,y:points.y,z:points.z,hoverinfo:"skip",marker:{{size,opacity:.96,color,line:{{width:0}}}}}};
}}

function layout(scene) {{
  const axis=range=>({{visible:false,showgrid:false,zeroline:false,range}});
  return {{
    margin:{{l:0,r:0,t:0,b:0}},paper_bgcolor:"#15171a",plot_bgcolor:"#15171a",showlegend:false,
    scene:{{bgcolor:"#15171a",aspectmode:"cube",xaxis:axis(scene.ranges[0]),yaxis:axis(scene.ranges[1]),zaxis:axis(scene.ranges[2]),camera:initialCamera,dragmode:"orbit"}},
    uirevision:scene.label
  }};
}}

async function render(index) {{
  loading.style.display="grid";
  await new Promise(resolve=>requestAnimationFrame(resolve));
  const scene=decodedScene(index);
  const config={{responsive:true,scrollZoom:true,displaylogo:false}};
  metrics.textContent=`${{scene.observed.count.toLocaleString()}} 个原始点  ·  ${{scene.completed.count.toLocaleString()}} 个补全点  ·  覆盖率 ${{(scene.coverage*100).toFixed(1)}}%  ·  接缝 P90 ${{(scene.seamP90*100).toFixed(2)}}%`;
  await Promise.all([
    Plotly.react(before,[trace(scene.observed,"#4a90e2","原始可见环境",1.25)],layout(scene),config),
    Plotly.react(after,[trace(scene.observed,"#4a90e2","原始可见环境",1.25),trace(scene.completed,"#4fbe7d","DepthLab 新增区域",2.2)],layout(scene),config)
  ]);
  loading.style.display="none";
}}

function synchronize(source,target) {{
  source.on("plotly_relayout",event=>{{
    if(syncing||!event["scene.camera"]) return;
    syncing=true;
    Plotly.relayout(target,{{"scene.camera":event["scene.camera"]}}).finally(()=>{{syncing=false;}});
  }});
}}

scenes.forEach((scene,index)=>{{
  const option=document.createElement("option");
  option.value=String(index);
  option.textContent=scene.label.replace(/^\\d{{2}}_/,"");
  selector.appendChild(option);
}});
selector.addEventListener("change",()=>render(Number(selector.value)));
render(0).then(()=>{{synchronize(before,after);synchronize(after,before);}});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    scene_dirs = sorted(path for path in root.glob("[0-9][0-9]_*") if path.is_dir())
    if not scene_dirs:
        raise ValueError(f"no reconstruction scenes found in {root}")
    scenes = [build_scene(scene_dir) for scene_dir in scene_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(scenes), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scene_count": len(scenes),
                "observed_points": sum(scene["observed"]["count"] for scene in scenes),
                "completed_points": sum(scene["completed"]["count"] for scene in scenes),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
