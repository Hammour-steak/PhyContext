#!/usr/bin/env python3
"""Build an interactive audit page for exported point trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_MANIFEST = Path(
    "datasets/physweep_training/point_trajectories/manifest.json"
)
DEFAULT_SOURCE_MANIFEST = Path("datasets/physweep_training/manifest.jsonl")
DEFAULT_OUTPUT = Path(
    "outputs/point_trajectory_validation/point_trajectory_validation.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--display-points", type=int, default=256)
    return parser.parse_args()


def decode_ids(values: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def compact_array(value: np.ndarray, digits: int = 6) -> list:
    return np.round(np.asarray(value), digits).tolist()


def choose_records(records: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("sample-count must be positive")
    groups: dict[str, dict] = {}
    for record in records:
        source_scene = str(record.get("source_scene", record["sample_id"]))
        groups.setdefault(source_scene, record)
    candidates = list(groups.values())
    if len(candidates) <= count:
        return candidates
    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    return [candidates[int(index)] for index in np.unique(indices)]


def sample_payload(
    root: Path,
    record: dict,
    display_points: int,
    video_href: str,
    video_source: str,
) -> dict:
    trajectory_path = root / record["path"]
    with np.load(trajectory_path, allow_pickle=False) as archive:
        time_s = np.asarray(archive["time_s"], dtype=np.float32)
        points_world = np.asarray(archive["points_world_m"], dtype=np.float32)
        points_camera = np.asarray(archive["points_camera_m"], dtype=np.float32)
        tracks = np.asarray(archive["tracks_xy_px"], dtype=np.float32)
        valid = np.asarray(archive["valid"], dtype=np.uint8).astype(bool)
        object_ids = decode_ids(archive["object_ids"])
        image_size = [int(value) for value in archive["image_size_px"]]

    if points_world.ndim != 4 or points_world.shape[2] != 2048:
        raise ValueError(f"unexpected point shape in {trajectory_path}: {points_world.shape}")
    point_count = points_world.shape[2]
    display_count = min(int(display_points), point_count)
    point_indices = np.rint(
        np.linspace(0, point_count - 1, display_count)
    ).astype(np.int64)
    point_indices = np.unique(point_indices)
    world_display = points_world[:, :, point_indices]
    camera_display = points_camera[:, :, point_indices]
    tracks_display = tracks[:, :, point_indices]
    valid_display = valid[:, :, point_indices]

    centers = points_world.mean(axis=2)
    center_displacement = np.linalg.norm(centers - centers[0:1], axis=-1)
    relative = points_world - centers[:, :, None, :]
    initial_radius = np.linalg.norm(relative[0], axis=-1)
    radius_error = np.abs(np.linalg.norm(relative, axis=-1) - initial_radius[None, :])
    per_object_valid = valid.mean(axis=(0, 2))
    camera_width, camera_height = image_size
    normalized_tracks = tracks_display.copy()
    normalized_tracks[..., 0] /= max(camera_width, 1)
    normalized_tracks[..., 1] /= max(camera_height, 1)
    normalized_tracks[~valid_display] = np.nan

    return {
        "sample_id": record["sample_id"],
        "source_scene": record.get("source_scene", ""),
        "video": video_href,
        "video_source": video_source,
        "object_ids": object_ids,
        "time_s": compact_array(time_s, 5),
        "image_size_px": image_size,
        "full_point_count": point_count,
        "display_point_count": int(len(point_indices)),
        "world_tracks": compact_array(np.transpose(world_display, (1, 0, 2, 3))),
        "camera_tracks": compact_array(np.transpose(normalized_tracks, (1, 0, 2, 3))),
        "valid_tracks": np.transpose(valid_display, (1, 0, 2)).tolist(),
        "initial_surface_world": compact_array(
            np.transpose(points_world[0], (0, 1, 2))
        ),
        "current_surface_world": compact_array(points_world[0]),
        "metrics": {
            "object_count": len(object_ids),
            "frame_count": int(len(time_s)),
            "duration_s": round(float(time_s[-1] - time_s[0]), 4),
            "max_center_displacement_m": round(float(center_displacement.max()), 5),
            "valid_ratio": [round(float(value), 5) for value in per_object_valid],
            "max_rigid_shape_error_m": round(float(radius_error.max()), 8),
            "initial_alignment_error_m": [
                round(float(value), 10)
                for value in record.get("initial_alignment_error_m", [])
            ],
            "max_depth_m": round(float(np.nanmax(np.where(valid, points_camera[..., 2], np.nan))), 4),
            "min_depth_m": round(float(np.nanmin(np.where(valid, points_camera[..., 2], np.nan))), 4),
        },
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhyContext Point-Trajectory Validation</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{color-scheme:dark;--bg:#101419;--panel:#1a222b;--line:#344250;--text:#edf3f7;--muted:#a8b7c2;--cyan:#59c7ff;--orange:#f4a261;--green:#69d7a2;--good:#79e2aa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:0 auto;padding:22px 18px 40px}h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:24px 0 10px}.lead{color:var(--muted);max-width:1100px}.toolbar,.metrics,.audit,.plot,.video-panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:16px 0}.toolbar label{color:var(--muted)}select,input[type=range]{accent-color:var(--cyan)}select{background:#0f151a;color:var(--text);border:1px solid var(--line);padding:7px 9px;border-radius:5px;max-width:100%;min-width:260px}.frame{display:flex;align-items:center;gap:10px;flex:1;min-width:260px}.frame input{width:100%}.frame output{width:150px;color:var(--cyan);text-align:right}.metrics{display:grid;grid-template-columns:repeat(6,minmax(100px,1fr));gap:10px;margin-bottom:12px}.metric small{display:block;color:var(--muted);font-size:11px}.metric b{display:block;margin-top:3px}.pass{color:var(--good)}.video-panel{margin-bottom:12px}.video-panel h2{margin:0 0 8px}.video-wrap{position:relative;width:min(100%,900px);aspect-ratio:16/9;background:#050708;border-radius:5px;overflow:hidden}.video-wrap video,.video-wrap canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.video-wrap video{object-fit:contain;background:#050708}.video-wrap canvas{pointer-events:none}.video-note{color:var(--muted);font-size:12px;margin:8px 0 0}.plots{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:12px}.plot{min-width:0}.plot h3{margin:0 0 4px;font-size:15px}.plot p{margin:0 0 4px;color:var(--muted);font-size:12px}.chart{height:590px}.audit{margin-top:14px;overflow:auto}table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}@media(max-width:1000px){.plots{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,minmax(100px,1fr))}.chart{height:470px}}@media(max-width:600px){main{padding:15px 10px}.metrics{grid-template-columns:repeat(2,minmax(100px,1fr))}.chart{height:390px}select{min-width:0;width:100%}}
</style>
</head>
<body><main>
<h1>PhyContext Point-Trajectory Validation</h1>
<p class="lead">Each sample stores 2048 fixed-identity surface points per dynamic object for every simulator frame. The viewer displays a thinned 256-point subset for responsiveness, while the audit values are computed from all 2048 points. The video overlay, 3D tracks, and camera projection are synchronized to the same simulator frame.</p>
<div class="toolbar"><label for="sampleSelect">sample</label><select id="sampleSelect"></select><div class="frame"><label for="frameSlider">frame</label><input id="frameSlider" type="range" min="0" value="0"><output id="frameOutput"></output></div></div>
<div class="metrics" id="metrics"></div>
<section class="video-panel"><h2>Rendered video + projected point tracks</h2><div class="video-wrap"><video id="videoPlayer" controls playsinline preload="metadata"></video><canvas id="videoOverlay"></canvas></div><p class="video-note" id="videoNote"></p></section>
<div class="plots"><section class="plot"><h3>World-space point tracks</h3><p>Gray points: initial full 2048-point surface. Colored points: current displayed subset.</p><div id="worldChart" class="chart"></div></section><section class="plot"><h3>Camera projection</h3><p>Normalized image coordinates. Dashed lines show the projected tracks and dots show the current frame.</p><div id="cameraChart" class="chart"></div></section></div>
<div class="audit"><h2>Selected-sample audit</h2><table><thead><tr><th>sample</th><th>objects</th><th>points/object</th><th>frames</th><th>valid ratio</th><th>max shape error (m)</th><th>max center displacement (m)</th><th>initial alignment (m)</th></tr></thead><tbody id="auditBody"></tbody></table></div>
</main>
<script>
const DATA=__DATA__;
const COLORS=['#59c7ff','#f4a261','#69d7a2'];
let sampleIndex=0,frameIndex=0;
const sampleSelect=document.getElementById('sampleSelect');const frameSlider=document.getElementById('frameSlider');const frameOutput=document.getElementById('frameOutput');
const video=document.getElementById('videoPlayer');const videoOverlay=document.getElementById('videoOverlay');const videoNote=document.getElementById('videoNote');
DATA.samples.forEach((sample,index)=>{const o=document.createElement('option');o.value=index;o.textContent=`${String(index+1).padStart(2,'0')}  ${sample.sample_id}`;sampleSelect.appendChild(o)});
function metric(label,value,good=false){return `<div class="metric"><small>${label}</small><b class="${good?'pass':''}">${value}</b></div>`}
function fmt(value,d=4){return Number(value).toFixed(d)}
function currentSample(){return DATA.samples[sampleIndex]}
function lineTrace3d(sample,objectIndex){const tracks=sample.world_tracks[objectIndex];const x=[],y=[],z=[];for(let p=0;p<tracks[0].length;p++){for(let t=0;t<tracks.length;t++){x.push(tracks[t][p][0]);y.push(tracks[t][p][1]);z.push(tracks[t][p][2])}x.push(null);y.push(null);z.push(null)}return {type:'scatter3d',mode:'lines',x,y,z,line:{color:COLORS[objectIndex%COLORS.length],width:2},name:`${sample.object_ids[objectIndex]} tracks`,hoverinfo:'skip'} }
function currentTrace3d(sample,objectIndex){const tracks=sample.world_tracks[objectIndex][frameIndex];return {type:'scatter3d',mode:'markers',x:tracks.map(p=>p[0]),y:tracks.map(p=>p[1]),z:tracks.map(p=>p[2]),marker:{size:3,color:COLORS[objectIndex%COLORS.length]},name:`${sample.object_ids[objectIndex]} current`}}
function initialSurfaceTrace(sample,objectIndex){const points=sample.initial_surface_world[objectIndex];return {type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size:1.5,color:'#8b98a5',opacity:.35},name:`${sample.object_ids[objectIndex]} initial surface`,hoverinfo:'skip'}}
function lineTrace2d(sample,objectIndex){const tracks=sample.camera_tracks[objectIndex];const x=[],y=[];for(let p=0;p<tracks[0].length;p++){for(let t=0;t<tracks.length;t++){const point=tracks[t][p];if(point&&Number.isFinite(point[0])){x.push(point[0]);y.push(point[1])}else{x.push(null);y.push(null)}}x.push(null);y.push(null)}return {type:'scattergl',mode:'lines',x,y,line:{color:COLORS[objectIndex%COLORS.length],width:1},name:`${sample.object_ids[objectIndex]} projected tracks`,hoverinfo:'skip'}}
function currentTrace2d(sample,objectIndex){const points=sample.camera_tracks[objectIndex][frameIndex];const valid=sample.valid_tracks[objectIndex][frameIndex];const x=[],y=[];points.forEach((p,i)=>{if(p&&valid[i]&&Number.isFinite(p[0])){x.push(p[0]);y.push(p[1])}});return {type:'scattergl',mode:'markers',x,y,marker:{size:5,color:COLORS[objectIndex%COLORS.length]},name:`${sample.object_ids[objectIndex]} current`}}
function updateFrameLabel(sample){frameSlider.max=sample.time_s.length-1;frameSlider.value=frameIndex;frameOutput.value=`${frameIndex+1} / ${sample.time_s.length}  (${sample.time_s[frameIndex]}s)`}
function drawVideoOverlay(){const sample=currentSample();const width=video.videoWidth||1280,height=video.videoHeight||720;if(videoOverlay.width!==width||videoOverlay.height!==height){videoOverlay.width=width;videoOverlay.height=height}const ctx=videoOverlay.getContext('2d');ctx.clearRect(0,0,width,height);for(let o=0;o<sample.object_ids.length;o++){const tracks=sample.camera_tracks[o];ctx.strokeStyle=COLORS[o%COLORS.length];ctx.fillStyle=COLORS[o%COLORS.length];ctx.globalAlpha=.4;ctx.lineWidth=Math.max(1,width/900);for(let p=0;p<tracks[0].length;p++){let started=false;ctx.beginPath();for(let t=0;t<=frameIndex;t++){const point=tracks[t][p];if(point&&Number.isFinite(point[0])){const x=point[0]*width,y=point[1]*height;if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}else started=false}if(started)ctx.stroke()}ctx.globalAlpha=.95;const current=tracks[frameIndex];const valid=sample.valid_tracks[o][frameIndex];current.forEach((point,i)=>{if(point&&valid[i]&&Number.isFinite(point[0])){ctx.beginPath();ctx.arc(point[0]*width,point[1]*height,Math.max(2,width/360),0,Math.PI*2);ctx.fill()}})}ctx.globalAlpha=1;ctx.fillStyle='rgba(0,0,0,.65)';ctx.fillRect(8,8,180,26);ctx.fillStyle='#fff';ctx.font='14px system-ui';ctx.fillText(`frame ${frameIndex+1} / ${sample.time_s.length}`,16,26)}
function nearestVideoFrame(sample){let best=0,bestDelta=Infinity;const currentTime=Number(video.currentTime)||0;sample.time_s.forEach((t,i)=>{const d=Math.abs(Number(t)-currentTime);if(d<bestDelta){best=i;bestDelta=d}});return best}
function syncVideoFrame(){const sample=currentSample();const best=nearestVideoFrame(sample);if(best!==frameIndex){frameIndex=best;updateFrameLabel(sample);drawVideoOverlay()}}
function setVideoSource(sample){video.pause();video.src=sample.video||'';video.load();videoNote.textContent=sample.video?`Video: ${sample.video_source}`:'No rendered video was found for this sample.'}
function render(){const sample=currentSample();frameIndex=Math.max(0,Math.min(frameIndex,sample.time_s.length-1));updateFrameLabel(sample);const m=sample.metrics;document.getElementById('metrics').innerHTML=metric('sample',sample.sample_id)+metric('objects',m.object_count)+metric('points/object',sample.full_point_count)+metric('valid ratio',m.valid_ratio.map(v=>fmt(v*100,1)+'%').join(' / '),m.valid_ratio.every(v=>v>0))+metric('max shape error',fmt(m.max_rigid_shape_error_m,8)+' m',m.max_rigid_shape_error_m<1e-5)+metric('initial alignment',m.initial_alignment_error_m.map(v=>v.toExponential(2)).join(' / '),m.initial_alignment_error_m.every(v=>v<1e-5));
const world=[];const camera=[];for(let o=0;o<sample.object_ids.length;o++){world.push(initialSurfaceTrace(sample,o),lineTrace3d(sample,o),currentTrace3d(sample,o));camera.push(lineTrace2d(sample,o),currentTrace2d(sample,o))}
Plotly.react('worldChart',world,{paper_bgcolor:'#1a222b',plot_bgcolor:'#1a222b',font:{color:'#edf3f7'},margin:{l:0,r:0,t:5,b:0},showlegend:true,legend:{orientation:'h',y:-.02},scene:{aspectmode:'data',xaxis:{title:'world X'},yaxis:{title:'world Y'},zaxis:{title:'world Z'},bgcolor:'#11161b'},uirevision:sample.sample_id}, {responsive:true,displaylogo:false});
Plotly.react('cameraChart',camera,{paper_bgcolor:'#1a222b',plot_bgcolor:'#11161b',font:{color:'#edf3f7'},margin:{l:46,r:12,t:5,b:40},showlegend:true,legend:{orientation:'h',y:-.14},xaxis:{title:'u / image width',range:[0,1]},yaxis:{title:'v / image height',range:[1,0]},uirevision:sample.sample_id}, {responsive:true,displaylogo:false});
drawVideoOverlay();
}
sampleSelect.addEventListener('change',()=>{sampleIndex=Number(sampleSelect.value);frameIndex=0;setVideoSource(currentSample());render()});frameSlider.addEventListener('input',()=>{frameIndex=Number(frameSlider.value);if(video.readyState>=1)video.currentTime=currentSample().time_s[frameIndex];render()});video.addEventListener('timeupdate',syncVideoFrame);video.addEventListener('loadedmetadata',()=>{syncVideoFrame();drawVideoOverlay()});
const body=document.getElementById('auditBody');DATA.samples.forEach(s=>{const m=s.metrics;const tr=document.createElement('tr');tr.innerHTML=`<td>${s.sample_id}</td><td>${m.object_count}</td><td>${s.full_point_count}</td><td>${m.frame_count}</td><td>${m.valid_ratio.map(v=>fmt(v*100,1)+'%').join(' / ')}</td><td>${fmt(m.max_rigid_shape_error_m,8)}</td><td>${fmt(m.max_center_displacement_m,5)}</td><td>${m.initial_alignment_error_m.map(v=>v.toExponential(2)).join(' / ')}</td>`;body.appendChild(tr)});
setVideoSource(currentSample());render();window.addEventListener('resize',()=>{Plotly.Plots.resize('worldChart');Plotly.Plots.resize('cameraChart');drawVideoOverlay()});
function videoOverlayLoop(){syncVideoFrame();window.requestAnimationFrame(videoOverlayLoop)}
window.requestAnimationFrame(videoOverlayLoop);
</script></body></html>'''


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    if manifest.get("schema") != "physweep.point_trajectory_manifest.v1":
        raise ValueError("unsupported point trajectory manifest schema")
    records = manifest.get("records", [])
    selected = choose_records(records, args.sample_count)
    source_manifest_path = (
        args.source_manifest
        if args.source_manifest.is_absolute()
        else root / args.source_manifest
    ).resolve()
    source_records = {
        record["sample_id"]: record
        for record in (
            json.loads(line)
            for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    video_root = output.parent / "videos"
    samples = []
    for index, record in enumerate(selected, 1):
        source = source_records.get(record["sample_id"], {})
        video_source = str(source.get("target", {}).get("video", ""))
        video_path = root / video_source if video_source else None
        video_href = ""
        if video_path is not None and video_path.is_file():
            video_root.mkdir(parents=True, exist_ok=True)
            video_name = f"sample_{index:02d}.mp4"
            shutil.copy2(video_path, video_root / video_name)
            video_href = f"videos/{video_name}"
        samples.append(
            sample_payload(
                root,
                record,
                args.display_points,
                video_href,
                video_source,
            )
        )
    document = {
        "schema": "phycontext.point_trajectory_validation.v1",
        "source_manifest": str(manifest_path.resolve()),
        "source_record_count": len(records),
        "point_count": manifest.get("point_count"),
        "sample_count": len(samples),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML.replace("__DATA__", json.dumps(document, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "source_records": len(records), "samples": len(samples), "point_count": manifest.get("point_count")}, indent=2))


if __name__ == "__main__":
    main()
