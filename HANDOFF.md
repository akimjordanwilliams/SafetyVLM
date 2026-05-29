# SafetyVLM — Workspace handoff

## Purpose

Dashcam **safety detection** demo: **YOLOv8** on `video.mp4` → **`detections.csv`** → **DuckDB analysis** → **agent/human triage** in **`TRIAGE_REPORT.md`**.

## Repository layout

| Path | Role |
|------|------|
| `safety_pipeline.py` | OpenCV + YOLOv8n → `detections.csv` + printed flags |
| `safety_queries.py` | DuckDB SQL over `detections.csv` |
| `TRIAGE_REPORT.md` | Severity-ranked triage report (committed) |
| `detections_sample.csv` | Reproducible CSV slice for GitHub readers |
| `render_first_minute_yolo.py` | First 60s with YOLO overlays |
| `vlm_triage.py` | *Legacy optional* — Anthropic API → `triage_results.csv` |
| `requirements.txt` | `ultralytics`, `opencv-python-headless`, `pandas`, `duckdb` |
| `video.mp4` | Input (~162k frames @ 640×360, 30 FPS; gitignored) |
| `detections.csv` | Full generated output (gitignored) |
| `video_first_1min_yolo_h264.mp4` | Demo overlay clip (committed) |

## Setup

```powershell
cd C:\Users\akimj\OneDrive\Desktop\SafetyVLM
python -m pip install -r requirements.txt
```

## Running

**Full video:**

```powershell
Remove-Item Env:SAFETY_PIPELINE_MAX_FRAMES -ErrorAction SilentlyContinue
python safety_pipeline.py
```

**Partial run:**

```powershell
$env:SAFETY_PIPELINE_MAX_FRAMES = "500"
python safety_pipeline.py
```

**Analysis:**

```powershell
python safety_queries.py
```

**Agent triage:** open the repo in Cursor and ask the agent to read `detections.csv` + query output and write `TRIAGE_REPORT.md` (see `README.md`).

## Configuration

| Item | Where | Notes |
|------|--------|--------|
| Input video | `VIDEO_PATH` in `safety_pipeline.py` | Windows raw string |
| Output CSV | `OUTPUT_CSV` | Default `detections.csv` |
| Max frames | `SAFETY_PIPELINE_MAX_FRAMES` | Optional env cap |
| Model | `yolov8n.pt` | Swap for larger weights if needed |
| Detector threshold | `conf=0.25` | Ultralytics default-style gate |
| Low-conf traffic light | `LOW_CONF_TRAFFIC_LIGHT = 0.5` | Heuristic flag threshold |

**Classes:** person (0), car/motorcycle/bus/truck (2,3,5,7), traffic light (9), stop sign (11).

## Safety flags (console only)

1. Pedestrian + vehicle in the **same frame** (co-occurrence, not distance).
2. Traffic-light boxes with `confidence < 0.5`.
3. Stop-sign detection count.

## Performance

- Full pass on **CPU** can take **many hours** for ~162k frames; GPU helps without code changes.
- `detections.csv` is written when the run **finishes** (not streamed per frame).

## GitHub artifacts

- Committed: code, `TRIAGE_REPORT.md`, `detections_sample.csv`, demo MP4.
- Local only: `video.mp4`, full `detections.csv`, weights `*.pt`.

## Legacy VLM path

`vlm_triage.py` + `ANTHROPIC_API_KEY` → `triage_results.csv`. Superseded by agent triage for the main demo story.

## Extensions

- Spatial heuristics (IoU / image distance).
- Temporal episode clustering before triage.
- Frame stride for faster archive scans.
- Tracking (ByteTrack) for stable IDs.
