# SafetyVLM — Workspace handoff

## Purpose

Small **dashcam safety detection** demo for an interview-style project: run **YOLOv8** on `video.mp4` frame-by-frame, log detections to **CSV**, then print a **console summary** of simple safety-oriented flags (person near traffic, dubious traffic-light scores, stop signs).

## Repository layout

| Path | Role |
|------|------|
| `safety_pipeline.py` | Main entry: OpenCV decode → Ultralytics YOLOv8n → `detections.csv` + printed flags |
| `safety_queries.py` | DuckDB SQL analysis queries over `detections.csv` |
| `vlm_triage.py` | Frame-level LLM triage via Anthropic Claude → `triage_results.csv` |
| `render_first_minute_yolo.py` | Utility to render first 60s with YOLO overlays |
| `prompt_engineering_notes.md` | Prompt iteration and design rationale for VLM triage |
| `requirements.txt` | `ultralytics`, `opencv-python-headless`, `pandas`, `anthropic` |
| `video.mp4` | Input dashcam (~162k frames @ 640×360, 30 FPS as reported by OpenCV — very long wall-clock on CPU if fully processed) |
| `yolov8n.pt` | Weights downloaded on first run (may already exist beside the script) |
| `detections.csv` | **Generated output** — one row per bounding box |
| `triage_results.csv` | **Generated output** — LLM severity classification per high-risk frame |
| `video_first_1min_yolo.mp4` | First minute YOLO-annotated render (MPEG-4 profile) |
| `video_first_1min_yolo_h264.mp4` | H.264/yuv420p compatibility render for broad playback support |

## Prerequisites

- **Python 3** with `pip`.
- Windows paths in the script are explicit; editing `VIDEO_PATH` is enough to point at another file.

## Setup

```powershell
cd C:\Users\akimj\OneDrive\Desktop\SafetyVLM
python -m pip install -r requirements.txt
```

First run triggers Ultralytics defaults (settings file under `%APPDATA%\Ultralytics\`, model fetch if `.pt` missing).

## Running

**Full video** (long on CPU):

```powershell
Remove-Item Env:SAFETY_PIPELINE_MAX_FRAMES -ErrorAction SilentlyContinue
python safety_pipeline.py
```

**Partial run** (recommended for demos / iteration). Stops after *N* frames and notes that in the summary:

```powershell
$env:SAFETY_PIPELINE_MAX_FRAMES = "500"
python safety_pipeline.py
```

Unset the variable (or leave empty) for a full pass.

## VLM triage run (`vlm_triage.py`)

Requires `ANTHROPIC_API_KEY` in environment.

```powershell
cd C:\Users\akimj\OneDrive\Desktop\SafetyVLM
$env:ANTHROPIC_API_KEY = "your_key_here"
python vlm_triage.py
```

This writes `triage_results.csv` with:

- `frame_index`
- `timestamp_sec`
- `event_description`
- `severity` (`LOW|MEDIUM|HIGH`)
- `reasoning`

High-risk candidates are generated deterministically before the API call:

1. frame contains `person` and any vehicle (`car|truck|bus|motorcycle`), or
2. frame contains `traffic light` with confidence `< 0.5`.

If API call/parsing fails for a frame, the script records fallback severity `MEDIUM` and logs the failure reason in `reasoning`.

## YOLO annotated video generation

Render first minute:

```powershell
cd C:\Users\akimj\OneDrive\Desktop\SafetyVLM
python render_first_minute_yolo.py
```

If editor playback is codec-sensitive, use the H.264 version:

- `video_first_1min_yolo_h264.mp4`

This file is optimized for compatibility (`avc1`, `yuv420p`, fast-start metadata).

## Configuration (code / env)

| Item | Where | Notes |
|------|--------|--------|
| Input video | `VIDEO_PATH` in `safety_pipeline.py` | Raw string, e.g. `r"C:\...\video.mp4"` |
| Output CSV | `OUTPUT_CSV` | Defaults to `detections.csv` next to the script |
| Max frames | `SAFETY_PIPELINE_MAX_FRAMES` | Optional env; empty = no cap |
| Model | `YOLO("yolov8n.pt")` | Swap for `yolov8s.pt` / etc. for accuracy vs speed |
| Detector threshold | `conf=0.25` in `model.predict(...)` | Standard Ultralytics inference threshold |
| “Low conf” traffic light | `LOW_CONF_TRAFFIC_LIGHT = 0.5` | Flag any traffic_light box below this |

**Classes** are COCO IDs filtered in code: **person** (0), **car / motorcycle / bus / truck** (2, 3, 5, 7), **traffic light** (9), **stop sign** (11).

## Output: `detections.csv`

UTF-8 CSV with header:

| Column | Meaning |
|--------|---------|
| `frame_index` | 0-based frame number in the processed sequence |
| `timestamp_sec` | `frame_index / fps` |
| `class_id`, `class_name` | COCO class |
| `confidence` | Detector score |
| `x1`, `y1`, `x2`, `y2` | Pixel bbox (xyxy) |

Re-running the pipeline **overwrites** the CSV.

## Safety flags (printed only)

Computed **after** writing the CSV:

1. **Pedestrian + vehicle (same frame):** frame contains at least one **person** and at least one of **car, motorcycle, bus, truck**. This is coarse co-occurrence, not proximity in image space or world space.
2. **Low-confidence traffic light:** count of traffic_light boxes with `confidence < LOW_CONF_TRAFFIC_LIGHT`.
3. **Stop sign:** count of all stop-sign detections over the processed frames.

Extend the script if you need a second artifact (e.g. `flags.csv` keyed by frame).

## Performance & ops notes

- **~162k frames** at full resolution through a small model is still heavy on **CPU**; use `SAFETY_PIPELINE_MAX_FRAMES` or **frame stride** (not implemented yet) for faster demos.
- **GPU:** if CUDA is available to PyTorch/Ultralytics, the same script should benefit without code changes.
- **Artifacts:** Ultralytics may create run folders under the working directory depending on version/settings; this pipeline uses `predict(..., verbose=False)` and does not depend on those for the CSV.

## Suggested next steps (if continuing the project)

- Add **spatial heuristics** (IoU / distance in image, or homography if calibration exists) instead of same-frame class co-occurrence.
- Export **flagged frames** or a **timeline JSON** for a dashboard.
- **Sample every k-th frame** for long archive video; document effective timebase.
- Optional ** tracking** (ByteTrack, etc.) for stable object IDs across frames.

## Interview talking points from this run

- **High-risk window:** second `11` had `100` pedestrian detections and `30` vehicle detections, suggesting a dense vulnerable-road-user period worth escalation.
- **Planning-relevant co-occurrence:** second `10` had `72` pedestrians and `52` traffic lights detected together, useful for corridor complexity and control-zone planning.
- **Sustained uncertainty event:** frames `299-312` repeatedly showed low-confidence traffic-light detections around the `10` second mark; treat this as persistent uncertainty rather than a one-off miss.
- **No stop signs in clip:** stop-sign query path is valid and ready, but this particular video segment produced zero stop-sign detections.

## Contact / context

Built as a **portfolio / interview** slice: detection + logging + simple rule-based “safety” signals. No git remote or CI is assumed in this folder unless you add it.
