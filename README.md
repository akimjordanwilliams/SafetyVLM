## SafetyVLM — AV Safety Event Triage Demo

### Overview

SafetyVLM is a small end‑to‑end demo pipeline for **autonomous vehicle (AV) safety event detection** on dashcam video:

- **Stage 1 — Perception:** run YOLOv8 on `video.mp4` to detect people, vehicles, traffic lights, and stop signs frame‑by‑frame and log them to `detections.csv`.
- **Stage 2 — Rule filters:** compute simple, explainable safety flags (pedestrian + vehicle co‑occurrence, low‑confidence traffic lights, stop signs).
- **Stage 3 — VLM triage:** summarize high‑risk frames as text and call Anthropic Claude to classify severity (`LOW | MEDIUM | HIGH`) with short reasoning into `triage_results.csv`.
- **Optional — Visualization:** render the first minute of video with YOLO overlays for interview/demo use.

This repo is sized for a take‑home / interview project and is intentionally simple and self‑contained.

### Repo layout

- `safety_pipeline.py` — main YOLO pipeline → `detections.csv` + printed safety flags.
- `safety_queries.py` — DuckDB SQL analysis over `detections.csv` (busiest windows, low‑confidence frames, etc.).
- `vlm_triage.py` — VLM triage using Anthropic Claude (`claude-sonnet-4-20250514`) → `triage_results.csv`.
- `render_first_minute_yolo.py` — utility to render the first 60s of `video.mp4` with YOLO overlays.
- `prompt_engineering_notes.md` — engineering notes on the triage prompt iterations and final design.
- `HANDOFF.md` — higher‑level handoff doc for new engineers.
- `requirements.txt` — Python dependencies.
- `video.mp4` — input dashcam clip (not committed in typical setups; treat as local data artifact).

Generated artifacts:

- `detections.csv` — raw detection rows, one per box.
- `triage_results.csv` — high‑risk frames with `severity` and `reasoning`.
- `video_first_1min_yolo.mp4` / `video_first_1min_yolo_h264.mp4` — first minute with YOLO overlays.

### Setup

```bash
cd SafetyVLM
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place your dashcam clip at `video.mp4` (or edit `VIDEO_PATH` in `safety_pipeline.py`).

### Stage 1: run YOLO over video

```bash
# full video (can be slow on CPU)
python safety_pipeline.py

# or partial run for faster iteration, e.g. first 500 frames
SAFETY_PIPELINE_MAX_FRAMES=500 python safety_pipeline.py
```

Outputs:

- `detections.csv`
- console summary of:
  - pedestrian + vehicle co‑occurrence frames,
  - low‑confidence traffic lights,
  - stop‑sign detections.

### Stage 2: offline safety queries

```bash
python safety_queries.py
```

Runs a set of DuckDB queries over `detections.csv` (class counts, busy windows, low‑confidence frames, etc.) and prints tables to stdout.

### Stage 3: VLM triage with Claude

This step turns high‑risk frames into **severity‑labeled events** using a VLM.

1. Set your Anthropic API key:

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."   # PowerShell: $env:ANTHROPIC_API_KEY="..."
   ```

2. Run triage:

   ```bash
   python vlm_triage.py
   ```

This:

- groups detections by `frame_index`,
- filters to frames with:
  - pedestrian + vehicle co‑occurrence, or
  - traffic light with `confidence < 0.5`,
- builds a per‑frame textual summary, and
- calls Claude (`claude-sonnet-4-20250514`) to return JSON:

  ```json
  {
    "severity": "LOW|MEDIUM|HIGH",
    "reasoning": "short explanation (1–3 sentences)"
  }
  ```

Results are written to `triage_results.csv` with columns:

- `frame_index`, `timestamp_sec`, `event_description`, `severity`, `reasoning`.

If the API call fails, the script records `severity="MEDIUM"` and logs the error in `reasoning`.

### YOLO visualization (first 60 seconds)

```bash
python render_first_minute_yolo.py
```

Produces:

- `video_first_1min_yolo.mp4` (MPEG‑4)
- `video_first_1min_yolo_h264.mp4` (H.264/yuv420p, best for embedded players)

### What to talk about in an interview

- Simple, explainable **safety heuristics** (co‑occurrence, model uncertainty) as a first pass.
- Separation of concerns:
  - detection → logging → analysis → triage,
  - rule‑based candidate selection vs. LLM‑based severity scoring.
- Prompt‑engineering choices documented in `prompt_engineering_notes.md`:
  - strict JSON output contract,
  - fixed severity taxonomy,
  - short, auditable reasoning.
- Extension ideas: tracking, spatial reasoning (distance / IoU), aggregations over time windows, and dashboarding.

