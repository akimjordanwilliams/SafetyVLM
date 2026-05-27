"""
AV safety event detection pipeline: YOLOv8 on dashcam video -> detections.csv + safety flags.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# Dashcam video (raw string for Windows paths)
VIDEO_PATH = r"C:\Users\akimj\OneDrive\Desktop\SafetyVLM\video.mp4"
OUTPUT_CSV = Path(__file__).resolve().parent / "detections.csv"

# COCO class IDs used by default YOLOv8 models
PERSON_ID = 0
VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
TRAFFIC_LIGHT_ID = 9
STOP_SIGN_ID = 11

# Safety heuristics
LOW_CONF_TRAFFIC_LIGHT = 0.5

# Optional cap for long videos (full run: unset or empty). Example: set SAFETY_PIPELINE_MAX_FRAMES=500
_max = os.environ.get("SAFETY_PIPELINE_MAX_FRAMES", "").strip()
MAX_FRAMES = int(_max) if _max else None


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO("yolov8n.pt")
    # Only run detector on classes we care about (faster)
    target_ids = [PERSON_ID, *sorted(VEHICLE_IDS), TRAFFIC_LIGHT_ID, STOP_SIGN_ID]

    rows: list[dict] = []
    # Per-frame class sets for flagging
    frame_classes: dict[int, set[int]] = defaultdict(set)
    frame_low_conf_lights: list[tuple[int, float, float]] = []  # frame, ts, conf
    frame_stop_signs: list[tuple[int, float]] = []  # frame, ts

    frame_idx = 0
    while True:
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = frame_idx / fps if fps > 0 else 0.0
        results = model.predict(
            frame,
            verbose=False,
            classes=target_ids,
            conf=0.25,
        )
        r0 = results[0]
        names = r0.names

        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                frame_classes[frame_idx].add(cls_id)
                class_name = names.get(cls_id, str(cls_id))

                rows.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 4),
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(float(conf), 4),
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                    }
                )

                if cls_id == TRAFFIC_LIGHT_ID and float(conf) < LOW_CONF_TRAFFIC_LIGHT:
                    frame_low_conf_lights.append((frame_idx, timestamp_sec, float(conf)))
                if cls_id == STOP_SIGN_ID:
                    frame_stop_signs.append((frame_idx, timestamp_sec))

        frame_idx += 1

    cap.release()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # --- Safety flagging (same-frame co-occurrence and attention events) ---
    ped_vehicle_frames: list[tuple[int, float]] = []
    for fi, clsset in frame_classes.items():
        if PERSON_ID in clsset and clsset & VEHICLE_IDS:
            ped_vehicle_frames.append((fi, fi / fps if fps > 0 else 0.0))

    # Summary
    print("=" * 60)
    print("SAFETY PIPELINE — OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height}  |  FPS: {fps:.2f}  |  Frames processed: {frame_idx}")
    if MAX_FRAMES is not None:
        print(
            f"Note: SAFETY_PIPELINE_MAX_FRAMES={MAX_FRAMES} — partial run "
            f"(video has ~{total_frames} frames; unset env for full video)."
        )
    print(f"Total detection rows written: {len(rows)}")
    print(f"CSV: {OUTPUT_CSV}")
    print()
    print("--- Safety flags ---")
    if ped_vehicle_frames:
        ex = ", ".join(str(fi) for fi, _ in ped_vehicle_frames[:8])
        more = " …" if len(ped_vehicle_frames) > 8 else ""
        ped_ex = f" (example frame indices: {ex}{more})"
    else:
        ped_ex = ""
    print(
        f"Pedestrian + vehicle same frame: {len(ped_vehicle_frames)} frame(s){ped_ex}"
    )
    print(
        f"Low-confidence traffic light (< {LOW_CONF_TRAFFIC_LIGHT}): {len(frame_low_conf_lights)} detection(s)"
    )
    print(f"Stop sign detection(s): {len(frame_stop_signs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
