from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


INPUT_VIDEO = Path(r"C:\Users\akimj\OneDrive\Desktop\SafetyVLM\video.mp4")
OUTPUT_VIDEO = Path(r"C:\Users\akimj\OneDrive\Desktop\SafetyVLM\video_first_1min_yolo.mp4")
MODEL_WEIGHTS = "yolov8n.pt"
SECONDS_TO_RENDER = 60


def main() -> None:
    if not INPUT_VIDEO.exists():
        raise SystemExit(f"Input video not found: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * SECONDS_TO_RENDER)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to create output video: {OUTPUT_VIDEO}")

    model = YOLO(MODEL_WEIGHTS)

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False, conf=0.25)
        annotated = results[0].plot()
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{max_frames} frames...")

    cap.release()
    writer.release()
    print(f"Done. Wrote {frame_idx} frames to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
