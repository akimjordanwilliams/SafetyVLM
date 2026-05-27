"""
VLM triage layer for AV safety events.

Reads detections.csv, identifies high-risk frames, asks Claude to classify severity,
and writes triage_results.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

INPUT_CSV = Path(__file__).resolve().parent / "detections.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "triage_results.csv"
MODEL_NAME = "claude-sonnet-4-20250514"
LOW_CONF_TRAFFIC_LIGHT_THRESHOLD = 0.5
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def build_event_description(frame_df: pd.DataFrame) -> str:
    """Build a concise text summary of one frame's detection context."""
    frame_index = int(frame_df["frame_index"].iloc[0])
    timestamp_sec = float(frame_df["timestamp_sec"].iloc[0])
    total_detections = int(len(frame_df))

    counts = frame_df["class_name"].value_counts().to_dict()
    count_parts = [f"{name}: {count}" for name, count in sorted(counts.items())]
    counts_str = ", ".join(count_parts)

    conf_stats = (
        frame_df.groupby("class_name")["confidence"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    conf_lines = []
    for _, row in conf_stats.sort_values("class_name").iterrows():
        conf_lines.append(
            (
                f"- {row['class_name']}: n={int(row['count'])}, "
                f"mean={row['mean']:.3f}, min={row['min']:.3f}, max={row['max']:.3f}"
            )
        )

    low_conf_lights = frame_df[
        (frame_df["class_name"] == "traffic light")
        & (frame_df["confidence"] < LOW_CONF_TRAFFIC_LIGHT_THRESHOLD)
    ]

    person_count = int(counts.get("person", 0))
    vehicle_count = int(sum(counts.get(v, 0) for v in VEHICLE_CLASSES))

    risk_reasons: list[str] = []
    if person_count > 0 and vehicle_count > 0:
        risk_reasons.append(
            f"pedestrian_vehicle_cooccurrence(persons={person_count}, vehicles={vehicle_count})"
        )
    if not low_conf_lights.empty:
        min_light_conf = float(low_conf_lights["confidence"].min())
        risk_reasons.append(
            f"low_conf_traffic_light(count={len(low_conf_lights)}, min_conf={min_light_conf:.3f})"
        )

    risk_str = ", ".join(risk_reasons) if risk_reasons else "none"
    conf_summary = "\n".join(conf_lines)

    return (
        f"Frame index: {frame_index}\n"
        f"Timestamp (sec): {timestamp_sec:.3f}\n"
        f"Total detections: {total_detections}\n"
        f"Class counts: {counts_str}\n"
        f"Risk triggers: {risk_str}\n"
        f"Per-class confidence stats:\n{conf_summary}"
    )


def call_claude_for_severity(client: Anthropic, event_description: str) -> tuple[str, str]:
    """Ask Claude to classify severity and provide reasoning."""
    prompt = (
        "You are an AV safety triage analyst.\n"
        "Given this frame-level event description, classify safety severity as exactly one of:\n"
        "LOW, MEDIUM, HIGH.\n\n"
        "Return valid JSON only with keys:\n"
        '- "severity": one of LOW|MEDIUM|HIGH\n'
        '- "reasoning": short explanation (1-3 sentences)\n\n'
        "Event description:\n"
        f"{event_description}\n"
    )

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=220,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text += block.text

    parsed = json.loads(text.strip())
    severity = str(parsed.get("severity", "MEDIUM")).upper().strip()
    if severity not in {"LOW", "MEDIUM", "HIGH"}:
        severity = "MEDIUM"
    reasoning = str(parsed.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = "No reasoning returned by model."
    return severity, reasoning


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Set it in your environment and rerun."
        )

    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required_cols = {
        "frame_index",
        "timestamp_sec",
        "class_name",
        "confidence",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {INPUT_CSV.name}: {sorted(missing)}")

    # High-risk frame criteria:
    # 1) person + vehicle co-occurrence in same frame
    # 2) traffic light detection below threshold
    grouped = df.groupby("frame_index")
    high_risk_frames: list[int] = []
    for frame_index, g in grouped:
        classes = set(g["class_name"].astype(str))
        has_person_vehicle = ("person" in classes) and bool(classes & VEHICLE_CLASSES)
        has_low_conf_light = bool(
            (
                (g["class_name"] == "traffic light")
                & (g["confidence"] < LOW_CONF_TRAFFIC_LIGHT_THRESHOLD)
            ).any()
        )
        if has_person_vehicle or has_low_conf_light:
            high_risk_frames.append(int(frame_index))

    high_risk_frames = sorted(high_risk_frames)
    if not high_risk_frames:
        pd.DataFrame(
            columns=[
                "frame_index",
                "timestamp_sec",
                "event_description",
                "severity",
                "reasoning",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        print("No high-risk frames found. Wrote empty triage_results.csv.")
        return

    client = Anthropic(api_key=api_key)
    records: list[dict] = []

    for frame_index in high_risk_frames:
        g = df[df["frame_index"] == frame_index]
        timestamp_sec = float(g["timestamp_sec"].iloc[0])
        event_description = build_event_description(g)

        try:
            severity, reasoning = call_claude_for_severity(client, event_description)
        except Exception as exc:  # noqa: BLE001
            severity = "MEDIUM"
            reasoning = f"Claude request failed: {exc}"

        records.append(
            {
                "frame_index": int(frame_index),
                "timestamp_sec": round(timestamp_sec, 4),
                "event_description": event_description,
                "severity": severity,
                "reasoning": reasoning,
            }
        )

    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values("frame_index").reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 60)
    print("VLM TRIAGE COMPLETE")
    print("=" * 60)
    print(f"Input detections: {INPUT_CSV}")
    print(f"High-risk frames triaged: {len(out_df)}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("Severity counts:")
    print(out_df["severity"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
