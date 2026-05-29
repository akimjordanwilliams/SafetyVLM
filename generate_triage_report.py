"""
Build TRIAGE_REPORT.md from detections.csv (agent-style triage, no API).
Run after safety_pipeline.py completes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path(__file__).resolve().parent / "detections.csv"
OUTPUT_MD = Path(__file__).resolve().parent / "TRIAGE_REPORT.md"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
LOW_CONF_LIGHT = 0.5


def frame_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fi, g in df.groupby("frame_index"):
        classes = set(g["class_name"].astype(str))
        ts = float(g["timestamp_sec"].iloc[0])
        persons = int((g["class_name"] == "person").sum())
        vehicles = int(g["class_name"].isin(VEHICLE_CLASSES).sum())
        low_lights = int(
            ((g["class_name"] == "traffic light") & (g["confidence"] < LOW_CONF_LIGHT)).sum()
        )
        stop_signs = int((g["class_name"] == "stop sign").sum())
        ped_vehicle = ("person" in classes) and bool(classes & VEHICLE_CLASSES)
        rows.append(
            {
                "frame_index": int(fi),
                "timestamp_sec": ts,
                "persons": persons,
                "vehicles": vehicles,
                "low_conf_lights": low_lights,
                "stop_signs": stop_signs,
                "ped_vehicle": ped_vehicle,
                "low_light_frame": low_lights > 0,
            }
        )
    return pd.DataFrame(rows)


def cluster_episodes(ff: pd.DataFrame, flag_col: str, gap_frames: int = 15) -> list[dict]:
    flagged = ff[ff[flag_col]].sort_values("frame_index")
    if flagged.empty:
        return []
    episodes: list[dict] = []
    start = end = int(flagged.iloc[0]["frame_index"])
    start_ts = end_ts = float(flagged.iloc[0]["timestamp_sec"])
    for _, row in flagged.iloc[1:].iterrows():
        fi = int(row["frame_index"])
        ts = float(row["timestamp_sec"])
        if fi - end <= gap_frames:
            end, end_ts = fi, ts
        else:
            episodes.append(
                {"start_frame": start, "end_frame": end, "start_sec": start_ts, "end_sec": end_ts}
            )
            start, end, start_ts, end_ts = fi, fi, ts, ts
    episodes.append(
        {"start_frame": start, "end_frame": end, "start_sec": start_ts, "end_sec": end_ts}
    )
    return episodes


def severity_for_episode(ep: dict, ff: pd.DataFrame, kind: str) -> str:
    mask = (ff["frame_index"] >= ep["start_frame"]) & (ff["frame_index"] <= ep["end_frame"])
    seg = ff[mask]
    max_persons = int(seg["persons"].max()) if len(seg) else 0
    max_vehicles = int(seg["vehicles"].max()) if len(seg) else 0
    duration = ep["end_sec"] - ep["start_sec"]
    if kind == "ped_vehicle":
        if max_persons >= 20 or (max_persons >= 5 and max_vehicles >= 5):
            return "HIGH"
        if max_persons >= 1 and max_vehicles >= 1:
            return "MEDIUM"
        return "LOW"
    # low_conf_light
    if duration >= 0.4 or int(seg["low_conf_lights"].sum()) >= 5:
        return "MEDIUM"
    return "LOW"


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    max_frame = int(df["frame_index"].max())
    max_ts = float(df["timestamp_sec"].max())
    total_rows = len(df)

    ff = frame_flags(df)
    ped_eps = cluster_episodes(ff, "ped_vehicle")
    light_eps = cluster_episodes(ff, "low_light_frame")

    class_summary = (
        df.groupby("class_name")
        .agg(count=("confidence", "count"), avg_conf=("confidence", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    busy = (
        df.assign(second_bucket=df["timestamp_sec"].astype(float).floordiv(1))
        .groupby("second_bucket")
        .size()
        .reset_index(name="detections")
        .sort_values("detections", ascending=False)
        .head(5)
    )

    lines = [
        "# SafetyVLM — Triage Report",
        "",
        "## Executive summary",
        "",
        f"- **Video coverage:** frames `0`–`{max_frame}` (~{max_ts:.1f} s at logged timestamps).",
        f"- **Detection rows:** {total_rows:,} bounding boxes.",
        f"- **Pedestrian + vehicle episodes:** {len(ped_eps)} (same-frame co-occurrence).",
        f"- **Low-confidence traffic-light episodes:** {len(light_eps)} (confidence < {LOW_CONF_LIGHT}).",
        f"- **Stop signs:** {int((df['class_name'] == 'stop sign').sum())} box detections.",
        "",
        "Triage uses **rule-based signals** on YOLOv8n outputs, then severity labels for review prioritization. "
        "This is not a substitute for human review or calibrated collision risk.",
        "",
        "## Detection summary",
        "",
        "| Class | Count | Avg confidence |",
        "|-------|------:|---------------:|",
    ]
    for _, r in class_summary.iterrows():
        lines.append(f"| {r['class_name']} | {int(r['count'])} | {r['avg_conf']:.3f} |")

    lines += [
        "",
        "## Severity-ranked events",
        "",
        "### HIGH / MEDIUM — pedestrian + vehicle co-occurrence",
        "",
    ]
    ranked_ped = []
    for ep in ped_eps:
        sev = severity_for_episode(ep, ff, "ped_vehicle")
        if sev in ("HIGH", "MEDIUM"):
            ranked_ped.append((sev, ep))
    ranked_ped.sort(key=lambda x: (0 if x[0] == "HIGH" else 1, x[1]["start_sec"]))
    if ranked_ped:
        for sev, ep in ranked_ped[:15]:
            lines.append(
                f"- **{sev}** — {ep['start_sec']:.2f}s–{ep['end_sec']:.2f}s "
                f"(frames {ep['start_frame']}–{ep['end_frame']})"
            )
        if len(ranked_ped) > 15:
            lines.append(f"- … and {len(ranked_ped) - 15} more episodes")
    else:
        lines.append("- None flagged above LOW threshold in this run.")

    lines += [
        "",
        "### MEDIUM / LOW — low-confidence traffic lights",
        "",
    ]
    ranked_light = []
    for ep in light_eps:
        sev = severity_for_episode(ep, ff, "low_conf_light")
        ranked_light.append((sev, ep))
    ranked_light.sort(key=lambda x: (0 if x[0] == "MEDIUM" else 1, x[1]["start_sec"]))
    for sev, ep in ranked_light[:12]:
        lines.append(
            f"- **{sev}** — {ep['start_sec']:.2f}s–{ep['end_sec']:.2f}s "
            f"(frames {ep['start_frame']}–{ep['end_frame']})"
        )

    lines += [
        "",
        "## Busiest 1-second windows",
        "",
        "| Second | Detections |",
        "|-------:|-----------:|",
    ]
    for _, r in busy.iterrows():
        lines.append(f"| {int(r['second_bucket'])} | {int(r['detections'])} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- **Same-frame co-occurrence** does not mean spatial proximity or imminent conflict.",
        "- **Box counts** can exceed unique objects (crowds, duplicate detections).",
        "- **No tracking** across frames; episodes are clustered by frame gaps only.",
        "- **No ego-motion** or map context; severity is heuristic.",
        "",
        "## Interview talking points",
        "",
        "1. Funnel design: cheap detector + SQL characterization before expensive human/agent review.",
        "2. Explainable triggers (ped+vehicle, uncertain signals) vs. black-box end-to-end scoring.",
        "3. Temporal clustering turns frame spam into reviewable **episodes**.",
        "4. Next steps: IoU/proximity, ByteTrack IDs, calibrated severity, dashboard timeline.",
        "",
        "---",
        "",
        "*Generated by `generate_triage_report.py` from `detections.csv`.*",
    ]

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
