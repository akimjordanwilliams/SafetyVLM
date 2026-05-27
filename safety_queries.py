"""
AV Safety Event SQL Analysis
Runs complex queries against detections.csv using DuckDB
"""

import duckdb

# Load CSV into DuckDB in-memory table
con = duckdb.connect()
con.execute("CREATE TABLE detections AS SELECT * FROM read_csv_auto('detections.csv')")

print("=" * 60)
print("AV SAFETY EVENT ANALYSIS")
print("=" * 60)


# --- QUERY 1: Detection summary by class ---
print("\n[1] Detection count by object class:")
print("-" * 40)
result = con.execute("""
    SELECT 
        class_name,
        COUNT(*) AS total_detections,
        ROUND(AVG(confidence), 3) AS avg_confidence,
        ROUND(MIN(confidence), 3) AS min_confidence,
        ROUND(MAX(confidence), 3) AS max_confidence
    FROM detections
    GROUP BY class_name
    ORDER BY total_detections DESC
""").fetchdf()
print(result.to_string(index=False))


# --- QUERY 2: Pedestrian + vehicle co-occurrence ---
# Find frames where a person AND a vehicle appear at the same time
print("\n[2] Frames with pedestrian + vehicle co-occurrence (high-risk):")
print("-" * 40)
result = con.execute("""
    SELECT 
        p.frame_index,
        ROUND(p.timestamp_sec, 2) AS timestamp_sec,
        COUNT(DISTINCT p.class_name) AS pedestrian_count,
        COUNT(DISTINCT v.class_name) AS vehicle_types_present
    FROM detections p
    JOIN detections v 
        ON p.frame_index = v.frame_index
        AND v.class_name IN ('car', 'truck', 'bus', 'motorcycle')
    WHERE p.class_name = 'person'
    GROUP BY p.frame_index, p.timestamp_sec
    ORDER BY p.timestamp_sec
    LIMIT 10
""").fetchdf()
print(result.to_string(index=False))


# --- QUERY 3: Low-confidence traffic light detections ---
# These are frames where the model was uncertain — safety risk
print("\n[3] Low-confidence traffic light detections (model uncertainty):")
print("-" * 40)
result = con.execute("""
    SELECT 
        frame_index,
        ROUND(timestamp_sec, 2) AS timestamp_sec,
        ROUND(confidence, 3) AS confidence,
        x1, y1, x2, y2
    FROM detections
    WHERE class_name = 'traffic light'
      AND confidence < 0.5
    ORDER BY confidence ASC
    LIMIT 10
""").fetchdf()
print(result.to_string(index=False))


# --- QUERY 4: Rolling window — detection density over time ---
# Find time windows with the most safety-relevant activity
print("\n[4] Busiest 1-second windows by safety object count:")
print("-" * 40)
result = con.execute("""
    SELECT 
        FLOOR(timestamp_sec) AS second_bucket,
        COUNT(*) AS total_detections,
        COUNT(CASE WHEN class_name = 'person' THEN 1 END) AS pedestrians,
        COUNT(CASE WHEN class_name IN ('car','truck','bus','motorcycle') THEN 1 END) AS vehicles,
        COUNT(CASE WHEN class_name = 'traffic light' THEN 1 END) AS traffic_lights,
        COUNT(CASE WHEN class_name = 'stop sign' THEN 1 END) AS stop_signs
    FROM detections
    GROUP BY second_bucket
    ORDER BY total_detections DESC
    LIMIT 10
""").fetchdf()
print(result.to_string(index=False))


# --- QUERY 5: Confidence drop events ---
# Frames where avg confidence across all detections was unusually low
# Could indicate bad lighting, occlusion, or adversarial conditions
print("\n[5] Frames with low average confidence (potential sensor degradation):")
print("-" * 40)
result = con.execute("""
    WITH frame_stats AS (
        SELECT 
            frame_index,
            ROUND(timestamp_sec, 2) AS timestamp_sec,
            COUNT(*) AS num_detections,
            ROUND(AVG(confidence), 3) AS avg_confidence
        FROM detections
        GROUP BY frame_index, timestamp_sec
    )
    SELECT *
    FROM frame_stats
    WHERE avg_confidence < 0.4
      AND num_detections >= 2
    ORDER BY avg_confidence ASC
    LIMIT 10
""").fetchdf()
print(result.to_string(index=False))


# --- QUERY 6: Time-based join — objects appearing within 10 frames of a stop sign ---
print("\n[6] Objects detected within 10 frames of a stop sign:")
print("-" * 40)
result = con.execute("""
    SELECT 
        d.class_name,
        COUNT(*) AS occurrences,
        ROUND(AVG(d.confidence), 3) AS avg_confidence
    FROM detections d
    JOIN detections s 
        ON ABS(d.frame_index - s.frame_index) <= 10
        AND s.class_name = 'stop sign'
        AND d.class_name != 'stop sign'
    GROUP BY d.class_name
    ORDER BY occurrences DESC
""").fetchdf()
print(result.to_string(index=False))


print("\n" + "=" * 60)
print("Analysis complete.")
print("=" * 60)
