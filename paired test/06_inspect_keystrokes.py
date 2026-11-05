import sqlite3
import pandas as pd

DB_PATH = "instance/data.db"

conn = sqlite3.connect(DB_PATH)

# 1. Inspect event types
print("\n========== EVENT TYPES ==========")

events = pd.read_sql_query("""
    SELECT event_type, COUNT(*) AS total
    FROM keystroke_events
    GROUP BY event_type
""", conn)

print(events.to_string(index=False))


# 2. Inspect sample pairing
print("\n========== SAMPLE PAIRING ==========")

pairs = pd.read_sql_query("""
    SELECT
        participant_id,
        session_id,
        video_filename,
        self_reported_emotion,
        COUNT(CASE WHEN sample_type = 'video_response'
                   THEN 1 END) AS honest_count,
        COUNT(CASE WHEN sample_type = 'opposite_response'
                   THEN 1 END) AS masked_count
    FROM typing_samples
    WHERE sample_type IN ('video_response', 'opposite_response')
    GROUP BY
        participant_id,
        session_id,
        video_filename,
        self_reported_emotion
""", conn)

print(pairs.to_string(index=False))

print("\nTotal pairs:", len(pairs))

print(
    "Complete pairs:",
    len(pairs[
        (pairs["honest_count"] == 1) &
        (pairs["masked_count"] == 1)
    ])
)


# 3. Inspect one sample's keystroke events
print("\n========== SAMPLE KEYSTROKE EVENTS ==========")

sample = pd.read_sql_query("""
    SELECT
        ts.id AS sample_id,
        ts.sample_type,
        ts.text_content,
        ts.duration_seconds,
        ke.key,
        ke.event_type,
        ke.timestamp_ms,
        ke.sequence_index
    FROM typing_samples ts
    JOIN keystroke_events ke
        ON ts.id = ke.typing_sample_id
    WHERE ts.sample_type = 'video_response'
    ORDER BY ts.id, ke.sequence_index
    LIMIT 20
""", conn)

print(sample.to_string(index=False))

conn.close()