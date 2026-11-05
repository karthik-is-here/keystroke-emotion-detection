import sqlite3
import pandas as pd

DB_PATH = r"C:\Karthik\Projects\keystroke_emotion_app\instance\data.db"

connection = sqlite3.connect(DB_PATH)

query = """
SELECT
    id,
    participant_id,
    session_id,
    label,
    sample_type,
    target_emotion,
    video_filename,
    self_reported_emotion,
    LENGTH(text_content) AS text_length,
    SUBSTR(text_content, 1, 100) AS text_preview
FROM typing_samples
ORDER BY participant_id, id;
"""

df = pd.read_sql_query(query, connection)

print("\nTOTAL SAMPLES:", len(df))

print("\nSAMPLE TYPE DISTRIBUTION")
print(df["sample_type"].value_counts(dropna=False))

print("\nLABEL DISTRIBUTION")
print(df["label"].value_counts(dropna=False))

print("\nSELF-REPORTED EMOTION DISTRIBUTION")
print(df["self_reported_emotion"].value_counts(dropna=False))

print("\nFIRST 10 RECORDS")
print(df.head(10).to_string(index=False))

connection.close()