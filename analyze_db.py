
import sqlite3
import pandas as pd
from pathlib import Path

# Database location
db_path = Path(__file__).parent / "instance" / "data.db"
# Verify the database exists
if not db_path.exists():
    raise FileNotFoundError(
        f"Database not found: {db_path}\n"
        "Check that data.db is in your project folder."
    )

# Connect to the database
conn = sqlite3.connect(str(db_path))

try:
    # Check available tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )
    print("\nAvailable tables:")
    print(tables)

    # Load typing samples
    samples = pd.read_sql_query("""
        SELECT
            participant_id,
            session_id,
            label,
            sample_type,
            target_emotion,
            self_reported_emotion,
            word_count,
            duration_seconds,
            wpm,
            error_rate
        FROM typing_samples
    """, conn)

    print("\nFirst five records:")
    print(samples.head())

    print("\nDataset information:")
    samples.info()

    print("\nMissing values:")
    print(samples.isnull().sum())

    print("\nNumber of samples per emotion:")
    print(samples["label"].value_counts(dropna=False))

    print("\nDescriptive statistics:")
    print(samples[
        ["word_count", "duration_seconds", "wpm", "error_rate"]
    ].describe().round(3))

finally:
    conn.close()
