import sqlite3
import pandas as pd

# Database path
DB_PATH = "instance/data.db"

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Extract emotion-related typing samples
query = """
SELECT
    id,
    participant_id,
    label,
    sample_type,
    target_emotion,
    video_filename,
    self_reported_emotion,
    text_content
FROM typing_samples
WHERE sample_type IN ('video_response', 'opposite_response')
"""

df = pd.read_sql_query(query, conn)

conn.close()

# Remove missing and empty responses
df = df.dropna(subset=[
    'text_content',
    'self_reported_emotion'
])

df['text_content'] = df['text_content'].str.strip()

df = df[df['text_content'] != '']

# Separate honest and masked responses
honest = df[df['sample_type'] == 'video_response'].copy()

masked = df[df['sample_type'] == 'opposite_response'].copy()

# Dataset summary
print("\n========== DATASET SUMMARY ==========")

print("Total usable responses:", len(df))

print("\nHonest responses:", len(honest))
print("Masked responses:", len(masked))

print("\nUnique participants:", df['participant_id'].nunique())

print("\nTrue emotion distribution:")
print(df.groupby('self_reported_emotion')['sample_type'].value_counts())

print("\nMasked emotion distribution:")
print(masked['target_emotion'].value_counts(dropna=False))

# Display sample responses
print("\n========== HONEST RESPONSES ==========")

print(
    honest[[
        'participant_id',
        'self_reported_emotion',
        'text_content'
    ]].head(5).to_string(index=False)
)

print("\n========== MASKED RESPONSES ==========")

print(
    masked[[
        'participant_id',
        'self_reported_emotion',
        'target_emotion',
        'text_content'
    ]].head(5).to_string(index=False)
)

# Save extracted dataset
df.to_csv("manipulation_check_data.csv", index=False)

print("\nExtracted dataset saved successfully!")