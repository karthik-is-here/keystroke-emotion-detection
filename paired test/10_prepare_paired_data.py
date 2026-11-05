import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD QUALITY-CONTROLLED DATA
# --------------------------------------------------

df = pd.read_csv("typing_features_quality_checked.csv")

print("=" * 65)
print("PREPARING PAIRED DATASET")
print("=" * 65)

# --------------------------------------------------
# 2. DEFINE PAIRING KEYS
# --------------------------------------------------

pair_keys = [
    "participant_id",
    "session_id",
    "video_filename",
    "self_reported_emotion"
]

# Verify required columns
required_columns = pair_keys + [
    "sample_type",
    "id",
    "text_content",
    "typing_speed_cps",
    "pause_rate",
    "backspace_rate",
    "valid_hold_count",
    "valid_flight_count",
    "clean_mean_hold_ms",
    "clean_std_hold_ms",
    "total_time_s"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# 3. SEPARATE GENUINE AND MASKED SAMPLES
# --------------------------------------------------

genuine = df[
    df["sample_type"] == "video_response"
].copy()

masked = df[
    df["sample_type"] == "opposite_response"
].copy()

# --------------------------------------------------
# 4. CHECK DUPLICATES
# --------------------------------------------------

if genuine.duplicated(pair_keys).any():
    raise ValueError("Duplicate genuine pairing keys found.")

if masked.duplicated(pair_keys).any():
    raise ValueError("Duplicate masked pairing keys found.")

# --------------------------------------------------
# 5. SELECT FEATURES
# --------------------------------------------------

features_to_keep = [
    "id",
    "text_content",
    "mean_hold_ms",
    "std_hold_ms",
    "clean_mean_hold_ms",
    "clean_std_hold_ms",
    "mean_flight_ms",
    "std_flight_ms",
    "typing_speed_cps",
    "pause_rate",
    "backspace_rate",
    "total_time_s",
    "active_typing_time_s",
    "valid_hold_count",
    "valid_flight_count",
    "printable_event_count",
    "total_keydowns",
    "unmatched_keyups",
    "unmatched_keydowns",
    "holds_over_5s"
]

features_to_keep = [
    col for col in features_to_keep
    if col in df.columns
]

genuine = genuine[
    pair_keys + features_to_keep
]

masked = masked[
    pair_keys + features_to_keep
]

# --------------------------------------------------
# 6. RENAME COLUMNS
# --------------------------------------------------

genuine = genuine.rename(
    columns={
        col: f"genuine_{col}"
        for col in features_to_keep
    }
)

masked = masked.rename(
    columns={
        col: f"masked_{col}"
        for col in features_to_keep
    }
)

# --------------------------------------------------
# 7. CREATE PAIRED DATASET
# --------------------------------------------------

paired = genuine.merge(
    masked,
    on=pair_keys,
    how="inner",
    validate="one_to_one"
)

print("\n--- PAIRING RESULTS ---")

print("Genuine samples:", len(genuine))
print("Masked samples:", len(masked))
print("Complete pairs:", len(paired))

if len(paired) != 84:
    print(
        "WARNING: Expected 84 pairs. "
        "Check missing or unmatched samples."
    )

# --------------------------------------------------
# 8. FEATURE ELIGIBILITY
# --------------------------------------------------

MIN_EVENTS = 10

# Hold-time eligibility
paired["hold_eligible"] = (
    (paired["genuine_valid_hold_count"] >= MIN_EVENTS)
    &
    (paired["masked_valid_hold_count"] >= MIN_EVENTS)
    &
    (paired["genuine_clean_std_hold_ms"].notna())
    &
    (paired["masked_clean_std_hold_ms"].notna())
)

# Flight-time eligibility
paired["flight_eligible"] = (
    (paired["genuine_valid_flight_count"] >= MIN_EVENTS)
    &
    (paired["masked_valid_flight_count"] >= MIN_EVENTS)
    &
    (paired["genuine_mean_flight_ms"].notna())
    &
    (paired["masked_mean_flight_ms"].notna())
)

# Typing speed eligibility
# Require sufficient recorded printable events in both conditions.
paired["speed_eligible"] = (
    (paired["genuine_printable_event_count"] >= MIN_EVENTS)
    &
    (paired["masked_printable_event_count"] >= MIN_EVENTS)
    &
    (paired["genuine_total_time_s"] > 0)
    &
    (paired["masked_total_time_s"] > 0)
    &
    (paired["genuine_typing_speed_cps"].notna())
    &
    (paired["masked_typing_speed_cps"].notna())
)

# Pause-rate eligibility
paired["pause_eligible"] = (
    (paired["genuine_valid_flight_count"] >= MIN_EVENTS)
    &
    (paired["masked_valid_flight_count"] >= MIN_EVENTS)
    &
    (paired["genuine_pause_rate"].notna())
    &
    (paired["masked_pause_rate"].notna())
)

# Backspace-rate eligibility
paired["backspace_eligible"] = (
    (paired["genuine_total_keydowns"] >= MIN_EVENTS)
    &
    (paired["masked_total_keydowns"] >= MIN_EVENTS)
    &
    (paired["genuine_backspace_rate"].notna())
    &
    (paired["masked_backspace_rate"].notna())
)

# Total task duration eligibility
paired["duration_eligible"] = (
    (paired["genuine_total_time_s"] > 0)
    &
    (paired["masked_total_time_s"] > 0)
)

# --------------------------------------------------
# 9. CREATE FEATURE DIFFERENCES
# --------------------------------------------------

# Difference = masked - genuine

paired["diff_mean_hold"] = (
    paired["masked_clean_mean_hold_ms"]
    - paired["genuine_clean_mean_hold_ms"]
)

paired["diff_std_hold"] = (
    paired["masked_clean_std_hold_ms"]
    - paired["genuine_clean_std_hold_ms"]
)

paired["diff_mean_flight"] = (
    paired["masked_mean_flight_ms"]
    - paired["genuine_mean_flight_ms"]
)

paired["diff_std_flight"] = (
    paired["masked_std_flight_ms"]
    - paired["genuine_std_flight_ms"]
)

paired["diff_typing_speed"] = (
    paired["masked_typing_speed_cps"]
    - paired["genuine_typing_speed_cps"]
)

paired["diff_pause_rate"] = (
    paired["masked_pause_rate"]
    - paired["genuine_pause_rate"]
)

paired["diff_backspace_rate"] = (
    paired["masked_backspace_rate"]
    - paired["genuine_backspace_rate"]
)

paired["diff_total_time"] = (
    paired["masked_total_time_s"]
    - paired["genuine_total_time_s"]
)

# --------------------------------------------------
# 10. PRINT ELIGIBILITY SUMMARY
# --------------------------------------------------

print("\n--- FEATURE ELIGIBILITY ---")

eligibility_columns = [
    "hold_eligible",
    "flight_eligible",
    "speed_eligible",
    "pause_eligible",
    "backspace_eligible",
    "duration_eligible"
]

for column in eligibility_columns:

    count = paired[column].sum()

    print(
        f"{column}: {count}/{len(paired)} pairs"
    )

# --------------------------------------------------
# 11. IDENTIFY EXCLUDED PAIRS
# --------------------------------------------------

print("\n--- PAIRS WITH INSUFFICIENT HOLD DATA ---")

print(
    paired.loc[
        ~paired["hold_eligible"],
        pair_keys
    ].to_string(index=False)
)

print("\n--- PAIRS WITH INSUFFICIENT FLIGHT DATA ---")

print(
    paired.loc[
        ~paired["flight_eligible"],
        pair_keys
    ].to_string(index=False)
)

print("\n--- PAIRS WITH INSUFFICIENT SPEED DATA ---")

print(
    paired.loc[
        ~paired["speed_eligible"],
        pair_keys
    ].to_string(index=False)
)

# --------------------------------------------------
# 12. SAVE PAIRED DATASET
# --------------------------------------------------

paired.to_csv(
    "paired_typing_features.csv",
    index=False
)

print("\nSaved: paired_typing_features.csv")

# --------------------------------------------------
# 13. SAVE ELIGIBILITY SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "feature": eligibility_columns,
    "eligible_pairs": [
        int(paired[col].sum())
        for col in eligibility_columns
    ],
    "total_pairs": len(paired)
})

summary.to_csv(
    "feature_eligibility_summary.csv",
    index=False
)

print("Saved: feature_eligibility_summary.csv")

print("\n" + "=" * 65)
print("PAIRED DATA PREPARATION COMPLETED")
print("=" * 65)