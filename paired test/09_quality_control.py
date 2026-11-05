import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD EXISTING DATA
# --------------------------------------------------

features = pd.read_csv("typing_features.csv")

holds = pd.read_csv("keystroke_hold_diagnostics.csv")

NON_PRINTABLE = {
    "Shift", "Control", "Alt", "AltGraph",
    "Meta", "CapsLock", "Tab", "Escape",
    "Enter", "ArrowUp", "ArrowDown",
    "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown",
    "Insert", "Delete", "Backspace"
}

print("=" * 65)
print("KEYSTROKE DATA QUALITY CONTROL")
print("=" * 65)

# --------------------------------------------------
# 2. INSPECT HOLD-TIME DISTRIBUTION
# --------------------------------------------------

print("\n--- HOLD TIME DISTRIBUTION ---")

print(
    holds["hold_ms"].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(2)
)

print("\nLong hold counts:")

for threshold in [1000, 2000, 5000, 10000]:
    count = (holds["hold_ms"] > threshold).sum()

    print(f"Holds above {threshold} ms: {count}")


# --------------------------------------------------
# 3. IDENTIFY POTENTIALLY UNRELIABLE SAMPLES
# --------------------------------------------------

print("\n--- SAMPLE QUALITY CHECK ---")

# Use printable key holds only
printable_holds = holds[
    ~holds["key"].isin(NON_PRINTABLE)
].copy()

# Count extreme holds per sample
hold_quality = printable_holds.groupby(
    "sample_id"
).agg(
    printable_hold_count=("hold_ms", "count"),

    holds_over_2s=(
        "hold_ms",
        lambda x: (x > 2000).sum()
    ),

    holds_over_5s=(
        "hold_ms",
        lambda x: (x > 5000).sum()
    ),

    holds_over_10s=(
        "hold_ms",
        lambda x: (x > 10000).sum()
    )
).reset_index()

# Merge diagnostic information
df = features.merge(
    hold_quality,
    left_on="id",
    right_on="sample_id",
    how="left"
)

count_columns = [
    "printable_hold_count",
    "holds_over_2s",
    "holds_over_5s",
    "holds_over_10s"
]

df[count_columns] = df[count_columns].fillna(0)

# Flag samples with insufficient events
df["low_event_count"] = (
    df["valid_hold_count"] < 10
)

# Flag samples containing very long holds
df["extreme_hold_flag"] = (
    df["holds_over_5s"] > 0
)

# Flag samples with potentially problematic timing
df["timing_quality_flag"] = (
    df["unmatched_keyups"] > 0
) | (
    df["unmatched_keydowns"] > 0
)

# General review flag
df["requires_review"] = (
    df["low_event_count"]
    | df["extreme_hold_flag"]
    | df["timing_quality_flag"]
)

print("\nQuality flag summary:")

for column in [
    "low_event_count",
    "extreme_hold_flag",
    "timing_quality_flag",
    "requires_review"
]:
    print(
        f"{column}: {df[column].sum()} samples"
    )


# --------------------------------------------------
# 4. CALCULATE CLEAN HOLD-TIME FEATURES
# --------------------------------------------------

print("\n--- CLEAN HOLD-TIME FEATURES ---")

# Five seconds is used as a provisional exclusion
# threshold for sensitivity analysis.
# Original measurements are preserved.

clean_holds = printable_holds[
    printable_holds["hold_ms"] <= 5000
].copy()

clean_hold_features = clean_holds.groupby(
    "sample_id"
).agg(
    clean_mean_hold_ms=("hold_ms", "mean"),

    clean_std_hold_ms=("hold_ms", "std"),

    clean_hold_count=("hold_ms", "count")
).reset_index()

df = df.merge(
    clean_hold_features,
    on="sample_id",
    how="left"
)

print(
    df[
        [
            "mean_hold_ms",
            "clean_mean_hold_ms",
            "std_hold_ms",
            "clean_std_hold_ms"
        ]
    ].describe().round(3)
)

print("\nSamples with no usable clean hold measurements:")

print(
    df[df["clean_hold_count"].isna()][
        ["id", "participant_id", "sample_type"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 5. IDENTIFY PROBLEMATIC SAMPLES
# --------------------------------------------------

print("\n--- SAMPLES REQUIRING REVIEW ---")

review_columns = [
    "id",
    "participant_id",
    "sample_type",
    "valid_hold_count",
    "printable_hold_count",
    "holds_over_2s",
    "holds_over_5s",
    "holds_over_10s",
    "unmatched_keyups",
    "unmatched_keydowns",
    "requires_review"
]

review = df[df["requires_review"]]

print("Samples requiring review:", len(review))

if not review.empty:
    print(
        review[review_columns].to_string(index=False)
    )


# --------------------------------------------------
# 6. CHECK GENUINE-MASKED PAIRS
# --------------------------------------------------

print("\n--- PAIRED SAMPLE CHECK ---")

pair_keys = [
    "participant_id",
    "session_id",
    "video_filename",
    "self_reported_emotion"
]

genuine = df[
    df["sample_type"] == "video_response"
].copy()

masked = df[
    df["sample_type"] == "opposite_response"
].copy()

genuine_pairs = genuine[pair_keys].drop_duplicates()

masked_pairs = masked[pair_keys].drop_duplicates()

paired = genuine_pairs.merge(
    masked_pairs,
    on=pair_keys,
    how="inner"
)

print("Genuine samples:", len(genuine))
print("Masked samples:", len(masked))
print("Complete matching pairs:", len(paired))

# Check duplicate pairing keys
print(
    "Duplicate genuine pairing keys:",
    genuine.duplicated(pair_keys).sum()
)

print(
    "Duplicate masked pairing keys:",
    masked.duplicated(pair_keys).sum()
)


# --------------------------------------------------
# 7. SAVE QUALITY-CONTROLLED DATA
# --------------------------------------------------

df.to_csv(
    "typing_features_quality_checked.csv",
    index=False
)

review.to_csv(
    "samples_requiring_review.csv",
    index=False
)

clean_holds.to_csv(
    "clean_keystroke_holds.csv",
    index=False
)

print("\n--- OUTPUT FILES ---")

print("1. typing_features_quality_checked.csv")
print("2. samples_requiring_review.csv")
print("3. clean_keystroke_holds.csv")

print("\n" + "=" * 65)
print("QUALITY CONTROL COMPLETED")
print("=" * 65)
