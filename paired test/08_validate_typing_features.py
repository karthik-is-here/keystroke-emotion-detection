import pandas as pd

df = pd.read_csv("typing_features.csv")

print("=" * 60)
print("TYPING FEATURE VALIDATION")
print("=" * 60)

# 1. Dataset information
print("\n--- DATASET INFORMATION ---")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Total samples:", len(df))

print("\nSample types:")
print(df["sample_type"].value_counts())

# 2. Feature summary
features = [
    "mean_hold_ms",
    "std_hold_ms",
    "mean_flight_ms",
    "std_flight_ms",
    "typing_speed_cps",
    "pause_rate",
    "backspace_rate",
    "total_time_s",
    "active_typing_time_s"
]

available_features = [
    col for col in features if col in df.columns
]

print("\n--- FEATURE SUMMARY ---")
print(df[available_features].describe().round(3).to_string())

# 3. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().to_string())

# 4. Highest typing speeds
print("\n--- HIGHEST TYPING SPEEDS ---")

print(
    df.nlargest(10, "typing_speed_cps")[
        [
            "id",
            "participant_id",
            "sample_type",
            "typing_speed_cps",
            "total_time_s",
            "active_typing_time_s"
        ]
    ].to_string(index=False)
)

# 5. Extreme hold times
print("\n--- EXTREME HOLD TIMES ---")

print(
    df.nlargest(10, "std_hold_ms")[
        [
            "id",
            "participant_id",
            "sample_type",
            "mean_hold_ms",
            "std_hold_ms",
            "valid_hold_count"
        ]
    ].to_string(index=False)
)

# 6. Negative flight times
print("\n--- NEGATIVE FLIGHT TIMES ---")

print(
    "Samples with negative mean flight time:",
    (df["mean_flight_ms"] < 0).sum()
)

print(
    df[df["mean_flight_ms"] < 0][
        [
            "id",
            "participant_id",
            "sample_type",
            "mean_flight_ms"
        ]
    ].to_string(index=False)
)

# 7. Low event counts
print("\n--- LOW EVENT COUNTS ---")

low_events = df[df["valid_hold_count"] < 10]

print("Samples with fewer than 10 valid hold events:", len(low_events))

if not low_events.empty:
    print(
        low_events[
            [
                "id",
                "participant_id",
                "sample_type",
                "valid_hold_count"
            ]
        ].to_string(index=False)
    )

# 8. Feature ranges
print("\n--- FEATURE RANGE CHECK ---")

for feature in available_features:
    print(
        f"{feature}: "
        f"Minimum = {df[feature].min():.3f}, "
        f"Maximum = {df[feature].max():.3f}"
    )

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)
