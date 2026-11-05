import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata
from pathlib import Path

INPUT_FILE = "paired test/paired_typing_features.csv"
OUTPUT_RESULTS = "emotion_specific_results.csv"
OUTPUT_PARTICIPANTS = "emotion_specific_participant_data.csv"

FEATURES = {
    "Clean mean hold time (ms)": ("clean_mean_hold_ms", "hold_eligible"),
    "Clean SD of hold time (ms)": ("clean_std_hold_ms", "hold_eligible"),
    "Mean flight time (ms)": ("mean_flight_ms", "flight_eligible"),
    "SD of flight time (ms)": ("std_flight_ms", "flight_eligible"),
    "Overall task throughput (characters/sec)": ("typing_speed_cps", "speed_eligible"),
    "Pause rate": ("pause_rate", "pause_eligible"),
    "Backspace rate": ("backspace_rate", "backspace_eligible"),
    "Total task duration (seconds)": ("total_time_s", "duration_eligible"),
}
EMOTIONS = ["happy", "sad", "calm", "angry"]


def find_column(df, feature, condition):
    for name in (f"{feature}_{condition}", f"{condition}_{feature}"):
        if name in df.columns:
            return name
    raise KeyError(f"Missing {feature} column for {condition}.")


def rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def median_iqr(values):
    values = pd.Series(values).dropna()
    return float(values.median()), float(values.quantile(.25)), float(values.quantile(.75))


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Put this script beside that CSV file."
        )

    df = pd.read_csv(INPUT_FILE)
    for col in ["participant_id", "self_reported_emotion"]:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")

    aliases = {
        "happiness": "happy", "sadness": "sad", "anger": "angry"
    }
    df["emotion_clean"] = (
        df["self_reported_emotion"].astype(str).str.strip().str.lower()
        .replace(aliases)
    )

    print(f"Rows: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print("\nEmotion counts:")
    print(df["emotion_clean"].value_counts().to_string())

    participant_outputs = []
    results = []

    for feature_label, (feature, eligibility) in FEATURES.items():
        genuine_col = find_column(df, feature, "genuine")
        masked_col = find_column(df, feature, "masked")
        if eligibility not in df.columns:
            raise KeyError(f"Eligibility column missing: {eligibility}")

        for emotion in EMOTIONS:
            subset = df[df["emotion_clean"] == emotion].copy()
            eligible = subset[
                subset[eligibility].fillna(False).astype(bool)
                & subset[genuine_col].notna()
                & subset[masked_col].notna()
            ]

            # One paired observation per participant and emotion.
            # Multiple eligible video pairs, if present, are averaged within participant.
            pdata = (
                eligible.groupby("participant_id", as_index=False)
                .agg(
                    genuine_mean=(genuine_col, "mean"),
                    masked_mean=(masked_col, "mean"),
                    eligible_pairs=(genuine_col, "count"),
                )
            )
            pdata["difference_masked_minus_genuine"] = (
                pdata["masked_mean"] - pdata["genuine_mean"]
            )
            pdata["emotion"] = emotion
            pdata["feature"] = feature_label
            participant_outputs.append(pdata)

            if len(pdata) < 2:
                print(f"Skipping {emotion} / {feature_label}: fewer than 2 participants.")
                continue

            genuine = pdata["genuine_mean"].to_numpy(float)
            masked = pdata["masked_mean"].to_numpy(float)
            differences = masked - genuine

            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    masked, genuine, alternative="two-sided",
                    zero_method="wilcox", method="auto"
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)

            gm, gq1, gq3 = median_iqr(genuine)
            mm, mq1, mq3 = median_iqr(masked)
            dm, dq1, dq3 = median_iqr(differences)

            results.append({
                "emotion": emotion,
                "feature": feature_label,
                "participants": len(pdata),
                "genuine_median": gm, "genuine_Q1": gq1, "genuine_Q3": gq3,
                "masked_median": mm, "masked_Q1": mq1, "masked_Q3": mq3,
                "median_difference_masked_minus_genuine": dm,
                "difference_Q1": dq1, "difference_Q3": dq3,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_effect": rank_biserial(differences),
            })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError("No emotion-specific tests could be calculated.")

    # Correct across all emotion-by-feature tests in this analysis.
    result_df["holm_adjusted_p"] = holm_adjust(result_df["p_value"])
    result_df["significant_after_holm"] = result_df["holm_adjusted_p"] < .05
    result_df["significance"] = np.where(
        result_df["significant_after_holm"], "Significant", "Not significant"
    )

    participant_df = pd.concat(participant_outputs, ignore_index=True)
    participant_df.to_csv(OUTPUT_PARTICIPANTS, index=False)
    result_df.to_csv(OUTPUT_RESULTS, index=False)

    columns = [
        "emotion", "feature", "participants", "genuine_median",
        "masked_median", "median_difference_masked_minus_genuine",
        "p_value", "holm_adjusted_p", "rank_biserial_effect", "significance"
    ]
    print("\n" + "=" * 110)
    print("EMOTION-SPECIFIC PAIRED ANALYSIS")
    print("=" * 110)
    print(result_df[columns].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print(f"\nSaved: {OUTPUT_RESULTS}")
    print(f"Saved: {OUTPUT_PARTICIPANTS}")
    print("\nAnalysis completed.")
    print(
        "Note: typing_speed_cps is overall task throughput, calculated using "
        "total task duration (including time before typing)."
    )


if __name__ == "__main__":
    main()
