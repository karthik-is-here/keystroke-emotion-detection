import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "paired_typing_features.csv"

MIN_ELIGIBLE_PAIRS = 3

OUTPUT_SUMMARY = "statistical_analysis_results.csv"
OUTPUT_PARTICIPANTS = "participant_level_analysis.csv"


# Features used in the primary analysis.
# Clean hold-time features exclude holds longer than 5 seconds.

FEATURES = {
    "Clean mean hold time (ms)": (
        "clean_mean_hold_ms",
        "hold_eligible"
    ),

    "Clean standard deviation of hold time (ms)": (
        "clean_std_hold_ms",
        "hold_eligible"
    ),

    "Mean flight time (ms)": (
        "mean_flight_ms",
        "flight_eligible"
    ),

    "Standard deviation of flight time (ms)": (
        "std_flight_ms",
        "flight_eligible"
    ),

    "Overall task throughput (characters/sec)": (
        "typing_speed_cps",
        "speed_eligible"
    ),

    "Pause rate": (
        "pause_rate",
        "pause_eligible"
    ),

    "Backspace rate": (
        "backspace_rate",
        "backspace_eligible"
    ),

    "Total task duration (seconds)": (
        "total_time_s",
        "duration_eligible"
    )
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, feature, condition):
    """
    Finds columns such as:
    clean_mean_hold_ms_genuine
    clean_mean_hold_ms_masked
    """

    possible_names = [
        f"{feature}_{condition}",
        f"{condition}_{feature}"
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    raise KeyError(
        f"Could not find column for {feature} ({condition}). "
        f"Expected one of: {possible_names}"
    )


def calculate_rank_biserial(differences):
    """
    Matched-pairs rank-biserial correlation.

    Positive values indicate higher masked values.
    Negative values indicate higher genuine values.
    """

    differences = np.asarray(differences, dtype=float)

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))

    positive_ranks = ranks[nonzero > 0].sum()
    negative_ranks = ranks[nonzero < 0].sum()

    total_ranks = positive_ranks + negative_ranks

    if total_ranks == 0:
        return 0.0

    return (positive_ranks - negative_ranks) / total_ranks


def holm_correction(p_values):
    """
    Holm-Bonferroni correction for multiple comparisons.
    """

    p_values = np.asarray(p_values, dtype=float)

    number_of_tests = len(p_values)

    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(number_of_tests)

    running_max = 0

    for rank, original_index in enumerate(sorted_indices):
        multiplier = number_of_tests - rank

        corrected = multiplier * p_values[original_index]

        running_max = max(running_max, corrected)

        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def median_iqr(values):
    values = pd.Series(values).dropna()

    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return median, q1, q3


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading paired typing data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total paired observations: {len(df)}")
print(f"Unique participants: {df['participant_id'].nunique()}")


# ============================================================
# PARTICIPANT-LEVEL AGGREGATION
# ============================================================

print("\nAggregating observations by participant...")

participant_results = []
statistical_results = []

for feature_name, (feature_column, eligibility_column) in FEATURES.items():

    genuine_column = find_column(
        df, feature_column, "genuine"
    )

    masked_column = find_column(
        df, feature_column, "masked"
    )

    required_columns = [
        "participant_id",
        eligibility_column,
        genuine_column,
        masked_column
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    # Only use eligible pairs with both measurements available.
    eligible = df[
        (df[eligibility_column] == True) &
        df[genuine_column].notna() &
        df[masked_column].notna()
    ].copy()

    # Average the same eligible video pairs for both conditions.
    participant_data = (
        eligible
        .groupby("participant_id")
        .agg(
            genuine_mean=(genuine_column, "mean"),
            masked_mean=(masked_column, "mean"),
            eligible_video_pairs=(genuine_column, "count")
        )
        .reset_index()
    )

    # Require at least 3 eligible pairs.
    participant_data = participant_data[
        participant_data["eligible_video_pairs"]
        >= MIN_ELIGIBLE_PAIRS
    ].copy()

    participant_data["difference_masked_minus_genuine"] = (
        participant_data["masked_mean"]
        - participant_data["genuine_mean"]
    )

    participant_data["feature"] = feature_name

    participant_results.append(participant_data)

    genuine_values = participant_data["genuine_mean"].to_numpy()
    masked_values = participant_data["masked_mean"].to_numpy()
    differences = participant_data[
        "difference_masked_minus_genuine"
    ].to_numpy()

    n = len(differences)

    if n < 2:
        print(f"Skipping {feature_name}: insufficient participants.")
        continue

    # Descriptive statistics
    genuine_median, genuine_q1, genuine_q3 = median_iqr(
        genuine_values
    )

    masked_median, masked_q1, masked_q3 = median_iqr(
        masked_values
    )

    difference_median, difference_q1, difference_q3 = median_iqr(
        differences
    )

    # Wilcoxon signed-rank test
    if np.all(differences == 0):
        statistic = 0.0
        p_value = 1.0
    else:
        test_result = wilcoxon(
            masked_values,
            genuine_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        statistic = test_result.statistic
        p_value = test_result.pvalue

    effect_size = calculate_rank_biserial(differences)

    statistical_results.append({
        "feature": feature_name,
        "participants": n,

        "genuine_median": genuine_median,
        "genuine_Q1": genuine_q1,
        "genuine_Q3": genuine_q3,

        "masked_median": masked_median,
        "masked_Q1": masked_q1,
        "masked_Q3": masked_q3,

        "median_difference": difference_median,
        "difference_Q1": difference_q1,
        "difference_Q3": difference_q3,

        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial_effect": effect_size
    })

    print(
        f"{feature_name}: "
        f"{n} participants, "
        f"p = {p_value:.4f}, "
        f"effect = {effect_size:.3f}"
    )


# ============================================================
# HOLM MULTIPLE-TEST CORRECTION
# ============================================================

results_df = pd.DataFrame(statistical_results)

if not results_df.empty:

    results_df["holm_adjusted_p"] = holm_correction(
        results_df["p_value"].to_numpy()
    )

    results_df["significant_after_holm"] = (
        results_df["holm_adjusted_p"] < 0.05
    )

    results_df["significance"] = results_df[
        "holm_adjusted_p"
    ].apply(
        lambda p: (
            "Significant"
            if p < 0.05
            else "Not significant"
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

if participant_results:
    participant_df = pd.concat(
        participant_results,
        ignore_index=True
    )

    participant_df.to_csv(
        OUTPUT_PARTICIPANTS,
        index=False
    )

    print(
        f"\nParticipant-level data saved to: "
        f"{OUTPUT_PARTICIPANTS}"
    )


results_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print(f"Statistical results saved to: {OUTPUT_SUMMARY}")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("PRIMARY STATISTICAL ANALYSIS")
print("=" * 75)

if not results_df.empty:

    display_columns = [
        "feature",
        "participants",
        "genuine_median",
        "masked_median",
        "median_difference",
        "p_value",
        "holm_adjusted_p",
        "rank_biserial_effect",
        "significance"
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

print("\nAnalysis completed.")