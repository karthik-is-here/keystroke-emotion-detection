
"""
Emotion-wise Keystroke Statistical Analysis
-------------------------------------------
Loads data from instance/data.db.

Analyses:
1. Descriptive statistics by emotion
2. Friedman repeated-measures tests
3. Holm-corrected post-hoc Wilcoxon tests
4. Effect size (Kendall's W)
5. Spearman correlations
6. Box plots and correlation heatmap
7. CSV and text report exports

Interpret findings cautiously when participant counts are small.
"""

import sqlite3
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
    spearmanr
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "instance" / "data.db"
OUTPUT_DIR = BASE_DIR / "emotion_analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

# Existing sample-level typing features in your database.
FEATURES = [
    "wpm",
    "error_rate",
    "duration_seconds",
    "word_count"
]

print("=" * 70)
print("KEYSTROKE EMOTION STATISTICAL ANALYSIS")
print("=" * 70)

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}\n"
        "Check the location of instance/data.db."
    )

# =========================================================
# 2. LOAD DATA
# =========================================================

with sqlite3.connect(DB_PATH) as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )["name"].tolist()

    if "typing_samples" not in tables:
        raise RuntimeError(
            f"'typing_samples' table not found. Tables: {tables}"
        )

    data = pd.read_sql_query(
        "SELECT * FROM typing_samples",
        conn
    )

print("\nDatabase:", DB_PATH)
print("Available tables:", tables)
print("Typing samples loaded:", len(data))
print("Available columns:", data.columns.tolist())

required = [
    "participant_id",
    "label",
    "sample_type"
]

missing = [c for c in required if c not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

features = [c for c in FEATURES if c in data.columns]

if not features:
    raise ValueError(
        "No expected typing features found. "
        "Inspect the typing_samples columns above."
    )

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["label"] = data["label"].astype("string").str.strip()
data["sample_type"] = (
    data["sample_type"].astype("string").str.strip()
)

print("\nEmotion counts:")
print(data["label"].value_counts(dropna=False))

print("\nSample type counts:")
print(data["sample_type"].value_counts(dropna=False))

print("\nUnique participants:", data["participant_id"].nunique())

# =========================================================
# 3. DESCRIPTIVE STATISTICS
# =========================================================

descriptive = (
    data.groupby(["sample_type", "label"], dropna=False)[features]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .round(4)
)

descriptive.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS BY SAMPLE TYPE AND EMOTION")
print("=" * 70)
print(descriptive.to_string())

# =========================================================
# 4. HELPER: HOLM MULTIPLE-TEST CORRECTION
# =========================================================

def holm_adjust(p_values):
    """Return Holm-adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if n == 0:
        return np.array([])

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.maximum.accumulate(
        [(n - i) * sorted_p[i] for i in range(n)]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return adjusted


# =========================================================
# 5. FRIEDMAN TEST AND POST-HOC PAIRWISE TESTS
# =========================================================

overall_results = []
pairwise_results = []

# Exclude baseline when comparing emotion-response conditions.
# Analyze video and opposite responses separately to avoid
# mixing different task conditions.
response_types = [
    "video_response",
    "opposite_response"
]

for sample_type in response_types:

    subset = data[data["sample_type"] == sample_type].copy()

    if subset.empty:
        print(f"\nNo records for sample type: {sample_type}")
        continue

    print("\n" + "=" * 70)
    print(f"REPEATED-MEASURES ANALYSIS: {sample_type}")
    print("=" * 70)

    for feature in features:

        # One value per participant per emotion.
        # If duplicate samples exist, average them first.
        wide = subset.pivot_table(
            index="participant_id",
            columns="label",
            values=feature,
            aggfunc="mean"
        )

        # Friedman requires complete matched observations.
        wide = wide.dropna(axis=0, how="any")

        labels = list(wide.columns)
        n_participants = len(wide)
        n_conditions = len(labels)

        print(
            f"\nFeature: {feature} | "
            f"Participants: {n_participants} | "
            f"Conditions: {n_conditions}"
        )

        if n_participants < 3 or n_conditions < 3:
            print(
                "Insufficient complete matched observations "
                "for the planned Friedman analysis."
            )
            continue

        if any(wide[label].nunique() == 0 for label in labels):
            continue

        try:
            result = friedmanchisquare(
                *[
                    wide[label].to_numpy(dtype=float)
                    for label in labels
                ]
            )

            statistic = float(result.statistic)
            p_value = float(result.pvalue)

            # Kendall's W effect size for Friedman test.
            kendall_w = statistic / (
                n_participants * (n_conditions - 1)
            )

            kendall_w = min(max(kendall_w, 0.0), 1.0)

            overall_results.append({
                "sample_type": sample_type,
                "feature": feature,
                "n_participants": n_participants,
                "n_conditions": n_conditions,
                "conditions": ", ".join(labels),
                "friedman_statistic": statistic,
                "p_value": p_value,
                "kendall_w": kendall_w
            })

            print(f"Friedman statistic: {statistic:.4f}")
            print(f"Raw p-value: {p_value:.6f}")
            print(f"Kendall's W: {kendall_w:.4f}")

            # Pairwise post-hoc comparisons.
            # Require at least 3 paired participants.
            if n_participants < 3:
                continue

            current_pairs = []

            for label_a, label_b in combinations(labels, 2):

                a = wide[label_a].to_numpy(dtype=float)
                b = wide[label_b].to_numpy(dtype=float)

                differences = a - b
                nonzero = differences[differences != 0]

                if len(nonzero) == 0:
                    pair_stat = 0.0
                    pair_p = 1.0
                else:
                    try:
                        pair_result = wilcoxon(
                            a,
                            b,
                            alternative="two-sided",
                            method="auto"
                        )
                        pair_stat = float(pair_result.statistic)
                        pair_p = float(pair_result.pvalue)
                    except ValueError:
                        pair_stat = np.nan
                        pair_p = np.nan

                # Paired rank-biserial effect size.
                abs_diff = np.abs(nonzero)

                if len(abs_diff):
                    ranks = pd.Series(abs_diff).rank().to_numpy()
                    positive_rank_sum = ranks[
                        nonzero > 0
                    ].sum()
                    negative_rank_sum = ranks[
                        nonzero < 0
                    ].sum()
                    denominator = positive_rank_sum + negative_rank_sum

                    rank_biserial = (
                        (positive_rank_sum - negative_rank_sum)
                        / denominator
                        if denominator else 0.0
                    )
                else:
                    rank_biserial = 0.0

                current_pairs.append({
                    "sample_type": sample_type,
                    "feature": feature,
                    "emotion_a": label_a,
                    "emotion_b": label_b,
                    "n_pairs": n_participants,
                    "median_a": float(wide[label_a].median()),
                    "median_b": float(wide[label_b].median()),
                    "wilcoxon_statistic": pair_stat,
                    "p_value": pair_p,
                    "rank_biserial": rank_biserial
                })

            valid_indices = [
                i for i, row in enumerate(current_pairs)
                if np.isfinite(row["p_value"])
            ]

            corrected = holm_adjust([
                current_pairs[i]["p_value"]
                for i in valid_indices
            ])

            for i, adjusted_p in zip(valid_indices, corrected):
                current_pairs[i]["holm_adjusted_p"] = float(adjusted_p)

            for row in current_pairs:
                row.setdefault("holm_adjusted_p", np.nan)
                row["significant_after_correction"] = (
                    row["holm_adjusted_p"] < ALPHA
                    if np.isfinite(row["holm_adjusted_p"])
                    else False
                )
                pairwise_results.append(row)

        except (ValueError, ZeroDivisionError) as exc:
            print("Test could not be completed:", exc)

# Correct overall Friedman p-values across all tested
# sample-type/feature combinations.
overall_df = pd.DataFrame(overall_results)

if not overall_df.empty:
    overall_df["holm_adjusted_p"] = holm_adjust(
        overall_df["p_value"].to_numpy()
    )
    overall_df["significant_after_correction"] = (
        overall_df["holm_adjusted_p"] < ALPHA
    )

    overall_df.to_csv(
        OUTPUT_DIR / "friedman_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("OVERALL FRIEDMAN RESULTS")
    print("=" * 70)
    print(overall_df.round(5).to_string(index=False))

pairwise_df = pd.DataFrame(pairwise_results)

if not pairwise_df.empty:
    pairwise_df.to_csv(
        OUTPUT_DIR / "pairwise_posthoc_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("PAIRWISE WILCOXON RESULTS")
    print("=" * 70)
    print(pairwise_df.round(5).to_string(index=False))

# =========================================================
# 6. SPEARMAN CORRELATION
# =========================================================

correlation_rows = []

for feature_a, feature_b in combinations(features, 2):
    pair = data[[feature_a, feature_b]].dropna()

    if len(pair) < 3:
        continue

    rho, p_value = spearmanr(
        pair[feature_a],
        pair[feature_b]
    )

    correlation_rows.append({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "n_samples": len(pair),
        "spearman_rho": rho,
        "p_value": p_value
    })

correlations = pd.DataFrame(correlation_rows)

if not correlations.empty:
    correlations["holm_adjusted_p"] = holm_adjust(
        correlations["p_value"].to_numpy()
    )
    correlations.to_csv(
        OUTPUT_DIR / "correlation_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATIONS")
    print("=" * 70)
    print(correlations.round(5).to_string(index=False))

# =========================================================
# 7. VISUALIZATIONS
# =========================================================

sns.set_theme(style="whitegrid")

for feature in features:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=data,
        x="label",
        y=feature,
        hue="sample_type"
    )

    plt.title(f"{feature} by Emotion and Sample Type")
    plt.xlabel("Emotion label")
    plt.ylabel(feature)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature}_by_emotion.png",
        dpi=200
    )
    plt.close()

# Correlation heatmap
if len(features) >= 2:
    corr = data[features].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title("Spearman Correlation of Typing Features")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=200
    )
    plt.close()

# =========================================================
# 8. SUMMARY REPORT
# =========================================================

report_path = OUTPUT_DIR / "analysis_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("KEYSTROKE EMOTION STATISTICAL ANALYSIS\n")
    report.write("=" * 60 + "\n")
    report.write(f"Database: {DB_PATH}\n")
    report.write(f"Typing samples: {len(data)}\n")
    report.write(
        f"Unique participants: {data['participant_id'].nunique()}\n"
    )
    report.write(f"Features analyzed: {features}\n\n")

    report.write("Emotion counts:\n")
    report.write(
        data["label"].value_counts(dropna=False).to_string()
    )
    report.write("\n\n")

    report.write(
        "Interpretation: Friedman tests evaluate matched "
        "emotion conditions within participants. "
        "Holm-adjusted p-values account for multiple tests. "
        "Kendall's W summarizes the Friedman effect size. "
        "Pairwise Wilcoxon results use Holm correction within "
        "each feature/sample-type family. Correlations pooled "
        "across repeated samples are descriptive, not valid "
        "independent-observation inference.\n\n"
    )

    if not overall_df.empty:
        report.write("Friedman results:\n")
        report.write(overall_df.to_string(index=False))
        report.write("\n\n")

    if not pairwise_df.empty:
        report.write("Pairwise results:\n")
        report.write(pairwise_df.to_string(index=False))
        report.write("\n\n")

    if not correlations.empty:
        report.write("Correlation results:\n")
        report.write(correlations.to_string(index=False))
        report.write("\n")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("Results saved to:")
print(OUTPUT_DIR.resolve())
print("\nFiles include:")
print("- descriptive_statistics.csv")
print("- friedman_results.csv")
print("- pairwise_posthoc_results.csv")
print("- correlation_results.csv")
print("- Feature comparison box plots")
print("- correlation_heatmap.png")
print("- analysis_report.txt")
