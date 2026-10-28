
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, friedmanchisquare

# Database and output paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "instance" / "data.db"
OUT_DIR = BASE_DIR / "statistical_results"
OUT_DIR.mkdir(exist_ok=True)

# Load the data
with sqlite3.connect(DB_PATH) as conn:
    samples = pd.read_sql_query(
        "SELECT * FROM typing_samples",
        conn
    )

metrics = [
    "word_count",
    "duration_seconds",
    "wpm",
    "error_rate"
]

# Convert numeric fields safely
for col in metrics:
    samples[col] = pd.to_numeric(
        samples[col], errors="coerce"
    )

# 1. Descriptive statistics by emotion label
summary = samples.groupby("label")[metrics].agg(
    ["count", "mean", "std", "median", "min", "max"]
)

print("\n=== DESCRIPTIVE STATISTICS BY EMOTION ===")
print(summary.round(3))

summary.to_csv(OUT_DIR / "emotion_summary.csv")

# 2. Spearman correlation analysis
corr_data = samples[metrics].dropna()

corr_matrix = corr_data.corr(method="spearman")

print("\n=== SPEARMAN CORRELATION MATRIX ===")
print(corr_matrix.round(3))

corr_matrix.to_csv(OUT_DIR / "spearman_correlations.csv")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Correlation Between Typing Metrics")
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=200)
plt.close()

# 3. Box plots
for metric in ["wpm", "error_rate", "duration_seconds"]:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=samples, x="label", y=metric)
    plt.title(f"{metric} by Emotion Label")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{metric}_by_emotion.png", dpi=200)
    plt.close()

# 4. Friedman tests for video-response samples
# Requires one observation per participant per emotion.
video = samples[
    samples["sample_type"] == "video_response"
]

print("\n=== FRIEDMAN TESTS: VIDEO RESPONSES ===")

for metric in ["wpm", "error_rate"]:
    wide = video.pivot_table(
        index="participant_id",
        columns="label",
        values=metric,
        aggfunc="mean"
    )

    # Use emotion categories available in all participants.
    complete = wide.dropna(axis=0, how="any")

    if (
        len(complete) >= 2
        and complete.shape[1] >= 3
    ):
        result = friedmanchisquare(
            *[
                complete[col].to_numpy()
                for col in complete.columns
            ]
        )

        print(f"\nMetric: {metric}")
        print("Participants:", len(complete))
        print("Emotions:", list(complete.columns))
        print("Friedman statistic:", round(result.statistic, 4))
        print("P-value:", round(result.pvalue, 4))
    else:
        print(
            f"\nCannot perform {metric} Friedman test: "
            "not enough complete participant data "
            "or fewer than three emotion categories."
        )

# 5. Individual sample counts
print("\n=== SAMPLE COUNTS BY LABEL ===")
print(samples["label"].value_counts(dropna=False))

print("\nAnalysis complete.")
print("Results saved to:", OUT_DIR)
