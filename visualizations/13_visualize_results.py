import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rankdata

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUT = PROJECT_DIR / "visualizations"
OUT.mkdir(exist_ok=True)

EMOTIONS = ["happy", "sad", "calm", "angry"]

FEATURES = [
    ("Clean mean hold time", "clean_mean_hold_ms", "hold_eligible", "ms"),
    ("Clean SD hold time", "clean_std_hold_ms", "hold_eligible", "ms"),
    ("Mean flight time", "mean_flight_ms", "flight_eligible", "ms"),
    ("SD flight time", "std_flight_ms", "flight_eligible", "ms"),
    ("Overall task throughput", "typing_speed_cps", "speed_eligible", "characters/sec"),
    ("Pause rate", "pause_rate", "pause_eligible", "proportion"),
    ("Backspace rate", "backspace_rate", "backspace_eligible", "proportion"),
    ("Total task duration", "total_time_s", "duration_eligible", "seconds"),
]


# ============================================================
# FIND FILES AUTOMATICALLY
# ============================================================

def find_file(filename):

    matches = list(PROJECT_DIR.rglob(filename))

    if not matches:
        return None

    return matches[0]


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

def getcol(df, feature, condition):

    possible = [
        f"{feature}_{condition}",
        f"{condition}_{feature}"
    ]

    for col in possible:

        if col in df.columns:
            return col

    raise KeyError(
        f"Missing column for {feature} ({condition})"
    )


# ============================================================
# BOXPLOT
# ============================================================

def boxplot(values, labels, title, ylabel, filename):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / filename,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLocating project files...")

    paired_file = find_file("paired_typing_features.csv")

    primary_file = find_file("participant_level_analysis.csv")

    emotion_file = find_file("emotion_specific_results.csv")

    if paired_file is None:

        raise FileNotFoundError(
            "paired_typing_features.csv was not found."
        )

    print(f"Paired data: {paired_file}")

    if primary_file:
        print(f"Primary results: {primary_file}")
    else:
        print("Primary results CSV not found. Effect-size chart will be skipped.")

    if emotion_file:
        print(f"Emotion results: {emotion_file}")
    else:
        print("Emotion results CSV not found. Emotion heatmap will be skipped.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(paired_file)

    df["emotion_clean"] = (
        df["self_reported_emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry"
        })
    )

    print(f"\nTotal rows: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")

    print("\nEmotion distribution:")
    print(df["emotion_clean"].value_counts().to_string())

    # ========================================================
    # 1. OVERALL BOXPLOTS
    # ========================================================

    print("\n[1/6] Generating overall boxplots...")

    summary = []

    for label, feature, eligible_col, ylabel in FEATURES:

        g = getcol(df, feature, "genuine")
        m = getcol(df, feature, "masked")

        valid = df[
            df[eligible_col].fillna(False).astype(bool)
            & df[g].notna()
            & df[m].notna()
        ]

        if valid.empty:
            continue

        boxplot(
            [
                valid[g].to_numpy(),
                valid[m].to_numpy()
            ],
            ["Genuine", "Masked"],
            f"Genuine vs Masked: {label}",
            ylabel,
            f"01_overall_{feature}.png"
        )

        for condition, col in [
            ("genuine", g),
            ("masked", m)
        ]:

            x = valid[col].dropna()

            summary.append({
                "feature": label,
                "condition": condition,
                "n": len(x),
                "median": x.median(),
                "Q1": x.quantile(0.25),
                "Q3": x.quantile(0.75),
                "mean": x.mean(),
                "sd": x.std()
            })

    pd.DataFrame(summary).to_csv(
        OUT / "descriptive_summary.csv",
        index=False
    )

    # ========================================================
    # 2. EMOTION-SPECIFIC BOXPLOTS
    # ========================================================

    print("[2/6] Generating emotion-specific boxplots...")

    for emotion in EMOTIONS:

        emotion_df = df[
            df["emotion_clean"] == emotion
        ]

        for label, feature, eligible_col, ylabel in FEATURES:

            g = getcol(df, feature, "genuine")
            m = getcol(df, feature, "masked")

            valid = emotion_df[
                emotion_df[eligible_col].fillna(False).astype(bool)
                & emotion_df[g].notna()
                & emotion_df[m].notna()
            ]

            if valid.empty:
                continue

            boxplot(
                [
                    valid[g].to_numpy(),
                    valid[m].to_numpy()
                ],
                ["Genuine", "Masked"],
                f"{emotion.title()}: {label}",
                ylabel,
                f"02_{emotion}_{feature}.png"
            )

    # ========================================================
    # 3. PARTICIPANT-LEVEL PAIRED PLOTS
    # ========================================================

    print("[3/6] Generating participant-level paired plots...")

    for label, feature, eligible_col, ylabel in FEATURES:

        g = getcol(df, feature, "genuine")
        m = getcol(df, feature, "masked")

        valid = df[
            df[eligible_col].fillna(False).astype(bool)
            & df[g].notna()
            & df[m].notna()
        ]

        participant = (
            valid.groupby("participant_id")
            .agg(
                genuine=(g, "mean"),
                masked=(m, "mean"),
                n_pairs=(g, "count")
            )
            .reset_index()
        )

        participant = participant[
            participant["n_pairs"] >= 3
        ]

        if participant.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for _, row in participant.iterrows():

            ax.plot(
                [0, 1],
                [row["genuine"], row["masked"]],
                marker="o",
                alpha=0.65
            )

        ax.set_xticks(
            [0, 1],
            ["Genuine", "Masked"]
        )

        ax.set_ylabel(ylabel)

        ax.set_title(
            f"Participant-level paired values: {label}"
        )

        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            OUT / f"03_paired_{feature}.png",
            dpi=180
        )

        plt.close(fig)

    # ========================================================
    # 4. OVERALL EFFECT-SIZE CHART
    # ========================================================

    print("[4/6] Generating overall effect-size chart...")

    if primary_file:

        primary = pd.read_csv(primary_file)

        if {
            "feature",
            "difference_masked_minus_genuine"
        }.issubset(primary.columns):

            rows = []

            for feature, group in primary.groupby("feature"):

                d = group[
                    "difference_masked_minus_genuine"
                ].dropna().to_numpy(float)

                d = d[d != 0]

                if len(d):

                    ranks = rankdata(np.abs(d))

                    effect = (
                        ranks[d > 0].sum()
                        - ranks[d < 0].sum()
                    ) / ranks.sum()

                else:
                    effect = 0.0

                rows.append({
                    "feature": feature,
                    "effect": effect
                })

            effects = pd.DataFrame(rows).sort_values("effect")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.barh(
                effects["feature"],
                effects["effect"]
            )

            ax.axvline(0)
            ax.set_xlim(-1, 1)

            ax.set_xlabel(
                "Rank-biserial effect (positive = masked higher)"
            )

            ax.set_title("Overall participant-level effect sizes")

            fig.tight_layout()

            fig.savefig(
                OUT / "04_overall_effect_sizes.png",
                dpi=180
            )

            plt.close(fig)

    # ========================================================
    # 5. EMOTION EFFECT-SIZE HEATMAP
    # ========================================================

    print("[5/6] Generating emotion effect-size heatmap...")

    if emotion_file:

        results = pd.read_csv(emotion_file)

        required = {
            "emotion",
            "feature",
            "rank_biserial_effect"
        }

        if required.issubset(results.columns):

            pivot = results.pivot(
                index="feature",
                columns="emotion",
                values="rank_biserial_effect"
            )

            pivot.to_csv(
                OUT / "emotion_effect_size_table.csv"
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            image = ax.imshow(
                pivot.to_numpy(float),
                vmin=-1,
                vmax=1,
                aspect="auto"
            )

            ax.set_xticks(
                range(len(pivot.columns)),
                pivot.columns
            )

            ax.set_yticks(
                range(len(pivot.index)),
                pivot.index
            )

            ax.set_title(
                "Emotion-specific rank-biserial effect sizes"
            )

            ax.set_xlabel("Emotion")
            ax.set_ylabel("Feature")

            fig.colorbar(
                image,
                ax=ax,
                label="Effect size"
            )

            fig.tight_layout()

            fig.savefig(
                OUT / "05_emotion_effect_size_heatmap.png",
                dpi=180
            )

            plt.close(fig)

    # ========================================================
    # 6. SPEARMAN CORRELATION HEATMAP
    # ========================================================

    print("[6/6] Generating correlation heatmap...")

    columns = []
    labels = []

    for label, feature, _, _ in FEATURES:

        columns.append(
            getcol(df, feature, "genuine")
        )

        labels.append(label)

    corr_data = df[columns].copy()
    corr_data.columns = labels

    corr = corr_data.corr(method="spearman")

    corr.to_csv(
        OUT / "genuine_feature_spearman_correlations.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        corr.to_numpy(float),
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(labels)),
        labels
    )

    ax.set_title(
        "Spearman correlation: genuine typing features"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Spearman correlation"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "06_genuine_feature_correlation_heatmap.png",
        dpi=180
    )

    plt.close(fig)

    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)

    print(f"Output folder: {OUT.resolve()}")
    print(f"Files generated: {len(list(OUT.iterdir()))}")

    print("\nRemember:")
    print("typing_speed_cps represents overall task throughput.")
    print("It is not pure active typing speed.")


if __name__ == "__main__":
    main()