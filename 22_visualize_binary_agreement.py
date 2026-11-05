from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

OUT = BASE / "binary_agreement_model"

PLOT_DIR = OUT / "visualizations"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD RESULTS
# ============================================================

agreement_file = OUT / "binary_agreement_results.csv"
performance_file = OUT / "binary_model_performance.csv"
confusion_file = OUT / "binary_confusion_matrices.csv"

for file in [
    agreement_file,
    performance_file,
    confusion_file
]:
    if not file.exists():
        raise FileNotFoundError(
            f"Missing file: {file}\n"
            "Run 21_compare_binary_predictions.py first."
        )

agreement = pd.read_csv(agreement_file)
performance = pd.read_csv(performance_file)
confusion = pd.read_csv(confusion_file)

print("Loaded all comparison results.")


# ============================================================
# COMMON PLOTTING FUNCTION
# ============================================================

def create_bar_chart(
    data,
    x,
    y,
    title,
    xlabel,
    ylabel,
    filename,
    ylim=True
):

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue="condition"
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(0, 1)

    plt.xticks(rotation=10, ha="right")
    plt.legend(title="Condition")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 1. TEXT–KEYSTROKE PREDICTION AGREEMENT
# ============================================================

create_bar_chart(
    agreement,
    x="keystroke_model",
    y="agreement_rate",
    title="Agreement Between Text and Keystroke Predictions",
    xlabel="Keystroke Model",
    ylabel="Agreement Rate",
    filename="01_prediction_agreement.png"
)


# ============================================================
# 2. COHEN'S KAPPA
# ============================================================

create_bar_chart(
    agreement,
    x="keystroke_model",
    y="cohen_kappa",
    title="Text–Keystroke Agreement: Cohen's Kappa",
    xlabel="Keystroke Model",
    ylabel="Cohen's Kappa",
    filename="02_cohen_kappa.png",
    ylim=False
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=agreement,
    x="condition",
    y="cohen_kappa",
    hue="keystroke_model"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.title("Cohen's Kappa: Genuine vs Masked")
plt.xlabel("Response Condition")
plt.ylabel("Cohen's Kappa")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "03_kappa_condition_comparison.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 3. ACCURACY COMPARISON
# ============================================================

create_bar_chart(
    performance,
    x="model",
    y="accuracy",
    title="Binary Classification Accuracy",
    xlabel="Prediction Model",
    ylabel="Accuracy",
    filename="04_accuracy_comparison.png"
)


# ============================================================
# 4. BALANCED ACCURACY
# ============================================================

create_bar_chart(
    performance,
    x="model",
    y="balanced_accuracy",
    title="Balanced Accuracy Comparison",
    xlabel="Prediction Model",
    ylabel="Balanced Accuracy",
    filename="05_balanced_accuracy.png"
)


# ============================================================
# 5. POSITIVE CLASS F1-SCORE
# ============================================================

create_bar_chart(
    performance,
    x="model",
    y="positive_f1",
    title="Positive Emotion: F1-Score Comparison",
    xlabel="Prediction Model",
    ylabel="Positive F1-Score",
    filename="06_positive_f1.png"
)


# ============================================================
# 6. POSITIVE PRECISION AND RECALL
# ============================================================

positive_metrics = performance.melt(
    id_vars=["model", "condition"],
    value_vars=[
        "positive_precision",
        "positive_recall"
    ],
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=positive_metrics,
    x="model",
    y="score",
    hue="metric"
)

plt.title("Positive Emotion: Precision and Recall")
plt.xlabel("Prediction Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=10, ha="right")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "07_positive_precision_recall.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 7. CORRECTNESS PATTERNS
# ============================================================

outcomes = agreement.melt(
    id_vars=[
        "keystroke_model",
        "condition"
    ],
    value_vars=[
        "both_correct_rate",
        "text_only_correct_rate",
        "keystroke_only_correct_rate",
        "both_wrong_rate"
    ],
    var_name="outcome",
    value_name="rate"
)

outcome_labels = {
    "both_correct_rate": "Both correct",
    "text_only_correct_rate": "Only text correct",
    "keystroke_only_correct_rate": "Only keystroke correct",
    "both_wrong_rate": "Both incorrect"
}

outcomes["outcome"] = outcomes["outcome"].map(
    outcome_labels
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=outcomes,
    x="outcome",
    y="rate",
    hue="condition"
)

plt.title("Correctness Patterns: Text vs Keystroke")
plt.xlabel("Prediction Outcome")
plt.ylabel("Proportion of Samples")
plt.ylim(0, 1)
plt.xticks(rotation=10, ha="right")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "08_correctness_patterns.png",
    dpi=300
)

plt.show()
plt.close()


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

for (model, condition), group in confusion.groupby(
    ["model", "condition"]
):

    matrix = group.pivot(
        index="actual",
        columns="predicted",
        values="count"
    )

    matrix = matrix.reindex(
        index=["negative", "positive"],
        columns=["negative", "positive"]
    ).fillna(0)

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5
    )

    plt.title(
        f"{model}\n{condition.title()} Confusion Matrix",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")

    plt.tight_layout()

    safe_name = (
        model.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    filename = (
        f"09_confusion_{safe_name}_{condition}.png"
    )

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 9. SAVE VISUALISATION DATA
# ============================================================

agreement.to_csv(
    PLOT_DIR / "agreement_chart_data.csv",
    index=False
)

performance.to_csv(
    PLOT_DIR / "performance_chart_data.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("BINARY AGREEMENT VISUALISATION COMPLETED")
print("=" * 65)

print("\nOutput folder:")
print(PLOT_DIR)

print("\nGenerated charts:")

for file in sorted(PLOT_DIR.glob("*.png")):
    print("-", file.name)
