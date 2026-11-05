from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "text_model_predictions.csv"
RESULTS_FILE = BASE_DIR / "text_model_results.csv"

OUTPUT_DIR = BASE_DIR / "text_model_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["negative", "neutral", "positive"]

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120
})


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(RESULTS_FILE)

    predictions = pd.read_csv(PREDICTIONS_FILE)
    results = pd.read_csv(RESULTS_FILE)

    print("Predictions loaded:", len(predictions))
    print("Results loaded:", len(results))

    return predictions, results


# ============================================================
# 1. ACCURACY COMPARISON
# ============================================================

def plot_accuracy(results):

    models = [
        "TF-IDF + Logistic Regression",
        "Majority Baseline"
    ]

    conditions = ["genuine", "masked"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, condition in enumerate(conditions):

        values = []

        for model in models:

            row = results[
                (results["model"] == model) &
                (results["condition"] == condition)
            ]

            values.append(
                row["accuracy"].iloc[0] * 100
            )

        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=condition.capitalize()
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3
        )

    ax.set_title("Text-only Model: Accuracy Comparison")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "01_accuracy_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 2. MACRO F1 COMPARISON
# ============================================================

def plot_macro_f1(results):

    models = [
        "TF-IDF + Logistic Regression",
        "Majority Baseline"
    ]

    conditions = ["genuine", "masked"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, condition in enumerate(conditions):

        values = []

        for model in models:

            row = results[
                (results["model"] == model) &
                (results["condition"] == condition)
            ]

            values.append(
                row["macro_f1"].iloc[0] * 100
            )

        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=condition.capitalize()
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3
        )

    ax.set_title("Text-only Model: Macro F1 Comparison")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "02_macro_f1_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 3. CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(predictions):

    conditions = ["genuine", "masked"]

    for condition in conditions:

        subset = predictions[
            predictions["condition"] == condition
        ]

        matrix = confusion_matrix(
            subset["true_valence"],
            subset["predicted_valence"],
            labels=CLASSES
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        image = ax.imshow(
            matrix,
            interpolation="nearest"
        )

        ax.set_title(
            f"Confusion Matrix - {condition.capitalize()} Text"
        )

        ax.set_xlabel("Predicted Emotion")
        ax.set_ylabel("Actual Emotion")

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))

        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)

        threshold = matrix.max() / 2

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                    fontsize=13
                )

        fig.colorbar(image, ax=ax)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"03_confusion_matrix_{condition}.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 4. EMOTION-WISE F1 COMPARISON
# ============================================================

def plot_emotion_f1(predictions):

    conditions = ["genuine", "masked"]

    scores = {
        "genuine": [],
        "masked": []
    }

    for condition in conditions:

        subset = predictions[
            predictions["condition"] == condition
        ]

        for emotion in CLASSES:

            score = f1_score(
                subset["true_valence"],
                subset["predicted_valence"],
                labels=[emotion],
                average="macro",
                zero_division=0
            )

            scores[condition].append(score * 100)

    x = np.arange(len(CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(
        x - width / 2,
        scores["genuine"],
        width,
        label="Genuine"
    )

    bars2 = ax.bar(
        x + width / 2,
        scores["masked"],
        width,
        label="Masked"
    )

    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)

    ax.set_title("Emotion-wise F1 Score: Genuine vs Masked")
    ax.set_ylabel("F1 Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_emotion_wise_f1.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("TEXT-ONLY MODEL VISUALISATION")
    print("=" * 65)

    predictions, results = load_data()

    plot_accuracy(results)
    print("Created accuracy comparison.")

    plot_macro_f1(results)
    print("Created macro F1 comparison.")

    plot_confusion_matrices(predictions)
    print("Created confusion matrices.")

    plot_emotion_f1(predictions)
    print("Created emotion-wise F1 comparison.")

    print("\nAll visualisations saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()