import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "combined_model_predictions.csv"
OUTPUT_DIR = BASE_DIR / "combined_model_visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)

LABELS = ["negative", "neutral", "positive"]

CONDITIONS = ["genuine", "masked"]

MODELS = ["Logistic Regression", "Random Forest"]


# ============================================================
# LOAD DATA
# ============================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Predictions file not found: {PREDICTIONS_FILE}"
    )

df = pd.read_csv(PREDICTIONS_FILE)

print("Loaded combined model predictions.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Identify prediction columns
true_col = next(
    (c for c in ["true_valence", "true_label", "actual_valence", "y_true"]
     if c in df.columns),
    None
)

pred_col = next(
    (c for c in ["predicted_valence", "predicted_label", "prediction", "y_pred"]
     if c in df.columns),
    None
)

model_col = next(
    (c for c in ["model", "model_name", "classifier"]
     if c in df.columns),
    None
)

condition_col = next(
    (c for c in ["condition", "sample_type"]
     if c in df.columns),
    None
)

if None in [true_col, pred_col, model_col, condition_col]:
    raise ValueError(
        "Could not identify required columns.\n"
        f"Available columns: {df.columns.tolist()}"
    )

df[condition_col] = df[condition_col].astype(str).str.lower()

print("\nModels:", df[model_col].unique())
print("Conditions:", df[condition_col].unique())


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

results = []

for model in df[model_col].unique():

    for condition in CONDITIONS:

        subset = df[
            (df[model_col] == model) &
            (df[condition_col] == condition)
        ]

        if subset.empty:
            continue

        y_true = subset[true_col]
        y_pred = subset[pred_col]

        results.append({
            "model": model,
            "condition": condition,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "combined_performance_metrics.csv",
    index=False
)

print("\nPerformance metrics:")
print(results_df.to_string(index=False))


# ============================================================
# 1. ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="model",
    y="accuracy",
    hue="condition"
)

plt.title("Combined Model: Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_accuracy_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. BALANCED ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="model",
    y="balanced_accuracy",
    hue="condition"
)

plt.title("Combined Model: Balanced Accuracy")
plt.xlabel("Model")
plt.ylabel("Balanced Accuracy")
plt.ylim(0, 1)
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_balanced_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. MACRO F1 COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="model",
    y="macro_f1",
    hue="condition"
)

plt.title("Combined Model: Macro F1 Comparison")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1)
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_macro_f1_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

for model in df[model_col].unique():

    for condition in CONDITIONS:

        subset = df[
            (df[model_col] == model) &
            (df[condition_col] == condition)
        ]

        if subset.empty:
            continue

        cm = confusion_matrix(
            subset[true_col],
            subset[pred_col],
            labels=LABELS
        )

        plt.figure(figsize=(7, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS
        )

        plt.title(f"{model} - {condition.title()} Confusion Matrix")
        plt.xlabel("Predicted Valence")
        plt.ylabel("Actual Valence")
        plt.tight_layout()

        filename = (
            f"04_confusion_{model.lower().replace(' ', '_')}_"
            f"{condition}.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=300
        )

        plt.show()


# ============================================================
# 5. CLASS-WISE F1 COMPARISON
# ============================================================

class_results = []

for model in df[model_col].unique():

    for condition in CONDITIONS:

        subset = df[
            (df[model_col] == model) &
            (df[condition_col] == condition)
        ]

        if subset.empty:
            continue

        report = classification_report(
            subset[true_col],
            subset[pred_col],
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )

        for label in LABELS:

            class_results.append({
                "model": model,
                "condition": condition,
                "valence": label,
                "f1_score": report[label]["f1-score"]
            })

class_df = pd.DataFrame(class_results)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=class_df,
    x="valence",
    y="f1_score",
    hue="condition"
)

plt.title("Combined Model: Class-wise F1-score")
plt.xlabel("Valence")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_classwise_f1.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. SAVE CLASS-WISE RESULTS
# ============================================================

class_df.to_csv(
    OUTPUT_DIR / "classwise_f1_results.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("VISUALISATION COMPLETED")
print("=" * 60)

print("\nCharts and result files saved in:")
print(OUTPUT_DIR)

print("\nGenerated visualisations:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file.name)