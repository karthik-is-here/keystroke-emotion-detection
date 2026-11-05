from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "binary_agreement_model"


def main():

    text_path = OUT / "binary_text_predictions.csv"
    key_path = OUT / "binary_keystroke_predictions.csv"

    if not text_path.exists() or not key_path.exists():
        raise FileNotFoundError(
            "Run 20_train_binary_models.py first."
        )

    text = pd.read_csv(text_path)
    keys = pd.read_csv(key_path)

    # Match predictions belonging to the same response.
    merged = text.merge(
        keys,
        on=[
            "sample_id",
            "participant_id",
            "condition",
            "true_label"
        ],
        how="inner",
        suffixes=("_text", "_key"),
        validate="one_to_many"
    )

    if merged.empty:
        raise ValueError("No matching predictions found.")

    # Compare predictions.
    merged["agree"] = (
        merged["predicted_label_text"] ==
        merged["predicted_label_key"]
    )

    merged["text_correct"] = (
        merged["predicted_label_text"] == merged["true_label"]
    )

    merged["key_correct"] = (
        merged["predicted_label_key"] == merged["true_label"]
    )

    merged["both_correct"] = (
        merged["text_correct"] & merged["key_correct"]
    )

    merged["text_only_correct"] = (
        merged["text_correct"] & ~merged["key_correct"]
    )

    merged["key_only_correct"] = (
        ~merged["text_correct"] & merged["key_correct"]
    )

    merged["both_wrong"] = (
        ~merged["text_correct"] & ~merged["key_correct"]
    )

    agreement_results = []
    performance_results = []
    confusion_results = []

    for key_model in sorted(merged["model_key"].unique()):

        model_data = merged[
            merged["model_key"] == key_model
        ]

        for condition in ["genuine", "masked"]:

            subset = model_data[
                model_data["condition"] == condition
            ]

            if subset.empty:
                continue

            # Agreement statistics.
            agreement_results.append({
                "keystroke_model": key_model,
                "condition": condition,
                "samples": len(subset),

                "agreement_rate":
                    subset["agree"].mean(),

                "cohen_kappa":
                    cohen_kappa_score(
                        subset["predicted_label_text"],
                        subset["predicted_label_key"],
                        labels=["negative", "positive"]
                    ),

                "both_correct_rate":
                    subset["both_correct"].mean(),

                "text_only_correct_rate":
                    subset["text_only_correct"].mean(),

                "keystroke_only_correct_rate":
                    subset["key_only_correct"].mean(),

                "both_wrong_rate":
                    subset["both_wrong"].mean()
            })

            # Individual model performance.
            model_columns = [
                (
                    "Text - Logistic Regression",
                    "predicted_label_text"
                ),
                (
                    key_model,
                    "predicted_label_key"
                )
            ]

            for model_name, prediction_column in model_columns:

                y_true = subset["true_label"]
                y_pred = subset[prediction_column]

                precision, recall, f1, _ = (
                    precision_recall_fscore_support(
                        y_true,
                        y_pred,
                        labels=["negative", "positive"],
                        average="binary",
                        pos_label="positive",
                        zero_division=0
                    )
                )

                performance_results.append({
                    "model": model_name,
                    "condition": condition,
                    "samples": len(subset),

                    "accuracy":
                        accuracy_score(y_true, y_pred),

                    "balanced_accuracy":
                        balanced_accuracy_score(y_true, y_pred),

                    "positive_precision": precision,
                    "positive_recall": recall,
                    "positive_f1": f1
                })

                # Confusion matrix.
                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=["negative", "positive"]
                )

                for i, actual in enumerate(
                    ["negative", "positive"]
                ):

                    for j, predicted in enumerate(
                        ["negative", "positive"]
                    ):

                        confusion_results.append({
                            "keystroke_model": key_model,
                            "model": model_name,
                            "condition": condition,
                            "actual": actual,
                            "predicted": predicted,
                            "count": int(cm[i, j])
                        })

    # Save outputs.
    agreement_df = pd.DataFrame(agreement_results)
    performance_df = pd.DataFrame(performance_results)
    confusion_df = pd.DataFrame(confusion_results)

    merged.to_csv(
        OUT / "binary_paired_predictions.csv",
        index=False
    )

    agreement_df.to_csv(
        OUT / "binary_agreement_results.csv",
        index=False
    )

    performance_df.to_csv(
        OUT / "binary_model_performance.csv",
        index=False
    )

    confusion_df.to_csv(
        OUT / "binary_confusion_matrices.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("TEXT–KEYSTROKE PREDICTION AGREEMENT")
    print("=" * 65)

    print(agreement_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 65)

    print(performance_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("FILES SAVED")
    print("=" * 65)

    print("binary_paired_predictions.csv")
    print("binary_agreement_results.csv")
    print("binary_model_performance.csv")
    print("binary_confusion_matrices.csv")


if __name__ == "__main__":
    main()
