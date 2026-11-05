import sqlite3
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "instance" / "data.db"

OUTPUT_PREDICTIONS = BASE_DIR / "text_model_predictions.csv"
OUTPUT_RESULTS = BASE_DIR / "text_model_results.csv"

VALENCE_MAP = {
    "happy": "positive",
    "sad": "negative",
    "angry": "negative",
    "calm": "neutral"
}

CLASS_ORDER = ["negative", "neutral", "positive"]


# ============================================================
# LOAD DATA FROM SQLITE
# ============================================================

def load_data():

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    connection = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id AS sample_id,
        participant_id,
        session_id,
        sample_type,
        video_filename,
        self_reported_emotion,
        text_content
    FROM typing_samples
    WHERE sample_type IN ('video_response', 'opposite_response')
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    print("\nDATABASE RECORDS:", len(df))

    df["self_reported_emotion"] = (
        df["self_reported_emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["text_content"] = df["text_content"].fillna("").astype(str)

    df["valence"] = df["self_reported_emotion"].map(VALENCE_MAP)

    df = df.dropna(subset=["valence"])

    df = df[df["text_content"].str.strip() != ""]

    df["condition"] = df["sample_type"].map({
        "video_response": "genuine",
        "opposite_response": "masked"
    })

    print("\nCONDITION DISTRIBUTION")
    print(df["condition"].value_counts())

    print("\nVALENCE DISTRIBUTION")
    print(df.groupby(["condition", "valence"]).size())

    print("\nPARTICIPANTS:", df["participant_id"].nunique())

    return df


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        )
    }


# ============================================================
# LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION
# ============================================================

def run_lopo_cv(df):

    participants = sorted(df["participant_id"].unique())

    all_predictions = []

    print("\n" + "=" * 70)
    print("LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
    print("=" * 70)

    for participant in participants:

        train_df = df[
            (df["participant_id"] != participant) &
            (df["condition"] == "genuine")
        ].copy()

        test_df = df[
            df["participant_id"] == participant
        ].copy()

        if train_df["valence"].nunique() < 3:
            print(
                f"Skipping participant {participant}: "
                "training data does not contain all three classes."
            )
            continue

        model = create_model()

        model.fit(
            train_df["text_content"],
            train_df["valence"]
        )

        predictions = model.predict(test_df["text_content"])

        # Majority-class baseline from genuine training responses
        majority_class = train_df["valence"].value_counts().idxmax()

        for (_, row), prediction in zip(
            test_df.iterrows(),
            predictions
        ):

            all_predictions.append({
                "sample_id": row["sample_id"],
                "participant_id": participant,
                "session_id": row["session_id"],
                "video_filename": row["video_filename"],
                "condition": row["condition"],
                "true_emotion": row["self_reported_emotion"],
                "true_valence": row["valence"],
                "predicted_valence": prediction,
                "majority_prediction": majority_class
            })

        print(
            f"Participant {participant}: "
            f"Training={len(train_df)}, "
            f"Testing={len(test_df)}"
        )

    predictions_df = pd.DataFrame(all_predictions)

    if predictions_df.empty:
        raise ValueError("No predictions were generated.")

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    return predictions_df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_results(predictions_df):

    results = []

    for condition in ["genuine", "masked"]:

        subset = predictions_df[
            predictions_df["condition"] == condition
        ]

        y_true = subset["true_valence"]

        model_predictions = subset["predicted_valence"]

        baseline_predictions = subset["majority_prediction"]

        model_metrics = calculate_metrics(
            y_true,
            model_predictions
        )

        baseline_metrics = calculate_metrics(
            y_true,
            baseline_predictions
        )

        results.append({
            "condition": condition,
            "model": "TF-IDF + Logistic Regression",
            "samples": len(subset),
            **model_metrics
        })

        results.append({
            "condition": condition,
            "model": "Majority Baseline",
            "samples": len(subset),
            **baseline_metrics
        })

        print("\n" + "=" * 70)
        print(f"{condition.upper()} TEXT RESULTS")
        print("=" * 70)

        print("\nMODEL CLASSIFICATION REPORT")

        print(
            classification_report(
                y_true,
                model_predictions,
                labels=CLASS_ORDER,
                zero_division=0
            )
        )

        print("MODEL CONFUSION MATRIX")

        print(
            confusion_matrix(
                y_true,
                model_predictions,
                labels=CLASS_ORDER
            )
        )

        print("\nMAJORITY BASELINE")

        print(
            classification_report(
                y_true,
                baseline_predictions,
                labels=CLASS_ORDER,
                zero_division=0
            )
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved files:")
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_RESULTS)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEXT-ONLY EMOTION PREDICTION")
    print("=" * 70)

    df = load_data()

    predictions_df = run_lopo_cv(df)

    evaluate_results(predictions_df)


if __name__ == "__main__":
    main()