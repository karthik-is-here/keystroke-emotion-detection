import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = "typing_features_quality_checked.csv"

OUTPUT_PREDICTIONS = BASE_DIR / "keystroke_model_predictions.csv"
OUTPUT_RESULTS = BASE_DIR / "keystroke_model_results.csv"

FEATURES = [
    "clean_mean_hold_ms",
    "clean_std_hold_ms",
    "mean_flight_ms",
    "std_flight_ms",
    "active_typing_speed_cps",
    "pause_rate",
    "backspace_rate",
    "total_time_s"
]

VALENCE_MAP = {
    "happy": "positive",
    "sad": "negative",
    "angry": "negative",
    "calm": "neutral"
}

CLASS_ORDER = ["negative", "neutral", "positive"]

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest"
]


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_input_file():

    matches = list(BASE_DIR.rglob(INPUT_FILE))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"{INPUT_FILE} was not found in {BASE_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple copies of the input file were found:\n"
            + "\n".join(str(p) for p in matches)
            + "\nKeep one authoritative copy or specify its path."
        )

    return matches[0]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    input_path = find_input_file()

    print("\nLoading:", input_path)

    df = pd.read_csv(input_path)

    required_columns = [
        "sample_id",
        "participant_id",
        "sample_type",
        "self_reported_emotion"
    ] + FEATURES

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    df = df[
        df["sample_type"].isin([
            "video_response",
            "opposite_response"
        ])
    ].copy()

    df["self_reported_emotion"] = (
        df["self_reported_emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["valence"] = df["self_reported_emotion"].map(
        VALENCE_MAP
    )

    df["condition"] = df["sample_type"].map({
        "video_response": "genuine",
        "opposite_response": "masked"
    })

    df = df.dropna(subset=["valence"])

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print("\nTOTAL SAMPLES:", len(df))

    print("\nCONDITION DISTRIBUTION")
    print(df["condition"].value_counts())

    print("\nVALENCE DISTRIBUTION")
    print(df.groupby(["condition", "valence"]).size())

    print("\nPARTICIPANTS:", df["participant_id"].nunique())

    return df


# ============================================================
# PARTICIPANT-WISE NORMALISATION
# ============================================================

def normalize_by_participant(df):

    result = df.copy()

    for participant in result["participant_id"].unique():

        participant_mask = (
            result["participant_id"] == participant
        )

        genuine_mask = (
            participant_mask &
            (result["condition"] == "genuine")
        )

        for feature in FEATURES:

            baseline_values = result.loc[
                genuine_mask, feature
            ].dropna()

            if len(baseline_values) == 0:
                result.loc[
                    participant_mask, feature
                ] = np.nan
                continue

            baseline_mean = baseline_values.mean()
            baseline_std = baseline_values.std(ddof=0)

            if (
                pd.isna(baseline_std)
                or baseline_std < 1e-10
            ):
                result.loc[
                    participant_mask, feature
                ] = (
                    result.loc[participant_mask, feature]
                    - baseline_mean
                )
            else:
                result.loc[
                    participant_mask, feature
                ] = (
                    result.loc[participant_mask, feature]
                    - baseline_mean
                ) / baseline_std

    return result


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    logistic_model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
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

    random_forest_model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        )
    }


# ============================================================
# LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION
# ============================================================

def run_lopo_cv(df):

    participants = sorted(
        df["participant_id"].unique()
    )

    all_predictions = []

    models = create_models()

    print("\n" + "=" * 70)
    print("LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
    print("=" * 70)

    for participant in participants:

        # Split participants first
        train_df = df[
            df["participant_id"] != participant
        ].copy()

        test_df = df[
            df["participant_id"] == participant
        ].copy()

        # Training uses genuine samples only
        train_df = train_df[
            train_df["condition"] == "genuine"
        ].copy()

        # Normalise using each participant's genuine baseline
        train_normalized = normalize_by_participant(
            train_df.assign(condition="genuine")
        )

        # For the held-out participant, calculate the baseline
        # from their genuine samples, then apply it to both
        # genuine and masked test samples.
        test_normalized = normalize_by_participant(
            test_df
        )

        X_train = train_normalized[FEATURES]
        y_train = train_normalized["valence"]

        X_test = test_normalized[FEATURES]
        y_test = test_normalized["valence"]

        if y_train.nunique() < 3:
            print(
                f"Skipping participant {participant}: "
                "training data lacks a valence class."
            )
            continue

        majority_class = y_train.value_counts().idxmax()

        print(
            f"\nParticipant {participant}: "
            f"Training={len(train_df)}, "
            f"Testing={len(test_df)}"
        )

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            for (_, row), prediction in zip(
                test_normalized.iterrows(),
                predictions
            ):

                all_predictions.append({
                    "sample_id": row["sample_id"],
                    "participant_id": participant,
                    "condition": row["condition"],
                    "true_emotion": row["self_reported_emotion"],
                    "true_valence": row["valence"],
                    "predicted_valence": prediction,
                    "majority_prediction": majority_class,
                    "model": model_name
                })

    predictions_df = pd.DataFrame(all_predictions)

    if predictions_df.empty:
        raise ValueError(
            "No predictions were generated."
        )

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

    for model_name in MODEL_NAMES:

        model_df = predictions_df[
            predictions_df["model"] == model_name
        ]

        for condition in ["genuine", "masked"]:

            subset = model_df[
                model_df["condition"] == condition
            ]

            y_true = subset["true_valence"]

            y_pred = subset["predicted_valence"]

            baseline_pred = subset["majority_prediction"]

            model_metrics = calculate_metrics(
                y_true, y_pred
            )

            baseline_metrics = calculate_metrics(
                y_true, baseline_pred
            )

            results.append({
                "model": model_name,
                "condition": condition,
                "samples": len(subset),
                **model_metrics
            })

            results.append({
                "model": "Majority Baseline",
                "condition": condition,
                "samples": len(subset),
                **baseline_metrics
            })

            print("\n" + "=" * 70)
            print(f"{model_name.upper()} - {condition.upper()}")
            print("=" * 70)

            print("\nCLASSIFICATION REPORT")

            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=CLASS_ORDER,
                    zero_division=0
                )
            )

            print("CONFUSION MATRIX")

            print(
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=CLASS_ORDER
                )
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
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
    print("KEYSTROKE-ONLY EMOTION PREDICTION")
    print("=" * 70)

    df = load_data()

    predictions_df = run_lopo_cv(df)

    evaluate_results(predictions_df)


if __name__ == "__main__":
    main()