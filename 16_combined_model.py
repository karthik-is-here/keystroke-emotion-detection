import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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

DB_PATH = BASE_DIR / "instance" / "data.db"

FEATURE_FILENAME = "typing_features_quality_checked.csv"

OUTPUT_PREDICTIONS = BASE_DIR / "combined_model_predictions.csv"
OUTPUT_RESULTS = BASE_DIR / "combined_model_results.csv"

KEYSTROKE_FEATURES = [
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

MODELS = [
    "Logistic Regression",
    "Random Forest"
]


# ============================================================
# FIND FEATURE FILE
# ============================================================

def find_feature_file():

    matches = list(
        BASE_DIR.rglob(FEATURE_FILENAME)
    )

    if not matches:
        raise FileNotFoundError(
            f"{FEATURE_FILENAME} was not found."
        )

    if len(matches) > 1:

        print("\nMultiple feature files found:")

        for file in matches:
            print(file)

        raise RuntimeError(
            "Multiple copies found. Keep one authoritative "
            "file or specify the correct path."
        )

    return matches[0]


# ============================================================
# LOAD TEXT DATA FROM SQLITE
# ============================================================

def load_text_data():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

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
    WHERE sample_type IN (
        'video_response',
        'opposite_response'
    )
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    df["self_reported_emotion"] = (
        df["self_reported_emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["valence"] = df[
        "self_reported_emotion"
    ].map(VALENCE_MAP)

    df["condition"] = df["sample_type"].map({
        "video_response": "genuine",
        "opposite_response": "masked"
    })

    df["text_content"] = (
        df["text_content"]
        .fillna("")
        .astype(str)
    )

    df = df.dropna(subset=["valence"])

    df = df[
        df["text_content"].str.strip() != ""
    ].copy()

    return df


# ============================================================
# LOAD KEYSTROKE FEATURES
# ============================================================

def load_keystroke_data():

    feature_path = find_feature_file()

    print("\nLoading keystroke features:")
    print(feature_path)

    df = pd.read_csv(feature_path)

    required = [
        "sample_id",
        "participant_id"
    ] + KEYSTROKE_FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing keystroke columns: {missing}"
        )

    df = df[
        ["sample_id", "participant_id"] +
        KEYSTROKE_FEATURES
    ].copy()

    for feature in KEYSTROKE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in feature CSV."
        )

    return df


# ============================================================
# MERGE TEXT AND KEYSTROKE DATA
# ============================================================

def prepare_dataset():

    print("\nLoading text data...")

    text_df = load_text_data()

    print("Text records:", len(text_df))

    print("\nLoading keystroke features...")

    feature_df = load_keystroke_data()

    print("Feature records:", len(feature_df))

    df = text_df.merge(
        feature_df,
        on=["sample_id", "participant_id"],
        how="inner",
        validate="one_to_one",
        indicator=True
    )

    unmatched = df["_merge"] != "both"

    if unmatched.any():
        raise ValueError(
            "Some text records did not match keystroke features."
        )

    df.drop(columns=["_merge"], inplace=True)

    if len(df) != len(text_df):

        raise ValueError(
            f"Only {len(df)} of {len(text_df)} "
            "text records matched."
        )

    print("\nMERGED DATA SUMMARY")
    print("-" * 60)

    print("Total samples:", len(df))

    print("\nConditions:")
    print(df["condition"].value_counts())

    print("\nValence distribution:")
    print(df.groupby(["condition", "valence"]).size())

    print("\nParticipants:", df["participant_id"].nunique())

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

        for feature in KEYSTROKE_FEATURES:

            baseline = result.loc[
                genuine_mask,
                feature
            ].dropna()

            if len(baseline) == 0:

                result.loc[
                    participant_mask,
                    feature
                ] = np.nan

                continue

            baseline_mean = baseline.mean()

            baseline_std = baseline.std(ddof=0)

            if (
                pd.isna(baseline_std)
                or baseline_std < 1e-10
            ):

                result.loc[
                    participant_mask,
                    feature
                ] = (
                    result.loc[
                        participant_mask,
                        feature
                    ] - baseline_mean
                )

            else:

                result.loc[
                    participant_mask,
                    feature
                ] = (
                    result.loc[
                        participant_mask,
                        feature
                    ] - baseline_mean
                ) / baseline_std

    return result


# ============================================================
# CREATE COMBINED FEATURE MATRICES
# ============================================================

def create_feature_matrices(train_df, test_df):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )

    X_train_text = vectorizer.fit_transform(
        train_df["text_content"]
    )

    X_test_text = vectorizer.transform(
        test_df["text_content"]
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_numeric = imputer.fit_transform(
        train_df[KEYSTROKE_FEATURES]
    )

    X_test_numeric = imputer.transform(
        test_df[KEYSTROKE_FEATURES]
    )

    scaler = StandardScaler()

    X_train_numeric = scaler.fit_transform(
        X_train_numeric
    )

    X_test_numeric = scaler.transform(
        X_test_numeric
    )

    X_train = hstack([
        X_train_text,
        csr_matrix(X_train_numeric)
    ]).tocsr()

    X_test = hstack([
        X_test_text,
        csr_matrix(X_test_numeric)
    ]).tocsr()

    return X_train, X_test


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    logistic_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }


# ============================================================
# LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION
# ============================================================

def run_lopo_cv(df):

    participants = sorted(
        df["participant_id"].unique()
    )

    all_predictions = []

    print("\n" + "=" * 70)
    print("COMBINED MODEL: LEAVE-ONE-PARTICIPANT-OUT")
    print("=" * 70)

    for participant in participants:

        train_df = df[
            df["participant_id"] != participant
        ].copy()

        test_df = df[
            df["participant_id"] == participant
        ].copy()

        # Train using genuine responses only
        train_df = train_df[
            train_df["condition"] == "genuine"
        ].copy()

        # Participant-specific normalisation
        train_normalized = normalize_by_participant(
            train_df
        )

        test_normalized = normalize_by_participant(
            test_df
        )

        y_train = train_normalized["valence"]

        y_test = test_normalized["valence"]

        if y_train.nunique() < 3:

            print(
                f"Skipping participant {participant}: "
                "training data lacks a valence class."
            )

            continue

        X_train, X_test = create_feature_matrices(
            train_normalized,
            test_normalized
        )

        models = create_models()

        majority_class = y_train.value_counts().idxmax()

        print(
            f"\nParticipant {participant}: "
            f"Training={len(train_df)}, "
            f"Testing={len(test_df)}, "
            f"Features={X_train.shape[1]}"
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
                    "session_id": row["session_id"],
                    "video_filename": row["video_filename"],
                    "condition": row["condition"],
                    "true_emotion": row["self_reported_emotion"],
                    "true_valence": row["valence"],
                    "predicted_valence": prediction,
                    "majority_prediction": majority_class,
                    "model": model_name
                })

    predictions_df = pd.DataFrame(
        all_predictions
    )

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
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
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
# EVALUATION
# ============================================================

def evaluate_results(predictions_df):

    results = []

    for model_name in MODELS:

        model_df = predictions_df[
            predictions_df["model"] == model_name
        ]

        for condition in ["genuine", "masked"]:

            subset = model_df[
                model_df["condition"] == condition
            ]

            y_true = subset["true_valence"]

            y_pred = subset["predicted_valence"]

            baseline_pred = subset[
                "majority_prediction"
            ]

            model_metrics = calculate_metrics(
                y_true,
                y_pred
            )

            baseline_metrics = calculate_metrics(
                y_true,
                baseline_pred
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
    print("FINAL COMBINED MODEL COMPARISON")
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
    print("COMBINED TEXT + KEYSTROKE EMOTION PREDICTION")
    print("=" * 70)

    df = prepare_dataset()

    predictions_df = run_lopo_cv(df)

    evaluate_results(predictions_df)


if __name__ == "__main__":
    main()