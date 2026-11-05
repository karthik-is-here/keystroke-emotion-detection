from pathlib import Path
import sqlite3
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

OUT = BASE / "binary_agreement_model"
OUT.mkdir(exist_ok=True)

DB = BASE / "instance" / "data.db"

EMOTION_MAP = {
    "happy": "positive",
    "sad": "negative",
    "angry": "negative"
}

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


# ============================================================
# LOAD TEXT DATA
# ============================================================

def load_text_data():

    if not DB.exists():
        raise FileNotFoundError(f"Database not found: {DB}")

    with sqlite3.connect(DB) as conn:

        columns = [
            row[1] for row in
            conn.execute("PRAGMA table_info(typing_samples)")
        ]

        # Actual response column in your database.
        text_column = "text_content"

        if text_column not in columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {columns}"
            )

        query = f"""
        SELECT
            id AS sample_id,
            participant_id,
            session_id,
            video_filename,
            sample_type,
            self_reported_emotion,
            text_content AS response_text
        FROM typing_samples
        WHERE sample_type IN (
            'video_response',
            'opposite_response'
        )
        """

        df = pd.read_sql_query(query, conn)

    # Identify genuine and masked responses.
    df["condition"] = df["sample_type"].map({
        "video_response": "genuine",
        "opposite_response": "masked"
    })

    # Convert original emotions into binary labels.
    df["true_label"] = (
        df["self_reported_emotion"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(EMOTION_MAP)
    )

    df["response_text"] = (
        df["response_text"]
        .fillna("")
        .astype(str)
    )

    # Exclude calm and empty responses.
    df = df[
        df["true_label"].notna() &
        df["response_text"].str.strip().ne("")
    ].copy()

    return df


# ============================================================
# LOAD KEYSTROKE FEATURES
# ============================================================

def load_keystroke_features():

    matches = list(
        BASE.rglob("typing_features_quality_checked.csv")
    )

    matches = [
        p for p in matches
        if OUT not in p.parents
    ]

    if not matches:
        raise FileNotFoundError(
            "typing_features_quality_checked.csv not found."
        )

    if len(matches) > 1:

        print("Multiple feature files found:")

        for path in matches:
            print(path)

        raise RuntimeError(
            "Multiple feature files detected. "
            "Keep only one or update the script."
        )

    path = matches[0]

    print("\nUsing feature file:")
    print(path)

    features = pd.read_csv(path)

    if "sample_id" not in features.columns:
        raise ValueError(
            "sample_id column missing from feature CSV."
        )

    available = [
        col for col in FEATURES
        if col in features.columns
    ]

    if len(available) < 3:
        raise ValueError(
            "Insufficient keystroke features found.\n"
            f"Available columns: {features.columns.tolist()}"
        )

    print("\nSelected keystroke features:")
    print(available)

    return features[["sample_id"] + available], available


# ============================================================
# PARTICIPANT NORMALISATION
# ============================================================

def normalize_features(train, test, feature_cols):

    train = train.copy()
    test = test.copy()

    # Genuine-response baseline for training participants.
    baselines = (
        train[train["condition"] == "genuine"]
        .groupby("participant_id")[feature_cols]
        .mean()
    )

    for participant, indices in train.groupby(
        "participant_id"
    ).groups.items():

        baseline = baselines.loc[participant]

        train.loc[indices, feature_cols] = (
            train.loc[indices, feature_cols] - baseline
        )

    # Held-out participant baseline.
    # Uses genuine test features without emotion labels.
    test_baseline = (
        test[test["condition"] == "genuine"][feature_cols]
        .mean()
    )

    if test_baseline.isna().all():
        raise ValueError(
            "No genuine samples available for test normalisation."
        )

    test.loc[:, feature_cols] = (
        test[feature_cols] - test_baseline
    )

    return train, test


# ============================================================
# MAIN TRAINING PROCESS
# ============================================================

def main():

    print("=" * 65)
    print("BINARY TEXT–KEYSTROKE AGREEMENT MODEL")
    print("=" * 65)

    # Load data.
    text_df = load_text_data()

    feature_df, feature_cols = (
        load_keystroke_features()
    )

    # Merge text and keystroke data.
    data = text_df.merge(
        feature_df,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feature")
    )

    if len(data) != len(text_df):
        raise ValueError(
            f"{len(text_df) - len(data)} text samples "
            "have no matching keystroke features."
        )

    # Verify participant IDs if available in feature CSV.
    if "participant_id_feature" in data.columns:

        if not (
            data["participant_id"].astype(str).reset_index(drop=True)
            .equals(
                data["participant_id_feature"]
                .astype(str).reset_index(drop=True)
            )
        ):
            raise ValueError(
                "Participant IDs do not match between files."
            )

        data.drop(
            columns=["participant_id_feature"],
            inplace=True
        )

    data["participant_id"] = (
        data["participant_id"].astype(str)
    )

    data["sample_id"] = (
        data["sample_id"].astype(str)
    )

    participants = sorted(
        data["participant_id"].unique()
    )

    print("\nTotal usable samples:", len(data))
    print("Total participants:", len(participants))

    print("\nBinary emotion distribution:")
    print(
        data.groupby(
            ["condition", "true_label"]
        ).size()
    )

    text_predictions = []
    key_predictions = []

    # ========================================================
    # LEAVE-ONE-PARTICIPANT-OUT CROSS VALIDATION
    # ========================================================

    for participant in participants:

        train = data[
            (data["participant_id"] != participant) &
            (data["condition"] == "genuine")
        ].copy()

        test = data[
            data["participant_id"] == participant
        ].copy()

        if train["true_label"].nunique() != 2:
            print(
                f"Skipping participant {participant}: "
                "training fold lacks both classes."
            )
            continue

        if test.empty:
            continue

        print(
            f"\nParticipant {participant}: "
            f"Training={len(train)}, Testing={len(test)}"
        )

        # ====================================================
        # MODEL 1: TEXT-ONLY
        # ====================================================

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            sublinear_tf=True
        )

        try:

            X_train = vectorizer.fit_transform(
                train["response_text"]
            )

            X_test = vectorizer.transform(
                test["response_text"]
            )

        except ValueError as error:

            print(
                f"Text vectorisation skipped for "
                f"participant {participant}: {error}"
            )

            continue

        text_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )

        text_model.fit(
            X_train,
            train["true_label"]
        )

        text_pred = text_model.predict(X_test)

        for (_, row), prediction in zip(
            test.iterrows(),
            text_pred
        ):

            text_predictions.append({
                "sample_id": row["sample_id"],
                "participant_id": participant,
                "condition": row["condition"],
                "true_label": row["true_label"],
                "predicted_label": prediction,
                "model": "Text - Logistic Regression"
            })

        # ====================================================
        # MODEL 2: KEYSTROKE-ONLY
        # ====================================================

        train_k, test_k = normalize_features(
            train,
            test,
            feature_cols
        )

        keystroke_models = {

            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        }

        for model_name, estimator in (
            keystroke_models.items()
        ):

            key_model = Pipeline([
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
                    estimator
                )
            ])

            key_model.fit(
                train_k[feature_cols],
                train_k["true_label"]
            )

            key_pred = key_model.predict(
                test_k[feature_cols]
            )

            for (_, row), prediction in zip(
                test_k.iterrows(),
                key_pred
            ):

                key_predictions.append({
                    "sample_id": row["sample_id"],
                    "participant_id": participant,
                    "condition": row["condition"],
                    "true_label": row["true_label"],
                    "predicted_label": prediction,
                    "model": f"Keystroke - {model_name}"
                })

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    text_output = pd.DataFrame(text_predictions)
    key_output = pd.DataFrame(key_predictions)

    if text_output.empty or key_output.empty:
        raise RuntimeError(
            "No predictions generated. Check the input data."
        )

    text_path = (
        OUT / "binary_text_predictions.csv"
    )

    key_path = (
        OUT / "binary_keystroke_predictions.csv"
    )

    text_output.to_csv(
        text_path,
        index=False
    )

    key_output.to_csv(
        key_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print("\nText predictions:", len(text_output))
    print("Keystroke predictions:", len(key_output))

    print("\nSaved files:")
    print(text_path)
    print(key_path)


if __name__ == "__main__":
    main()
