import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("manipulation_check_data.csv")

# 2. Load zero-shot classification model
print("Loading classification model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

emotions = ["happy", "sad", "calm", "angry"]

# 3. Classify every response
predictions = []
confidence_scores = []

print("\nClassifying responses...")

for index, row in df.iterrows():

    text = str(row["text_content"])

    result = classifier(
        text,
        candidate_labels=emotions,
        multi_label=False
    )

    predictions.append(result["labels"][0])
    confidence_scores.append(result["scores"][0])

    print(f"Processed {index + 1}/{len(df)}")

# 4. Store predictions
df["predicted_emotion"] = predictions
df["confidence"] = confidence_scores

# 5. Separate honest and masked responses
honest = df[df["sample_type"] == "video_response"].copy()
masked = df[df["sample_type"] == "opposite_response"].copy()

# 6. Evaluate honest responses
print("\n========== HONEST RESPONSES ==========")

honest_accuracy = accuracy_score(
    honest["self_reported_emotion"],
    honest["predicted_emotion"]
)

print("Accuracy:", round(honest_accuracy, 4))

print(classification_report(
    honest["self_reported_emotion"],
    honest["predicted_emotion"],
    labels=emotions,
    zero_division=0
))

# 7. Evaluate masked responses
print("\n========== MASKED RESPONSES ==========")

masked_accuracy = accuracy_score(
    masked["target_emotion"],
    masked["predicted_emotion"]
)

print("Accuracy:", round(masked_accuracy, 4))

print(classification_report(
    masked["target_emotion"],
    masked["predicted_emotion"],
    labels=emotions,
    zero_division=0
))

# 8. Save results
df.to_csv("manipulation_check_predictions.csv", index=False)

honest.to_csv("honest_predictions.csv", index=False)
masked.to_csv("masked_predictions.csv", index=False)

print("\nResults saved successfully!")