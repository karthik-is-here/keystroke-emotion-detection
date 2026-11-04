import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load classification results
df = pd.read_csv("manipulation check/manipulation_check_predictions.csv")

emotions = ["happy", "sad", "calm", "angry"]

# Separate honest and masked responses
honest = df[df["sample_type"] == "video_response"].copy()
masked = df[df["sample_type"] == "opposite_response"].copy()


# --------------------------------------------------
# 1. CONFUSION MATRICES
# --------------------------------------------------

def generate_confusion_matrix(data, true_column, title, filename):

    cm = confusion_matrix(
        data[true_column],
        data["predicted_emotion"],
        labels=emotions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=emotions
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"\n{title}")
    print(cm)


generate_confusion_matrix(
    honest,
    "self_reported_emotion",
    "Honest Responses Confusion Matrix",
    "honest_confusion_matrix.png"
)

generate_confusion_matrix(
    masked,
    "target_emotion",
    "Masked Responses Confusion Matrix",
    "masked_confusion_matrix.png"
)


# --------------------------------------------------
# 2. PARTICIPANT-WISE ACCURACY
# --------------------------------------------------

honest["correct"] = (
    honest["self_reported_emotion"] ==
    honest["predicted_emotion"]
)

masked["correct"] = (
    masked["target_emotion"] ==
    masked["predicted_emotion"]
)

honest_accuracy = (
    honest.groupby("participant_id")["correct"]
    .mean()
    .reset_index(name="honest_accuracy")
)

masked_accuracy = (
    masked.groupby("participant_id")["correct"]
    .mean()
    .reset_index(name="masked_accuracy")
)

participant_results = pd.merge(
    honest_accuracy,
    masked_accuracy,
    on="participant_id",
    how="outer"
)

participant_results["accuracy_difference"] = (
    participant_results["honest_accuracy"] -
    participant_results["masked_accuracy"]
)

participant_results.to_csv(
    "participant_wise_accuracy.csv",
    index=False
)


# --------------------------------------------------
# 3. PARTICIPANT-LEVEL SUMMARY
# --------------------------------------------------

print("\n========== PARTICIPANT-WISE RESULTS ==========")

print(participant_results.round(3).to_string(index=False))

print("\n========== OVERALL PARTICIPANT SUMMARY ==========")

print(
    participant_results[
        ["honest_accuracy", "masked_accuracy"]
    ].describe().round(3)
)


# --------------------------------------------------
# 4. COMPARE HONEST VS MASKED
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    participant_results["participant_id"],
    participant_results["honest_accuracy"],
    marker="o",
    label="Honest"
)

plt.plot(
    participant_results["participant_id"],
    participant_results["masked_accuracy"],
    marker="o",
    label="Masked"
)

plt.xlabel("Participant ID")
plt.ylabel("Classification Accuracy")
plt.title("Honest vs Masked Accuracy by Participant")

plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "participant_accuracy_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Number of participants:", len(participant_results))

print(
    "Mean honest participant accuracy:",
    round(participant_results["honest_accuracy"].mean(), 4)
)

print(
    "Mean masked participant accuracy:",
    round(participant_results["masked_accuracy"].mean(), 4)
)

print(
    "Participants with higher honest accuracy:",
    (participant_results["accuracy_difference"] > 0).sum()
)

print(
    "Participants with higher masked accuracy:",
    (participant_results["accuracy_difference"] < 0).sum()
)

print("\nAll analysis files generated successfully!")