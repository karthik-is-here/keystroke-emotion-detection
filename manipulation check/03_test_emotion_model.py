from transformers import pipeline

print("Loading emotion classification model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = """
I felt very happy watching the beautiful scenery.
The entire experience made me smile and feel positive.
"""

emotions = ["happy", "sad", "calm", "angry"]

result = classifier(
    text,
    candidate_labels=emotions,
    multi_label=False
)

print("\nClassification Result:")
print("----------------------")
print("Text:", text)
print("Predicted emotion:", result["labels"][0])

print("\nConfidence scores:")
for emotion, score in zip(result["labels"], result["scores"]):
    print(f"{emotion}: {score:.4f}")