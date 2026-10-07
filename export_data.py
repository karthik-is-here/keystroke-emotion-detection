"""
Exports every table to a CSV file in an `exports/` folder, for offline
analysis and model training. Run this any time after collecting data:

    python export_data.py
"""

import csv
import os

from app import app
from models import ComprehensionCheck, EmotionSession, KeystrokeEvent, Participant, TypingSample, db

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exports")


def _export(model, filename, columns):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in model.query.all():
            writer.writerow([getattr(row, col) for col in columns])
    print(f"Wrote {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with app.app_context():
        _export(
            Participant,
            "participants.csv",
            ["id", "name", "age", "nationality", "first_language",
             "typing_experience", "dominant_hand", "consented_at", "created_at"],
        )
        _export(
            EmotionSession,
            "emotion_sessions.csv",
            ["id", "participant_id", "emotion_order", "created_at"],
        )
        _export(
            TypingSample,
            "typing_samples.csv",
            ["id", "participant_id", "session_id", "label", "sample_type",
             "target_emotion", "video_filename", "self_reported_emotion",
             "prompt_text", "text_content", "word_count", "duration_seconds",
             "wpm", "error_rate", "started_at", "ended_at"],
        )
        _export(
            KeystrokeEvent,
            "keystroke_events.csv",
            ["id", "typing_sample_id", "key", "event_type", "timestamp_ms",
             "sequence_index"],
        )
        _export(
            ComprehensionCheck,
            "comprehension_checks.csv",
            ["id", "typing_sample_id", "question", "selected_answer",
             "correct_answer", "is_correct"],
        )


if __name__ == "__main__":
    main()
