"""
Extract the 10 keystroke features from the project's SQLite database.

Run from the project folder:
    python extract_features.py

Expected database:
    instance/data.db

Output:
    feature_extraction/keystroke_features.csv
"""

import sqlite3
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "instance" / "data.db"
OUTPUT_DIR = PROJECT_DIR / "feature_extraction"
OUTPUT_DIR.mkdir(exist_ok=True)

# Modifier/navigation keys are not treated as ordinary typing keys.
MODIFIER_KEYS = {
    "shift", "leftshift", "rightshift",
    "control", "ctrl", "leftctrl", "rightctrl",
    "alt", "leftalt", "rightalt", "altgr",
    "capslock", "numlock", "scrolllock",
    "meta", "windows", "command", "cmd",
    "escape", "tab", "arrowleft", "arrowright",
    "arrowup", "arrowdown", "home", "end",
    "pageup", "pagedown", "insert",
}


def normalise_key(key):
    """Return a comparable key name without changing printable characters."""
    return str(key if key is not None else "").strip()


def is_modifier_or_navigation(key):
    return normalise_key(key).lower() in MODIFIER_KEYS


def is_printable_key(key):
    """
    A printable character is a one-character key, including space.
    Named keys such as Backspace and Delete are not printable characters.
    """
    raw = str(key if key is not None else "")
    return len(raw) == 1


def extract_one_sample(events, sample):
    """
    events must be ordered by sequence_index.
    Match each key-up to the earliest unmatched key-down of the same key.
    """
    pending_downs = defaultdict(deque)
    completed = []
    typing_downs = []

    for event in events:
        key = str(event["key"] if event["key"] is not None else "")
        event_type = str(event["event_type"]).lower()
        timestamp = float(event["timestamp_ms"])

        if event_type == "down":
            if not is_modifier_or_navigation(key):
                typing_downs.append({
                    "key": key,
                    "down_ms": timestamp,
                    "up_ms": None,
                })
                pending_downs[key].append(typing_downs[-1])

        elif event_type == "up":
            if pending_downs[key]:
                stroke = pending_downs[key].popleft()
                stroke["up_ms"] = timestamp
                completed.append(stroke)

    # The first three non-modifier key-downs define D1/U1, D2/U2, D3/U3.
    first_three = typing_downs[:3]

    def hold_end(index):
        if len(first_three) > index:
            return first_three[index]["up_ms"]
        return None

    d1 = first_three[0]["down_ms"] if len(first_three) >= 1 else None
    u1 = hold_end(0)
    d2 = first_three[1]["down_ms"] if len(first_three) >= 2 else None
    u2 = hold_end(1)
    d3 = first_three[2]["down_ms"] if len(first_three) >= 3 else None
    u3 = hold_end(2)

    def difference(end, start):
        if end is None or start is None:
            return None
        return end - start

    # Count key-downs for frequency calculations.
    counted_keys = [
        str(e["key"] if e["key"] is not None else "")
        for e in events
        if str(e["event_type"]).lower() == "down"
        and not is_modifier_or_navigation(e["key"])
    ]

    total_counted = len(counted_keys)
    delete_count = sum(k.lower() == "delete" for k in counted_keys)
    backspace_count = sum(k.lower() == "backspace" for k in counted_keys)
    printable_count = sum(is_printable_key(k) for k in counted_keys)

    # Use the elapsed span of the recorded typing key-down events.
    # If only one timestamp exists, fall back to the sample duration.
    if len(typing_downs) >= 2:
        event_duration_seconds = (
            max(x["down_ms"] for x in typing_downs)
            - min(x["down_ms"] for x in typing_downs)
        ) / 1000.0
    else:
        event_duration_seconds = float(sample["duration_seconds"] or 0)

    typing_speed = (
        printable_count / event_duration_seconds
        if event_duration_seconds > 0 else None
    )

    return {
        "sample_id": sample["id"],
        "candidate_id": sample["participant_id"],
        "session_id": sample["session_id"],
        "emotion_label": sample["label"],
        "sample_type": sample["sample_type"],
        "target_emotion": sample["target_emotion"],
        "self_reported_emotion": sample["self_reported_emotion"],

        # Timing features: milliseconds
        "D1U1_ms": difference(u1, d1),
        "D1U2_ms": difference(u2, d1),
        "D1D2_ms": difference(d2, d1),
        "U1D2_ms": difference(d2, u1),
        "U1U2_ms": difference(u2, u1),
        "D1U3_ms": difference(u3, d1),
        "D1D3_ms": difference(d3, d1),

        # Speed and relative frequencies
        "typing_speed_chars_per_second": typing_speed,
        "DelFreq": delete_count / total_counted if total_counted else None,
        "LeftFreq": backspace_count / total_counted if total_counted else None,

        # Useful quality-control columns
        "printable_character_count": printable_count,
        "delete_count": delete_count,
        "backspace_count": backspace_count,
        "counted_key_down_events": total_counted,
        "event_duration_seconds": event_duration_seconds,
    }


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Check that the database is in the project's instance folder."
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        samples = conn.execute("""
            SELECT
                id, participant_id, session_id, label, sample_type,
                target_emotion, self_reported_emotion, duration_seconds
            FROM typing_samples
            ORDER BY id
        """).fetchall()

        results = []

        for sample in samples:
            events = conn.execute("""
                SELECT key, event_type, timestamp_ms, sequence_index
                FROM keystroke_events
                WHERE typing_sample_id = ?
                ORDER BY sequence_index, id
            """, (sample["id"],)).fetchall()

            results.append(extract_one_sample(events, sample))

    df = pd.DataFrame(results)

    output_csv = OUTPUT_DIR / "keystroke_features.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\nFEATURE EXTRACTION COMPLETED")
    print(f"Database: {DB_PATH}")
    print(f"Typing samples processed: {len(df)}")
    print(f"Feature rows created: {len(df)}")
    print(f"Output CSV: {output_csv}")

    feature_columns = [
        "D1U1_ms", "D1U2_ms", "D1D2_ms", "U1D2_ms",
        "U1U2_ms", "D1U3_ms", "D1D3_ms",
        "typing_speed_chars_per_second", "DelFreq", "LeftFreq"
    ]

    print("\nExtracted feature preview:")
    print(df[["sample_id", "candidate_id", "emotion_label",
              "sample_type"] + feature_columns].head(10).to_string(index=False))

    print("\nMissing values in extracted features:")
    print(df[feature_columns].isna().sum().to_string())

    print("\nRows per emotion:")
    print(df.groupby("emotion_label").size().to_string())


if __name__ == "__main__":
    main()
