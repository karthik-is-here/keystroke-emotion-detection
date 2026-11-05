import sqlite3
import pandas as pd

from collections import defaultdict, deque


DB_PATH = "instance/data.db"

conn = sqlite3.connect(DB_PATH)

# --------------------------------------------------
# 1. LOAD TYPING SAMPLES
# --------------------------------------------------

samples = pd.read_sql_query("""
    SELECT
        id,
        participant_id,
        session_id,
        video_filename,
        sample_type,
        self_reported_emotion,
        target_emotion,
        text_content,
        duration_seconds
    FROM typing_samples
    WHERE sample_type IN (
        'video_response',
        'opposite_response'
    )
""", conn)


# --------------------------------------------------
# 2. LOAD KEYSTROKE EVENTS
# --------------------------------------------------

events = pd.read_sql_query("""
    SELECT
        typing_sample_id,
        key,
        event_type,
        timestamp_ms,
        sequence_index
    FROM keystroke_events
    ORDER BY typing_sample_id, sequence_index
""", conn)

conn.close()


# --------------------------------------------------
# 3. DEFINE NON-PRINTABLE KEYS
# --------------------------------------------------

NON_PRINTABLE = {
    "Shift", "Control", "Alt", "AltGraph",
    "Meta", "CapsLock", "Tab", "Escape",
    "Enter", "ArrowUp", "ArrowDown",
    "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown",
    "Insert", "Delete", "Backspace"
}


# --------------------------------------------------
# 4. FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(sample, event_data):

    event_data = event_data.sort_values("sequence_index")

    key_events = event_data.to_dict("records")

    pending_down = defaultdict(deque)

    hold_times = []
    hold_records = []
    key_intervals = []

    backspaces = 0
    total_keydowns = 0

    unmatched_keyups = 0

    paired_events = []

    for event in key_events:

        key = event["key"]
        event_type = event["event_type"]
        timestamp = event["timestamp_ms"]

        if event_type == "down":

            total_keydowns += 1

            if key in ("Backspace", "Delete"):
                backspaces += 1

            pending_down[key].append(timestamp)

        elif event_type == "up":

            if pending_down[key]:

                down_time = pending_down[key].popleft()

                hold = timestamp - down_time

                if hold >= 0:

                    hold_records.append({
                        "sample_id": sample["id"],
                        "participant_id": sample["participant_id"],
                        "sample_type": sample["sample_type"],
                        "key": key,
                        "down_ms": down_time,
                        "up_ms": timestamp,
                        "hold_ms": hold
                    })

                    if key not in NON_PRINTABLE:
                        hold_times.append(hold)

                    if len(key) == 1 and key.isprintable():

                        paired_events.append({
                            "key": key,
                            "down": down_time,
                            "up": timestamp
                        })

            else:
                unmatched_keyups += 1

    # Count unmatched keydowns
    unmatched_keydowns = sum(
        len(queue) for queue in pending_down.values()
    )

    # Sort printable events
    paired_events.sort(key=lambda x: x["down"])

    # --------------------------------------------------
    # FLIGHT TIME
    # --------------------------------------------------

    for i in range(1, len(paired_events)):

        flight = (
            paired_events[i]["down"] -
            paired_events[i - 1]["up"]
        )

        key_intervals.append(flight)

    # --------------------------------------------------
    # HOLD TIME FEATURES
    # --------------------------------------------------

    mean_hold = (
        sum(hold_times) / len(hold_times)
        if hold_times else None
    )

    std_hold = (
        pd.Series(hold_times).std()
        if len(hold_times) > 1 else None
    )

    # --------------------------------------------------
    # FLIGHT TIME FEATURES
    # --------------------------------------------------

    mean_flight = (
        sum(key_intervals) / len(key_intervals)
        if key_intervals else None
    )

    std_flight = (
        pd.Series(key_intervals).std()
        if len(key_intervals) > 1 else None
    )

    # --------------------------------------------------
    # ACTIVE TYPING DURATION
    # --------------------------------------------------

    if paired_events:

        first_keydown = min(
            e["down"] for e in paired_events
        )

        last_keyup = max(
            e["up"] for e in paired_events
        )

        active_duration = (
            last_keyup - first_keydown
        ) / 1000

    else:
        active_duration = 0

    # --------------------------------------------------
    # TYPING SPEED
    # --------------------------------------------------

    character_count = len(
        sample["text_content"] or ""
    )

    task_duration = sample["duration_seconds"]

    typing_speed = (
        character_count / task_duration
        if pd.notna(task_duration) and task_duration > 0
        else None
    )

    active_typing_speed = (
        character_count / active_duration
        if active_duration > 0
        else None
    )

    # --------------------------------------------------
    # PAUSE RATE
    # --------------------------------------------------

    pause_rate = (
        sum(1 for x in key_intervals if x > 1000)
        / len(key_intervals)
        if key_intervals else None
    )

    # --------------------------------------------------
    # BACKSPACE RATE
    # --------------------------------------------------

    backspace_rate = (
        backspaces / total_keydowns
        if total_keydowns else None
    )

    # --------------------------------------------------
    # RETURN FEATURES AND DIAGNOSTICS
    # --------------------------------------------------

    return {

        "mean_hold_ms": mean_hold,
        "std_hold_ms": std_hold,

        "mean_flight_ms": mean_flight,
        "std_flight_ms": std_flight,

        "typing_speed_cps": typing_speed,
        "active_typing_speed_cps": active_typing_speed,

        "pause_rate": pause_rate,
        "backspace_rate": backspace_rate,

        "total_time_s": task_duration,
        "active_typing_time_s": active_duration,

        "valid_hold_count": len(hold_times),
        "valid_flight_count": len(key_intervals),

        "total_keydowns": total_keydowns,

        "unmatched_keyups": unmatched_keyups,
        "unmatched_keydowns": unmatched_keydowns,

        "max_hold_ms": max(hold_times) if hold_times else None,

        "holds_over_1000ms": sum(
            1 for x in hold_times if x > 1000
        ),

        "holds_over_5000ms": sum(
            1 for x in hold_times if x > 5000
        ),

        "printable_event_count": len(paired_events),

        "text_character_count": character_count,

        "_hold_records": hold_records
    }


# --------------------------------------------------
# 5. EXTRACT FEATURES
# --------------------------------------------------

print("Extracting typing features...")

feature_rows = []

all_hold_records = []

for _, sample in samples.iterrows():

    sample_events = events[
        events["typing_sample_id"] == sample["id"]
    ]

    features = extract_features(
        sample,
        sample_events
    )

    all_hold_records.extend(
        features.pop("_hold_records")
    )

    row = sample.to_dict()

    row.update(features)

    feature_rows.append(row)

    print(
        f"Processed sample {sample['id']} "
        f"({len(feature_rows)}/{len(samples)})"
    )


features_df = pd.DataFrame(feature_rows)


# --------------------------------------------------
# 6. SAVE FEATURE DATASET
# --------------------------------------------------

features_df.to_csv(
    "typing_features.csv",
    index=False
)

print("\nFeature extraction completed!")

print("\nDataset shape:", features_df.shape)

feature_columns = [
    "mean_hold_ms",
    "std_hold_ms",
    "mean_flight_ms",
    "std_flight_ms",
    "typing_speed_cps",
    "active_typing_speed_cps",
    "pause_rate",
    "backspace_rate",
    "total_time_s",
    "active_typing_time_s"
]

print("\nFeature summary:")

print(
    features_df[feature_columns]
    .describe()
    .round(3)
)

print("\nSaved: typing_features.csv")


# --------------------------------------------------
# 7. SAVE LONG HOLD DIAGNOSTICS
# --------------------------------------------------

hold_df = pd.DataFrame(all_hold_records)

if not hold_df.empty:

    hold_df.to_csv(
        "keystroke_hold_diagnostics.csv",
        index=False
    )

    long_holds = hold_df[
        hold_df["hold_ms"] > 1000
    ].sort_values(
        "hold_ms",
        ascending=False
    )

    print("\n--- LONG KEY HOLDS ---")

    print("Holds exceeding 1 second:", len(long_holds))

    print("\nLongest 20 key holds:")

    print(
        long_holds.head(20).to_string(index=False)
    )

    print("\nSaved: keystroke_hold_diagnostics.csv")


# --------------------------------------------------
# 8. PARTICIPANT 15 DIAGNOSTICS
# --------------------------------------------------

print("\n--- PARTICIPANT 15 DIAGNOSTICS ---")

participant_15 = features_df[
    features_df["participant_id"] == 15
]

diagnostic_columns = [
    "id",
    "sample_type",
    "text_character_count",
    "total_time_s",
    "active_typing_time_s",
    "typing_speed_cps",
    "active_typing_speed_cps",
    "valid_hold_count",
    "printable_event_count",
    "total_keydowns",
    "unmatched_keyups",
    "unmatched_keydowns"
]

print(
    participant_15[diagnostic_columns].to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. GENERAL EVENT DIAGNOSTICS
# --------------------------------------------------

print("\n--- EVENT DIAGNOSTICS ---")

print(
    features_df[
        [
            "unmatched_keyups",
            "unmatched_keydowns",
            "valid_hold_count",
            "printable_event_count",
            "max_hold_ms"
        ]
    ].describe().round(3)
)


# --------------------------------------------------
# 10. SAVE SAMPLE DIAGNOSTICS
# --------------------------------------------------

diagnostic_df = features_df[
    [
        "id",
        "participant_id",
        "sample_type",
        "text_character_count",
        "total_time_s",
        "active_typing_time_s",
        "typing_speed_cps",
        "active_typing_speed_cps",
        "valid_hold_count",
        "printable_event_count",
        "unmatched_keyups",
        "unmatched_keydowns",
        "max_hold_ms",
        "holds_over_1000ms",
        "holds_over_5000ms"
    ]
]

diagnostic_df.to_csv(
    "sample_event_diagnostics.csv",
    index=False
)

print("\nSaved: sample_event_diagnostics.csv")

print("\nALL EXTRACTION AND DIAGNOSTIC TASKS COMPLETED.")
