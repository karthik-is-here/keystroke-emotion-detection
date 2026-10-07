"""
Data-collection web app for the keystroke-dynamics + text emotion project.

This app ONLY collects and stores data (typed text, keystroke timing, and
the participant's self-reported emotion). It does not run any emotion
prediction itself — that happens later, offline, once enough data has
been collected.

Flow per participant:
  consent -> demographics -> baseline (fixed text) -> baseline (question)
  -> for each of 4 emotions, in random order:
       video intro -> video (1 of 3 for that emotion, random) ->
       self-report current emotion -> write about the video ->
       write about a contrasting emotion -> comprehension check ->
       10s break (skipped after the last round)
  -> thank you

Run with:
    python app.py
Then open http://127.0.0.1:5000/ in a browser.
"""

import json
import os
import random
from datetime import datetime

from flask import Flask, redirect, render_template, request, session, url_for

from config import (
    BASELINE_FIXED_TEXT,
    BASELINE_MIN_WORD_COUNT,
    BASELINE_QUESTIONS,
    Config,
    EMOTIONS,
    MASTER_SKIP_KEY,
    OPPOSITE_EMOTION_MAP,
    OPPOSITE_MAX_WORD_COUNT,
    OPPOSITE_MIN_WORD_COUNT,
    ROUND_GAP_SECONDS,
    SELF_REPORT_LABELS,
    VIDEO_LIBRARY,
)
from models import ComprehensionCheck, EmotionSession, KeystrokeEvent, Participant, TypingSample, db

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)
db.init_app(app)


def _add_missing_columns():
    """If a column exists on a model but not yet in the actual SQLite
    table (e.g. after models.py gains a new field), add it in place
    instead of requiring the whole database to be deleted and recreated.
    This only handles adding columns — removing/renaming a column, or
    changing its type, still needs a manual fix."""
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    existing_tables = set(inspector.get_table_names())

    for table in db.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # brand-new table; db.create_all() already made it
        existing_columns = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            col_type = column.type.compile(db.engine.dialect)
            with db.engine.begin() as conn:
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}')
                )
            print(f"[schema] added missing column: {table.name}.{column.name}")


with app.app_context():
    db.create_all()
    _add_missing_columns()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_wpm(word_count, duration_seconds):
    if not duration_seconds or duration_seconds <= 0:
        return None
    minutes = duration_seconds / 60.0
    return round(word_count / minutes, 2)


def _compute_error_rate(events):
    keydowns = [e for e in events if e.get("type") == "down"]
    if not keydowns:
        return None
    backspaces = sum(1 for e in keydowns if e.get("key") in ("Backspace", "Delete"))
    return round(backspaces / len(keydowns), 4)


def _save_typing_sample(label, sample_type, participant_id, session_id, text_content,
                         events, duration_ms, self_reported_emotion=None,
                         prompt_text=None, target_emotion=None, video_filename=None):
    word_count = len(text_content.split())
    duration_seconds = (duration_ms or 0) / 1000.0

    sample = TypingSample(
        participant_id=participant_id,
        session_id=session_id,
        label=label,
        sample_type=sample_type,
        self_reported_emotion=self_reported_emotion,
        prompt_text=prompt_text,
        target_emotion=target_emotion,
        video_filename=video_filename,
        text_content=text_content,
        word_count=word_count,
        duration_seconds=duration_seconds,
        wpm=_compute_wpm(word_count, duration_seconds),
        error_rate=_compute_error_rate(events),
        ended_at=datetime.utcnow(),
    )
    db.session.add(sample)
    db.session.flush()  # get sample.id before committing

    for idx, event in enumerate(events):
        db.session.add(
            KeystrokeEvent(
                typing_sample_id=sample.id,
                key=str(event.get("key", ""))[:50],
                event_type=event.get("type", ""),
                timestamp_ms=float(event.get("t", 0)),
                sequence_index=idx,
            )
        )
    db.session.commit()
    return sample


def _pick_balanced_target(chosen):
    """Pick whichever opposite emotion has been used least so far for this
    self-reported emotion, so that e.g. angry and sad each split evenly
    between happy and calm. Ties are broken randomly."""
    candidates = OPPOSITE_EMOTION_MAP.get(chosen, EMOTIONS)
    if len(candidates) == 1:
        return candidates[0]

    counts = dict.fromkeys(candidates, 0)
    rows = (
        db.session.query(TypingSample.target_emotion, db.func.count(TypingSample.id))
        .filter(
            TypingSample.sample_type == "opposite_response",
            TypingSample.self_reported_emotion == chosen,
            TypingSample.target_emotion.in_(candidates),
        )
        .group_by(TypingSample.target_emotion)
        .all()
    )
    for target, n in rows:
        counts[target] = n

    fewest = min(counts.values())
    return random.choice([c for c, n in counts.items() if n == fewest])


def _advance_to_next_emotion():
    """Pop the next emotion off the queue into session['current_emotion'],
    and randomly pick one of its 3 videos. Returns True if there was one,
    False if the queue is empty."""
    queue = session.get("emotion_queue", [])
    if queue:
        emotion = queue.pop(0)
        session["emotion_queue"] = queue
        session["current_emotion"] = emotion
        session["current_video_index"] = random.randrange(len(VIDEO_LIBRARY[emotion]))
        return True
    session.pop("current_emotion", None)
    session.pop("current_video_index", None)
    return False


def _current_video_config():
    emotion = session["current_emotion"]
    index = session.get("current_video_index", 0)
    return VIDEO_LIBRARY[emotion][index]


def _require_participant():
    return "participant_id" in session


def _require_current_emotion():
    return "current_emotion" in session


def _round_number():
    """1-indexed position of the current emotion within this participant's
    randomized order of the 4 video-induced emotions, or None."""
    if "current_emotion" not in session or "session_id" not in session:
        return None
    emotion_session = EmotionSession.query.get(session["session_id"])
    if not emotion_session:
        return None
    order = emotion_session.order_list()
    try:
        return order.index(session["current_emotion"]) + 1
    except ValueError:
        return None


def _step_label():
    """Coarse 'Step X of 7' label: 1=demographics, 2=baseline (fixed),
    3=baseline (question), 4-7=the four emotion rounds."""
    n = _round_number()
    if n is None:
        return ""
    return f"Step {3 + n} of 7"


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("consent.html")


@app.route("/consent", methods=["POST"])
def consent():
    session.clear()
    session["consented"] = True
    return redirect(url_for("demographics"))


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

@app.route("/demographics", methods=["GET", "POST"])
def demographics():
    if not session.get("consented"):
        return redirect(url_for("index"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        nationality = request.form.get("nationality", "").strip()
        first_language = request.form.get("first_language", "").strip()
        typing_experience = request.form.get("typing_experience", "").strip()
        dominant_hand = request.form.get("dominant_hand", "").strip()

        if not name:
            return render_template(
                "demographics.html", error="Please enter your name.",
                step_label="Step 1 of 7",
            )

        participant = Participant(
            name=name,
            age=int(age) if age.isdigit() else None,
            nationality=nationality or None,
            first_language=first_language or None,
            typing_experience=typing_experience or None,
            dominant_hand=dominant_hand or None,
        )
        db.session.add(participant)
        db.session.flush()

        order = EMOTIONS.copy()
        random.shuffle(order)
        emotion_session = EmotionSession(
            participant_id=participant.id, emotion_order=",".join(order)
        )
        db.session.add(emotion_session)
        db.session.commit()

        session["participant_id"] = participant.id
        session["session_id"] = emotion_session.id
        session["emotion_queue"] = order

        return redirect(url_for("baseline_fixed"))

    return render_template("demographics.html", error=None, step_label="Step 1 of 7")


# ---------------------------------------------------------------------------
# Baseline task 1: transcribe a short fixed passage
# ---------------------------------------------------------------------------

@app.route("/baseline", methods=["GET", "POST"])
def baseline_fixed():
    if not _require_participant():
        return redirect(url_for("index"))

    if request.method == "POST":
        text_content = request.form.get("text_content", "")
        duration_ms = request.form.get("duration_ms", "0")
        keystroke_json = request.form.get("keystroke_data", "[]")
        try:
            events = json.loads(keystroke_json)
        except (ValueError, TypeError):
            events = []

        _save_typing_sample(
            label="neutral",
            sample_type="baseline_fixed",
            participant_id=session["participant_id"],
            session_id=session["session_id"],
            text_content=text_content,
            events=events,
            duration_ms=float(duration_ms) if duration_ms else 0,
            prompt_text=BASELINE_FIXED_TEXT,
        )
        return redirect(url_for("baseline_question"))

    return render_template(
        "baseline_fixed.html", baseline_text=BASELINE_FIXED_TEXT,
        step_label="Step 2 of 7",
    )


# ---------------------------------------------------------------------------
# Baseline task 2: a random neutral question, answered freely
# ---------------------------------------------------------------------------

@app.route("/baseline_question", methods=["GET", "POST"])
def baseline_question():
    if not _require_participant():
        return redirect(url_for("index"))

    question = session.get("baseline_question")
    if not question:
        question = random.choice(BASELINE_QUESTIONS)
        session["baseline_question"] = question

    if request.method == "POST":
        text_content = request.form.get("text_content", "")
        duration_ms = request.form.get("duration_ms", "0")
        keystroke_json = request.form.get("keystroke_data", "[]")
        try:
            events = json.loads(keystroke_json)
        except (ValueError, TypeError):
            events = []

        word_count = len(text_content.split())
        if word_count < BASELINE_MIN_WORD_COUNT:
            return render_template(
                "baseline_question.html",
                question=question,
                min_words=BASELINE_MIN_WORD_COUNT,
                error=f"Please write at least {BASELINE_MIN_WORD_COUNT} words "
                      f"(currently {word_count}).",
                step_label="Step 3 of 7",
            )

        _save_typing_sample(
            label="neutral",
            sample_type="baseline_question",
            participant_id=session["participant_id"],
            session_id=session["session_id"],
            text_content=text_content,
            events=events,
            duration_ms=float(duration_ms) if duration_ms else 0,
            prompt_text=question,
        )
        session.pop("baseline_question", None)

        if _advance_to_next_emotion():
            return redirect(url_for("video_intro"))
        return redirect(url_for("thank_you"))

    return render_template(
        "baseline_question.html", question=question, min_words=BASELINE_MIN_WORD_COUNT,
        error=None, step_label="Step 3 of 7",
    )


# ---------------------------------------------------------------------------
# Per-emotion round
# ---------------------------------------------------------------------------

@app.route("/video_intro", methods=["GET"])
def video_intro():
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))
    emotion = session["current_emotion"]
    config = _current_video_config()
    return render_template(
        "video_intro.html", emotion=emotion, config=config, step_label=_step_label()
    )


@app.route("/video", methods=["GET"])
def video():
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))
    emotion = session["current_emotion"]
    config = _current_video_config()
    return render_template(
        "video.html", emotion=emotion, config=config, step_label=_step_label(),
        skip_error=None,
    )


@app.route("/video/skip", methods=["POST"])
def video_skip():
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))

    if request.form.get("master_key", "") == MASTER_SKIP_KEY:
        return redirect(url_for("self_report"))

    emotion = session["current_emotion"]
    config = _current_video_config()
    return render_template(
        "video.html", emotion=emotion, config=config, step_label=_step_label(),
        skip_error="Incorrect key.",
    )


@app.route("/self_report", methods=["GET", "POST"])
def self_report():
    """How the video actually made them feel — asked right after
    watching, before either writing task."""
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))

    if request.method == "POST":
        chosen = request.form.get("emotion")
        if chosen not in SELF_REPORT_LABELS:
            return render_template(
                "self_report.html", labels=SELF_REPORT_LABELS,
                error="Please choose one option.", step_label=_step_label(),
            )
        session["current_self_report"] = chosen
        session["current_target_emotion"] = _pick_balanced_target(chosen)
        return redirect(url_for("typing"))

    return render_template(
        "self_report.html", labels=SELF_REPORT_LABELS, error=None,
        step_label=_step_label(),
    )


@app.route("/typing", methods=["GET", "POST"])
def typing():
    """Write about the video just watched."""
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))
    if "current_self_report" not in session:
        return redirect(url_for("self_report"))
    emotion = session["current_emotion"]

    if request.method == "POST":
        text_content = request.form.get("text_content", "")
        duration_ms = request.form.get("duration_ms", "0")
        keystroke_json = request.form.get("keystroke_data", "[]")
        try:
            events = json.loads(keystroke_json)
        except (ValueError, TypeError):
            events = []

        word_count = len(text_content.split())
        if word_count < app.config["MIN_WORD_COUNT"]:
            return render_template(
                "typing.html",
                emotion=emotion,
                min_words=app.config["MIN_WORD_COUNT"],
                error=f"Please write at least {app.config['MIN_WORD_COUNT']} words "
                      f"(currently {word_count}).",
                step_label=_step_label(),
            )

        sample = _save_typing_sample(
            label=emotion,
            sample_type="video_response",
            participant_id=session["participant_id"],
            session_id=session["session_id"],
            text_content=text_content,
            events=events,
            duration_ms=float(duration_ms) if duration_ms else 0,
            self_reported_emotion=session["current_self_report"],
            video_filename=_current_video_config()["filename"],
        )
        session["current_video_sample_id"] = sample.id
        return redirect(url_for("opposite_typing"))

    return render_template(
        "typing.html", emotion=emotion, min_words=app.config["MIN_WORD_COUNT"], error=None,
        step_label=_step_label(),
    )


@app.route("/opposite_typing", methods=["GET", "POST"])
def opposite_typing():
    """Write about a contrasting emotion to the one they just reported."""
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))
    if "current_target_emotion" not in session:
        return redirect(url_for("self_report"))
    emotion = session["current_emotion"]
    target_emotion = session["current_target_emotion"]

    if request.method == "POST":
        text_content = request.form.get("text_content", "")
        duration_ms = request.form.get("duration_ms", "0")
        keystroke_json = request.form.get("keystroke_data", "[]")
        try:
            events = json.loads(keystroke_json)
        except (ValueError, TypeError):
            events = []

        word_count = len(text_content.split())
        if word_count < OPPOSITE_MIN_WORD_COUNT or word_count > OPPOSITE_MAX_WORD_COUNT:
            return render_template(
                "opposite_typing.html",
                target_emotion=target_emotion,
                min_words=OPPOSITE_MIN_WORD_COUNT,
                max_words=OPPOSITE_MAX_WORD_COUNT,
                error=f"Please write between {OPPOSITE_MIN_WORD_COUNT} and "
                      f"{OPPOSITE_MAX_WORD_COUNT} words (currently {word_count}).",
                step_label=_step_label(),
            )

        _save_typing_sample(
            label=emotion,
            sample_type="opposite_response",
            participant_id=session["participant_id"],
            session_id=session["session_id"],
            text_content=text_content,
            events=events,
            duration_ms=float(duration_ms) if duration_ms else 0,
            self_reported_emotion=session["current_self_report"],
            target_emotion=target_emotion,
            video_filename=_current_video_config()["filename"],
        )
        session.pop("current_target_emotion", None)
        return redirect(url_for("comprehension"))

    return render_template(
        "opposite_typing.html", target_emotion=target_emotion,
        min_words=OPPOSITE_MIN_WORD_COUNT, max_words=OPPOSITE_MAX_WORD_COUNT,
        error=None, step_label=_step_label(),
    )


@app.route("/comprehension", methods=["GET", "POST"])
def comprehension():
    if not _require_participant() or not _require_current_emotion():
        return redirect(url_for("index"))
    config = _current_video_config()

    if request.method == "POST":
        selected = request.form.get("answer", "")
        check = ComprehensionCheck(
            typing_sample_id=session["current_video_sample_id"],
            question=config["comprehension_question"],
            selected_answer=selected,
            correct_answer=config["correct_answer"],
            is_correct=(selected == config["correct_answer"]),
        )
        db.session.add(check)
        db.session.commit()

        session.pop("current_video_sample_id", None)
        session.pop("current_self_report", None)

        if _advance_to_next_emotion():
            session["gap_next"] = "video_intro"
            return redirect(url_for("round_gap"))
        return redirect(url_for("thank_you"))

    return render_template(
        "comprehension.html", config=config, step_label=_step_label()
    )


@app.route("/round_gap", methods=["GET", "POST"])
def round_gap():
    """A short mandatory break between rounds. gap_next is intentionally
    NOT cleared here — it's overwritten fresh by comprehension() before
    every visit, and leaving it in place means a page refresh during the
    break still lands on the correct next step."""
    if not _require_participant():
        return redirect(url_for("index"))
    next_endpoint = session.get("gap_next", "thank_you")

    if request.method == "POST":
        if request.form.get("master_key", "") == MASTER_SKIP_KEY:
            return redirect(url_for(next_endpoint))
        return render_template(
            "round_gap.html", seconds=ROUND_GAP_SECONDS,
            next_url=url_for(next_endpoint), skip_error="Incorrect key.",
        )

    return render_template(
        "round_gap.html", seconds=ROUND_GAP_SECONDS,
        next_url=url_for(next_endpoint), skip_error=None,
    )


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

@app.route("/thank_you", methods=["GET"])
def thank_you():
    session.clear()
    return render_template("thank_you.html")


if __name__ == "__main__":
    app.run(debug=True)