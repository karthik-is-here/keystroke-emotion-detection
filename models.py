"""
Database models.

Per participant: one EmotionSession, plus TypingSamples of several kinds
(see `sample_type` below) — two baseline samples, then for each of the 4
emotion rounds, a 'video_response' sample and an 'opposite_response'
sample. Each TypingSample has many KeystrokeEvents. Each round's
'video_response' sample also gets one ComprehensionCheck.
"""

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Participant(db.Model):
    __tablename__ = "participants"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    nationality = db.Column(db.String(100), nullable=True)
    first_language = db.Column(db.String(100), nullable=True)
    typing_experience = db.Column(db.String(50), nullable=True)
    dominant_hand = db.Column(db.String(20), nullable=True)
    consented_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sessions = db.relationship("EmotionSession", backref="participant", lazy=True)
    typing_samples = db.relationship("TypingSample", backref="participant", lazy=True)


class EmotionSession(db.Model):
    __tablename__ = "emotion_sessions"

    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(
        db.Integer, db.ForeignKey("participants.id"), nullable=False
    )
    # Randomized order of the four video-induced emotions for this
    # participant, stored as a comma-separated string, e.g. "sad,calm,angry,happy"
    emotion_order = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    typing_samples = db.relationship("TypingSample", backref="session", lazy=True)

    def order_list(self):
        return self.emotion_order.split(",")


class TypingSample(db.Model):
    __tablename__ = "typing_samples"

    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(
        db.Integer, db.ForeignKey("participants.id"), nullable=False
    )
    session_id = db.Column(
        db.Integer, db.ForeignKey("emotion_sessions.id"), nullable=False
    )

    # The task label: 'neutral' for both baseline samples, or one of the
    # four video-induced emotions for a round's two samples.
    label = db.Column(db.String(20), nullable=False)

    # What kind of writing task this was:
    #   'baseline_fixed'    - transcribed the fixed passage
    #   'baseline_question' - answered a random neutral question freely
    #   'video_response'    - wrote about the video just watched
    #   'opposite_response' - wrote about a contrasting emotion
    sample_type = db.Column(db.String(20), nullable=False)

    # For 'opposite_response' samples, which emotion they were actually
    # asked to write about (may differ from `label`, which stays the
    # round's induced emotion). Null for every other sample type.
    target_emotion = db.Column(db.String(20), nullable=True)

    # Which of the 3 videos for this emotion was shown this round. Null
    # for baseline samples.
    video_filename = db.Column(db.String(100), nullable=True)

    # What the participant actually reported feeling right after the
    # video (before either writing task). Null for baseline samples,
    # which are neutral by design and skip self-report.
    self_reported_emotion = db.Column(db.String(20), nullable=True)

    # The question/passage shown for this task. Populated for both
    # baseline samples; null for video-round samples (their prompt is
    # implicit in sample_type + target_emotion).
    prompt_text = db.Column(db.Text, nullable=True)

    text_content = db.Column(db.Text, nullable=False)
    word_count = db.Column(db.Integer, nullable=False)
    duration_seconds = db.Column(db.Float, nullable=False)
    wpm = db.Column(db.Float, nullable=True)
    error_rate = db.Column(db.Float, nullable=True)

    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime, nullable=True)

    keystroke_events = db.relationship(
        "KeystrokeEvent", backref="typing_sample", lazy=True, cascade="all, delete-orphan"
    )
    comprehension_check = db.relationship(
        "ComprehensionCheck",
        backref="typing_sample",
        uselist=False,
        cascade="all, delete-orphan",
    )


class KeystrokeEvent(db.Model):
    __tablename__ = "keystroke_events"

    id = db.Column(db.Integer, primary_key=True)
    typing_sample_id = db.Column(
        db.Integer, db.ForeignKey("typing_samples.id"), nullable=False
    )
    key = db.Column(db.String(50), nullable=False)
    event_type = db.Column(db.String(4), nullable=False)  # 'down' or 'up'
    timestamp_ms = db.Column(db.Float, nullable=False)  # relative to sample start
    sequence_index = db.Column(db.Integer, nullable=False)


class ComprehensionCheck(db.Model):
    __tablename__ = "comprehension_checks"

    id = db.Column(db.Integer, primary_key=True)
    typing_sample_id = db.Column(
        db.Integer, db.ForeignKey("typing_samples.id"), nullable=False
    )
    question = db.Column(db.Text, nullable=False)
    selected_answer = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.Text, nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
