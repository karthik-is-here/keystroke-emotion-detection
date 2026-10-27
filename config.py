"""
Configuration for the keystroke + text emotion data-collection app.

Nothing sensitive is hard-coded here beyond a placeholder SECRET_KEY.
Before running a real session, set FLASK_SECRET_KEY as an environment
variable instead of relying on the default below.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Used to sign the session cookie that tracks a participant's progress
    # through the experiment.
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

    # SQLite file lives in the instance/ folder, which Flask keeps outside
    # of version control by default.
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(
        BASE_DIR, "instance", "data.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Minimum words required when writing about the video just watched.
    MIN_WORD_COUNT = 50


# The four video-induced emotions.
EMOTIONS = ["happy", "sad", "angry", "calm"]

# All labels a participant can choose from on the self-report screen.
SELF_REPORT_LABELS = ["happy", "sad", "angry", "calm"]

# --- Baseline task 1: fixed passage (transcribed verbatim) -----------------
# Kept short (max two sentences) on purpose — this is just to get a quick,
# consistent-across-participants typing sample.
BASELINE_FIXED_TEXT = (
    "Many people use computers and phones every day for work and "
    "communication. Learning to type efficiently can save a lot of time."
)

# --- Baseline task 2: a random neutral question, answered freely -----------
BASELINE_QUESTIONS = [
    "What did you have for breakfast today?",
    "Describe your commute or walk to this session.",
    "What's a TV show or movie you've watched recently?",
    "What's your favorite way to spend a weekend?",
    "Describe today's weather in a couple of sentences.",
    "What did you do last weekend?",
    "Describe your favorite meal to cook or order.",
    "What's a hobby you enjoy in your free time?",
]
# Light minimum so the sample has enough keystrokes to be useful, without
# turning it into a real writing task. No hard maximum is enforced — the
# on-screen instruction just asks for "about two sentences."
BASELINE_MIN_WORD_COUNT = 15
OPPOSITE_EMOTION_MAP = {
    "angry": ["happy", "calm"],
    "sad": ["happy", "calm"],
    "happy": ["sad"],
    "calm": ["angry"],
    
}
OPPOSITE_MIN_WORD_COUNT = 40
OPPOSITE_MAX_WORD_COUNT = 60

# Typing this into a "skip" box (on the video page, or during the
# between-round break) jumps straight ahead. Meant for testing/development
# only — not shown or hinted at to real participants.
MASTER_SKIP_KEY = "1234"

# Seconds of mandatory break shown between the end of one emotion round
# (after the opposite-emotion writing task) and the start of the next.
ROUND_GAP_SECONDS = 10

# --- Video library -----------------------------------------------------
# Three candidate videos per emotion. One is picked at random per
# participant per emotion, so different participants see different videos
# within the same emotion category (variety across the dataset), while any
# single participant only ever sees one video per emotion.
#
# Put the 12 video files in static/videos/ using these exact filenames.
def _video(filename):
    return {
        "filename": filename,
        "intro_text": "You're about to watch a short video.",
    }


VIDEO_LIBRARY = {
    "happy": [_video("happy_1.mp4"), _video("happy_2.mp4"), _video("happy_3.mp4")],
    "sad": [_video("sad_1.mp4"), _video("sad_2.mp4"), _video("sad_3.mp4")],
    "angry": [_video("angry_1.mp4"), _video("angry_2.mp4"), _video("angry_3.mp4")],
    "calm": [_video("calm_1.mp4"), _video("calm_2.mp4"), _video("calm_3.mp4")],
}
