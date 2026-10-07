# Keystroke + Text Emotion Data Collection

A Flask web app for collecting labeled data for the keystroke-dynamics +
text emotion detection project. It only **collects** data — it does not
run any emotion prediction. That happens later, offline, once enough
data exists.

## What it does

For each participant:

1. Consent screen, then a demographic form: name, age, nationality,
   first language, typing skill, dominant hand.
2. **Baseline, part 1** — transcribe a short (two-sentence) fixed
   passage, the same for every participant.
3. **Baseline, part 2** — answer a random neutral question freely, in
   about two sentences (e.g. "What did you have for breakfast today?").
   Both baseline samples are labeled `neutral`.
4. Four rounds, one per emotion (`happy`, `sad`, `angry`, `calm`), shown
   in a random order per participant. Each emotion has a pool of 3
   candidate videos — one is picked at random per participant, so
   different participants may see different videos within the same
   emotion category, but any single participant only ever sees one video
   per emotion. Each round:
   - Info screen, then the video plays automatically with no visible
     controls.
   - **Immediately after the video**: self-report — "how did that make
     you feel?" — from all five labels (`happy`/`sad`/`angry`/`calm`/
     `neutral`). This, not the video's intended emotion, is the
     ground-truth label.
   - Write at least 80 words about the video itself.
   - Write **40–80 words** about a *contrasting* emotion — e.g. if they
     just reported feeling angry, they're asked to write about something
     that makes them happy or calm instead. (The exact mapping is in
     `OPPOSITE_EMOTION_MAP` in `config.py`.)
   - A short comprehension question, to confirm they watched/attended to
     the video.
   - A mandatory 10-second break before the next round (skipped after
     the last one).
5. Thank-you screen.

Every keystroke (key, down/up, millisecond-precision timestamp) is
recorded for every typing task, along with the full typed text.

## Setup

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Adding the induction videos

This app expects **12 video files** in `static/videos/` — 3 per emotion:

```
static/videos/happy_1.mp4   static/videos/happy_2.mp4   static/videos/happy_3.mp4
static/videos/sad_1.mp4     static/videos/sad_2.mp4     static/videos/sad_3.mp4
static/videos/angry_1.mp4   static/videos/angry_2.mp4   static/videos/angry_3.mp4
static/videos/calm_1.mp4    static/videos/calm_2.mp4    static/videos/calm_3.mp4
```

**These files are not included** — you need to source and add them
yourself. A few notes:

- Keep each to roughly 1–2 minutes.
- `.mp4` (H.264) plays reliably in all major browsers.
- Avoid anything genuinely disturbing or graphic, especially for the
  "angry" videos — searching for "angry induction video" tends to surface
  real violence/abuse footage, which is not appropriate here. Safer
  options that still reliably induce frustration: sports-injustice clips
  (a bad referee call), or a consumer-scam call-out video.
- For each of the 12 videos, update its comprehension question, answer
  options, and correct answer in `VIDEO_LIBRARY` in `config.py` — every
  one is currently a "PLACEHOLDER" that won't make sense against real
  content.
- For the eventual full study (100+ participants, aiming for a
  publishable dataset), consider using clips from an academically
  validated emotion-elicitation film set instead of ad-hoc video finds,
  for stronger methodological rigor.

## Running

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser. For in-lab data
collection, this also works fine on a machine with no internet access,
since everything (fonts, video, JS) is served locally — nothing is
loaded from a CDN.

To make it reachable from other machines on the same network, run:

```bash
flask --app app run --host=0.0.0.0
```

and have participants visit `http://<your-machine-ip>:5000/`.

### Testing shortcuts (staff only)

Two places have a small collapsed "Staff: skip..." control, for
development/testing — not meant for real participants to find or use:

- On the video page: typing `1234` (the default `MASTER_SKIP_KEY` in
  `config.py`) skips playing the video and jumps to self-report.
- On the between-round break page: the same key skips the 10-second
  wait immediately.

Change `MASTER_SKIP_KEY` before running real sessions.

## Where the data goes

Everything is stored in a local SQLite database at `instance/data.db`,
created automatically the first time you run the app. If you edit
`models.py` to add a new column, you don't need to delete the database —
the app checks for and adds missing columns automatically on startup.

To export everything to CSV files for analysis:

```bash
python export_data.py
```

This writes `participants.csv`, `emotion_sessions.csv`,
`typing_samples.csv`, `keystroke_events.csv`, and
`comprehension_checks.csv` into an `exports/` folder.

`typing_samples.csv` now has 10 rows per participant (2 baseline + 4
rounds × 2 samples each). The `sample_type` column tells you which kind
each row is (`baseline_fixed`, `baseline_question`, `video_response`,
`opposite_response`); `label` is the round's induced emotion;
`self_reported_emotion` is what the participant actually said they felt;
`target_emotion` is only set for `opposite_response` rows, and tells you
which contrasting emotion they were asked to write about.

Deriving per-letter dwell/flight-time features (as used in the
keystroke-dynamics literature) from `keystroke_events.csv` is a matter
of pairing up each key's "down" and "up" rows (dwell time) and looking
at the gap between one key's "up" and the next key's "down" (flight
time), grouped by `typing_sample_id`.

## Things to double check before running with real participants

- [ ] Replace the placeholder consent text in `templates/consent.html`
      with your institution's actual consent/ethics language.
- [ ] Add all 12 real video files and update `VIDEO_LIBRARY` in
      `config.py` to match them (comprehension questions especially).
- [ ] Set a real `FLASK_SECRET_KEY` environment variable instead of the
      default dev key in `config.py`.
- [ ] Change `MASTER_SKIP_KEY` in `config.py` from the default `1234`.
- [ ] Decide whether the word-count bounds in `config.py`
      (`MIN_WORD_COUNT`, `BASELINE_MIN_WORD_COUNT`,
      `OPPOSITE_MIN_WORD_COUNT`/`OPPOSITE_MAX_WORD_COUNT`) match what you
      actually want to enforce.
- [ ] `OPPOSITE_EMOTION_MAP` has no defined opposite for a `neutral`
      self-report — it currently falls back to a random pick among all
      four emotions. Adjust if you'd rather handle that case differently.
- [ ] The comprehension check is still included, placed at the end of
      each round (this wasn't explicitly mentioned in the latest protocol
      update, so it was kept by default — remove it if that was
      intentional).

## Project structure

```
keystroke_emotion_app/
├── app.py                  # Flask routes / the whole experiment flow
├── config.py                 # Settings, emotion map, baseline text, video library
├── models.py                  # Database tables (SQLAlchemy)
├── export_data.py              # Dumps the database to CSV
├── requirements.txt
├── static/
│   ├── css/style.css             # All styling
│   ├── js/keystroke.js            # Keystroke capture + word-count logic
│   └── videos/                     # Put the 12 video files here
├── templates/                      # One HTML page per screen
└── instance/
    └── data.db                       # SQLite database (created on first run)
```
