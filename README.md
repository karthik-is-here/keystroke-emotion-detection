# Keystroke + Text Emotion Data Collection

A Flask web app that collects labeled data for the keystroke-dynamics + text
emotion detection project. It only **collects** data. It does not predict
anything; modeling happens later, offline.

## Flow per participant

1. Consent screen, then demographics: name, age, nationality, first language,
   typing skill, dominant hand.
2. **Baseline 1:** transcribe a short fixed passage (two sentences).
3. **Baseline 2:** answer a random neutral question in about two sentences.
   Both baseline samples are labeled `neutral`.
4. Four rounds (`happy`, `sad`, `angry`, `calm`) in random order. Each emotion
   has 3 candidate videos; one is picked at random per participant, so nobody
   sees two videos of the same emotion. Each round:
   - Info screen, then the video autoplays with no controls.
   - **Self-report:** "How did that make you feel?" (happy / sad / angry /
     calm). This answer, not the video's intended emotion, is the ground truth.
   - Write about the video (minimum words: `MIN_WORD_COUNT` in `config.py`).
   - Write about the **same video as if it had made them feel a contrasting
     emotion** (`OPPOSITE_MIN_WORD_COUNT` to `OPPOSITE_MAX_WORD_COUNT` words).
   - A 10-second break before the next round (not after the last one).
5. Thank-you screen.

There is no comprehension check.

Every keystroke (key, down/up, millisecond timestamp) is recorded for every
typing task, along with the typed text.

### Opposite emotion

Which emotion they are asked to write about depends on the self-report
(`OPPOSITE_EMOTION_MAP` in `config.py`): angry and sad -> happy or calm,
happy -> sad, calm -> angry. When there are two options, the app picks
whichever has been assigned fewer times so far for that self-reported emotion
(ties are random), so the dataset stays balanced. This counts rows already in
the database, so clear test data before a real trial.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`. Everything is served locally, so it works
offline.

## Videos

Put 12 files in `static/videos/` with these exact names:

```
happy_1.mp4  happy_2.mp4  happy_3.mp4
sad_1.mp4    sad_2.mp4    sad_3.mp4
angry_1.mp4  angry_2.mp4  angry_3.mp4
calm_1.mp4   calm_2.mp4   calm_3.mp4
```

- Roughly 1-2 minutes each, H.264 `.mp4`.
- Check each plays at `http://127.0.0.1:5000/static/videos/<name>`.
- The `.gitignore` keeps videos out of git (size and copyright).
- Filenames and intro text are set in `VIDEO_LIBRARY` in `config.py`.

## Testing shortcuts (staff only)

A collapsed "Staff: skip..." control on the video page and the break page
accepts `MASTER_SKIP_KEY` (default `1234`, set in `config.py`). The video skip
jumps straight to self-report; the break skip ends the wait.

## Data

Stored in SQLite at `instance/data.db`, created on first run. New model
columns are added automatically on startup, so you do not need to delete the
database after editing `models.py`. An older database may still contain an
unused `comprehension_checks` table; it is harmless.

Export to CSV with `python export_data.py` (writes `participants.csv`,
`emotion_sessions.csv`, `typing_samples.csv`, `keystroke_events.csv` into
`exports/`). Each participant has 10 rows in `typing_samples` (2 baseline +
4 rounds x 2). Key columns:

- `sample_type`: `baseline_fixed`, `baseline_question`, `video_response`,
  `opposite_response`
- `label`: the round's induced emotion (`neutral` for baselines)
- `self_reported_emotion`: what the participant said they felt
- `target_emotion`: only on `opposite_response` rows
- `video_filename`: which of the 3 videos was shown

Dwell time = a key's `up` minus its `down`. Flight time = a key's `down` minus
the previous key's `up`, grouped by `typing_sample_id`.

`instance/data.db` and `exports/` contain participant names. Never commit or
share them.

## Before real participants

- [ ] Replace the placeholder consent text in `templates/consent.html`.
- [ ] Add the 12 videos.
- [ ] Change `MASTER_SKIP_KEY` from `1234`.
- [ ] Set a real `FLASK_SECRET_KEY` environment variable.
- [ ] Confirm the word limits in `config.py`.
- [ ] Delete `instance/data.db` once, after your last test run and before the
      first real participant. Never delete it after that.

## Structure

```
app.py          routes and experiment flow
config.py       settings, emotion map, baseline text, video library
models.py       database tables
export_data.py  CSV export
static/         css, js, videos
templates/      one page per screen
instance/       data.db (created on first run)
```
