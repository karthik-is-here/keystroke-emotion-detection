# Step 12 — Emotion-specific paired analysis

## Files
- `12_emotion_specific_analysis.py` — complete runnable analysis.
- `requirements.txt` — required packages.

## Setup
Copy `12_emotion_specific_analysis.py` into:
`C:\Karthik\Projects\keystroke_emotion_app`

Make sure `paired_typing_features.csv` is in the same folder.

Run in VS Code PowerShell:

```powershell
& "C:\Program Files\Python314\python.exe" 12_emotion_specific_analysis.py
```

If required packages are missing:

```powershell
& "C:\Program Files\Python314\python.exe" -m pip install -r requirements.txt
```

## Output
- `emotion_specific_results.csv`
- `emotion_specific_participant_data.csv`

## Method
The script compares genuine and masked typing separately for happy, sad, calm, and angry sessions. It pairs values within participant. If multiple eligible rows exist for one participant and emotion, it averages those rows within that participant. It uses Wilcoxon signed-rank tests and applies Holm correction across all emotion-by-feature tests (up to 32).

`typing_speed_cps` is overall task throughput based on total task duration, including time before typing. It should not be described as pure active typing speed.
