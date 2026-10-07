/*
 * Captures keydown/keyup timing on a textarea for keystroke-dynamics
 * analysis (dwell time, flight time, etc. are all derivable later from
 * the raw down/up timestamps stored here).
 *
 * Usage (see templates/baseline.html and templates/typing.html):
 *
 *   const recorder = new KeystrokeRecorder(document.getElementById('text_content'));
 *   recorder.start();
 *   // ... on form submit, before the form actually submits:
 *   const data = recorder.stop();
 *   document.getElementById('keystroke_data').value = JSON.stringify(data.events);
 *   document.getElementById('duration_ms').value = data.durationMs;
 */

class KeystrokeRecorder {
  constructor(textareaEl) {
    this.textarea = textareaEl;
    this.events = [];
    this.startTime = null;
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
  }

  start() {
    this.events = [];
    this.startTime = performance.now();
    this.textarea.addEventListener("keydown", this._onKeyDown);
    this.textarea.addEventListener("keyup", this._onKeyUp);
  }

  stop() {
    this.textarea.removeEventListener("keydown", this._onKeyDown);
    this.textarea.removeEventListener("keyup", this._onKeyUp);
    return {
      durationMs: performance.now() - this.startTime,
      events: this.events,
    };
  }

  _onKeyDown(e) {
    // Ignore auto-repeat events fired while a key is held down; only the
    // first press and the eventual release matter for dwell time.
    if (e.repeat) return;
    this.events.push({
      key: e.key,
      type: "down",
      t: performance.now() - this.startTime,
    });
  }

  _onKeyUp(e) {
    this.events.push({
      key: e.key,
      type: "up",
      t: performance.now() - this.startTime,
    });
  }
}

/*
 * Wires up a textarea + form so that: a live word count is shown, the
 * submit button is disabled until the minimum word count is reached
 * (when minWords > 0), and keystroke data is attached to the form right
 * before it submits.
 */
function setupTypingForm({ textareaId, formId, wordCountId, submitId, minWords, maxWords }) {
  const textarea = document.getElementById(textareaId);
  const form = document.getElementById(formId);
  const wordCountEl = wordCountId ? document.getElementById(wordCountId) : null;
  const submitBtn = submitId ? document.getElementById(submitId) : null;

  const recorder = new KeystrokeRecorder(textarea);
  recorder.start();

  function countWords(text) {
    const trimmed = text.trim();
    return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
  }

  function updateWordCount() {
    const count = countWords(textarea.value);
    if (wordCountEl) {
      if (minWords && maxWords) {
        wordCountEl.textContent = `${count} words (need ${minWords}\u2013${maxWords})`;
      } else if (minWords) {
        wordCountEl.textContent = `${count} / ${minWords} words`;
      } else {
        wordCountEl.textContent = `${count} words`;
      }
    }
    if (submitBtn) {
      const tooFew = minWords ? count < minWords : false;
      const tooMany = maxWords ? count > maxWords : false;
      submitBtn.disabled = tooFew || tooMany;
    }
  }

  textarea.addEventListener("input", updateWordCount);
  updateWordCount();

  form.addEventListener("submit", function () {
    const data = recorder.stop();
    document.getElementById("keystroke_data").value = JSON.stringify(data.events);
    document.getElementById("duration_ms").value = data.durationMs;
  });
}
