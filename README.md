# Personalized Driver Monitor

A local webcam application with personalized eye thresholds, fatigue trends, supporting facial signals, explainable alerts, a native dashboard, and recorded-video evaluation. MediaPipe is the default landmark backend; OpenCV LBF remains available as a fallback.

## Start on this Windows machine

From the project folder:

```powershell
.\run.cmd
```

The launcher prefers a project `.venv`, then the installed Python under `%LOCALAPPDATA%\Python\bin`, then the Python launcher. This avoids the nonworking Windows Store `python` alias found on this machine. It does not install or change packages.

Look forward with your eyes naturally open and mouth relaxed. Calibration collects **8 seconds of acceptable observations**, excluding poor visibility and unsuitable head angles. Blink normally. Progress can pause or restart if the samples are unstable. The personalized thresholds remain fixed until you recalibrate or select a new driver.

| Control | Action |
|---|---|
| Q / Esc / close window | Stop, release camera, silence audio, finalize report |
| C | Recalibrate and clear rolling history; preserve an active prolonged-closure alert |
| R | Reselect driver and reset calibration/history; preserve an active prolonged-closure alert |
| Click a face | Explicitly select that driver and start a fresh monitoring session within the report |
| M | Toggle audio; visual detection continues |

The image is **not mirrored**. `--driver-region left` and `right` refer to the displayed camera image. If multiple faces are eligible, click the driver. Tracking uses spatial continuity, not face recognition. After more than 2 seconds without a reliable match, use R or click to select again. This deliberately prevents automatic reassignment after a long loss; it cannot guarantee identity during overlapping faces or a very fast substitution.

## Fresh installation

Use Python 3.10+ with compatible wheels for the selected packages. The current Windows installation was exercised with Python 3.14.6, OpenCV contrib 5.0.0.93, MediaPipe 1.0.1, and NumPy 2.5.2. Other Python/package combinations need verification on their target platform.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.cmd
```

Install only **one OpenCV wheel variant** in the environment. This project uses `opencv-contrib-python` for the LBF fallback and desktop display; do not also install `opencv-python` or a headless variant. SciPy is no longer required. Sound uses Windows `winsound`; unsupported platforms retain visual alerts and show that audio is unavailable.

Keep these model files beside `main.py`:

- `face_landmarker.task` for MediaPipe.
- `lbfmodel.yaml` and `haarcascade_frontalface_default.xml` for OpenCV LBF.
- `alarm.wav`, generated automatically if absent.

Phone detection is enabled by default and additionally uses `models/efficientdet_lite0.tflite` and `models/hand_landmarker.task`, already included in this workspace. Download sources are recorded in `THIRD_PARTY.md`. If either model is missing or cannot load, the dashboard explicitly shows phone detection as unavailable; eye, mouth and head monitoring continue.

## What changed

1. **Reliable state handling.** Tracking loss and poor visibility are explicit states. Neither counts as open eyes nor clears a prolonged-closure alert. Recovery requires 0.8 seconds of reliable open-eye readings. Timing uses a monotonic clock for live input; gaps over 0.5 seconds invalidate the sample and break continuity.
2. **Personalized calibration.** The stable median open-eye EAR sets closing and reopening thresholds at 70% and 82% of baseline. Their separation avoids rapid switching around one threshold. During calibration, a conservative threshold of `0.22 * 0.65` still detects severe sustained closure. Detection is not fully personalized until calibration completes. `--no-calibration` uses fixed thresholds.
3. **Temporal fatigue signals.** A time-weighted 60-second window tracks eye-closure fraction, blink count, long blinks, possible yawns, and nods. The closure fraction requires at least 20 observed seconds and 70% coverage. Both endpoints of an interval must have valid eye readings; missing time is excluded. The fraction includes blinks and is an EAR-based proxy, **not validated clinical PERCLOS**.
4. **Supporting signals.** Sustained inner-mouth opening produces a possible-yawn event. Mouth reliability is checked independently from the eyes, so closing your eyes while yawning does not cancel it. A smoothed downward head excursion must reach 10 degrees relative to the calibrated pose and return within 3 seconds to count as a nod; no pause at the bottom is required. Both detectors tolerate brief unreliable measurements without counting missing time as evidence. Repeated yawns and nods together support a warning; neither alone declares drowsiness. Sustained head turns/tilts are reported separately as possible distraction, not measured gaze.
5. **Landmarks, selection and quality.** MediaPipe video mode provides face landmarks; LBF supports both older and newer OpenCV landmark array layouts. Driver selection rejects ambiguous matches. Lighting, blur, face size, eye geometry, head angle and pose fit are checked before eye measurements are trusted.
6. **Alerts and dashboard.** The native dashboard shows the selected face, current state and reasons, calibration, closure duration, rolling coverage, counters and EAR history. Warning sounds are rate-limited; critical alerts loop asynchronously without playback threads. Mute and audio errors are visible.
7. **Evaluation and reporting.** Replay uses video time, not processing speed. Every run can save numeric measurements, event intervals and a summary. With exhaustive annotations, reports compare drowsiness alerts against the original rule and calculate event recall, precision, missed events, false alert events per hour and detection delay.

An immediate alert follows 2 seconds of continuous closure. Rolling closure above 20% warns, and above 40% alerts once coverage is sufficient. Three long blinks in the window also warn. These defaults are configurable engineering heuristics, not learned probabilities or established accuracy claims.

## Configure and run

```powershell
# Create a complete editable configuration.
.\run.cmd --write-config settings.json
.\run.cmd --config settings.json

# Choose camera, driver location, audio and backend.
.\run.cmd --camera 1 --driver-region right --mute
.\run.cmd --backend mediapipe
.\run.cmd --backend lbf

# Disable numeric logging if desired.
.\run.cmd --no-report
```

`settings.example.json` lists all defaults. Partial JSON configurations are accepted; unknown keys, nonfinite numbers and inconsistent settings are rejected. `auto` falls back to LBF if MediaPipe cannot initialize and prints the reason. A failure during processing stops the run and saves an error outcome rather than silently changing measurement models mid-session.

## Replay and evaluate

```powershell
# Visible replay, muted by default. Add --audio to enable sound.
.\run.cmd --video recording.mp4

# Process without a window, as quickly as the detector permits.
.\run.cmd --video recording.mp4 --headless

# Compare against human event labels.
.\run.cmd --video recording.mp4 --headless --annotations examples\annotations.json
```

Use constant-frame-rate videos. Timestamps are `frame_index / FPS`; supply `--video-fps 30` only when the metadata is missing or incorrect. Variable-frame-rate material must first be converted to constant frame rate. Headless mode is for replay only. Use `--driver-region` to disambiguate the initial driver; headless replay holds an unknown state after a long tracking loss because it cannot receive a manual reselection.

Copy and edit `examples/annotations.json` for the actual video. It is an illustrative 60-second example, not labels for your recordings. `duration_seconds` must match the whole clip. Each event has `label` (`drowsy`, `distracted`, or `phone_use`), `start`, and `end` in seconds. Merge overlapping intervals of the same label. Empty `events` means the entire clip was reviewed and contains no such events.

Matching is one-to-one chronological interval overlap. One sustained alert can match only one true event; unmatched predicted intervals are false alert events. Detection delay is `max(0, alert_start - truth_start)`. Undefined precision/recall is saved as `null`. Partial runs and mismatched annotation durations skip evaluation with an explicit explanation. Do not compare numbers from clips with different annotation definitions.

The baseline is the original **EAR < 0.22 for 2 seconds** rule, including resetting to awake on a missing face, applied to the **same selected face and landmark measurements**. This isolates rule differences; it is not a benchmark of the full original Haar/LBF pipeline against MediaPipe. Use `--backend lbf` for a separate backend comparison. Human labels and varied driver footage are still required to establish whether the new system improves accuracy.

## Yawns, nods and phone distraction

Finish calibration while facing forward before testing gestures. The dashboard now shows mouth openness versus its personalized threshold, opening duration, relative head dip, and whether a nod is in progress. A yawn-like opening counts after 1 second by default; the mouth must close briefly before another can count. The default mouth threshold is 0.38 or the calibrated resting-mouth value plus 0.25, whichever is larger. Nods count after the down-and-up movement completes, not when the head first dips. A sustained downward head position instead produces a looking-down distraction warning after 2 seconds.

The phone feature runs an object detector and hand landmark model locally, approximately five times per second. A phone within the selected driver's interaction region triggers a warning after 2 seconds of detection, even if the hand is hidden. Hand overlap allows an earlier warning after 1.2 seconds. Brief missed checks (up to 0.45 seconds) preserve progress but do not add evidence time; stale results cannot trigger a warning. This warns about a nearby phone, not proven phone use: a mounted phone inside that region can also trigger it. Attribution to a nearby passenger is rejected when ambiguous. Phone and hand boxes, detection progress, and warnings appear on the dashboard and in reports. Audible warnings use the bundled alarm recording; press M to toggle mute.

The camera must include the driver's face and phone. A phone below the camera view cannot be identified. Occlusion, reflections and small objects can prevent detection; a looking-down warning can still fire without claiming a phone was seen. Head-distraction warnings remain active even when the head angle makes eye measurements unreliable. Tune `phone_visible_seconds` for the visible-phone delay and `phone_gap_seconds` for dropout tolerance.

```powershell
# Optional: run only facial monitoring, without the phone/hand models.
.\run.cmd --no-phone-detection
```

The default configuration has been updated. If you pass an older custom `settings.json`, its old yawn/nod values still override the new defaults. Generate a new file with `--write-config new-settings.json` and transfer any custom settings you want to retain. `phone_seconds`, `phone_confidence`, `yawn_seconds`, `yawn_ratio`, and `nod_degrees` are configurable.

Annotations may also use `phone_use`. For a phone-use event that is also intended as a general distraction label, include both intervals: labels are evaluated independently. Phone metrics follow the same interval matching rules as other events.

## Local output

Each run creates a unique folder under `reports/` (override with `--report-dir`):

- `measurements.csv`: timestamped measurements, quality, decisions, thresholds, baseline and processing time.
- `events.json`: enhanced drowsiness, distraction, possible phone use, visibility and baseline intervals.
- `summary.json`: configuration, source, backend, outcome, processing speed, manual actions and optional evaluation.

No camera images, video, embeddings or identity profiles are saved by the monitor, and processing does not upload data. Reports do contain the video path and numerical facial measurements. `--no-report` disables report writing.

## Validation

```powershell
python -m unittest discover -s tests -v

# Optional plumbing check using an existing public portrait image.
python tools\smoke_check.py --portrait artifacts\mediapipe-portrait.jpg
.\run.cmd --video artifacts\smoke\visibility-replay.avi --headless --annotations artifacts\smoke\annotations.json
```

Use the actual installed Python executable if the `python` alias does not work. Unit tests cover timing, missing data, calibration, hysteresis, recovery, supporting signals, driver continuity, pose geometry, audio transitions, annotation validation and report generation. The smoke utility makes a static-portrait replay with black and dim frames and renders dashboard previews. It verifies integration, not fatigue accuracy. Sources are listed in `THIRD_PARTY.md`.

This remains a prototype: ordinary webcams cannot reliably see eyes through dark sunglasses or in darkness; estimated pose, mouth opening and EAR can be wrong. Reliable night operation would require suitable hardware and separate validation. Evaluate with labeled recordings across drivers, lighting, glasses and camera positions before drawing real-world performance conclusions.

## Source layout

| File | Purpose |
|---|---|
| `main.py` | CLI, capture/replay, controls, lifecycle |
| `config.py` | Validated settings |
| `vision.py` | Landmark backends, driver selection, pose and quality |
| `monitoring.py` | Calibration, rolling signals, decisions and baseline |
| `behavior.py` | Yawn and nod event detection with gap tolerance |
| `phone_detection.py` | Phone/hand models and driver association |
| `alerts.py` | Nonblocking Windows audio |
| `dashboard.py` | Native dashboard and history graph |
| `reporting.py` | CSV/JSON output and event evaluation |
| `utils.py` | Eye and mouth geometry |
| `tests/` | Deterministic regression checks |
| `tools/smoke_check.py` | Optional integration fixture and preview generator |
