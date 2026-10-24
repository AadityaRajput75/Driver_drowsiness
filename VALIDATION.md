# Validation results

## 21 October 2026: yawn/nod repair and phone warnings

57 regression tests pass. Added coverage includes yawns with unreliable eyes, short measurement dropouts, opening/rearming hysteresis, talking-like short openings, down-and-up nods without a pause, calibrated head offsets, long occlusions, looking-down warnings when eyes cannot be measured, sustained phone evidence, stale detections, unheld phones and passenger attribution.

The default MediaPipe CLI completed the 240-frame visibility replay with phone detection enabled and no phone warnings on the no-phone portrait. Output: `artifacts/validation/phone-update/20261021-123358-68ec6fc9/`. The LBF replay with `--no-phone-detection` also completed: `artifacts/validation/phone-disabled/20261021-123819-e1056ca0/`.

Real face, phone and hand models detected an interaction in a cropped, closer-framed public test photograph. The time-based possible-phone-use warning began at 1.4 seconds with a configured 1.2-second requirement (sampling determines the exact frame). Both distraction and phone-use events were saved. Output: `artifacts/phone-validation/20261021-123757-5e7581dc/`. A dashboard preview was rendered and visually inspected. This repeated-image fixture also produced an uncalibrated eye-closure alert on the subject's low measured EAR; it is not evidence of drowsiness accuracy.

The tests also exposed useful limits: a phone can be found while a hand hidden behind it is missed; a distant or unrecognized face prevents driver attribution. Cropping the test photo to face-and-upper-body framing enabled the complete detection path. The application therefore requires suitable camera framing and labels the result as possible use. Tests do not establish accuracy on actual driving footage.

Replaying the prior session's saved measurements with independent mouth validity and the new movement logic yielded four yawn-like events and one nod. No video or human event labels were available for that session, and old logs lack the new per-signal validity flags, so this is diagnostic evidence only, not a measured recall result.

## Initial implementation

Implementation checks completed on 15 October 2026.

## Environment exercised

- Windows, Python 3.14.6.
- OpenCV contrib 5.0.0.93, MediaPipe 1.0.1, NumPy 2.5.2.
- Both existing landmark models load. A missing Haar cascade asset was added locally.
- OpenCV threads are capped at two to avoid excessive worker creation and the native-library initialization failure encountered during integration testing.

## Automated checks

41 regression tests passed. They cover:

- Prolonged closure, reliable recovery, normal/long blinks and threshold hysteresis.
- Calibration of small naturally open eyes, unstable samples, poor visibility and severe closure before calibration finishes.
- Missing frames, low coverage, large timestamp gaps, nonfinite inputs and time-weighted results across frame rates.
- Distraction separate from drowsiness; sustained mouth opening; downward nod and return; rejection of a prolonged head tilt.
- Driver selection despite reordered detections, ambiguous candidates, initial regions and tracking timeouts.
- Both OpenCV landmark array layouts and recovery of known projected head angles.
- Mocked audio transitions, warning cooldown and muting/unmuting. Speaker output was not acoustically verified.
- Event matching, false alarms, missed events, undefined metrics, invalid labels and partial-run reports.

Report tests used a temporary directory under `artifacts/test-temp` because this execution sandbox blocks writes inside its default temporary subdirectories. Normal installations can run `python -m unittest discover -s tests -v` directly.

## Integration checks

The Windows launcher starts the installed Python without relying on the Windows Store alias. CLI help and configuration generation work.

Both MediaPipe and LBF processed a 240-frame, 10 FPS, 24-second fixture to completion. The fixture repeats a public frontal portrait, briefly removes it, dims it, and restores it. Both completed calibration at 8 seconds, reported the missing face, and recovered after the short loss. MediaPipe explicitly reported insufficient lighting on the dim interval; LBF lost the face on that interval.

Reports were inspected at:

- `artifacts/validation/mediapipe/20261015-125638-b8414afc/`
- `artifacts/validation/lbf/20261015-125635-067cf6b9/`

Each contains measurements, event intervals and a completed summary with evaluation results. The test had no drowsiness labels, so recall is correctly `null`; the new monitor produced no drowsiness events. The fixed original rule generated alerts on the portrait's naturally low EAR. This is a useful regression example, not an accuracy benchmark.

The native dashboard was rendered and visually inspected in monitoring and tracking-loss states:

- `artifacts/smoke/dashboard-preview.png`
- `artifacts/smoke/dashboard-missing.png`

## What remains empirical

The implementation is complete for the seven approved improvements. Accuracy across real drivers, glasses, lighting, camera positions, actual yawns/nods, and true fatigue has not been established. The head-pose geometry and event rules are heuristics. Evaluation needs representative, exhaustively labeled recordings; synthetic tests and a static portrait cannot establish road suitability or a validated fatigue metric.
# Phone-warning correction — 22 October 2026

The saved camera session `reports/20261021-125008-e56fb573` contained 483 frames with an associated visible phone, only 11 with hand overlap, and no phone warnings. These counts include cached detections, not independent model predictions. Requiring hand overlap prevented the visible-phone evidence from producing an alert.

The warning now accepts sustained driver-associated phone visibility (2 seconds), retains the faster hand-supported path (1.2 seconds), and tolerates short missed checks without adding evidence time. Warnings use the bundled WAV instead of a Windows notification alias. Replaying the saved phone measurements through the revised monitor produces warning onsets at 37.57 and 189.63 seconds. This verifies warning logic against the recorded detector output; no recorded video or ground-truth phone-use labels were available to measure detection accuracy. Mounted phones within the driver's region can trigger this broader warning.

All 60 automated tests pass, including hidden-hand warnings, dropout recovery, stale results, isolated detections, and the alarm sound selection. Live camera performance still requires checking in the user's framing.

