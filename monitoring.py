"""Time-based monitoring, independent of camera, UI, and machine learning."""
from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median

from config import Config
from behavior import NodDetector, YawnDetector


@dataclass
class Observation:
    time: float
    ear: float | None = None
    mouth: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    face_found: bool = False
    quality_ok: bool = False
    eyes_valid: bool = False
    quality_reason: str = 'No driver detected'
    mouth_valid: bool | None = None
    pose_valid: bool | None = None
    phone_visible: bool = False
    phone_held: bool = False
    phone_checked_at: float | None = None
    phone_detection_status: str = 'OFF'
    phone_score: float | None = None


@dataclass
class State:
    time: float
    status: str
    reasons: list[str]
    fatigue: str
    visibility: str
    distracted: bool
    alarm_level: int
    ear: float | None
    close_threshold: float
    open_threshold: float
    closed_seconds: float
    closure_fraction: float | None
    observed_seconds: float
    coverage: float
    blinks: int
    long_blinks: int
    yawns: int
    nods: int
    calibrated: bool
    calibration_progress: float
    calibration_message: str
    yawn_active: bool = False
    yawn_duration: float = 0.0
    yawn_threshold: float = 0.38
    nod_active: bool = False
    pitch_delta: float | None = None
    phone_warning: bool = False
    phone_duration: float = 0.0
    phone_status: str = 'OFF'


def finite(value):
    return value is not None and isfinite(value)


class Monitor:
    def __init__(self, config=None, calibrate=True):
        self.config = (config or Config()).validate()
        self.calibration_enabled = calibrate
        self.calibrated = False
        self.close_threshold = self.config.ear_threshold * (self.config.uncalibrated_close_ratio if calibrate else 1.0)
        self.open_threshold = self.close_threshold * self.config.reopen_ratio
        self.calibration_samples = []
        self.calibration_elapsed = 0.0
        self.calibration_previous = False
        self.calibration_message = 'Look forward, eyes naturally open; blink normally'
        if not calibrate:
            self.calibration_message = 'Fixed thresholds (calibration disabled)'
        self.pitch_baseline = 0.0
        self.yaw_baseline = 0.0
        self.last_time = None
        self.started = None
        self.previous_valid = False
        self.previous_closed = False
        self.closed = False
        self.closed_start = None
        self.open_start = None
        self.latched_alert = False
        self.intervals = deque()
        self.blinks = deque()
        self.yawns = deque()
        self.nods = deque()
        self.yawn_detector = YawnDetector(self.config)
        self.nod_detector = NodDetector(self.config)
        self.yawn_threshold = self.config.yawn_ratio
        self.phone_start = None
        self.phone_last_positive = None
        self.phone_evidence_seconds = 0.0
        self.phone_previous_positive = False
        self.distraction_start = None
        self.invalid_start = None

    def _calibrate(self, obs, dt, eyes_valid):
        if self.calibrated or not self.calibration_enabled:
            return
        eligible = (eyes_valid and 0.16 <= obs.ear <= 0.45
                    and finite(obs.pitch) and finite(obs.yaw)
                    and abs(obs.pitch) < self.config.calibration_max_pitch
                    and abs(obs.yaw) < self.config.calibration_max_yaw
                    and finite(obs.mouth) and obs.mouth < 0.30)
        if not eligible:
            self.calibration_previous = False
            self.calibration_message = 'Need clear frontal view, open eyes and relaxed mouth'
            return
        if self.calibration_previous:
            self.calibration_elapsed += dt
        self.calibration_previous = True
        self.calibration_samples.append((obs.ear, obs.pitch, obs.yaw, obs.mouth))
        self.calibration_samples = self.calibration_samples[-2400:]
        self.calibration_message = 'Calibrating: look forward, eyes naturally open'
        if (self.calibration_elapsed < self.config.calibration_seconds
                or len(self.calibration_samples) < self.config.calibration_min_samples):
            return
        values = sorted(s[0] for s in self.calibration_samples)
        baseline = median(values)
        spread = values[int(len(values) * 0.85)] - values[int(len(values) * 0.15)]
        if spread > baseline * 0.25:
            self.calibration_samples.clear()
            self.calibration_elapsed = 0.0
            self.calibration_message = 'Unstable calibration; retrying with a steady frontal view'
            return
        self.close_threshold = baseline * self.config.calibrated_close_ratio
        self.open_threshold = baseline * self.config.calibrated_open_ratio
        self.pitch_baseline = median(s[1] for s in self.calibration_samples)
        self.yaw_baseline = median(s[2] for s in self.calibration_samples)
        self.yawn_threshold = max(self.config.yawn_ratio,
                                  median(s[3] for s in self.calibration_samples) + self.config.yawn_baseline_delta)
        self.nod_detector = NodDetector(self.config)
        self.calibrated = True
        self.calibration_message = f'Personal baseline EAR {baseline:.3f}'
        # Do not mix measurements classified under different thresholds.
        self.intervals.clear()
        self.blinks.clear()
        self.closed = False
        self.closed_start = self.open_start = None
        self.previous_valid = False

    def update(self, obs):
        c = self.config
        t = obs.time
        if not isfinite(t) or t < 0 or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Observation timestamps must be finite, nonnegative and increasing')
        if self.started is None:
            self.started = t
        raw_dt = 0.0 if self.last_time is None else t - self.last_time
        gap = raw_dt > c.max_sample_gap
        dt = 0.0 if gap else raw_dt
        eyes_valid = (not gap and obs.face_found and obs.quality_ok and obs.eyes_valid
                      and finite(obs.ear) and 0 <= obs.ear <= 1)
        if gap:
            self.previous_valid = False
            self.closed_start = self.open_start = None
            self.closed = False
            self.distraction_start = self.phone_start = None
            self.calibration_previous = False
        self._calibrate(obs, dt, eyes_valid)
        # Only intervals bounded by two reliable measurements enter the denominator.
        if eyes_valid and self.previous_valid and dt > 0:
            self.intervals.append((t - dt, t, self.previous_closed))
        duration = 0.0
        if eyes_valid:
            self.invalid_start = None
            if not self.closed and obs.ear < self.close_threshold:
                self.closed = True
                self.closed_start = t
                self.open_start = None
            elif self.closed and obs.ear >= self.open_threshold:
                blink_duration = t - self.closed_start if self.closed_start is not None else 0
                if blink_duration >= 0.08:
                    self.blinks.append((t, blink_duration))
                self.closed = False
                self.closed_start = None
            if self.closed:
                duration = t - self.closed_start
                self.open_start = None
                if duration >= c.closed_seconds:
                    self.latched_alert = True
            elif obs.ear >= self.open_threshold:
                if self.open_start is None:
                    self.open_start = t
                if t - self.open_start >= c.recovery_seconds:
                    self.latched_alert = False
            else:
                self.open_start = None
        else:
            if self.invalid_start is None:
                self.invalid_start = t
            self.closed = False
            self.closed_start = self.open_start = None
            # Missing data is unknown. It cannot resolve an existing alert.

        pose_valid = (not gap and obs.face_found
                      and (obs.quality_ok if obs.pose_valid is None else obs.pose_valid)
                      and finite(obs.pitch) and finite(obs.yaw))
        yaw_delta = obs.yaw - self.yaw_baseline if pose_valid else 0.0
        pitch_delta = obs.pitch - self.pitch_baseline if pose_valid else 0.0
        looking_down = pose_valid and pitch_delta >= c.looking_down_degrees
        away = pose_valid and (abs(yaw_delta) >= c.distraction_degrees
                               or abs(pitch_delta) >= c.distraction_degrees or looking_down)
        if away:
            if self.distraction_start is None:
                self.distraction_start = t
        else:
            self.distraction_start = None
        head_distracted = self.distraction_start is not None and t - self.distraction_start >= c.distraction_seconds
        fresh_phone = (not gap and obs.face_found and finite(obs.phone_checked_at)
                       and 0 <= t - obs.phone_checked_at <= c.phone_result_max_age
                       and obs.phone_detection_status == 'READY')
        positive = fresh_phone and (obs.phone_held or obs.phone_visible)
        expired = (not fresh_phone or self.phone_last_positive is None
                   or obs.phone_checked_at - self.phone_last_positive > c.phone_gap_seconds)
        if expired:
            self.phone_evidence_seconds = 0.0
            self.phone_last_positive = None
            self.phone_previous_positive = False
        if positive:
            # Count new positive observations only; cached results and missed checks
            # cannot increase the evidence duration.
            if self.phone_last_positive is not None and self.phone_previous_positive:
                self.phone_evidence_seconds += max(0.0, obs.phone_checked_at - self.phone_last_positive)
            self.phone_last_positive = obs.phone_checked_at
        self.phone_previous_positive = bool(positive)
        phone_duration = self.phone_evidence_seconds
        required_phone_seconds = c.phone_seconds if obs.phone_held else c.phone_visible_seconds
        phone_warning = fresh_phone and phone_duration >= required_phone_seconds
        distracted = head_distracted or phone_warning
        mouth_valid = (not gap and obs.face_found
                       and (obs.quality_ok if obs.mouth_valid is None else obs.mouth_valid)
                       and finite(obs.mouth) and 0 <= obs.mouth <= 2)
        if self.yawn_detector.update(t, obs.mouth, mouth_valid, self.yawn_threshold):
            self.yawns.append(t)
        if self.nod_detector.update(t, pitch_delta, pose_valid and abs(yaw_delta) < 35):
            self.nods.append(t)
        cutoff = t - c.window_seconds
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        while self.blinks and self.blinks[0][0] < cutoff:
            self.blinks.popleft()
        for events in (self.yawns, self.nods):
            while events and events[0] < cutoff:
                events.popleft()
        observed = closed_time = 0.0
        for start, end, closed in self.intervals:
            length = end - max(start, cutoff)
            observed += length
            if closed:
                closed_time += length
        elapsed = min(c.window_seconds, t - self.started)
        coverage = min(1.0, observed / elapsed) if elapsed > 0 else 0.0
        enough = observed >= c.min_observed_seconds and coverage >= c.min_coverage
        fraction = closed_time / observed if enough else None
        long_blinks = sum(c.long_blink_seconds <= d < c.closed_seconds for _, d in self.blinks)
        reasons = []
        fatigue = 'LOW' if enough else 'COLLECTING'
        if fraction is not None and fraction >= c.closure_warning:
            fatigue = 'ELEVATED'
            reasons.append('Frequent eye closure in the rolling window')
        if long_blinks >= c.repeated_events:
            fatigue = 'ELEVATED'
            reasons.append('Repeated long blinks')
        if len(self.yawns) >= c.repeated_events:
            reasons.append('Repeated sustained mouth opening (possible yawns)')
        if len(self.nods) >= c.repeated_events:
            reasons.append('Repeated downward head nods')
        if len(self.yawns) >= c.repeated_events and len(self.nods) >= c.repeated_events:
            fatigue = 'ELEVATED'
        trend_alert = fraction is not None and fraction >= c.closure_alert
        if trend_alert:
            fatigue = 'HIGH'
        if self.latched_alert:
            fatigue = 'HIGH'
            reasons.insert(0, 'Prolonged eye closure; waiting for clear open-eye recovery')
        if self.yawn_detector.counted and self.yawn_detector.active:
            reasons.append('Yawn-like mouth opening detected')
        elif self.yawn_detector.active:
            reasons.append('Measuring sustained mouth opening')
        if self.nods and t - self.nods[-1] < 2:
            reasons.append('Head nod detected')
        if phone_warning:
            reasons.insert(0, 'Phone near driver; put it away and focus on the road')
        if head_distracted:
            reasons.append('Looking down for too long' if looking_down else 'Sustained head turn or tilt away from forward view')
        visibility = 'GOOD' if eyes_valid else ('POOR VISIBILITY' if obs.face_found else 'TRACKING LOST')
        if not eyes_valid:
            reasons.append('Frame gap: sampling too slow or interrupted' if gap else
                           (obs.quality_reason or 'Eye measurements unavailable'))
        if self.latched_alert or trend_alert:
            status, level = 'DROWSY', 2
        elif distracted:
            status, level = 'DISTRACTED', 1
        elif not eyes_valid:
            status = visibility
            level = int(t - self.invalid_start >= c.visibility_warning_seconds)
        elif fatigue == 'ELEVATED':
            status, level = 'WARNING', 1
        elif self.closed and duration >= c.long_blink_seconds:
            status, level = 'WARNING', 0
            reasons.append('Eyes closed; monitoring duration')
        elif self.calibration_enabled and not self.calibrated:
            status, level = 'CALIBRATING', 0
        else:
            status, level = 'MONITORING', 0
        if not reasons:
            reasons = ['No current alert' if enough else 'Collecting reliable observations']
        self.last_time = t
        self.previous_valid = eyes_valid
        self.previous_closed = self.closed
        return State(t, status, reasons, fatigue, visibility, distracted, level,
                     obs.ear if eyes_valid else None, self.close_threshold, self.open_threshold,
                     duration, fraction, observed, coverage, len(self.blinks), long_blinks,
                     len(self.yawns), len(self.nods), self.calibrated,
                     min(1.0, self.calibration_elapsed / c.calibration_seconds), self.calibration_message,
                     self.yawn_detector.active, self.yawn_detector.duration, self.yawn_threshold,
                     self.nod_detector.active, pitch_delta if pose_valid else None,
                     phone_warning, phone_duration, obs.phone_detection_status)


class TraditionalBaseline:
    """Original fixed 0.22 / 2-second rule, evaluated on the SAME landmarks."""
    def __init__(self):
        self.start = None

    def update(self, obs):
        if obs.face_found and finite(obs.ear) and obs.ear < 0.22:
            if self.start is None:
                self.start = obs.time
            return 'DROWSY' if obs.time - self.start >= 2.0 else 'WARNING'
        self.start = None
        return 'AWAKE'
