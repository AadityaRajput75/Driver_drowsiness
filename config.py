"""Validated settings shared by live monitoring and replay."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class Config:
    ear_threshold: float = 0.22
    reopen_ratio: float = 1.15
    closed_seconds: float = 2.0
    recovery_seconds: float = 0.8
    calibration_seconds: float = 8.0
    uncalibrated_close_ratio: float = 0.65
    calibration_min_samples: int = 30
    calibration_max_pitch: float = 25.0
    calibration_max_yaw: float = 20.0
    calibrated_close_ratio: float = 0.70
    calibrated_open_ratio: float = 0.82
    window_seconds: float = 60.0
    min_observed_seconds: float = 20.0
    min_coverage: float = 0.70
    closure_warning: float = 0.20
    closure_alert: float = 0.40
    long_blink_seconds: float = 0.50
    repeated_events: int = 3
    yawn_ratio: float = 0.38
    yawn_seconds: float = 1.0
    yawn_baseline_delta: float = 0.25
    event_gap_seconds: float = 0.30
    event_rearm_seconds: float = 0.25
    nod_degrees: float = 10.0
    nod_seconds: float = 0.18
    nod_max_seconds: float = 3.0
    nod_smoothing_seconds: float = 0.08
    distraction_degrees: float = 30.0
    looking_down_degrees: float = 18.0
    distraction_seconds: float = 2.0
    phone_seconds: float = 1.2
    phone_visible_seconds: float = 2.0
    phone_gap_seconds: float = 0.45
    phone_confidence: float = 0.40
    phone_interval_seconds: float = 0.20
    phone_result_max_age: float = 0.50
    visibility_warning_seconds: float = 3.0
    max_sample_gap: float = 0.5
    min_brightness: float = 35.0
    min_sharpness: float = 15.0
    min_face_pixels: int = 80
    eye_pose_limit: float = 35.0
    track_timeout_seconds: float = 2.0
    warning_repeat_seconds: float = 8.0

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f'{field.name} must be a number')
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name} must be finite and positive')
        for name in ('calibration_min_samples', 'repeated_events', 'min_face_pixels'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if not 0 < self.ear_threshold < 1:
            raise ValueError('ear_threshold must be between 0 and 1')
        if not 0 < self.uncalibrated_close_ratio <= 1:
            raise ValueError('uncalibrated_close_ratio must be in (0, 1]')
        if not 0 < self.calibrated_close_ratio < self.calibrated_open_ratio < 1:
            raise ValueError('Calibration ratios must satisfy 0 < close < open < 1')
        if not 0 < self.closure_warning < self.closure_alert < 1:
            raise ValueError('Closure thresholds must satisfy 0 < warning < alert < 1')
        if self.min_coverage > 1 or self.min_observed_seconds > self.window_seconds:
            raise ValueError('Invalid rolling-window coverage or duration')
        if self.reopen_ratio <= 1 or self.long_blink_seconds >= self.closed_seconds:
            raise ValueError('Reopen ratio must exceed 1; long blinks must be shorter than the alarm delay')
        if self.min_brightness > 255:
            raise ValueError('min_brightness cannot exceed 255')
        if self.nod_max_seconds <= self.nod_seconds:
            raise ValueError('nod_max_seconds must exceed nod_seconds')
        if self.phone_confidence > 1:
            raise ValueError('phone_confidence must be in (0, 1]')
        if self.phone_result_max_age <= self.phone_interval_seconds:
            raise ValueError('phone_result_max_age must exceed phone_interval_seconds')
        if self.event_gap_seconds >= self.yawn_seconds:
            raise ValueError('event_gap_seconds must be shorter than yawn_seconds')
        return self


def load_config(path=None):
    values = {} if path is None else json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f'Unknown settings: {", ".join(sorted(unknown))}')
    return Config(**values).validate()


def save_config(path):
    Path(path).write_text(json.dumps(asdict(Config()), indent=2) + '\n', encoding='utf-8')
