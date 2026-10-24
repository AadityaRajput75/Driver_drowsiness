"""Stateful mouth-opening and down/up movement detectors with short-gap tolerance."""
import math


class YawnDetector:
    def __init__(self, config):
        self.c = config
        self.duration = 0.0
        self.active = False
        self.counted = False
        self.armed = True
        self.last_time = None
        self.last_valid = None
        self.previous_open = False
        self.close_start = None

    def update(self, t, value, valid, threshold):
        dt = 0 if self.last_time is None else t - self.last_time
        self.last_time = t
        expired = self.last_valid is not None and t - self.last_valid > self.c.event_gap_seconds
        if expired:
            if self.active:
                self.armed = False  # Do not double-count one opening across an occlusion.
            self.active = False
            self.duration = 0.0
            self.previous_open = False
            self.close_start = None
        if not valid or value is None or not math.isfinite(value):
            self.previous_open = False
            return False
        self.last_valid = t
        if value < threshold * 0.65:
            if self.close_start is None:
                self.close_start = t
            if t - self.close_start >= self.c.event_rearm_seconds:
                self.active = False
                self.duration = 0.0
                self.counted = False
                self.armed = True
            self.previous_open = False
            return False
        self.close_start = None
        if value >= threshold and self.armed and not self.active:
            self.active = True
            self.duration = 0.0
        if self.active and self.previous_open and dt <= self.c.event_gap_seconds:
            self.duration += dt
        self.previous_open = self.active
        if self.active and not self.counted and self.duration >= self.c.yawn_seconds:
            self.counted = True
            return True
        return False


class NodDetector:
    def __init__(self, config):
        self.c = config
        self.smoothed = None
        self.last_valid = None
        self.start = None
        self.peak = 0.0
        self.armed = False
        self.active = False

    def update(self, t, pitch_delta, valid):
        if self.last_valid is not None and t - self.last_valid > self.c.event_gap_seconds:
            self.start = None
            self.active = False
            self.armed = False
            self.smoothed = None
        if not valid or pitch_delta is None or not math.isfinite(pitch_delta):
            return False
        dt = 0 if self.last_valid is None else t - self.last_valid
        self.last_valid = t
        alpha = 1 - math.exp(-dt / self.c.nod_smoothing_seconds)
        self.smoothed = pitch_delta if self.smoothed is None else self.smoothed + alpha * (pitch_delta - self.smoothed)
        value = self.smoothed
        return_level = self.c.nod_degrees * 0.35
        if value <= return_level:
            completed = (self.active and self.start is not None
                         and self.c.nod_seconds <= t - self.start <= self.c.nod_max_seconds
                         and self.peak >= self.c.nod_degrees)
            self.start = None
            self.peak = 0.0
            self.active = False
            self.armed = True
            return completed
        if self.armed and not self.active:
            self.start = t
            self.peak = value
            self.active = True
            self.armed = False
        if self.active:
            self.peak = max(self.peak, value)
            if t - self.start > self.c.nod_max_seconds:
                self.start = None
                self.active = False
        return False
