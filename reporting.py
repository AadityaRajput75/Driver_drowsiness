"""Local numeric reports and interval-based evaluation against human labels."""
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean
from uuid import uuid4


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Annotations must be an object with duration_seconds and events')
    duration = data.get('duration_seconds')
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Annotation duration_seconds must be finite and positive')
    events = data.get('events')
    if not isinstance(events, list):
        raise ValueError('Annotation events must be a list')
    checked = []
    for event in events:
        if not isinstance(event, dict) or event.get('label') not in ('drowsy', 'distracted', 'phone_use'):
            raise ValueError('Each annotation needs label drowsy, distracted or phone_use')
        start, end = event.get('start'), event.get('end')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
            raise ValueError('Annotation start/end must be finite numbers')
        if not 0 <= start < end <= duration:
            raise ValueError('Annotation intervals must satisfy 0 <= start < end <= duration_seconds')
        checked.append({'start': start, 'end': end, 'label': event['label']})
    checked.sort(key=lambda e: e['start'])
    for label in ('drowsy', 'distracted', 'phone_use'):
        same = [e for e in checked if e['label'] == label]
        if any(a['end'] > b['start'] for a, b in zip(same, same[1:])):
            raise ValueError(f'Overlapping {label} annotations must be merged')
    return {'duration_seconds': duration, 'events': checked}


def evaluate_events(predictions, annotations, label='drowsy'):
    """One-to-one chronological matching of strictly overlapping intervals."""
    truth = [e for e in annotations['events'] if e['label'] == label]
    predictions = sorted([e for e in predictions if e['label'] == label], key=lambda e: e['start'])
    used = set()
    latencies = []
    matches = []
    for target in truth:
        for i, prediction in enumerate(predictions):
            if i not in used and min(target['end'], prediction['end']) > max(target['start'], prediction['start']):
                used.add(i)
                delay = max(0.0, prediction['start'] - target['start'])
                latencies.append(delay)
                matches.append({'truth_start': target['start'], 'alert_start': prediction['start'], 'delay_seconds': delay})
                break
    tp = len(used)
    false = len(predictions) - tp
    return {'true_events': len(truth), 'detected_events': tp, 'missed_events': len(truth) - tp,
            'false_alert_events': false, 'event_precision': tp / len(predictions) if predictions else None,
            'event_recall': tp / len(truth) if truth else None,
            'false_alerts_per_hour': false * 3600 / annotations['duration_seconds'],
            'mean_detection_delay_seconds': mean(latencies) if latencies else None,
            'matches': matches}


class SessionReport:
    def __init__(self, root, config, metadata):
        name = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]
        self.directory = Path(root) / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = asdict(config)
        self.metadata = metadata
        self.file = (self.directory / 'measurements.csv').open('w', newline='', encoding='utf-8')
        names = ['time_seconds', 'status', 'fatigue', 'visibility', 'distracted', 'alarm_level',
                 'raw_ear', 'valid_ear', 'mouth_ratio', 'pitch_degrees', 'yaw_degrees',
                 'close_threshold', 'open_threshold', 'closure_seconds', 'rolling_closure_fraction',
                 'observed_seconds', 'coverage', 'blinks', 'long_blinks', 'yawns', 'nods',
                 'calibrated', 'calibration_progress', 'baseline_status', 'processing_ms', 'reasons',
                 'mouth_valid', 'pose_valid', 'yawn_active', 'yawn_duration', 'yawn_threshold',
                 'nod_active', 'pitch_delta', 'phone_visible', 'phone_held', 'phone_score',
                 'phone_checked_at', 'phone_status', 'phone_warning', 'phone_duration']
        self.writer = csv.DictWriter(self.file, fieldnames=names)
        self.writer.writeheader()
        self.events = []
        self.open_events = {}
        self.actions = []
        self.frames = 0
        self.valid_frames = 0
        self.processing_total = 0.0
        self.processing_max = 0.0
        self.last_time = 0.0
        self.first_time = None
        self.last_state = None
        self.finished = False

    def action(self, timestamp, action):
        self.actions.append({'time': timestamp, 'action': action})

    def add(self, state, obs, baseline, processing_ms):
        self.writer.writerow(dict(zip(self.writer.fieldnames, [
            state.time, state.status, state.fatigue, state.visibility, state.distracted, state.alarm_level,
            obs.ear, state.ear, obs.mouth, obs.pitch, obs.yaw, state.close_threshold, state.open_threshold,
            state.closed_seconds, state.closure_fraction, state.observed_seconds, state.coverage,
            state.blinks, state.long_blinks, state.yawns, state.nods, state.calibrated,
            state.calibration_progress, baseline, round(processing_ms, 3), ' | '.join(state.reasons),
            obs.mouth_valid, obs.pose_valid, state.yawn_active, state.yawn_duration, state.yawn_threshold,
            state.nod_active, state.pitch_delta, obs.phone_visible, obs.phone_held, obs.phone_score,
            obs.phone_checked_at, state.phone_status, state.phone_warning, state.phone_duration])))
        signals = {'enhanced:drowsy': state.status == 'DROWSY', 'enhanced:distracted': state.distracted,
                   'enhanced:visibility': state.visibility != 'GOOD', 'baseline:drowsy': baseline == 'DROWSY',
                   'enhanced:phone_use': state.phone_warning}
        for name, active in signals.items():
            if active and name not in self.open_events:
                self.open_events[name] = {'start': state.time, 'reason': ' | '.join(state.reasons)}
            elif not active and name in self.open_events:
                self._end_event(name, state.time)
        self.frames += 1
        self.valid_frames += int(state.ear is not None)
        self.processing_total += processing_ms
        self.processing_max = max(self.processing_max, processing_ms)
        self.first_time = state.time if self.first_time is None else self.first_time
        self.last_time = state.time
        self.last_state = state
        if self.frames % 60 == 0:
            self.file.flush()

    def _end_event(self, name, end):
        event = self.open_events.pop(name)
        if end <= event['start']:
            return
        system, label = name.split(':')
        self.events.append(dict(event, end=end, system=system, label=label))

    def finish(self, end_time, annotations=None, outcome='completed', tolerance=0.1):
        if self.finished:
            return self.directory
        for name in list(self.open_events):
            self._end_event(name, end_time)
        self.file.close()
        summary = {'metadata': self.metadata, 'configuration': self.config, 'outcome': outcome,
                   'duration_seconds': end_time, 'frames': self.frames,
                   'valid_eye_frame_fraction': self.valid_frames / self.frames if self.frames else None,
                   'mean_processing_ms': self.processing_total / self.frames if self.frames else None,
                   'max_processing_ms': self.processing_max,
                   'processing_fps': 1000 * self.frames / self.processing_total if self.processing_total else None,
                   'last_state': asdict(self.last_state) if self.last_state else None,
                   'actions': self.actions,
                   'baseline_note': 'Original 0.22 / 2 s rule on the same selected face and landmark backend; not a full original-pipeline benchmark.',
                   'metric_note': 'Rolling eye-closure fraction is an EAR proxy, includes blinks, excludes missing intervals; not clinical PERCLOS or a probability.',
                   'evaluation': None}
        if annotations is not None:
            if outcome != 'completed' or abs(end_time - annotations['duration_seconds']) > tolerance:
                summary['evaluation_error'] = 'Evaluation skipped: replay incomplete or annotation duration does not match processed video.'
            else:
                enhanced = [e for e in self.events if e['system'] == 'enhanced']
                baseline = [e for e in self.events if e['system'] == 'baseline']
                summary['evaluation'] = {'enhanced_drowsy': evaluate_events(enhanced, annotations),
                                         'baseline_drowsy': evaluate_events(baseline, annotations),
                                         'enhanced_distraction': evaluate_events(enhanced, annotations, 'distracted'),
                                         'enhanced_phone_use': evaluate_events(enhanced, annotations, 'phone_use'),
                                         'method': 'One-to-one chronological interval overlap; unmatched predictions are false events. Labels must exhaustively cover the entire clip.'}
        (self.directory / 'events.json').write_text(json.dumps(self.events, indent=2, allow_nan=False), encoding='utf-8')
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        self.finished = True
        return self.directory
