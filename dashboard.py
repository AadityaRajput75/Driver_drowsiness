"""Native OpenCV dashboard; no web service or video uploads."""
from collections import deque
import textwrap

import cv2
import numpy as np

BG = (23, 20, 17)
CARD = (39, 34, 28)
TEXT = (238, 234, 225)
MUTED = (168, 158, 144)
TEAL = (180, 218, 72)
AMBER = (81, 191, 255)
RED = (104, 105, 255)


class Dashboard:
    title = 'Driver Monitor'

    def __init__(self):
        self.history = deque()
        self.transform = (1, 0, 0)
        self.pending_click = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale, ox, oy = self.transform
            if 18 <= x < 818 and 86 <= y < 686:
                self.pending_click = ((x - ox) / scale, (y - oy) / scale)

    def render(self, frame, faces, selected, state, obs, backend, audio, fps, baseline, phone=None):
        canvas = np.full((800, 1280, 3), BG, dtype=np.uint8)

        def text(message, x, y, size=0.48, color=TEXT, weight=1):
            cv2.putText(canvas, str(message), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        size, color, weight, cv2.LINE_AA)

        text('DRIVER / MONITOR', 22, 36, 0.78, TEXT, 2)
        text('PERSONALIZED OBSERVATION  /  LOCAL PROCESSING', 22, 61, 0.40, MUTED)
        text(f'{backend.upper()}   |   {fps:.1f} FPS   |   {state.time:.0f}s', 860, 35, 0.48, MUTED)
        annotated = frame.copy()
        if phone is not None:
            for box in phone.boxes:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), AMBER, 2)
                cv2.putText(annotated, 'PHONE', (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)
            for box in phone.hand_boxes:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), MUTED, 1)
        for face in faces:
            x, y, w, h = [int(v) for v in face.box]
            color = TEAL if face is selected else MUTED
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, 'DRIVER' if face is selected else 'click to select', (x, max(y - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if face is selected:
                for points in (*face.eyes, face.mouth):
                    cv2.polylines(annotated, [points.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        h, w = annotated.shape[:2]
        scale = min(800 / w, 600 / h)
        rw, rh = round(w * scale), round(h * scale)
        ox, oy = 18 + (800 - rw) // 2, 86 + (600 - rh) // 2
        canvas[oy:oy + rh, ox:ox + rw] = cv2.resize(annotated, (rw, rh))
        self.transform = (scale, ox, oy)
        color = RED if state.alarm_level == 2 else (AMBER if state.status not in ('MONITORING', 'CALIBRATING') else TEAL)
        cv2.rectangle(canvas, (18, 86), (818, 686), color, 2)
        cv2.rectangle(canvas, (842, 86), (1258, 207), CARD, -1)
        text('CURRENT STATE', 860, 112, 0.40, MUTED)
        text(state.status, 860, 149, 0.78, color, 2)
        text(f'Fatigue: {state.fatigue}   |   Audio: {"ON" if audio.enabled else "OFF"}', 860, 180, 0.44)
        text(f'Visibility: {state.visibility}', 860, 233, 0.46, TEAL if state.visibility == 'GOOD' else AMBER)
        text(f'Distraction: {"DETECTED" if state.distracted else "No current signal"}', 860, 257, 0.44)
        ear = '--' if state.ear is None else f'{state.ear:.3f}'
        text(f'Eye openness  {ear}', 860, 292, 0.57, TEXT, 1)
        text(f'Close < {state.close_threshold:.3f}  /  Reopen >= {state.open_threshold:.3f}', 860, 317, 0.42, MUTED)
        text(f'Continuous closure: {state.closed_seconds:.1f}s', 860, 345)
        fraction = '-- (collecting)' if state.closure_fraction is None else f'{state.closure_fraction:.0%}'
        text(f'Rolling closure: {fraction}', 860, 374)
        text(f'Observed: {state.observed_seconds:.1f}s  |  Coverage: {state.coverage:.0%}', 860, 398, 0.42, MUTED)
        text(f'Yawns {state.yawns}  /  Nods {state.nods}  /  Long blinks {state.long_blinks}', 860, 428, 0.44)
        mouth = '--' if obs.mouth is None or obs.mouth_valid is False else f'{obs.mouth:.2f}'
        text(f'Mouth {mouth} / {state.yawn_threshold:.2f}  |  Opening {state.yawn_duration:.1f}s', 860, 452, 0.42,
             AMBER if state.yawn_active else MUTED)
        pitch = '--' if state.pitch_delta is None else f'{state.pitch_delta:+.1f} deg'
        nod_label = 'Unavailable' if state.pitch_delta is None else ('Nod in progress' if state.nod_active else 'Ready')
        text(f'Head dip {pitch}  |  {nod_label}', 860, 476, 0.42,
             AMBER if state.nod_active else MUTED)
        phone_text = state.phone_status
        if state.phone_status == 'READY':
            phone_text = (f'POSSIBLE USE {state.phone_duration:.1f}s' if state.phone_warning else
                          (f'Hand + phone {state.phone_duration:.1f}s' if obs.phone_held else
                           (f'Visible {state.phone_duration:.1f}s' if obs.phone_visible else 'No current evidence')))
        text(f'Phone: {phone_text}', 860, 503, 0.43, AMBER if state.phone_warning else MUTED)
        text('WHY THIS STATE', 860, 532, 0.40, MUTED)
        lines = [line for reason in state.reasons for line in textwrap.wrap(reason, width=48)]
        for i, line in enumerate(lines[:4]):
            text(line, 860, 554 + i * 19, 0.41, color)
        calibration = 'CALIBRATED' if state.calibrated else f'CALIBRATION {state.calibration_progress:.0%}'
        if 'disabled' in state.calibration_message:
            calibration = 'FIXED THRESHOLDS'
        text(calibration, 860, 642, 0.43, TEAL)
        for i, line in enumerate(textwrap.wrap(state.calibration_message, width=48)[:2]):
            text(line, 860, 664 + i * 19, 0.40, MUTED)
        text(f'Original rule: {baseline}', 860, 717, 0.42, MUTED)
        text('EAR trend / last 60 seconds', 26, 713, 0.40, MUTED)
        self.history.append((state.time, state.ear))
        while self.history and self.history[0][0] < state.time - 60:
            self.history.popleft()
        cv2.line(canvas, (25, 759), (810, 759), MUTED, 1)
        threshold_y = int(758 - min(state.close_threshold, 0.5) / 0.5 * 35)
        cv2.line(canvas, (25, threshold_y), (810, threshold_y), AMBER, 1)
        previous = None
        for t, value in self.history:
            if value is None:
                previous = None
                continue
            point = (int(25 + (t - max(0, state.time - 60)) / 60 * 785), int(758 - min(value, 0.5) / 0.5 * 35))
            if previous is not None:
                cv2.line(canvas, previous, point, TEAL, 1, cv2.LINE_AA)
            previous = point
        footer = audio.error or 'Q quit   |   C recalibrate   |   R reselect driver   |   M mute   |   Click a face to select'
        text(footer[:145], 22, 788, 0.43, AMBER if audio.error else MUTED)
        return canvas
