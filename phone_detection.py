"""Local phone + hand detection, associated conservatively with the selected driver."""
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass
class PhoneResult:
    checked_at: float | None = None
    visible: bool = False
    held: bool = False
    score: float | None = None
    boxes: list = field(default_factory=list)
    hand_boxes: list = field(default_factory=list)
    status: str = 'OFF'
    detail: str = 'Phone detection disabled'


def driver_region(box, width, height):
    x, y, w, h = box
    return (max(0, int(x - w)), max(0, int(y - h * 0.3)),
            min(width, int(x + 2 * w)), min(height, int(y + 3.5 * h)))


def hand_near_phone(phone, hand):
    px, py, pw, ph = phone
    hx, hy, hw, hh = hand
    # A small expansion allows for fingertips just beyond a detector box.
    pad = max(8, min(hw, hh) * 0.15)
    intersection = max(0, min(px + pw, hx + hw + pad) - max(px, hx - pad)) * max(
        0, min(py + ph, hy + hh + pad) - max(py, hy - pad))
    return intersection / max(1, min(pw * ph, (hw + pad * 2) * (hh + pad * 2))) >= 0.15


def belongs_to_driver(phone, driver, other_faces):
    px, py, pw, ph = phone
    cx, cy = px + pw / 2, py + ph / 2
    x, y, w, h = driver
    if not (x - w <= cx <= x + 2 * w and y - h * 0.3 <= cy <= y + 3.5 * h):
        return False

    def distance(box):
        bx, by, bw, bh = box
        return hypot((cx - bx - bw / 2) / max(bw, 1), (cy - by - bh) / max(bh, 1))

    own_distance = distance(driver)
    # Reject ambiguous attribution rather than calling a passenger's phone the driver's.
    return all(distance(other) > own_distance + 0.25 for other in other_faces)


def held_phone_evidence(phones, hands, driver, other_faces=()):
    eligible = [(box, score) for box, score in phones if belongs_to_driver(box, driver, other_faces)]
    held = [(box, score) for box, score in eligible if any(hand_near_phone(box, hand) for hand in hands)]
    return bool(eligible), bool(held), max((score for _, score in (held or eligible)), default=None)


class PhoneDetector:
    def __init__(self, config, enabled=True):
        self.config = config
        self.object_detector = self.hand_detector = None
        self.last_result = PhoneResult()
        self.last_ms = -1
        self.error = None
        self.enabled = enabled
        if not enabled:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self.mp = mp
            object_model = ROOT / 'models' / 'efficientdet_lite0.tflite'
            hand_model = ROOT / 'models' / 'hand_landmarker.task'
            if not object_model.is_file() or not hand_model.is_file():
                raise RuntimeError('Phone/hand models missing; see README model setup')
            self.object_detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=str(object_model)),
                running_mode=vision.RunningMode.VIDEO, category_allowlist=['cell phone'],
                score_threshold=config.phone_confidence, max_results=5))
            self.hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO, num_hands=4,
                min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4))
            self.last_result = PhoneResult(status='READY', detail='Phone detection ready')
        except Exception as exc:
            self.error = str(exc)
            self.close()
            self.last_result = PhoneResult(status='UNAVAILABLE', detail=self.error)

    def reset(self):
        if self.enabled and self.error is None:
            self.last_result = PhoneResult(status='READY', detail='Waiting for driver')

    def update(self, frame, driver, faces, timestamp):
        if not self.enabled or self.error is not None:
            return self.last_result
        if driver is None:
            self.last_result = PhoneResult(status='NO DRIVER', detail='Phone attribution needs the selected driver')
            return self.last_result
        if (self.last_result.checked_at is not None
                and timestamp - self.last_result.checked_at < self.config.phone_interval_seconds):
            return self.last_result
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = driver_region(driver.box, w, h)
            crop = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ms = max(self.last_ms + 1, round(timestamp * 1000))
            self.last_ms = ms
            result = self.object_detector.detect_for_video(image, ms)
            phones = []
            for detection in result.detections:
                category = detection.categories[0]
                if category.category_name != 'cell phone' or category.score < self.config.phone_confidence:
                    continue
                box = detection.bounding_box
                phones.append(((box.origin_x + x1, box.origin_y + y1, box.width, box.height), float(category.score)))
            hands = []
            if phones:
                result = self.hand_detector.detect_for_video(image, ms)
                for landmarks in result.hand_landmarks:
                    xs = [p.x * (x2 - x1) + x1 for p in landmarks]
                    ys = [p.y * (y2 - y1) + y1 for p in landmarks]
                    hands.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
            visible, held, score = held_phone_evidence(phones, hands, driver.box,
                                                      [f.box for f in faces if f is not driver])
            self.last_result = PhoneResult(timestamp, visible, held, score, [box for box, _ in phones], hands,
                                           'READY', 'Phone near hand' if held else ('Phone visible' if visible else 'No phone evidence'))
        except Exception as exc:
            self.error = str(exc)
            self.last_result = PhoneResult(status='ERROR', detail=self.error)
            self.close()
        return self.last_result

    def close(self):
        for name in ('object_detector', 'hand_detector'):
            detector = getattr(self, name)
            if detector is not None:
                detector.close()
                setattr(self, name, None)
