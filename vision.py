"""Interchangeable landmark backends, spatial driver tracking and quality gates."""
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from monitoring import Observation
from utils import eye_aspect_ratio, mouth_aspect_ratio

ROOT = Path(__file__).resolve().parent


@dataclass
class Face:
    box: tuple
    eyes: tuple
    mouth: np.ndarray
    pose_points: np.ndarray


def make_face(points, kind, shape):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < (468 if kind == 'mediapipe' else 68):
        return None
    if kind == 'mediapipe':
        left, right = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
        mouth, pose = [78, 82, 312, 308, 317, 87], [1, 152, 33, 263, 61, 291]
    else:
        left, right = [36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]
        mouth, pose = [60, 61, 63, 64, 65, 67], [30, 8, 36, 45, 48, 54]
    if not np.isfinite(points).all():
        return None
    h, w = shape[:2]
    lo = np.maximum(points.min(axis=0), [0, 0])
    hi = np.minimum(points.max(axis=0), [w - 1, h - 1])
    box = (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    if min(box[2:]) <= 0:
        return None
    return Face(box, (points[left], points[right]), points[mouth], points[pose])


class MediaPipeBackend:
    name = 'mediapipe'

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ROOT / 'face_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO, num_faces=3,
            min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1

    def detect(self, frame, timestamp):
        ms = max(self.last_ms + 1, round(timestamp * 1000))
        self.last_ms = ms
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(image, ms)
        h, w = frame.shape[:2]
        faces = []
        for landmarks in result.face_landmarks:
            points = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float64)
            face = make_face(points, self.name, frame.shape)
            if face is not None:
                faces.append(face)
        return faces

    def close(self):
        self.detector.close()


class LBFBackend:
    name = 'lbf'

    def __init__(self):
        if not hasattr(cv2, 'face'):
            raise RuntimeError('LBF requires opencv-contrib-python; install requirements.txt in a clean environment')
        path = ROOT / 'lbfmodel.yaml'
        if not path.is_file():
            raise RuntimeError(f'Missing model: {path}')
        cascade_path = ROOT / 'haarcascade_frontalface_default.xml'
        if not cascade_path.exists():
            cascade_path = Path(cv2.data.haarcascades) / cascade_path.name
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError('OpenCV face detector could not be loaded')
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(path))

    def detect(self, frame, timestamp):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(boxes) == 0:
            return []
        success, landmarks = self.facemark.fit(gray, boxes)
        if not success:
            return []
        return [face for marks in landmarks
                if (face := make_face(marks, self.name, frame.shape)) is not None]

    def close(self):
        pass


def create_backend(name):
    if name == 'lbf':
        return LBFBackend()
    try:
        return MediaPipeBackend()
    except Exception as exc:
        if name != 'auto':
            raise RuntimeError(f'MediaPipe initialization failed: {exc}') from exc
        print(f'[WARNING] MediaPipe unavailable ({exc}); using OpenCV LBF.')
        return LBFBackend()


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(1.0, aw * ah + bw * bh - intersection)


class DriverTracker:
    """Spatial continuity, NOT identity recognition. Ambiguous matches are rejected."""
    def __init__(self, region='center', timeout=2.0):
        self.region = region
        self.timeout = timeout
        self.box = None
        self.last_seen = None
        self.locked_out = False
        self.reason = 'Waiting for a driver'

    def choose(self, face, timestamp):
        self.box = face.box
        self.last_seen = timestamp
        self.locked_out = False

    def select(self, faces, timestamp, width):
        if self.last_seen is not None and timestamp - self.last_seen > self.timeout:
            self.locked_out = True
        if self.locked_out:
            self.reason = 'Driver lost: click the driver or press R to reselect'
            return None
        if self.box is None:
            def in_region(face):
                center = (face.box[0] + face.box[2] / 2) / width
                return (self.region == 'all' or (self.region == 'left' and center < 0.5)
                        or (self.region == 'right' and center >= 0.5)
                        or (self.region == 'center' and 0.25 <= center <= 0.75))
            eligible = [f for f in faces if in_region(f)]
            if len(eligible) != 1:
                self.reason = 'Multiple faces: click the driver' if eligible else 'No driver in the selected region'
                return None
            face = eligible[0]
        else:
            x, y, w, h = self.box
            candidates = []
            for f in faces:
                fx, fy, fw, fh = f.box
                distance = hypot(fx + fw / 2 - x - w / 2, fy + fh / 2 - y - h / 2) / max(1, hypot(w, h))
                size_ratio = fw * fh / max(1, w * h)
                iou = overlap(self.box, f.box)
                if distance < 0.35 and iou > 0.2 and 0.5 < size_ratio < 2.0:
                    candidates.append((iou - distance, f))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.15):
                self.reason = 'Driver tracking uncertain; holding selection'
                return None
            face = candidates[0][1]
        self.choose(face, timestamp)
        self.reason = 'Driver tracked'
        return face


POSE_MODEL = np.array([(0, 0, 0), (0, 330, -65), (-225, -170, -135),
                       (225, -170, -135), (-150, 150, -125), (150, 150, -125)], dtype=np.float64)


def head_pose(points, shape):
    h, w = shape[:2]
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros((4, 1))
    # EPNP supplies a front-of-camera initialization; a cold iterative solve can
    # converge to a mirrored solution behind the camera for angled faces.
    success, rotation, translation = cv2.solvePnP(POSE_MODEL, points, camera, distortion,
                                                 flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return None, None
    success, rotation, translation = cv2.solvePnP(POSE_MODEL, points, camera, distortion,
                                                 rotation, translation, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or translation[2, 0] <= 0:
        return None, None
    projected, _ = cv2.projectPoints(POSE_MODEL, rotation, translation, camera, distortion)
    error = np.linalg.norm(projected.reshape(-1, 2) - points, axis=1).mean()
    eye_width = np.linalg.norm(points[2] - points[3])
    if error > max(8.0, eye_width * 0.15):
        return None, None
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    pitch, yaw = float(angles[0]), float(angles[1])
    if not np.isfinite([pitch, yaw]).all() or abs(pitch) > 80 or abs(yaw) > 80:
        return None, None
    return pitch, yaw


def observe(face, frame, timestamp, config, missing_reason='No driver detected'):
    if face is None:
        return Observation(timestamp, quality_reason=missing_reason)
    left, right = [eye_aspect_ratio(eye) for eye in face.eyes]
    ear = (left + right) / 2 if left is not None and right is not None else None
    mouth = mouth_aspect_ratio(face.mouth)
    pitch, yaw = head_pose(face.pose_points, frame.shape)
    x, y, w, h = [int(v) for v in face.box]
    roi = frame[max(y, 0):y + h, max(x, 0):x + w]
    reasons = []
    if min(w, h) < config.min_face_pixels or roi.size == 0:
        reasons.append('Move closer: face too small')
    if roi.size:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if gray.mean() < config.min_brightness:
            reasons.append('Insufficient lighting')
        if cv2.Laplacian(gray, cv2.CV_64F).var() < config.min_sharpness:
            reasons.append('Image too blurred')
    quality_ok = not reasons
    if pitch is None or yaw is None:
        reasons.append('Head pose unreliable')
    elif abs(pitch) > config.eye_pose_limit or abs(yaw) > config.eye_pose_limit:
        reasons.append('Head angle too large for reliable eye measurements')
    if ear is None or not 0 <= ear <= 0.65:
        reasons.append('Eye landmarks unreliable')
    if any(np.linalg.norm(eye[0] - eye[3]) < 8 for eye in face.eyes):
        reasons.append('Eyes too small to measure')
    if left is not None and right is not None and abs(left - right) > 0.20:
        reasons.append('Eye measurements disagree; possible occlusion')
    mouth_ok = (quality_ok and mouth is not None and 0 <= mouth <= 2
                and np.linalg.norm(face.mouth[0] - face.mouth[3]) >= 10
                and (yaw is None or abs(yaw) < 50))
    pose_ok = quality_ok and pitch is not None and yaw is not None
    return Observation(timestamp, ear, mouth, pitch, yaw, True, quality_ok, not reasons,
                       '; '.join(reasons) if reasons else 'Good visibility',
                       mouth_valid=mouth_ok, pose_valid=pose_ok)
