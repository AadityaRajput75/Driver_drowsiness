"""Exercise real face, phone and hand models on a local static test photo.

This is a plumbing check, not a driving-accuracy benchmark.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2

from alerts import AlertPlayer
from config import Config
from dashboard import Dashboard
from monitoring import Monitor, TraditionalBaseline
from phone_detection import PhoneDetector
from reporting import SessionReport
from vision import create_backend, DriverTracker, observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError('Image could not be read')
    config = Config()
    backend = create_backend('mediapipe')
    phone_detector = PhoneDetector(config)
    audio = AlertPlayer(enabled=False)
    report = SessionReport(ROOT / 'artifacts' / 'phone-validation', config,
                           {'source': str(args.image), 'fixture': 'Repeated static public test image',
                            'backend': 'mediapipe', 'phone_detection': phone_detector.last_result.status})
    monitor, baseline = Monitor(config), TraditionalBaseline()
    tracker, dashboard = DriverTracker('all'), Dashboard()
    detected = False
    first_warning = None
    outcome = 'error'
    end_time = 0.0
    try:
        if phone_detector.error:
            raise RuntimeError(phone_detector.error)
        for i in range(40):
            t = i / 10
            started = time.perf_counter()
            faces = backend.detect(frame, t)
            driver = tracker.select(faces, t, frame.shape[1])
            obs = observe(driver, frame, t, config, tracker.reason)
            phone = phone_detector.update(frame, driver, faces, t)
            obs = replace(obs, phone_visible=phone.visible, phone_held=phone.held,
                          phone_checked_at=phone.checked_at, phone_score=phone.score,
                          phone_detection_status=phone.status)
            state = monitor.update(obs)
            old_state = baseline.update(obs)
            report.add(state, obs, old_state, (time.perf_counter() - started) * 1000)
            if state.phone_warning:
                detected = True
                if first_warning is None:
                    first_warning = t
                    preview = dashboard.render(frame, faces, driver, state, obs, 'mediapipe', audio, 10, old_state, phone)
                    cv2.imwrite(str(report.directory / 'phone-warning.png'), preview)
            end_time = t + 0.1
        if not detected:
            raise RuntimeError('No sustained phone/hand warning on this test photo; inspect its framing and the report')
        outcome = 'completed'
        print(f'Phone-use warning confirmed at {first_warning:.1f}s; output: {report.directory}')
    finally:
        audio.close()
        phone_detector.close()
        backend.close()
        report.finish(end_time, {'duration_seconds': 4, 'events': [
            {'label': 'phone_use', 'start': 0, 'end': 4},
            {'label': 'distracted', 'start': 0, 'end': 4}]}, outcome)


if __name__ == '__main__':
    main()
