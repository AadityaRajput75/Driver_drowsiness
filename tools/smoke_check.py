"""Create a deterministic replay fixture and dashboard preview from a public portrait.

Usage: python tools/smoke_check.py --portrait artifacts/mediapipe-portrait.jpg
This exercises plumbing and visibility transitions, not real drowsiness accuracy.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from alerts import AlertPlayer
from config import Config
from dashboard import Dashboard
from monitoring import Monitor, TraditionalBaseline
from vision import create_backend, observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portrait', required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    image = cv2.imread(str(args.portrait))
    if image is None:
        raise ValueError('Could not read portrait')
    # Letterbox without changing facial geometry.
    scale = min(800 / image.shape[1], 600 / image.shape[0])
    scaled = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    frame = np.full((600, 800, 3), 24, dtype=np.uint8)
    x, y = (800 - scaled.shape[1]) // 2, (600 - scaled.shape[0]) // 2
    frame[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    output = ROOT / 'artifacts' / 'smoke'
    output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output / 'visibility-replay.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (800, 600))
    if not writer.isOpened():
        raise RuntimeError('MJPG video writer unavailable')
    try:
        for i in range(240):
            if 100 <= i < 106:
                sample = np.zeros_like(frame)
            elif 206 <= i < 216:
                sample = (frame.astype(float) * 0.08).astype(np.uint8)
            else:
                sample = frame
            writer.write(sample)
    finally:
        writer.release()
    (output / 'annotations.json').write_text(json.dumps({'duration_seconds': 24, 'events': []}, indent=2), encoding='utf-8')
    backend = create_backend('mediapipe')
    audio = AlertPlayer(enabled=False)
    try:
        faces = backend.detect(frame, 0)
        if len(faces) != 1:
            raise RuntimeError(f'Expected one public test face, got {len(faces)}')
        obs = observe(faces[0], frame, 0, Config())
        if not obs.eyes_valid:
            raise RuntimeError(f'Public test face quality failed: {obs.quality_reason}')
        monitor = Monitor()
        baseline = TraditionalBaseline()
        dashboard = Dashboard()
        for i in range(301):
            current = replace(obs, time=i / 10)
            state = monitor.update(current)
            original = baseline.update(current)
            display = dashboard.render(frame, faces, faces[0], state, current, backend.name, audio, 10, original)
        if not state.calibrated:
            raise RuntimeError('Calibration did not complete on stable public portrait')
        cv2.imwrite(str(output / 'dashboard-preview.png'), display)
        # Also inspect maximum-length reason strings and the missing-face state.
        lost = monitor.update(replace(current, time=30.1, face_found=False, quality_ok=False,
                                      eyes_valid=False, quality_reason='Driver tracking uncertain; holding selection'))
        missing = dashboard.render(np.zeros_like(frame), [], None, lost, current, backend.name, audio, 10, 'AWAKE')
        cv2.imwrite(str(output / 'dashboard-missing.png'), missing)
        print(f'Created smoke fixtures and previews in {output}')
        print(f'Calibrated EAR: close={state.close_threshold:.3f}, reopen={state.open_threshold:.3f}')
    finally:
        backend.close()
        audio.close()


if __name__ == '__main__':
    main()
