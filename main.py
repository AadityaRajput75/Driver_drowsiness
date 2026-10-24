"""Personalized driver monitoring with live dashboard and reproducible video replay."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import time

from config import load_config, save_config
from monitoring import Monitor, TraditionalBaseline
from reporting import SessionReport, load_annotations

ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    source.add_argument('--video', type=Path, help='Replay a local constant-frame-rate video')
    parser.add_argument('--backend', choices=['auto', 'mediapipe', 'lbf'], default='auto')
    parser.add_argument('--driver-region', choices=['center', 'left', 'right', 'all'], default='center',
                        help='Initial driver location in the unmirrored camera image')
    parser.add_argument('--config', type=Path, help='JSON settings; unspecified settings keep defaults')
    parser.add_argument('--write-config', type=Path, help='Write default settings and exit')
    parser.add_argument('--no-calibration', action='store_true', help='Use fixed thresholds for controlled comparisons')
    parser.add_argument('--no-phone-detection', action='store_true', help='Disable phone/hand models; retain head-distraction warnings')
    parser.add_argument('--headless', action='store_true', help='Replay without displaying a window')
    sound = parser.add_mutually_exclusive_group()
    sound.add_argument('--mute', action='store_true', help='Disable sound (visual alerts remain)')
    sound.add_argument('--audio', action='store_true', help='Enable sound during video replay')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports')
    parser.add_argument('--no-report', action='store_true', help='Do not write session measurements')
    parser.add_argument('--annotations', type=Path, help='Exhaustive JSON event labels for video evaluation')
    parser.add_argument('--video-fps', type=float, help='Override video frame rate / timestamp scale')
    args = parser.parse_args(argv)
    if args.headless and not args.video:
        parser.error('--headless requires --video; live monitoring needs a visible dashboard')
    if args.annotations and (not args.video or args.no_report):
        parser.error('--annotations requires --video and session reporting')
    if args.video_fps is not None and (not args.video or not math.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error('--video-fps requires a video and a finite positive frame rate')
    if args.camera < 0:
        parser.error('--camera must be nonnegative')
    return args


def run(args):
    config = load_config(args.config)
    annotations = load_annotations(args.annotations) if args.annotations else None
    if args.video and not args.video.is_file():
        raise ValueError(f'Video not found: {args.video}')
    import cv2
    cv2.setNumThreads(2)
    from alerts import AlertPlayer
    from dashboard import Dashboard
    from vision import DriverTracker, create_backend, observe
    from phone_detection import PhoneDetector

    backend = cap = audio = report = phone_detector = None
    frame_count = 0
    timestamp = 0.0
    end_time = 0.0
    video_fps = 0.0
    outcome = 'error'
    try:
        backend = create_backend(args.backend)
        cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        if not cap.isOpened():
            raise RuntimeError('Could not open video source. Check the path or camera index/permissions.')
        if args.video:
            video_fps = args.video_fps or cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(video_fps) or video_fps <= 0:
                raise ValueError('Video has no usable frame rate. Supply --video-fps.')
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if args.video else 0
        phone_detector = PhoneDetector(config, enabled=not args.no_phone_detection)
        if phone_detector.error:
            print(f'[WARNING] Phone detection unavailable: {phone_detector.error}')
        audio = AlertPlayer(enabled=not args.mute and (not args.video or args.audio),
                            warning_repeat=config.warning_repeat_seconds)
        if audio.error:
            print(f'[WARNING] {audio.error}')
        monitor = Monitor(config, calibrate=not args.no_calibration)
        baseline = TraditionalBaseline()
        tracker = DriverTracker(args.driver_region, config.track_timeout_seconds)
        dashboard = Dashboard()
        if not args.no_report:
            report = SessionReport(args.report_dir, config, {
                'backend': backend.name, 'source': str(args.video.resolve()) if args.video else f'camera:{args.camera}',
                'video_fps': video_fps or None, 'calibration_enabled': not args.no_calibration,
                'driver_region': args.driver_region,
                'phone_detection': phone_detector.last_result.status,
                'clock': 'frame index / video FPS (constant-frame-rate replay)' if args.video else 'monotonic capture time',
                'video_saved': False})
        if not args.headless:
            cv2.namedWindow(dashboard.title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(dashboard.title, dashboard.mouse)
        print(f'[INFO] Backend: {backend.name}. Calibration: {not args.no_calibration}.')
        print('[INFO] Look forward with naturally open eyes for calibration. Q quits; C recalibrates; R reselects; M mutes.')
        print('[INFO] Video is processed locally. Reports contain measurements, not images.')
        origin = time.monotonic()
        previous_capture = None
        previous_status = None
        next_display = time.monotonic()
        while True:
            ok, frame = cap.read()
            captured = time.monotonic()
            if not ok:
                if not args.video:
                    raise RuntimeError('Camera stopped delivering frames; monitoring ended.')
                if frame_count == 0 or (math.isfinite(total_frames) and total_frames > frame_count + 1):
                    raise RuntimeError(f'Video decoding stopped early after {frame_count} frames.')
                outcome = 'completed'
                break
            timestamp = frame_count / video_fps if args.video else captured - origin
            processing_start = time.perf_counter()
            faces = backend.detect(frame, timestamp)
            if dashboard.pending_click is not None:
                px, py = dashboard.pending_click
                dashboard.pending_click = None
                clicked = [f for f in faces if f.box[0] <= px <= f.box[0] + f.box[2]
                           and f.box[1] <= py <= f.box[1] + f.box[3]]
                if len(clicked) == 1:
                    tracker.choose(clicked[0], timestamp)
                    monitor = Monitor(config, calibrate=not args.no_calibration)
                    baseline = TraditionalBaseline()
                    dashboard.history.clear()
                    phone_detector.reset()
                    if report:
                        report.action(timestamp, 'Driver manually selected; calibration and history reset')
            selected = tracker.select(faces, timestamp, frame.shape[1])
            obs = observe(selected, frame, timestamp, config, tracker.reason)
            phone = phone_detector.update(frame, selected, faces, timestamp)
            obs = replace(obs, phone_visible=phone.visible, phone_held=phone.held,
                          phone_checked_at=phone.checked_at, phone_detection_status=phone.status,
                          phone_score=phone.score)
            state = monitor.update(obs)
            baseline_status = baseline.update(obs)
            audio.update(state.alarm_level)
            processing_ms = (time.perf_counter() - processing_start) * 1000
            fps = 1 / (captured - previous_capture) if previous_capture is not None and captured > previous_capture else 0
            previous_capture = captured
            if report:
                report.add(state, obs, baseline_status, processing_ms)
            frame_count += 1
            end_time = frame_count / video_fps if args.video else timestamp
            if state.status != previous_status:
                print(f'[{timestamp:7.2f}s] {state.status}: {"; ".join(state.reasons)}')
                previous_status = state.status
            if not args.headless:
                display = dashboard.render(frame, faces, selected, state, obs, backend.name, audio, fps, baseline_status, phone)
                cv2.imshow(dashboard.title, display)
                delay = 1
                if args.video:
                    next_display += 1 / video_fps
                    delay = max(1, min(1000, round((next_display - time.monotonic()) * 1000)))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(dashboard.title, cv2.WND_PROP_VISIBLE) < 1:
                    outcome = 'stopped_by_user'
                    break
                if key in (ord('c'), ord('r')):
                    # Recalibration must not dismiss an existing closure alert.
                    latch = monitor.latched_alert
                    monitor = Monitor(config, calibrate=not args.no_calibration if key == ord('r') else True)
                    monitor.latched_alert = latch
                    dashboard.history.clear()
                    phone_detector.reset()
                    if key == ord('r'):
                        tracker = DriverTracker(args.driver_region, config.track_timeout_seconds)
                        baseline = TraditionalBaseline()
                    if report:
                        report.action(timestamp, 'Reselect driver' if key == ord('r') else 'Recalibrate')
                elif key == ord('m'):
                    audio.toggle()
                    if report:
                        report.action(timestamp, 'Audio enabled' if audio.enabled else 'Audio muted')
        return 0
    except KeyboardInterrupt:
        outcome = 'interrupted'
        print('[INFO] Interrupted; saving available results.')
        return 130
    finally:
        if audio is not None:
            audio.close()
        if phone_detector is not None:
            phone_detector.close()
        if cap is not None:
            cap.release()
        if backend is not None:
            backend.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if report is not None:
            directory = report.finish(end_time, annotations, outcome, tolerance=max(0.1, 2 / video_fps) if video_fps else 0.1)
            print(f'[REPORT] {directory.resolve()}')
            if annotations and (outcome != 'completed' or abs(end_time - annotations['duration_seconds']) > max(0.1, 2 / video_fps)):
                print('[WARNING] Evaluation skipped: incomplete replay or annotation duration mismatch. See summary.json.')


def main(argv=None):
    args = arguments(argv)
    try:
        if args.write_config:
            save_config(args.write_config)
            print(f'Default settings written to {args.write_config.resolve()}')
            return 0
        return run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
