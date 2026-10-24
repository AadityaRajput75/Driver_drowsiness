import math
import unittest

from config import Config
from monitoring import Monitor, Observation, TraditionalBaseline
from utils import eye_aspect_ratio


def observation(t, ear=0.30, **kwargs):
    values = dict(time=t, ear=ear, mouth=0.10, pitch=0.0, yaw=0.0,
                  face_found=True, quality_ok=True, eyes_valid=True, quality_reason='Good')
    values.update(kwargs)
    return Observation(**values)


def feed(monitor, start, end, ear=0.30, fps=20, **kwargs):
    for i in range(round((end - start) * fps) + 1):
        state = monitor.update(observation(round(start + i / fps, 6), ear, **kwargs))
    return state


class MonitorTests(unittest.TestCase):
    def test_prolonged_closure_and_recovery(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 2.1, 0.1)
        self.assertEqual(state.status, 'DROWSY')
        self.assertEqual(m.update(observation(2.15)).status, 'DROWSY')
        self.assertEqual(feed(m, 2.2, 3.1).status, 'MONITORING')

    def test_lost_face_never_clears_alert(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 2.1, 0.1)
        state = m.update(Observation(2.2))
        self.assertEqual(state.status, 'DROWSY')
        self.assertEqual(state.visibility, 'TRACKING LOST')
        self.assertEqual(state.alarm_level, 2)
        self.assertIsNone(state.ear)

    def test_missing_data_has_its_own_state(self):
        m = Monitor(calibrate=False)
        self.assertEqual(m.update(Observation(0)).status, 'TRACKING LOST')
        self.assertEqual(m.update(Observation(3.1)).alarm_level, 1)

    def test_gap_does_not_count_as_closed_time(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 1.0, 0.1)
        state = m.update(observation(5, 0.1))
        self.assertNotEqual(state.status, 'DROWSY')
        self.assertEqual(state.closed_seconds, 0)
        self.assertLessEqual(state.observed_seconds, 1.01)

    def test_invalid_frame_breaks_closure(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 1, 0.1)
        m.update(observation(1.05, 0.1, quality_ok=False))
        state = feed(m, 1.1, 2.1, 0.1)
        self.assertNotEqual(state.status, 'DROWSY')
        self.assertAlmostEqual(state.closed_seconds, 1)

    def test_hysteresis_does_not_chatter(self):
        m = Monitor(calibrate=False)
        m.update(observation(0, 0.1))
        state = feed(m, 0.05, 2.1, 0.23)
        self.assertEqual(state.status, 'DROWSY')
        self.assertEqual(state.blinks, 0)

    def test_normal_blinks_do_not_alarm(self):
        m = Monitor(calibrate=False)
        for i in range(120):
            state = m.update(observation(i / 20, 0.1 if i % 40 < 3 else 0.3))
            self.assertNotEqual(state.status, 'DROWSY')
        self.assertEqual(state.blinks, 3)
        self.assertEqual(state.long_blinks, 0)

    def test_repeated_long_blinks_warn(self):
        m = Monitor(calibrate=False)
        for i in range(100):
            state = m.update(observation(i / 20, 0.1 if i % 30 < 12 else 0.3))
        self.assertEqual(state.long_blinks, 3)
        self.assertEqual(state.fatigue, 'ELEVATED')

    def test_calibration_personalizes_thresholds(self):
        m = Monitor(Config(calibration_seconds=2, calibration_min_samples=10))
        state = feed(m, 0, 2.1, 0.32)
        self.assertTrue(state.calibrated)
        self.assertAlmostEqual(state.close_threshold, 0.32 * 0.70)
        self.assertAlmostEqual(state.open_threshold, 0.32 * 0.82)

    def test_small_open_eyes_do_not_trigger_during_calibration(self):
        m = Monitor()
        state = feed(m, 0, 8.2, 0.20)
        self.assertTrue(state.calibrated)
        self.assertEqual(state.status, 'MONITORING')
        self.assertFalse(m.latched_alert)

    def test_severe_closure_still_alerts_during_calibration(self):
        state = feed(Monitor(), 0, 2.1, 0.08)
        self.assertEqual(state.status, 'DROWSY')

    def test_held_head_tilt_is_not_a_nod(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 8, pitch=25)
        state = m.update(observation(8.05))
        self.assertEqual(state.nods, 0)

    def test_calibration_rejects_poor_or_closed_eyes(self):
        for kwargs in ({'quality_ok': False}, {'yaw': 40}, {'ear': 0.08}):
            m = Monitor(Config(calibration_seconds=1, calibration_min_samples=5))
            state = feed(m, 0, 3, **kwargs)
            self.assertFalse(state.calibrated)

    def test_calibration_rejects_unstable_samples(self):
        m = Monitor(Config(calibration_seconds=1, calibration_min_samples=5))
        for i in range(25):
            state = m.update(observation(i / 20, 0.18 if i % 2 else 0.4))
        self.assertFalse(state.calibrated)

    def test_time_weighted_window_is_frame_rate_independent(self):
        values = []
        for fps in (10, 20, 30):
            m = Monitor(Config(window_seconds=10, min_observed_seconds=5), calibrate=False)
            for i in range(10 * fps + 1):
                t = i / fps
                state = m.update(observation(t, 0.1 if 2 <= t < 5 else 0.3))
            values.append(state.closure_fraction)
        for value in values:
            self.assertAlmostEqual(value, 0.3, places=5)

    def test_insufficient_coverage_suppresses_fraction(self):
        m = Monitor(Config(window_seconds=10, min_observed_seconds=2), calibrate=False)
        feed(m, 0, 3)
        state = m.update(Observation(10))
        self.assertIsNone(state.closure_fraction)
        self.assertLess(state.coverage, 0.7)

    def test_distraction_is_not_drowsiness(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 2.1, yaw=32)
        self.assertTrue(state.distracted)
        self.assertEqual(state.status, 'DISTRACTED')
        self.assertNotEqual(state.fatigue, 'HIGH')

    def test_one_sustained_mouth_opening_counts_once(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 6, mouth=0.6)
        self.assertEqual(state.yawns, 1)
        self.assertNotEqual(state.status, 'DROWSY')

    def test_nod_requires_downward_motion_and_return(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 0.3)
        state = feed(m, 0.35, 1.5, pitch=23)
        self.assertEqual(state.nods, 0)
        state = feed(m, 1.55, 2.0)
        self.assertEqual(state.nods, 1)
        state = feed(m, 2.1, 3, pitch=-25)
        state = m.update(observation(3.05))
        self.assertEqual(state.nods, 1)

    def test_repeated_yawns_and_nods_support_warning(self):
        m = Monitor(Config(repeated_events=2), calibrate=False)
        feed(m, 0, 0.3)
        feed(m, 0.35, 1.95, mouth=0.6, pitch=20)
        feed(m, 2.0, 2.4)
        feed(m, 2.45, 4.05, mouth=0.6, pitch=20)
        state = feed(m, 4.1, 4.5)
        self.assertEqual((state.yawns, state.nods), (2, 2))
        self.assertEqual(state.status, 'WARNING')

    def test_bad_timestamps_and_nonfinite_ear(self):
        m = Monitor(calibrate=False)
        state = m.update(observation(0, math.nan))
        self.assertEqual(state.visibility, 'POOR VISIBILITY')
        with self.assertRaises(ValueError):
            m.update(observation(0))
        with self.assertRaises(ValueError):
            m.update(observation(float('inf')))

    def test_baseline_reproduces_original_missing_face_behavior(self):
        baseline = TraditionalBaseline()
        baseline.update(observation(0, 0.1))
        self.assertEqual(baseline.update(observation(2.1, 0.1)), 'DROWSY')
        self.assertEqual(baseline.update(Observation(2.2)), 'AWAKE')

    def test_degenerate_eye_is_unknown(self):
        self.assertIsNone(eye_aspect_ratio([(0, 0)] * 6))
        self.assertIsNone(eye_aspect_ratio([(math.nan, 0)] * 6))

    def test_invalid_configuration_is_rejected(self):
        for config in (Config(ear_threshold=-1), Config(window_seconds=2), Config(repeated_events=2.5),
                       Config(closure_alert=0.1), Config(min_coverage=2), Config(closed_seconds=math.inf)):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()
