import unittest

from config import Config
from monitoring import Monitor
from phone_detection import belongs_to_driver, held_phone_evidence
from test_monitoring import observation, feed


class BehaviorRegressionTests(unittest.TestCase):
    def test_yawn_does_not_require_open_or_measurable_eyes(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 1.2, ear=0.05, mouth=0.6, eyes_valid=False, mouth_valid=True)
        self.assertEqual(state.yawns, 1)
        self.assertTrue(state.yawn_active)

    def test_yawn_continues_through_short_quality_dropout(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 0.6, mouth=0.6)
        feed(m, 0.65, 0.75, mouth=0.6, quality_ok=False, mouth_valid=False)
        state = feed(m, 0.8, 1.5, mouth=0.6)
        self.assertEqual(state.yawns, 1)

    def test_long_occlusion_cannot_complete_or_duplicate_yawn(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 1.2, mouth=0.6)
        feed(m, 1.25, 2, mouth=0.6, face_found=False)
        state = feed(m, 2.05, 4, mouth=0.6)
        self.assertEqual(state.yawns, 1)
        feed(m, 4.05, 4.5)
        state = feed(m, 4.55, 5.75, mouth=0.6)
        self.assertEqual(state.yawns, 2)

    def test_short_talking_like_openings_are_not_yawns(self):
        m = Monitor(calibrate=False)
        for i in range(120):
            state = m.update(observation(i * 0.05, mouth=0.6 if i % 16 < 8 else 0.1))
        self.assertEqual(state.yawns, 0)

    def test_yawn_hysteresis_avoids_reset_near_threshold(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 0.5, mouth=0.5)
        state = feed(m, 0.55, 1.2, mouth=0.36)
        self.assertEqual(state.yawns, 1)

    def test_natural_down_up_motion_without_peak_hold_counts(self):
        m = Monitor(calibrate=False)
        angles = [0, 0, 0, 5, 12, 18, 12, 5, 0, 0, 0, 0]
        for i, pitch in enumerate(angles):
            state = m.update(observation(i * 0.1, pitch=pitch))
        self.assertEqual(state.nods, 1)

    def test_nod_survives_a_brief_bad_pose(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 0.3)
        feed(m, 0.35, 0.65, pitch=18)
        m.update(observation(0.70, pitch=18, pose_valid=False))
        feed(m, 0.75, 0.85, pitch=15)
        state = feed(m, 0.90, 1.3)
        self.assertEqual(state.nods, 1)

    def test_nod_cannot_cross_long_tracking_loss(self):
        m = Monitor(calibrate=False)
        feed(m, 0, 0.3)
        feed(m, 0.35, 0.65, pitch=18)
        feed(m, 0.70, 1.2, face_found=False)
        state = feed(m, 1.25, 1.6)
        self.assertEqual(state.nods, 0)

    def test_calibrated_pose_offset_is_removed(self):
        m = Monitor(Config(calibration_seconds=1, calibration_min_samples=10))
        feed(m, 0, 1.2, pitch=-12)
        self.assertTrue(m.calibrated)
        angles = [-12, -7, 0, 6, 0, -7, -12, -12, -12, -12]
        for i, pitch in enumerate(angles):
            state = m.update(observation(1.3 + i * 0.1, pitch=pitch))
        self.assertEqual(state.nods, 1)

    def test_looking_down_warns_even_if_eyes_are_unreliable(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 2.2, pitch=22, eyes_valid=False, pose_valid=True)
        self.assertEqual(state.status, 'DISTRACTED')
        self.assertIn('Looking down for too long', state.reasons)
        self.assertFalse(state.phone_warning)


class PhoneWarningTests(unittest.TestCase):
    def test_sustained_hand_phone_evidence_warns(self):
        m = Monitor(calibrate=False)
        for i in range(31):
            t = i / 20
            state = m.update(observation(t, phone_visible=True, phone_held=True,
                                        phone_checked_at=t, phone_detection_status='READY', eyes_valid=False))
        self.assertTrue(state.phone_warning)
        self.assertEqual(state.status, 'DISTRACTED')
        self.assertEqual(state.alarm_level, 1)

    def test_sustained_phone_warns_even_when_hand_is_hidden(self):
        m = Monitor(calibrate=False)
        for i in range(61):
            t = i / 20
            state = m.update(observation(t, phone_visible=True, phone_checked_at=t, phone_detection_status='READY'))
        self.assertTrue(state.phone_warning)
        self.assertTrue(state.distracted)

    def test_brief_phone_visibility_does_not_warn(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 1.5, phone_visible=True, phone_checked_at=0,
                     phone_detection_status='READY')
        self.assertFalse(state.phone_warning)

    def test_one_missed_check_does_not_erase_phone_progress(self):
        m = Monitor(calibrate=False)
        for i in range(18):
            t = i * 0.2
            state = m.update(observation(t, phone_visible=i != 6,
                                        phone_checked_at=t, phone_detection_status='READY'))
        self.assertTrue(state.phone_warning)

    def test_repeated_isolated_detections_cannot_accumulate(self):
        m = Monitor(calibrate=False)
        for i in range(40):
            t = i * 0.2
            state = m.update(observation(t, phone_visible=i % 2 == 0,
                                        phone_checked_at=t, phone_detection_status='READY'))
        self.assertFalse(state.phone_warning)

    def test_cached_detection_cannot_create_a_phone_warning(self):
        m = Monitor(calibrate=False)
        state = feed(m, 0, 2, phone_held=True, phone_checked_at=0, phone_detection_status='READY')
        self.assertFalse(state.phone_warning)

    def test_missing_phone_evidence_clears_warning(self):
        m = Monitor(calibrate=False)
        for i in range(31):
            t = i / 20
            state = m.update(observation(t, phone_held=True, phone_checked_at=t, phone_detection_status='READY'))
        self.assertTrue(state.phone_warning)
        for t in (1.6, 1.8, 2.0):
            state = m.update(observation(t, phone_checked_at=t, phone_detection_status='READY'))
        self.assertFalse(state.phone_warning)

    def test_phone_needs_driver_and_hand_association(self):
        driver = (200, 100, 100, 120)
        phone = (240, 280, 40, 70)
        hand = (230, 290, 60, 90)
        self.assertEqual(held_phone_evidence([(phone, 0.8)], [hand], driver), (True, True, 0.8))
        self.assertEqual(held_phone_evidence([(phone, 0.8)], [], driver), (True, False, 0.8))

    def test_passenger_phone_is_not_attributed_to_driver(self):
        driver = (200, 100, 100, 120)
        passenger = (370, 100, 100, 120)
        phone = (380, 280, 40, 70)
        self.assertFalse(belongs_to_driver(phone, driver, [passenger]))


if __name__ == '__main__':
    unittest.main()
