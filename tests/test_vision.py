import unittest
import cv2
import numpy as np

from config import Config
from vision import DriverTracker, Face, POSE_MODEL, head_pose, observe, make_face


def face(x=200, y=100, w=160, h=200):
    return Face((x, y, w, h), (np.zeros((6, 2)), np.zeros((6, 2))),
                np.zeros((6, 2)), np.zeros((6, 2)))


class TrackingTests(unittest.TestCase):
    def test_detection_order_changes_do_not_change_driver(self):
        tracker = DriverTracker('all')
        driver = face()
        self.assertIs(tracker.select([driver], 0, 800), driver)
        passenger = face(600)
        moved = face(205)
        self.assertIs(tracker.select([passenger, moved], 0.1, 800), moved)

    def test_multiple_faces_require_selection(self):
        tracker = DriverTracker('all')
        self.assertIsNone(tracker.select([face(), face(600)], 0, 800))

    def test_region_selects_requested_side(self):
        tracker = DriverTracker('right')
        right = face(600)
        self.assertIs(tracker.select([face(), right], 0, 800), right)

    def test_ambiguous_matches_rejected(self):
        tracker = DriverTracker('all')
        tracker.select([face()], 0, 800)
        self.assertIsNone(tracker.select([face(199), face(201)], 0.1, 800))

    def test_driver_not_replaced_after_loss(self):
        tracker = DriverTracker('all')
        tracker.select([face()], 0, 800)
        self.assertIsNone(tracker.select([face(600)], 0.1, 800))
        self.assertIsNone(tracker.select([face()], 3, 800))
        self.assertTrue(tracker.locked_out)
        selected = face()
        tracker.choose(selected, 3.1)
        self.assertIs(tracker.select([selected], 3.1, 800), selected)

    def test_short_gap_reacquires_spatial_match(self):
        tracker = DriverTracker()
        tracker.select([face()], 0, 800)
        tracker.select([], 0.1, 800)
        returned = face(205)
        self.assertIs(tracker.select([returned], 0.2, 800), returned)


class PoseTests(unittest.TestCase):
    def test_opencv_landmark_layouts_are_supported(self):
        points = np.arange(136, dtype=float).reshape(68, 2)
        for shape in ((1, 68, 2), (68, 1, 2), (68, 2)):
            result = make_face(points.reshape(shape), 'lbf', (600, 800, 3))
            self.assertIsNotNone(result)
            self.assertEqual(result.eyes[0].shape, (6, 2))

    def test_known_projection_recovers_downward_pitch(self):
        camera = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=float)
        for degrees in (-20, 0, 20):
            rvec = np.array([np.deg2rad(degrees), 0, 0], dtype=float)
            points, _ = cv2.projectPoints(POSE_MODEL, rvec, np.array([0, 0, 1300.], dtype=float), camera, np.zeros(4))
            pitch, yaw = head_pose(points.reshape(-1, 2), (600, 800, 3))
            self.assertAlmostEqual(pitch, degrees, places=3)
            self.assertAlmostEqual(yaw, 0, places=3)

    def test_missing_face_produces_unknown_observation(self):
        obs = observe(None, np.zeros((600, 800, 3), dtype=np.uint8), 0, Config())
        self.assertFalse(obs.face_found)
        self.assertFalse(obs.eyes_valid)
        self.assertIsNone(obs.ear)


if __name__ == '__main__':
    unittest.main()
