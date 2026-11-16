"""CPU-only frame/sign and validity checks for recorded posture review."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/diagnostics'))
from review_vd14_play import distribution, posture, rotation_wxyz


class RecordedPosture(unittest.TestCase):
    def fixture(self):
        feet = np.array([[[.4, .25, -.5], [.4, -.25, -.5],
                          [-.4, .25, -.5], [-.4, -.25, -.5]]])
        hips = np.array([[[.3, .1, 0.], [.3, -.1, 0.],
                          [-.3, .1, 0.], [-.3, -.1, 0.]]])
        return feet, hips

    def test_yaw_does_not_change_body_width_or_level_height(self):
        feet, hips = self.fixture()
        rot = rotation_wxyz([2**-.5, 0., 0., 2**-.5])
        result = posture(feet, hips, rot[None])
        self.assertAlmostEqual(result['front_width_m']['median'], .5)
        self.assertAlmostEqual(result['front_above_rear_hip_m']['median'], 0.)
        self.assertEqual(result['front_width_below_0p42_fraction'], 0.)

    def test_pitch_height_gap_sign_and_angle(self):
        feet, hips = self.fixture()
        angle = -.2  # Negative body pitch raises front hips.
        rot = rotation_wxyz([np.cos(angle/2), 0., np.sin(angle/2), 0.])
        result = posture(feet, hips, rot[None])
        self.assertAlmostEqual(result['front_above_rear_hip_m']['median'], .6*np.sin(.2))
        self.assertAlmostEqual(result['inclination_deg']['median'], np.degrees(.2))
        self.assertAlmostEqual(result['abs_pitch_deg']['median'], np.degrees(.2))

    def test_crossed_feet_not_hidden_by_absolute_distance(self):
        feet, hips = self.fixture()
        feet[:, [0, 1]] = feet[:, [1, 0]]
        result = posture(feet, hips, np.eye(3)[None])
        self.assertAlmostEqual(result['front_width_m']['median'], -.5)
        self.assertEqual(result['front_width_below_0p42_fraction'], 1.)

    def test_empty_and_invalid_windows_cannot_report_zero_quality(self):
        self.assertIsNone(distribution([]))
        feet, hips = self.fixture()
        for args in ((feet[:0], hips[:0], np.empty((0, 3, 3))),
                     (feet, hips[:, :2], np.eye(3)[None]),
                     (feet * np.nan, hips, np.eye(3)[None])):
            with self.assertRaises(ValueError):
                posture(*args)

    def test_quaternion_order_and_invalid_norm(self):
        np.testing.assert_allclose(rotation_wxyz([1., 0., 0., 0.]), np.eye(3))
        for q in ([0., 0., 0., 0.], [2., 0., 0., 0.], [float('nan'), 0., 0., 1.], [1., 0., 0.]):
            with self.assertRaises(ValueError):
                rotation_wxyz(q)


if __name__ == '__main__':
    unittest.main()
