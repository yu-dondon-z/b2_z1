"""Pure CPU tests for full-pose/full-gun episode-startup witnesses."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vd12_seam_path_cpu", ROOT / "scripts/diagnostics/check_vd12_seam_path.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class StartupMathTests(unittest.TestCase):
    def test_quintic_endpoints_midpoint_and_clipping(self):
        for time, expected in ((-1., 0.), (0., 0.), (1., .5), (2., 1.), (3., 1.)):
            self.assertAlmostEqual(check.quintic_fraction(time, 2.), expected)
        with self.assertRaises(ValueError):
            check.quintic_fraction(1., 0.)

    def test_nlerp_shortest_quaternion_and_not_linear_pitch(self):
        first = Rotation.from_euler("xyz", (0., .3, math.pi)).as_quat()
        second = Rotation.from_euler("xyz", (0., math.pi / 2 - .45, math.pi)).as_quat()
        result = check.nlerp(first, -second, .25)
        np.testing.assert_allclose(result, check.nlerp(first, second, .25), atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(result), 1.)
        pitch = Rotation.from_quat(result).as_euler("xyz")[1]
        self.assertGreater(abs(pitch - (.3 + .25 * (math.pi / 2 - .45 - .3))), 1e-4)

    def test_zoh_base_integration_is_source_clock_not_continuous_ramp(self):
        self.assertEqual(check.zoh_base_x(.02, 2., .2), 0.)
        expected = sum(check.quintic_fraction(k * .02, 2.) for k in range(100)) * .02 * .2
        self.assertAlmostEqual(check.zoh_base_x(2., 2., .2), expected)
        self.assertAlmostEqual(expected, .198)


class FullGunPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = check.SeamPathModel()

    def test_current_source_profile_and_actual_reset_fk(self):
        p = self.model.profile
        self.assertEqual(p["duration_s"], 2.)
        self.assertEqual(p["speed_m_s"], .2)
        self.assertEqual(p["trailing_distance_m"], .8)
        np.testing.assert_allclose(self.model.start_pose[:3, 3], (-.95, 0., .35), atol=1e-8)
        self.assertEqual(self.model.audit.metadata["tube"]["length_m"], 1.1)
        self.assertEqual(len(self.model.nominal_joints), 12)

    def test_target_preserves_measured_start_and_existing_endpoint(self):
        start = self.model.target(0.)
        np.testing.assert_allclose(start["position_root_m"], self.model.start_pose[:3, 3], atol=1e-15)
        np.testing.assert_allclose(start["rotation"], self.model.start_pose[:3, :3], atol=1e-15)
        end = self.model.target(2.)
        expected_height = .005 + .028 * math.cos(math.pi / 2 - .45)
        np.testing.assert_allclose(end["position_root_m"], (-.8, 0., expected_height - .5), atol=1e-14)

    def test_position_rim_correction_uses_current_nlerp_attitude(self):
        row = self.model.target(.8)
        s = check.quintic_fraction(.8, 2.)
        expected = (1 - s) * self.model.start_pose[2, 3] + s * (-.5)
        expected += s * (.005 + self.model.floor_support(row["rotation"][2, 0]))
        self.assertAlmostEqual(row["position_root_m"][2], expected)
        end = self.model.target(2.)
        linear_compensated_end = (1 - s) * self.model.start_pose[2, 3] + s * end["position_root_m"][2]
        self.assertGreater(abs(expected - linear_compensated_end), .001)

    def test_moving_base_keeps_world_start_not_base_relative_start(self):
        fixed, moving = self.model.target(1.), self.model.target(1., .2)
        travel = check.zoh_base_x(1., 2., .2)
        self.assertAlmostEqual(moving["position_root_m"][0] - fixed["position_root_m"][0], -.5 * travel)
        self.assertAlmostEqual(self.model.target(2., .2)["position_root_m"][0], -.8)

    def test_50hz_world_target_hold_between_commands(self):
        first, second = self.model.target(1.005, .2), self.model.target(1.015, .2)
        self.assertEqual(first["command_time_s"], second["command_time_s"])
        np.testing.assert_allclose(first["position_world_m"], second["position_world_m"], atol=1e-15)
        self.assertNotEqual(first["position_root_m"][0], second["position_root_m"][0])

    def test_full_pose_ik_and_whole_gun_independent_geometry_on_101_points(self):
        for speed in (0., .2):
            with self.subTest(speed=speed):
                report = check.validate_path(self.model, points=101, speed_m_s=speed)
                self.assertTrue(report["sampled_path_passed"])
                self.assertEqual(report["geometry_status_counts"], {"safe": 101})
                self.assertGreater(report["min_hard_limit_margin_rad"], .4)
                self.assertGreater(report["min_robot_clearance_lower_bound_m"], .03)
                self.assertGreaterEqual(report["min_finite_tube_floor_clearance_m"], .005 - 1e-8)
                self.assertLess(report["max_position_error_m"], check.POSITION_TOL)
                self.assertLess(report["max_full_orientation_error_rad"], check.ROTATION_TOL)
                self.assertFalse(report["continuous_motion_certified"])
                self.assertLess(report["adjacent_joint_continuity"]["max_absolute_step_rad"], .1)
                json.dumps(report, allow_nan=False)

    def test_exact_tcp_success_does_not_accept_intersecting_whole_gun(self):
        # Existing Isaac env0 at t=2.0 s: TCP itself is above the floor,
        # yet a positive length of the tube centreline passes through the base.
        q = np.array((-.14608973264694214, .0001682684087427333, .0003486844361759722,
                      .7668989896774292, .08212888240814209, -.27894189953804016))
        actual = self.model.forward(q)
        target = self.model.target(2.)
        target["position_root_m"], target["rotation"] = actual[:3, 3], actual[:3, :3]
        result = self.model.evidence(q, target)
        self.assertLess(result["position_error_m"], 1e-12)
        self.assertLess(result["full_orientation_error_rad"], 1e-12)
        self.assertEqual(result["geometry"]["status"], "invalid")
        self.assertFalse(result["accepted"])

    def test_limit_violation_and_axis_only_attitude_are_not_accepted(self):
        target = self.model.target(0.)
        bad_q = self.model.start_q.copy()
        bad_q[1] = -.1
        pose = self.model.forward(bad_q)
        target["position_root_m"], target["rotation"] = pose[:3, 3], pose[:3, :3]
        self.assertFalse(self.model.evidence(bad_q, target)["full_pose_within_hard_limits"])
        target = self.model.target(0.)
        target["rotation"] = target["rotation"] @ Rotation.from_rotvec((.1, 0., 0.)).as_matrix()
        result = self.model.evidence(self.model.start_q, target)
        self.assertAlmostEqual(result["full_orientation_error_rad"], .1)
        self.assertFalse(result["accepted"])

    def test_unsolved_point_stops_without_changing_goal_or_claiming_unreachable(self):
        class UnsolvedFixture:
            profile = {"duration_s": 2.}
            start_q = np.zeros(6)

            def target(self, time_s, speed_m_s):
                return {"time_s": time_s, "rotation": np.eye(3),
                        "position_root_m": np.array((1., 2., 3.))}

            def solve(self, target, previous_q, seed_count):
                return None, [{"accepted": False, "reason": "fixture search unresolved"}]

        result = check.validate_path(UnsolvedFixture(), points=101)
        self.assertFalse(result["sampled_path_passed"])
        self.assertFalse(result["unreachability_proved"])
        self.assertEqual(result["attempted_sample_count"], 1)
        self.assertEqual(result["points"][0]["position_root_m"], [1., 2., 3.])
        self.assertIn("not an unreachability proof", result["stop_reason"])


if __name__ == "__main__":
    unittest.main()
