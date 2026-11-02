"""CPU checks of the real capture contract and external visual feedback."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands/dual_vision_contract.py"
spec = importlib.util.spec_from_file_location("dual_contract", SOURCE)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class DualControlTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(sequence=8, capture_generation=3, capture_timestamp=20.0)
        self.path = np.column_stack((np.linspace(-1, -.55, 20), np.zeros(20), np.full(20, -.5)))
        self.packet = dict(self.snapshot, valid=True, points_b=self.path.tolist(),
                           tcp_measured_b=[-.775, .04, -.41], ground_plane_b=[0, 0, 1, .5])

    def test_late_publication_cannot_refresh_capture_age(self):
        self.packet["timestamp"] = 100000
        with self.assertRaisesRegex(ValueError, "expired"):
            control.validate_stream(self.packet, self.snapshot, 3, 21.01, 1, rear=True)

    def test_changed_capture_and_reset_are_rejected(self):
        for field, value in (("sequence", 9), ("capture_timestamp", 20.1), ("capture_generation", 4)):
            packet = dict(self.packet, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.validate_stream(packet, self.snapshot, 3, 20.2, 1, rear=True)

    def test_real_capture_is_accepted(self):
        result = control.validate_stream(self.packet, self.snapshot, 3, 20.2, 1, rear=True)
        np.testing.assert_allclose(result["tcp_measured_b"], [-.775, .04, -.41])

    def test_invalid_tcp_and_path_cannot_reach_control(self):
        for field, value in (("tcp_measured_b", [0, float("nan"), 0]),
                             ("points_b", [[0, 0, 0]] * 3), ("ground_plane_b", [1, 0, 0, 1])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.validate_stream(dict(self.packet, **{field: value}), self.snapshot, 3, 20.2, 1, rear=True)

    def target(self, measured):
        return control.rear_target(self.path, [0, 0, 1, .5], measured, [0, 0, 1],
                                   (-1, -.55), .09, .35, .03)

    def test_observed_tcp_changes_command_in_correct_direction(self):
        left = self.target([-.775, .04, -.41])
        right = self.target([-.775, -.04, -.41])
        self.assertLess(left[0][1], 0)
        self.assertGreater(right[0][1], 0)
        self.assertAlmostEqual(left[3], .04)
        self.assertAlmostEqual(right[3], .04)

    def test_tcp_height_and_projection_error_are_separate(self):
        low = self.target([-.775, 0, -.49])
        high = self.target([-.775, 0, -.2])
        self.assertAlmostEqual(low[3], 0)
        self.assertAlmostEqual(high[3], 0)
        self.assertGreater(low[2][2], 0)
        self.assertLess(high[2][2], 0)

    def test_large_visual_error_has_bounded_residual(self):
        result = self.target([-.775, 10, 2])
        self.assertLessEqual(np.linalg.norm(result[2]), .030000001)

    def test_invisible_rear_workspace_is_not_extrapolated(self):
        with self.assertRaisesRegex(ValueError, "workspace"):
            control.rear_target(self.path + [2, 0, 0], [0, 0, 1, .5], [0, 0, 0],
                                [0, 0, 1], (-1, -.55), .09, .35, .03)


if __name__ == "__main__":
    unittest.main()
