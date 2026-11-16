"""CPU-only recorded-state renderer geometry tests."""
import importlib.util
import copy
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).parents[1] / "scripts/diagnostics/render_vd12_recorded_play.py"
spec = importlib.util.spec_from_file_location("renderer", MODULE)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class RecordedPlayTests(unittest.TestCase):
    def complete_fixture(self, samples=120):
        duration = samples * .005
        manifest = dict(iteration=2000, num_envs=1, physics_dt_s=.005, control_dt_s=.02,
                        requested_duration_s=duration)
        summary = dict(status="completed_sampled_play", completed_requested_duration=True,
                       initial_episode_only=True, duration_s=duration, requested_duration_s=duration,
                       physics_steps=samples, terminal=dict(causes=dict(time_out=[0], vd12_gun_exclusion=[])))
        rows = [dict(physics_step=i, time_s=i * .005, episode_actor_step=[(i - 1) // 4],
                     joint_position_rad=[[0.] * 18], base_position_w=[[0., 0., .5]],
                     base_quat_wxyz=[[1., 0., 0., 0.]], tcp_position_w=[[0., 0., .5]],
                     tcp_command_world_50hz_zoh=[[0., 0., .5]],
                     minimum_clearance_lower_bound_m=[.02], geometry_input_valid=[True],
                     exclusion_latched=[False]) for i in range(1, samples + 1)]
        return manifest, summary, rows

    def test_quaternion_wxyz(self):
        np.testing.assert_allclose(renderer.quaternion_matrix([1, 0, 0, 0]), np.eye(3))

    def test_fk_uses_logged_joint(self):
        joints = [dict(name="q", kind="revolute", parent="base_link", child="arm",
                       origin=np.eye(4), axis=np.array([0, 0, 1]))]
        fk = renderer.forward_kinematics(joints, ["q"], [np.pi / 2], [1, 2, 3], [1, 0, 0, 0])
        np.testing.assert_allclose(fk["arm"] @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-10)

    def test_missing_joints_rejected(self):
        joints = [dict(name="q", kind="revolute", parent="base_link", child="arm")]
        with self.assertRaises(ValueError):
            renderer.forward_kinematics(joints, [], [], [0, 0, 0], [1, 0, 0, 0])

    def test_wireframes(self):
        for kind, count in [("box", 12), ("cylinder", 6), ("sphere", 3)]:
            lines = renderer.primitive_lines(dict(primitive=kind, half_size_m=[1, 2, 3],
                                                   radius_m=.1, half_length_m=.5))
            self.assertEqual(len(lines), count)
            self.assertTrue(all(line.shape[1] == 3 and np.isfinite(line).all() for line in lines))

    def test_complete_recording_and_exact_video_timebase(self):
        manifest, summary, rows = self.complete_fixture()
        stride = renderer.validate_summary(manifest, summary, 10)
        selected, count = renderer.select_complete_records(rows, manifest, summary, stride)
        self.assertEqual(count, 120)
        self.assertEqual(len(selected), 6)
        np.testing.assert_allclose([row["time_s"] for row in selected], np.arange(1, 7) / 10)

    def test_incompatible_fps_rejected(self):
        manifest, summary, _ = self.complete_fixture()
        for fps in (0, 7, 30, 51, 10.0):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "divisor"):
                renderer.validate_summary(manifest, summary, fps)

    def test_partial_or_failed_summary_rejected(self):
        manifest, summary, _ = self.complete_fixture()
        for key, value in (("status", "rejected"), ("completed_requested_duration", False),
                           ("initial_episode_only", False), ("duration_s", .5)):
            bad = copy.deepcopy(summary)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                renderer.validate_summary(manifest, bad, 10)
        summary["terminal"]["causes"]["vd12_gun_exclusion"] = [0]
        with self.assertRaisesRegex(ValueError, "failure/geometry"):
            renderer.validate_summary(manifest, summary, 10)

    def test_gap_and_reset_rejected(self):
        manifest, summary, rows = self.complete_fixture()
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            renderer.select_complete_records(rows[:30] + rows[31:], manifest, summary, 20)
        rows[40]["episode_actor_step"] = [0]
        with self.assertRaisesRegex(ValueError, "reset/discontinuity"):
            renderer.select_complete_records(rows, manifest, summary, 20)

    def test_exclusion_or_invalid_state_rejected(self):
        manifest, summary, rows = self.complete_fixture()
        for key, value in (("minimum_clearance_lower_bound_m", [-.001]), ("geometry_input_valid", [False]),
                           ("exclusion_latched", [True]), ("joint_position_rad", [[float("nan")] * 18])):
            bad = copy.deepcopy(rows)
            bad[12][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                renderer.select_complete_records(bad, manifest, summary, 20)

    def test_empty_short_and_truncated_logs_rejected(self):
        manifest, summary, rows = self.complete_fixture()
        with self.assertRaisesRegex(ValueError, "empty"):
            renderer.select_complete_records([], manifest, summary, 20)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            renderer.select_complete_records(rows[:-1], manifest, summary, 20)
        manifest, summary, rows = self.complete_fixture(samples=100)
        with self.assertRaisesRegex(ValueError, "six distinct"):
            renderer.select_complete_records(rows, manifest, summary, 20)

    def test_video_warning_does_not_claim_jitter_proof(self):
        self.assertIn("must not be used to establish high-frequency jitter", renderer.VIDEO_USE)


if __name__ == "__main__":
    unittest.main()
