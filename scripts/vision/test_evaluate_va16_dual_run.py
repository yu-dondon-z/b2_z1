"""Check that report acceptance cannot hide missing or excessive TCP error."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_va16_dual_run import evaluate_run


class RunEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="va16_report_test_")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def statuses(self, rows):
        (self.directory / "controller_history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_no_active_tracking_fails_with_json_null_instead_of_nan(self):
        self.statuses([{"simulation_time_s": 1., "tool_target_active": False,
            "evaluation_only_tcp_projection_to_physical_crack_m": .2}])
        report = evaluate_run(self.directory)
        self.assertFalse(report["pass"])
        self.assertEqual(report["active_ratio_by_recorded_samples"], 0)
        self.assertIsNone(report["tcp_ground_projection_error_m"]["active"]["max"])
        json.dumps(report, allow_nan=False)

    def test_six_cm_active_error_fails_even_when_average_is_small(self):
        self.statuses([{"simulation_time_s": float(index), "tool_target_active": True,
            "evaluation_only_tcp_projection_to_physical_crack_m": error}
            for index, error in enumerate([.01, .01, .06])])
        report = evaluate_run(self.directory)
        self.assertFalse(report["pass"])
        self.assertAlmostEqual(report["tcp_ground_projection_error_m"]["active"]["max"], .06)

    def test_preparation_error_stays_global_but_is_excluded_from_tracking_score(self):
        self.statuses([{"simulation_time_s": float(index), "tool_target_active": active,
            "evaluation_only_tcp_projection_to_physical_crack_m": error}
            for index, (active, error) in enumerate([(False, .3), (True, .04), (True, .05)])])
        report = evaluate_run(self.directory)
        self.assertTrue(report["pass"])
        self.assertEqual(report["tracking_start_delay_from_first_record_s"], 1.)
        self.assertEqual(report["tcp_ground_projection_error_m"]["active"]["max"], .05)
        self.assertEqual(report["tcp_ground_projection_error_m"]["all_including_preparation"]["max"], .3)

    def test_actual_rate_uses_capture_clock_instead_of_nominal_twenty_five_hz(self):
        self.statuses([{"simulation_time_s": 0., "tool_target_active": True,
            "evaluation_only_tcp_projection_to_physical_crack_m": .02}])
        packets = [{stream: {"capture_generation": 1, "sequence": sequence,
            "capture_timestamp": timestamp, "valid": True} for stream in ("front", "rear")}
            for sequence, timestamp in [(0, 10.), (2, 10.5), (4, 11.)]]
        (self.directory / "dual_vision_packets.jsonl").write_text("".join(json.dumps(packet) + "\n" for packet in packets))
        report = evaluate_run(self.directory)
        rate = report["vision"]["streams"]["front"]["effective_rate"]
        self.assertEqual(rate["processed_frames_per_wall_s"], 2.)
        self.assertEqual(rate["exported_sequence_intervals_per_wall_s"], 4.)

    def test_missing_independent_error_on_active_sample_cannot_pass(self):
        self.statuses([{"simulation_time_s": 0., "tool_target_active": True}])
        self.assertFalse(evaluate_run(self.directory)["pass"])

    def test_full_rate_trace_catches_peak_between_controller_records(self):
        geometry = {"amplitude_m": 0., "wavelength_m": 20., "curve_half_length_m": 20.,
                    "road_half_length_m": 95., "mesh_step_m": .05, "env_origin_w": [0., 0., 0.]}
        self.statuses([{"simulation_time_s": timestamp, "tool_target_active": True,
            "evaluation_only_tcp_projection_to_physical_crack_m": .01,
            "evaluation_scene_geometry": geometry} for timestamp in (0., .06)])
        with (self.directory / "motion_trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["time_s", "actual_x", "actual_y", "base_x", "base_y"])
            writer.writeheader()
            writer.writerows([{"time_s": timestamp, "actual_x": 0., "actual_y": error,
                               "base_x": 0., "base_y": 0.}
                              for timestamp, error in ((0., .01), (.02, .08), (.04, .01))])
        report = evaluate_run(self.directory)
        self.assertFalse(report["pass"])
        self.assertEqual(report["tcp_ground_projection_error_m"]["controller_active"]["max"], .01)
        self.assertEqual(report["tcp_ground_projection_error_m"]["active"]["max"], .08)
        self.assertEqual(report["tcp_ground_projection_error_m"]["criterion_source"], "full-rate motion trace")

    def test_exact_trace_active_flag_excludes_preparation_and_reports_foot_centre_margin(self):
        geometry = {"amplitude_m": 0., "wavelength_m": 20., "curve_half_length_m": 20.,
                    "road_half_length_m": 95., "mesh_step_m": .05, "env_origin_w": [0., 0., 0.]}
        self.statuses([{"simulation_time_s": timestamp, "tool_target_active": False,
            "evaluation_only_tcp_projection_to_physical_crack_m": .3,
            "evaluation_scene_geometry": geometry} for timestamp in (.02, .06)])
        fields = ["time_s", "actual_x", "actual_y", "base_x", "base_y", "visual_target_active"]
        fields.extend(f"foot_{label}_{field}" for label in ("FL", "FR", "RL", "RR")
                      for field in ("x_w", "y_w", "z_w", "contact_force_n"))
        with (self.directory / "motion_trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for timestamp, error, active in ((0., .3, 0), (.02, .02, 1), (.04, .03, 1)):
                row = {"time_s": timestamp, "actual_x": 0., "actual_y": error,
                       "base_x": 0., "base_y": 0., "visual_target_active": active}
                row.update({f"foot_{label}_{field}": value for label in ("FL", "FR", "RL", "RR")
                            for field, value in (("x_w", .2), ("y_w", .02), ("z_w", .03), ("contact_force_n", 25.))})
                writer.writerow(row)
        report = evaluate_run(self.directory)
        self.assertTrue(report["pass"])
        self.assertEqual(report["active_samples"], 2)
        self.assertEqual(report["active_sample_source"], "motion trace visual_target_active")
        self.assertEqual(report["tcp_ground_projection_error_m"]["active"]["max"], .03)
        self.assertEqual(report["tcp_ground_projection_error_m"]["all_including_preparation"]["max"], .3)
        gait = report["tcp_ground_projection_error_m"]["full_rate_trace"]["foot_gait"]
        self.assertFalse(gait["reference_margin_is_acceptance_criterion"])
        self.assertAlmostEqual(gait["all_support_centres_to_crack_m_active"]["min"], .02)
        self.assertLessEqual(report["active_ratio_by_recorded_intervals"], 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
