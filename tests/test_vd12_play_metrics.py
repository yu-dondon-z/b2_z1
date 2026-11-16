"""Synthetic CPU regressions for censored VD12 checked-play motion metrics."""

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vd12_play_metrics_cpu', ROOT / 'scripts/diagnostics/analyze_vd12_play.py')
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def synthetic(seconds=8., num_envs=1):
    count = round(seconds / .005)
    times = (np.arange(count) + 1) * .005
    arrays = {name: np.zeros((count, num_envs, *trailing)) for name, trailing in metrics.REQUIRED.items()}
    arrays['episode_actor_step'][:] = (np.arange(count) // 4)[:, None]
    arrays['base_quat_wxyz'][:, :, 0] = 1.
    arrays['base_position_w'][:, :, 2] = .5
    arrays['minimum_clearance_lower_bound_m'][:] = .04
    arrays['geometry_input_valid'][:] = 1
    manifest = dict(num_envs=num_envs, physics_dt_s=.005, control_dt_s=.02,
        joint_names=['leg' + str(k) for k in range(12)] + ['joint' + str(k) for k in range(1, 7)],
        foot_names=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'], resetmode='nominal',
        restored_curriculum={'ground_blend': .1}, checkpoint='synthetic.pt')
    summary = dict(duration_s=seconds, requested_duration_s=seconds, terminal=None, completed_requested_duration=True)
    return arrays, times, manifest, summary


def rows(arrays, times):
    return [dict(time_s=float(time), **{name: values[index].tolist() for name, values in arrays.items()})
            for index, time in enumerate(times)]


class VD12PlayMetricsTests(unittest.TestCase):
    def test_highpass_reuses_legacy_and_reports_world_y_xyz(self):
        arrays, times, manifest, summary = synthetic(12)
        arrays['tcp_position_w'][:, 0, 1] = .01 * np.sin(2 * np.pi * 8 * times)
        arrays['tcp_position_w'][:, 0, 2] = .02 * np.sin(2 * np.pi * times)
        arrays['tcp_command_world_50hz_zoh'][:, 0, 2] = arrays['tcp_position_w'][:, 0, 2]
        # A 90-degree base yaw must NOT rotate WORLD Y out of the primary axis.
        arrays['base_quat_wxyz'][:, 0] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        measured = result['tcp_world_highpass_3hz']
        self.assertAlmostEqual(measured['y_rms_mm'], 10 / np.sqrt(2), delta=.03)
        self.assertLess(measured['z_rms_mm'], .08)
        self.assertEqual(measured['x_rms_mm'], 0.)
        source = arrays['tcp_position_w'][times >= 2., 0]
        expected = metrics.legacy_metrics().high_frequency_rms(source, .005) * 1000
        self.assertAlmostEqual(measured['xyz_rms_mm'], expected, places=10)
        self.assertIn('NOT rotating base-yaw', result['method']['tcp_frame'])

    def test_low_band_is_tracking_error_not_intended_actual_motion(self):
        arrays, times, manifest, summary = synthetic(16)
        target = .03 * np.sin(2 * np.pi * 1.5 * times)
        arrays['tcp_command_world_50hz_zoh'][:, 0, 1] = target
        arrays['tcp_position_w'][:, 0, 1] = target + .01 * np.sin(2 * np.pi * 1.5 * times)
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertAlmostEqual(result['tracking_error_world_bandpass_0p5_3hz']['y_rms_mm'],
                               10 / np.sqrt(2), delta=.3)
        self.assertLess(result['tcp_world_highpass_3hz']['y_rms_mm'], .6)

    def test_too_short_does_not_become_zero_jitter(self):
        arrays, times, manifest, summary = synthetic(3.)
        summary.update(terminal={'kind': 'evaluation_stop_guard'}, completed_requested_duration=False,
                       requested_duration_s=30.)
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertIsNone(result['tcp_world_highpass_3hz']['y_rms_mm'])
        self.assertIsNone(result['tracking_error_world_bandpass_0p5_3hz']['xyz_rms_mm'])
        self.assertFalse(result['assessment']['can_assess_sampled_jitter'])
        self.assertIn('不能验收抖动和砸地', result['assessment']['message'])
        self.assertFalse(result['assessment']['deployment_approved'])
        self.assertIsNone(result['foot_contacts']['by_foot']['FL_foot']['upward_force_peak_50ms_n']['p95'])
        json.dumps(result, allow_nan=False)

    def test_requires_full_two_second_scored_time_span(self):
        arrays, times, manifest, summary = synthetic(5.995)
        self.assertFalse(metrics.analyze(rows(arrays, times), manifest, summary)['environments'][0]['motion_window']['valid'])
        arrays, times, manifest, summary = synthetic(6.)
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        window = result['environments'][0]['motion_window']
        self.assertTrue(window['valid'])
        self.assertAlmostEqual(window['scored_span_s'], 2.)
        self.assertAlmostEqual(window['first_scored_age_s'], 3.)
        self.assertAlmostEqual(window['last_scored_age_s'], 5.)

    def test_initial_support_and_subthreshold_force_are_not_touchdown(self):
        arrays, times, manifest, summary = synthetic()
        arrays['foot_contact_force_w'][:, 0, 0, 2] = 100.
        arrays['foot_contact_force_w'][600:, 0, 1, 2] = 30.
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertEqual(result['foot_contacts']['total_valid_events'], 0)
        self.assertEqual(result['environments'][0]['foot_contacts']['by_foot']['FL_foot']['initial_support'], 1)
        self.assertFalse(result['assessment']['can_assess_sampled_foot_impacts'])

    def test_touchdown_50ms_peak_impulse_and_strict_pre20ms_speed(self):
        arrays, times, manifest, summary = synthetic()
        onset = 600
        arrays['foot_contact_force_w'][onset:, 0, 2, 2] = 100.
        arrays['foot_contact_force_w'][onset + 5, 0, 2, 2] = 300.
        arrays['foot_velocity_w'][onset - 4:onset, 0, 2, 2] = [-.2, -.4, -.3, -.1]
        arrays['foot_velocity_w'][onset, 0, 2, 2] = -9.  # Must not contaminate PRE-onset speed.
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertEqual(result['foot_contacts']['total_valid_events'], 1)
        event = result['foot_contacts']['events'][0]
        self.assertEqual(event['foot'], 'RL_foot')
        self.assertEqual(event['upward_force_peak_50ms_n'], 300.)
        self.assertAlmostEqual(event['upward_impulse_50ms_n_s'], 6.)
        self.assertEqual(event['max_downward_speed_pre20ms_m_s'], .4)
        self.assertEqual(event['vertical_velocity_immediately_before_m_s'], -.1)

    def test_touchdown_chatter_and_incomplete_post_window_are_excluded(self):
        arrays, times, manifest, summary = synthetic()
        force = arrays['foot_contact_force_w'][:, 0, 0, 2]
        force[:] = 100.
        force[599:602] = 0.  # 15 ms cannot rearm.
        force[799:803] = 0.  # 20 ms can rearm.
        arrays['foot_contact_force_w'][-5:, 0, 1, 2] = 200.
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertEqual(result['foot_contacts']['total_valid_events'], 1)
        feet = result['environments'][0]['foot_contacts']['by_foot']
        self.assertEqual(feet['FL_foot']['not_rearmed_crossings'], 1)
        self.assertEqual(feet['FR_foot']['excluded_incomplete_post_window'], 1)

    def test_only_first_episode_is_used_and_contact_window_cannot_cross_reset(self):
        arrays, times, manifest, summary = synthetic(10., 2)
        split = 1200
        arrays['episode_actor_step'][split:, 0] -= split // 4
        arrays['tcp_position_w'][split:, 0] = 100.
        arrays['foot_contact_force_w'][split - 5:, 0, 0, 2] = 100.
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        first, other = result['environments']
        self.assertEqual(first['retained_physics_samples'], split)
        self.assertEqual(other['retained_physics_samples'], len(times))
        self.assertEqual(first['motion_window']['actual_tcp_world_highpass_3hz']['xyz_rms_mm'], 0.)
        self.assertEqual(first['foot_contacts']['by_foot']['FL_foot']['excluded_incomplete_post_window'], 1)
        self.assertEqual(result['foot_contacts']['total_valid_events'], 0)
        self.assertEqual(result['segmentation']['later_episodes_analyzed'], 0)

    def test_time_gap_and_any_env_keepout_defensively_censor_all_motion(self):
        arrays, times, manifest, summary = synthetic(8., 2)
        arrays['minimum_clearance_lower_bound_m'][599, 1] = -.001
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertEqual(result['segmentation']['global_defensive_retained_rows'], 600)
        self.assertEqual([r['retained_physics_samples'] for r in result['environments']], [600, 600])
        self.assertIsNone(result['tcp_world_highpass_3hz']['xyz_rms_mm'])
        arrays, times, manifest, summary = synthetic(8.)
        times[600:] += .02
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        self.assertEqual(result['segmentation']['time_gap_count'], 1)
        self.assertEqual(result['environments'][0]['retained_physics_samples'], 600)

    def test_arm_response_uses_joint_names_actual_velocity_and_manifest_limits(self):
        arrays, times, manifest, summary = synthetic(1.)
        arrays['joint_position_rad'][:, 0, 12] = .2
        arrays['joint_velocity_radps'][:, 0, 12] = -.3
        arrays['joint_target_rad'][:, 0, 12] = 1.2
        manifest['joint_limits_rad'] = {name: [-1., 1.] for name in manifest['joint_names']}
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        arm = result['environments'][0]['arm_response']['full_initial_episode']
        self.assertTrue(arm['limits_available'])
        row = arm['by_joint']['joint1']
        self.assertAlmostEqual(row['velocity_rms_radps'], .3)
        self.assertAlmostEqual(row['target_tracking_error_rms_rad'], 1.)
        self.assertEqual(row['actual_hard_limit_violation_samples'], 0)
        self.assertEqual(row['target_hard_limit_violation_samples'], len(times))
        self.assertAlmostEqual(row['minimum_limit_margin_rad'], .8)
        self.assertIsNone(result['environments'][0]['arm_response']['post_startup']['by_joint']['joint1']['velocity_rms_radps'])

    def test_runtime_per_environment_joint_position_limits_are_supported(self):
        arrays, times, manifest, summary = synthetic(.1, 2)
        limits = np.tile(np.array([-1., 1.]), (2, 18, 1))
        limits[1, 12] = [-.2, .3]
        manifest['joint_position_limits_rad'] = limits.tolist()
        arrays['joint_position_rad'][:, 1, 12] = .4
        result = metrics.analyze(rows(arrays, times), manifest, summary)
        first, second = [row['arm_response']['full_initial_episode']['by_joint']['joint1']
                         for row in result['environments']]
        self.assertEqual(first['limit_upper_rad'], 1.)
        self.assertEqual(second['limit_upper_rad'], .3)
        self.assertEqual(second['actual_hard_limit_violation_samples'], len(times))

    def test_reset_exclusion_with_no_physics_rows_is_reported_not_faked(self):
        _, _, manifest, summary = synthetic(.1, 2)
        summary.update(duration_s=0., completed_requested_duration=False,
                       terminal={'reason': 'exclusion_at_reset', 'time_s': 0., 'env_ids': [1]})
        result = metrics.analyze([], manifest, summary)
        self.assertEqual(result['logged_physics_rows'], 0)
        self.assertIsNone(result['actual_logged_end_time_s'])
        self.assertIsNone(result['tcp_world_highpass_3hz']['xyz_rms_mm'])
        self.assertIn('不能验收抖动和砸地', result['assessment']['message'])
        self.assertIsNone(result['environments'][1]['geometry']['minimum_lower_bound_m'])
        self.assertEqual(result['summary']['terminal']['reason'], 'exclusion_at_reset')
        json.dumps(result, allow_nan=False)

    def test_pooled_rms_is_sample_weighted_not_mean_of_env_rms(self):
        records = [dict(samples=100, x_rms_mm=1., y_rms_mm=2., z_rms_mm=3., xyz_rms_mm=np.sqrt(14.)),
                   dict(samples=300, x_rms_mm=3., y_rms_mm=4., z_rms_mm=5., xyz_rms_mm=np.sqrt(50.))]
        result = metrics.pool_rms(records)
        self.assertAlmostEqual(result['y_rms_mm'], np.sqrt(13.))
        self.assertEqual(result['samples'], 400)

    def test_gzip_directory_analysis_writes_unique_output_without_overwrite(self):
        arrays, times, manifest, summary = synthetic(.1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            (directory / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with gzip.open(directory / 'physics_200hz.jsonl.gz', 'wt', encoding='utf-8') as stream:
                for row in rows(arrays, times):
                    stream.write(json.dumps(row) + '\n')
            source_before = {path.name: metrics.sha256(path) for path in directory.iterdir()}
            report = metrics.analyze_log_dir(directory)
            first = metrics.write_unique_analysis(report, directory / 'reports')
            second = metrics.write_unique_analysis(report, directory / 'reports')
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual({name: metrics.sha256(directory / name) for name in source_before}, source_before)
            self.assertEqual(len(report['source_files']), 3)

    def test_nonfinite_or_missing_required_measurements_are_rejected(self):
        arrays, times, manifest, summary = synthetic(.1)
        trace = rows(arrays, times)
        del trace[0]['joint_velocity_radps']
        with self.assertRaisesRegex(ValueError, 'joint_velocity_radps'):
            metrics.analyze(trace, manifest, summary)
        arrays['tcp_position_w'][0, 0, 1] = float('nan')
        with self.assertRaisesRegex(ValueError, 'tcp_position_w'):
            metrics.analyze(rows(arrays, times), manifest, summary)


if __name__ == '__main__':
    unittest.main()
