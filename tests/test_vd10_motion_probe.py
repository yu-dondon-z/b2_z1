"""CPU-only regressions for reset-safe VD10 motion/landing diagnostics."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vd10_motion_probe_cpu", ROOT / "scripts/diagnostics/analyze_vd10_motion_probe.py")
motion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion
SPEC.loader.exec_module(motion)


def synthetic_trace(seconds=8., num_envs=1):
    count = round(seconds * 200)
    clock = np.arange(count)
    arrays = {
        "env_tcp_world_m": np.zeros((count, num_envs, 3)),
        "env_tcp_target_world_m": np.zeros((count, num_envs, 3)),
        "env_base_quat_wxyz": np.zeros((count, num_envs, 4)),
        "env_gun_axis_error_rad": np.zeros((count, num_envs)),
        "env_foot_force_world_n": np.zeros((count, num_envs, 4, 3)),
        "env_foot_velocity_world_m_s": np.zeros((count, num_envs, 4, 3)),
        "env_foot_height_m": np.zeros((count, num_envs, 4)),
        "env_episode_actor_step": np.tile((clock // 4)[:, None], (1, num_envs)),
    }
    arrays["env_base_quat_wxyz"][:, :, 0] = 1
    metadata = {"num_envs": num_envs, "control_dt_s": .02, "physics_dt_s": .005,
                "decimation": 4, "action_mode": "deterministic_actor", "reset_events": [],
                "foot_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]}
    return arrays, metadata, (clock + 1) * .005


def rows(arrays, steps=None):
    count = len(arrays["env_episode_actor_step"])
    steps = np.arange(1, count + 1) if steps is None else steps
    return [{"physics_step": int(steps[index]),
             **{key: values[index].tolist() for key, values in arrays.items()}} for index in range(count)]


def test_synthetic_sine_lateral_rms_and_low_frequency_rejection():
    arrays, metadata, time = synthetic_trace(10)
    arrays["env_tcp_world_m"][:, 0, 1] = .01 * np.sin(2 * np.pi * 6 * time)
    arrays["env_tcp_world_m"][:, 0, 2] = .04 * np.sin(2 * np.pi * 1 * time)
    result = motion.analyze(rows(arrays), metadata)
    actual = result["tcp_high_frequency"]["actual_tcp"]
    assert actual["lateral_rms_mm"] == pytest.approx(10 / np.sqrt(2), rel=.025)
    assert actual["vertical_rms_mm"] < .15
    assert actual["longitudinal_rms_mm"] == 0
    assert result["episodes"][0]["first_scored_age_s"] == pytest.approx(2.5)
    assert result["episodes"][0]["last_scored_age_s"] == pytest.approx(9.5)


def test_same_frame_current_yaw_projection_not_seam_direction():
    vectors = np.array([[1., 0., 2.], [0., 1., 3.], [1., 0., 4.]])
    angle = np.array([0., np.pi / 2, np.pi / 2])
    quaternions = np.column_stack((np.cos(angle / 2), np.zeros((3, 2)), np.sin(angle / 2)))
    expected = [[1., 0., 2.], [1., 0., 3.], [0., -1., 4.]]
    np.testing.assert_allclose(motion.project_yaw(vectors, quaternions), expected, atol=1e-15)


def test_low_frequency_sway_is_reported_separately_from_high_frequency():
    arrays, metadata, time = synthetic_trace(16)
    arrays["env_tcp_world_m"][:, 0, 1] = .01 * np.sin(2 * np.pi * 1.5 * time)
    result = motion.analyze(rows(arrays), metadata)
    high = result["tcp_high_frequency"]["actual_tcp"]["lateral_rms_mm"]
    low = result["tcp_low_frequency_band"]["actual_tcp"]["lateral_rms_mm"]
    assert high < .1
    assert low == pytest.approx(10 / np.sqrt(2), rel=.06)
    assert result["field_version"] == 2
    assert result["tcp_low_frequency_band"]["actual_tcp"]["samples"] == result[
        "tcp_high_frequency"]["actual_tcp"]["samples"]


def test_angle_bias_mean_std_and_band_are_not_signed_yaw():
    arrays, metadata, time = synthetic_trace(16)
    arrays["env_gun_axis_error_rad"][:, 0] = .4 + .05 * np.sin(2 * np.pi * 1.5 * time)
    result = motion.analyze(rows(arrays), metadata)
    angles = result["gun_axis_error_deg"]
    assert angles["mean"] == pytest.approx(np.degrees(.4), abs=.06)
    assert angles["std_deg"] == pytest.approx(np.degrees(.05 / np.sqrt(2)), rel=.02)
    assert angles["low_frequency_band_rms_deg"] == pytest.approx(np.degrees(.05 / np.sqrt(2)), rel=.06)
    assert angles["high_frequency_rms_deg"] < .03
    assert result["episodes"][0]["gun_axis_error_deg"]["count"] == angles["count"]
    assert sum(window["count"] for window in result["gun_axis_error_by_episode_age_window"]) == angles["count"]
    assert "NOT left/right" in result["method"]["gun_axis_error"]


def test_target_staircase_is_not_actual_tcp_jitter():
    arrays, metadata, time = synthetic_trace(10)
    arrays["env_tcp_world_m"][:, 0, 0] = .2 * time
    knots = ((np.arange(len(time)) // 4) * 4 + 1) * .005
    arrays["env_tcp_target_world_m"][:, 0, 0] = .2 * knots
    result = motion.analyze(rows(arrays), metadata)["tcp_high_frequency"]
    assert result["actual_tcp"]["longitudinal_rms_mm"] < .003
    assert result["tracking_error_logged_zoh"]["longitudinal_rms_mm"] > .8
    # Holding the final synthetic target contributes a tiny padded-edge residue.
    assert result["tracking_error_diagnostic_linear_target"]["longitudinal_rms_mm"] < .01


def test_reset_offsets_never_become_tcp_jitter_or_contact_onsets():
    arrays, metadata, _ = synthetic_trace(12)
    split = 1200
    arrays["env_tcp_world_m"][split:, 0, :] = [100., -10., 20.]
    arrays["env_tcp_target_world_m"][split:, 0, :] = [100., -10., 20.]
    arrays["env_episode_actor_step"][split:] -= split // 4
    arrays["env_foot_force_world_n"][split:, 0, :, 2] = 100.
    metadata["reset_events"] = [{"actor_step": split // 4, "env_ids": [0]}]
    result = motion.analyze(rows(arrays), metadata)
    assert result["tcp_high_frequency"]["actual_tcp"]["xyz_rms_mm"] == 0
    assert result["tcp_low_frequency_band"]["actual_tcp"]["xyz_rms_mm"] == 0
    assert result["segmentation"]["environment_episode_segments"] == 2
    assert result["segmentation"]["additional_segments_after_reset_or_gap"] == 1
    assert result["foot_contacts"]["total_valid_events"] == 0
    assert result["foot_contacts"]["by_foot"]["FL"]["initial_support_segments"] == 1


def test_reset_metadata_splits_even_if_episode_clock_still_zero():
    steps = np.arange(1, 13)
    age = np.zeros(12)
    events = [{"actor_step": 1, "env_ids": [0]}, {"actor_step": 2, "env_ids": [0]}]
    assert motion.segments(steps, age, events, 0, 4) == [(0, 4), (4, 8), (8, 12)]


def test_constant_stand_support_is_not_a_touchdown():
    arrays, metadata, _ = synthetic_trace()
    arrays["env_foot_force_world_n"][:, :, :, 2] = 200.
    result = motion.analyze(rows(arrays), metadata)
    assert result["foot_contacts"]["total_valid_events"] == 0
    for foot in motion.FEET:
        summary = result["foot_contacts"]["by_foot"][foot]
        assert summary["valid_event_count"] == 0
        assert summary["initial_support_segments"] == 1
        assert summary["upward_force_peak_50ms_n"]["p95"] is None


def test_onset_50ms_force_impulse_and_pre_descent_speed():
    arrays, metadata, _ = synthetic_trace()
    onset = 600
    arrays["env_foot_force_world_n"][onset:, 0, 2, 2] = 100.
    arrays["env_foot_force_world_n"][onset + 5, 0, 2, 2] = 300.
    arrays["env_foot_velocity_world_m_s"][onset - 4:onset, 0, 2, 2] = [-.2, -.4, -.3, -.1]
    arrays["env_foot_height_m"][onset, 0, 2] = .001
    result = motion.analyze(rows(arrays), metadata)
    assert result["foot_contacts"]["total_valid_events"] == 1
    event = result["foot_contacts"]["events"][0]
    assert event["foot"] == "RL"
    assert event["physics_step"] == onset + 1
    assert event["upward_force_peak_50ms_n"] == 300.
    assert event["upward_impulse_50ms_n_s"] == pytest.approx(6.)
    assert event["vertical_velocity_immediately_before_m_s"] == -.1
    assert event["max_downward_speed_pre20ms_m_s"] == .4
    assert event["foot_height_at_onset_m"] == .001


def test_threshold_chatter_requires_twenty_ms_rearm():
    arrays, metadata, _ = synthetic_trace()
    forces = arrays["env_foot_force_world_n"][:, 0, 0, 2]
    forces[:] = 100.
    forces[599:602] = 0.  # Only 15 ms, must not rearm.
    forces[799:803] = 0.  # 20 ms, one valid event.
    result = motion.analyze(rows(arrays), metadata)["foot_contacts"]
    assert result["total_valid_events"] == 1
    assert result["events"][0]["physics_step"] == 804
    assert result["by_foot"]["FL"]["not_rearmed_crossings"] == 1


def test_post_onset_window_cannot_cross_reset():
    arrays, metadata, _ = synthetic_trace(10)
    split, onset = 800, 795
    arrays["env_foot_force_world_n"][onset:split, 0, 0, 2] = 100.
    arrays["env_foot_force_world_n"][split:, 0, 0, 2] = 1000.
    arrays["env_episode_actor_step"][split:] -= split // 4
    metadata["reset_events"] = [{"actor_step": split // 4, "env_ids": [0]}]
    result = motion.analyze(rows(arrays), metadata)["foot_contacts"]
    assert result["total_valid_events"] == 0
    assert result["by_foot"]["FL"]["excluded_incomplete_post_window"] == 1
    assert result["by_foot"]["FL"]["initial_support_segments"] == 1


def test_gaps_split_trace_and_exclude_short_filter_segments():
    arrays, metadata, _ = synthetic_trace(1)
    steps = np.arange(1, 201)
    steps[100:] += 4
    result = motion.analyze(rows(arrays, steps), metadata)
    assert result["segmentation"]["trace_gap_count"] == 1
    assert result["segmentation"]["environment_episode_segments"] == 2
    assert result["segmentation"]["excluded_short_segments_from_tcp_filter"] == 2
    assert result["tcp_high_frequency"]["actual_tcp"]["samples"] == 0
    assert result["tcp_low_frequency_band"]["actual_tcp"]["samples"] == 0
    assert result["gun_axis_error_deg"]["std_deg"] is None
    json.dumps(result, allow_nan=False)


def test_async_reset_does_not_split_other_environments():
    arrays, metadata, _ = synthetic_trace(8, num_envs=2)
    arrays["env_episode_actor_step"][800:, 0] -= 200
    metadata["reset_events"] = [{"actor_step": 200, "env_ids": [0]}]
    result = motion.analyze(rows(arrays), metadata)
    assert result["segmentation"]["environment_episode_segments"] == 3
    assert sum(episode["env_id"] == 1 for episode in result["episodes"]) == 1


def test_nonfinite_and_old_schema_are_rejected():
    arrays, metadata, _ = synthetic_trace(.1)
    trace = rows(arrays)
    del trace[0]["env_tcp_world_m"]
    with pytest.raises(ValueError, match="schema lacks env_tcp_world_m"):
        motion.analyze(trace, metadata)
    trace = rows(arrays)
    trace[0]["env_tcp_world_m"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid env_tcp_world_m"):
        motion.analyze(trace, metadata)


def test_log_directory_and_multiple_probe_comparison():
    arrays, metadata, _ = synthetic_trace(4)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "vd10_policy_probe.json").write_text(json.dumps(metadata))
        (directory / "vd10_200hz_trace.json").write_text(json.dumps(rows(arrays)))
        first = motion.analyze_log_dir(directory)
    second = dict(first, source_log_dir="second")
    compared = motion.compare_reports([first, second])
    assert compared[0]["actual_tcp_high_frequency_delta_mm"]["lateral_rms_mm"] == 0
