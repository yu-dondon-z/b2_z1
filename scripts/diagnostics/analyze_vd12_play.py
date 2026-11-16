#!/usr/bin/env python3
"""CPU-only first-episode motion analysis for checked VD12 Isaac play logs.

This never imports Isaac, torch, an actor, or a simulator. Empty/short motion
windows yield null metrics, not zero jitter. Actual WORLD-Y motion is the
primary lateral measurement: it is intentionally not base-yaw projected like
some older VD10 reports. A negative geometry lower bound is a conservative
keep-out violation, not a measured contact force or penetration depth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
LEGACY_METRICS = Path('/home/dong/rl_sar_b2_z1/scripts/vd5_gait_metrics.py')
REQUIRED = {
    'joint_position_rad': (18,), 'joint_velocity_radps': (18,),
    'joint_target_rad': (18,), 'episode_actor_step': (),
    'base_position_w': (3,), 'base_quat_wxyz': (4,),
    'tcp_position_w': (3,), 'tcp_command_world_50hz_zoh': (3,),
    'gun_axis_error_rad': (), 'tool_contact_force_w': (3,),
    'foot_contact_force_w': (4, 3), 'foot_velocity_w': (4, 3),
    'foot_surface_height_m': (4,), 'minimum_clearance_lower_bound_m': (),
    'geometry_input_valid': (), 'minimum_pair_index': (), 'exclusion_latched': (),
}


@dataclass(frozen=True)
class Settings:
    physics_dt_s: float = .005
    control_dt_s: float = .02
    startup_s: float = 2.
    filter_edge_guard_s: float = 1.
    minimum_scored_span_s: float = 2.
    highpass_cutoff_hz: float = 3.
    highpass_order: int = 3
    bandpass_low_hz: float = .5
    bandpass_high_hz: float = 3.
    bandpass_order: int = 3
    touchdown_threshold_n: float = 30.
    touchdown_rearm_s: float = .02
    touchdown_pre_window_s: float = .02
    touchdown_window_s: float = .05

    def validate(self):
        if self.physics_dt_s <= 0 or self.control_dt_s <= 0:
            raise ValueError('Both time steps must be positive')
        ratio = self.control_dt_s / self.physics_dt_s
        if not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
            raise ValueError('Control dt must be an integer multiple of physics dt')
        # The reused VD5 implementation fixes these settings. Do not silently
        # label a different high-pass/edge convention as a legacy comparison.
        if (self.highpass_order != 3 or self.highpass_cutoff_hz != 3.
                or self.filter_edge_guard_s != 1.):
            raise ValueError('VD5 reuse requires order=3, cutoff=3 Hz, edge guard=1 s')
        nyquist = .5 / self.physics_dt_s
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist:
            raise ValueError('Invalid band-pass cutoffs')
        if not self.highpass_cutoff_hz < nyquist or self.bandpass_order < 1:
            raise ValueError('Invalid filter settings')
        if self.startup_s < 0 or self.minimum_scored_span_s < 2.:
            raise ValueError('Require nonnegative startup and at least 2 s scored span')
        if self.touchdown_threshold_n <= 0:
            raise ValueError('Touchdown threshold must be positive')
        for key in ('filter_edge_guard_s', 'touchdown_rearm_s',
                    'touchdown_pre_window_s', 'touchdown_window_s'):
            intervals = getattr(self, key) / self.physics_dt_s
            if intervals < 1 or not np.isclose(intervals, round(intervals), atol=1e-8, rtol=0):
                raise ValueError(key + ' must contain an integer number of physics intervals')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_metrics(path=LEGACY_METRICS):
    """Reuse the existing pure-NumPy/SciPy VD5 metric, not a simulator module."""
    spec = importlib.util.spec_from_file_location('vd12_legacy_vd5_gait_metrics', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load existing VD5 metrics source: ' + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict(count=0, mean=None, median=None, p95=None, max=None)
    return dict(count=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def vector_rms(values):
    """WORLD XYZ RMS in mm; null denotes an unavailable window, not zero."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(samples=0, x_rms_mm=None, y_rms_mm=None, z_rms_mm=None, xyz_rms_mm=None)
    meansquare = np.mean(values ** 2, axis=0)
    return dict(samples=int(len(values)), x_rms_mm=float(1000 * np.sqrt(meansquare[0])),
                y_rms_mm=float(1000 * np.sqrt(meansquare[1])),
                z_rms_mm=float(1000 * np.sqrt(meansquare[2])),
                xyz_rms_mm=float(1000 * np.sqrt(meansquare.sum())))


def filtered_metrics(positions, commands, age_s, settings, legacy):
    """Filter only after startup, then score >=2 s after both 1 s edge guards.

    VD5 high_frequency_rms is reused verbatim for scalar and vector RMS. Its
    minimum four-second array check is strengthened here to require a full
    four-second TIME SPAN, leaving a full >=2 s span after edge exclusion.
    The 0.5--3 Hz error uses the same prototype order and scoring window.
    """
    after = np.flatnonzero(age_s >= settings.startup_s - 1e-10)
    empty = vector_rms(np.empty((0, 3)))
    report = dict(valid=False, reason=None, post_startup_samples=int(len(after)),
                  post_startup_span_s=float(age_s[after[-1]] - age_s[after[0]]) if len(after) else 0.,
                  required_post_startup_span_s=2 * settings.filter_edge_guard_s + settings.minimum_scored_span_s,
                  scored_samples=0, scored_span_s=0., first_scored_age_s=None, last_scored_age_s=None,
                  actual_tcp_world_highpass_3hz=dict(empty),
                  tracking_error_world_bandpass_0p5_3hz=dict(empty))
    if report['post_startup_span_s'] + 1e-9 < report['required_post_startup_span_s']:
        report['reason'] = 'Insufficient post-startup span: require >=2 s scored plus 1 s at EACH filter edge'
        return report
    guard = round(settings.filter_edge_guard_s / settings.physics_dt_s)
    selected = after[guard:-guard]
    if len(selected) < 2 or age_s[selected[-1]] - age_s[selected[0]] + 1e-9 < settings.minimum_scored_span_s:
        report['reason'] = 'Insufficient scored duration after filter edge exclusion'
        return report
    actual = positions[after] - positions[after[0]]
    hp_values = [legacy.high_frequency_rms(actual[:, axis], settings.physics_dt_s,
                                          settings.highpass_cutoff_hz) for axis in range(3)]
    hp_vector = legacy.high_frequency_rms(actual, settings.physics_dt_s, settings.highpass_cutoff_hz)
    if hp_vector is None or any(value is None for value in hp_values):
        report['reason'] = 'Existing VD5 high-pass implementation rejected the short array'
        return report
    error = positions[after] - commands[after]
    error -= error[0]
    sos = butter(settings.bandpass_order, (settings.bandpass_low_hz, settings.bandpass_high_hz),
                 btype='bandpass', fs=1 / settings.physics_dt_s, output='sos')
    band = sosfiltfilt(sos, error, axis=0)[guard:-guard]
    report.update(valid=True, reason=None, scored_samples=int(len(selected)),
                  scored_span_s=float(age_s[selected[-1]] - age_s[selected[0]]),
                  first_scored_age_s=float(age_s[selected[0]]), last_scored_age_s=float(age_s[selected[-1]]),
                  actual_tcp_world_highpass_3hz=dict(samples=int(len(selected)),
                      x_rms_mm=float(hp_values[0] * 1000), y_rms_mm=float(hp_values[1] * 1000),
                      z_rms_mm=float(hp_values[2] * 1000), xyz_rms_mm=float(hp_vector * 1000)),
                  tracking_error_world_bandpass_0p5_3hz=vector_rms(band))
    return report


def touchdowns(force, velocity, height, times, age_s, settings, foot_names):
    """30 N onset; same reset-safe 50 ms trapezoid semantics as VD10 audit.

    The older VD5 utility used >10 N, >=60 ms flight, >=10 ms contact and a
    40 ms peak only. Those counts are NOT interchangeable with this explicit
    30 N / 20 ms rearm / 50 ms force-and-impulse protocol. Initial support is
    never treated as landing. All pre/post samples must remain in this same
    first episode and all event onsets must be after startup.
    """
    pre = round(settings.touchdown_pre_window_s / settings.physics_dt_s)
    post = round(settings.touchdown_window_s / settings.physics_dt_s)
    rearm = round(settings.touchdown_rearm_s / settings.physics_dt_s)
    by_foot, events = {}, []
    for foot_id, foot in enumerate(foot_names):
        fz = np.maximum(force[:, foot_id, 2], 0.)
        counts = dict(initial_support=int(len(fz) > 0 and fz[0] > settings.touchdown_threshold_n),
                      not_rearmed_crossings=0, excluded_startup=0,
                      excluded_incomplete_pre_window=0, excluded_incomplete_post_window=0)
        previous_unloaded, unloaded_count, armed = False, 0, False
        foot_events = []
        for index, value in enumerate(fz):
            unloaded = value <= settings.touchdown_threshold_n
            if unloaded:
                unloaded_count += 1
                armed |= unloaded_count >= rearm
            else:
                if previous_unloaded:
                    if not armed:
                        counts['not_rearmed_crossings'] += 1
                    elif age_s[index] < settings.startup_s - 1e-10:
                        counts['excluded_startup'] += 1
                    elif index < pre:
                        counts['excluded_incomplete_pre_window'] += 1
                    elif index + post >= len(fz):
                        counts['excluded_incomplete_post_window'] += 1
                    else:
                        window = fz[index:index + post + 1]
                        before_v = velocity[index - pre:index, foot_id, 2]
                        event = dict(foot=foot, time_s=float(times[index]), episode_age_s=float(age_s[index]),
                            upward_force_peak_50ms_n=float(window.max()),
                            upward_impulse_50ms_n_s=float(np.trapz(window, dx=settings.physics_dt_s)),
                            vertical_velocity_immediately_before_m_s=float(before_v[-1]),
                            max_downward_speed_pre20ms_m_s=float(max(0., -before_v.min())),
                            foot_surface_height_at_onset_m=float(height[index, foot_id]))
                        foot_events.append(event)
                unloaded_count, armed = 0, False
            previous_unloaded = unloaded
        by_foot[foot] = dict(valid_event_count=len(foot_events), **counts)
        for key in ('upward_force_peak_50ms_n', 'upward_impulse_50ms_n_s',
                    'max_downward_speed_pre20ms_m_s', 'vertical_velocity_immediately_before_m_s',
                    'foot_surface_height_at_onset_m'):
            by_foot[foot][key] = distribution([event[key] for event in foot_events])
        events.extend(foot_events)
    return dict(total_valid_events=len(events), by_foot=by_foot, events=events)


def joint_limits(manifest, joint_names, num_envs):
    raw = manifest.get('joint_position_limits_rad', manifest.get('joint_limits_rad'))
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = [raw[name] for name in joint_names]
    limits = np.asarray(raw, dtype=float)
    if limits.shape == (18, 2):
        limits = np.broadcast_to(limits, (num_envs, 18, 2)).copy()
    if (limits.shape != (num_envs, 18, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, :, 0] > limits[:, :, 1])):
        raise ValueError('Joint limits must be finite [18,2] or [N,18,2] bounds in manifest joint order')
    return limits


def arm_response(position, velocity, target, selected, joint_names, arm_names, limits):
    result = dict(samples=int(len(selected)), limits_available=limits is not None, by_joint={})
    for name in arm_names:
        index = joint_names.index(name)
        q, qd, qt = position[selected, index], velocity[selected, index], target[selected, index]
        row = dict(position_min_rad=float(q.min()) if len(q) else None,
                   position_max_rad=float(q.max()) if len(q) else None,
                   velocity_rms_radps=float(np.sqrt(np.mean(qd ** 2))) if len(qd) else None,
                   target_tracking_error_rms_rad=float(np.sqrt(np.mean((q - qt) ** 2))) if len(q) else None,
                   target_min_rad=float(qt.min()) if len(qt) else None,
                   target_max_rad=float(qt.max()) if len(qt) else None,
                   limit_lower_rad=None, limit_upper_rad=None, minimum_limit_margin_rad=None,
                   actual_hard_limit_violation_samples=None, target_hard_limit_violation_samples=None)
        if limits is not None:
            lower, upper = limits[index]
            row.update(limit_lower_rad=float(lower), limit_upper_rad=float(upper),
                       minimum_limit_margin_rad=float(np.minimum(q - lower, upper - q).min()) if len(q) else None,
                       actual_hard_limit_violation_samples=int(np.count_nonzero((q < lower) | (q > upper))),
                       target_hard_limit_violation_samples=int(np.count_nonzero((qt < lower) | (qt > upper))))
        result['by_joint'][name] = row
    return result


def validate_rows(rows, manifest, settings):
    joint_names, foot_names = manifest.get('joint_names'), manifest.get('foot_names')
    if not joint_names or len(joint_names) != 18 or len(set(joint_names)) != 18:
        raise ValueError('Manifest must record 18 unique actual joint_names')
    if not foot_names or len(foot_names) != 4 or len(set(foot_names)) != 4:
        raise ValueError('Manifest must record four unique actual foot_names')
    if not rows:
        num_envs = manifest.get('num_envs')
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError('Empty trace requires actual num_envs in manifest')
        fields = {name: np.empty((0, num_envs, *trailing)) for name, trailing in REQUIRED.items()}
        return fields, np.empty(0), list(joint_names), list(foot_names), num_envs
    first = np.asarray(rows[0].get('joint_position_rad', []))
    if first.ndim != 2 or first.shape[1] != 18:
        raise ValueError('joint_position_rad must have [N,18] rows')
    num_envs = first.shape[0]
    if num_envs < 1 or manifest.get('num_envs', num_envs) != num_envs:
        raise ValueError('Invalid num_envs or manifest/trace mismatch')
    fields = {}
    for name, trailing in REQUIRED.items():
        try:
            values = np.asarray([row[name] for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Trace lacks valid field ' + name) from error
        expected = (len(rows), num_envs, *trailing)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError('Invalid ' + name + ': expected finite shape ' + str(expected))
        fields[name] = values
    for name in ('geometry_input_valid', 'exclusion_latched'):
        if not np.isin(fields[name], [0, 1]).all():
            raise ValueError(name + ' must contain real boolean values')
    for name in ('episode_actor_step', 'minimum_pair_index'):
        values = fields[name]
        if not np.equal(values, values.astype(np.int64)).all():
            raise ValueError(name + ' must be integral')
    if np.any(fields['episode_actor_step'] < 0):
        raise ValueError('Episode clock must be nonnegative')
    norms = np.linalg.norm(fields['base_quat_wxyz'], axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError('Recorded base quaternion has zero norm')
    times = np.asarray([row['time_s'] for row in rows], dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('time_s must be finite, nonnegative, strictly increasing')
    ticks = times / settings.physics_dt_s
    if not np.allclose(ticks, np.round(ticks), atol=1e-5, rtol=0):
        raise ValueError('time_s must be on the recorded physics grid')
    return fields, times, list(joint_names), list(foot_names), num_envs


def pool_rms(records):
    """Sample-weighted sum of squares, never mean environment RMS values."""
    count = sum(record['samples'] for record in records)
    if not count:
        return vector_rms(np.empty((0, 3)))
    return dict(samples=count, **{key: float(np.sqrt(sum(record['samples'] * record[key] ** 2
                        for record in records if record['samples']) / count))
                        for key in ('x_rms_mm', 'y_rms_mm', 'z_rms_mm', 'xyz_rms_mm')})


def analyze(rows, manifest, summary, settings=None, legacy_source=LEGACY_METRICS):
    settings = settings or Settings(physics_dt_s=manifest['physics_dt_s'],
                                    control_dt_s=manifest['control_dt_s'])
    settings.validate()
    fields, times, names, feet, num_envs = validate_rows(rows, manifest, settings)
    legacy = legacy_metrics(legacy_source)
    limits = joint_limits(manifest, names, num_envs)
    arm_names = list(manifest.get('arm_joint_names', ['joint' + str(k) for k in range(1, 7)]))
    if len(arm_names) != 6 or len(set(arm_names)) != 6 or any(name not in names for name in arm_names):
        raise ValueError('Manifest arm_joint_names must identify six unique actual joints')
    decimation = round(settings.control_dt_s / settings.physics_dt_s)
    phase = ((np.round(times / settings.physics_dt_s).astype(np.int64) - 1) % decimation + 1) * settings.physics_dt_s
    gap_indices = np.flatnonzero(~np.isclose(np.diff(times), settings.physics_dt_s, atol=1e-7, rtol=0)) + 1
    global_end = int(gap_indices[0]) if len(gap_indices) else len(times)
    geometry_bad = ((fields['minimum_clearance_lower_bound_m'] < 0)
                    | ~fields['geometry_input_valid'].astype(bool) | fields['exclusion_latched'].astype(bool))
    guard_indices = np.flatnonzero(np.any(geometry_bad, axis=1))
    first_guard = int(guard_indices[0]) if len(guard_indices) else None
    # Defensive censoring: the checked recorder stops at this first global
    # unsafe callback. If handed a longer trace, never count its later motion.
    if first_guard is not None:
        global_end = min(global_end, first_guard + 1)
    environments, all_events = [], []
    for env_id in range(num_envs):
        age = fields['episode_actor_step'][:, env_id]
        reset_indices = np.flatnonzero(np.diff(age) < 0) + 1
        jump_indices = np.flatnonzero(np.diff(age) > 1) + 1
        end = min(global_end, int(reset_indices[0]) if len(reset_indices) else len(times),
                  int(jump_indices[0]) if len(jump_indices) else len(times))
        selected = slice(0, end)
        age_s = age[selected] * settings.control_dt_s + phase[selected]
        if np.any(np.diff(age_s) <= 0):
            raise ValueError('Episode/substep clock is not strictly increasing before first reset')
        post = np.flatnonzero(age_s >= settings.startup_s - 1e-10)
        full = np.arange(end)
        pos, cmd = fields['tcp_position_w'][selected, env_id], fields['tcp_command_world_50hz_zoh'][selected, env_id]
        filtered = filtered_metrics(pos, cmd, age_s, settings, legacy)
        landing = touchdowns(fields['foot_contact_force_w'][selected, env_id],
            fields['foot_velocity_w'][selected, env_id], fields['foot_surface_height_m'][selected, env_id],
            times[selected], age_s, settings, feet)
        for event in landing['events']:
            event['env_id'] = env_id
        all_events.extend(landing['events'])
        q, qd, target = (fields[key][selected, env_id] for key in
                         ('joint_position_rad', 'joint_velocity_radps', 'joint_target_rad'))
        lower = fields['minimum_clearance_lower_bound_m'][selected, env_id]
        own_bad = np.flatnonzero(geometry_bad[selected, env_id])
        env_limits = limits[env_id] if limits is not None else None
        environments.append(dict(env_id=env_id, initial_episode_only=True,
            initial_actor_step=int(age[0]) if len(age) else None,
            starts_at_episode_reset=bool(age[0] == 0) if len(age) else None,
            retained_physics_samples=end, ignored_later_samples=len(times) - end,
            first_time_s=float(times[0]) if end else None, last_time_s=float(times[end - 1]) if end else None,
            first_episode_age_s=float(age_s[0]) if end else None,
            last_episode_age_s=float(age_s[-1]) if end else None,
            first_reset_boundary_time_s=float(times[reset_indices[0]]) if len(reset_indices) else None,
            first_actor_clock_jump_time_s=float(times[jump_indices[0]]) if len(jump_indices) else None,
            motion_window=filtered,
            raw_tracking_error_world=dict(full_initial_episode=vector_rms(pos - cmd),
                                           post_startup=vector_rms((pos - cmd)[post])),
            arm_response=dict(full_initial_episode=arm_response(q, qd, target, full, names, arm_names, env_limits),
                              post_startup=arm_response(q, qd, target, post, names, arm_names, env_limits)),
            gun_axis_error_deg=dict(full_initial_episode=distribution(np.degrees(fields['gun_axis_error_rad'][selected, env_id])),
                                    post_startup=distribution(np.degrees(fields['gun_axis_error_rad'][selected, env_id][post]))),
            tool_contact_force_norm_n=distribution(np.linalg.norm(fields['tool_contact_force_w'][selected, env_id], axis=-1)),
            geometry=dict(minimum_lower_bound_m=float(lower.min()) if len(lower) else None,
                input_invalid_samples=int(np.count_nonzero(
                ~fields['geometry_input_valid'][selected, env_id].astype(bool))),
                conservative_keepout_seen=bool(len(own_bad)) if end else None,
                first_keepout_time_s=float(times[own_bad[0]]) if len(own_bad) else None,
                first_keepout_pair_index=int(fields['minimum_pair_index'][own_bad[0], env_id]) if len(own_bad) else None),
            foot_contacts=landing))
    valid = [row for row in environments if row['motion_window']['valid']]
    feet_pooled = {}
    for foot in feet:
        events = [event for event in all_events if event['foot'] == foot]
        feet_pooled[foot] = dict(valid_event_count=len(events))
        for key in ('upward_force_peak_50ms_n', 'upward_impulse_50ms_n_s', 'max_downward_speed_pre20ms_m_s'):
            feet_pooled[foot][key] = distribution([event[key] for event in events])
    if not valid:
        assessment = '不能验收抖动和砸地：没有startup后满足滤波及至少2秒评分要求的连续运动窗口。'
    elif not all_events:
        assessment = '存在可用抖动诊断窗口，但没有完整触地事件，不能验收砸地问题；不自动判定部署通过。'
    else:
        assessment = '可报告当前样本的抖动与触地诊断；无匹配基线/验收阈值，不能自动判定改善或部署通过。'
    if not summary.get('completed_requested_duration', False) or first_guard is not None:
        assessment += ' 本次未完整跑完请求时长或触发保守禁区，不能作为运动质量验收通过。'
    return dict(schema_version=1, analysis_kind='CPU first-continuous-episode only', settings=asdict(settings),
        num_envs=num_envs, logged_physics_rows=len(times),
        logged_time_span_s=float(times[-1] - times[0]) if len(times) else 0.,
        actual_logged_end_time_s=float(times[-1]) if len(times) else None, summary=dict(summary),
        checkpoint=manifest.get('checkpoint'), resetmode=manifest.get('resetmode', manifest.get('reset_mode')),
        restored_curriculum=manifest.get('restored_curriculum'), joint_names=names, arm_joint_names=arm_names,
        foot_names=feet, joint_limits_available=limits is not None,
        segmentation=dict(initial_episodes_analyzed=num_envs, later_episodes_analyzed=0,
            time_gap_count=int(len(gap_indices)), first_gap_time_s=float(times[gap_indices[0]]) if len(gap_indices) else None,
            first_global_keepout_time_s=float(times[first_guard]) if first_guard is not None else None,
            global_defensive_retained_rows=global_end),
        usable_motion_window_env_ids=[row['env_id'] for row in valid],
        tcp_world_highpass_3hz=pool_rms([row['motion_window']['actual_tcp_world_highpass_3hz'] for row in environments]),
        tracking_error_world_bandpass_0p5_3hz=pool_rms([
            row['motion_window']['tracking_error_world_bandpass_0p5_3hz'] for row in environments]),
        foot_contacts=dict(total_valid_events=len(all_events), by_foot=feet_pooled, events=all_events),
        environments=environments,
        assessment=dict(message=assessment, can_assess_sampled_jitter=bool(valid),
            can_assess_sampled_foot_impacts=bool(valid and all_events), deployment_approved=False,
            all_requested_envs_have_motion_window=len(valid) == num_envs,
            comparison_to_old_long_window_valid=False),
        method=dict(tcp_frame='WORLD XYZ; Y is fixed world lateral, NOT rotating base-yaw or seam-tangent lateral.',
            highpass='Reuse existing vd5_gait_metrics.high_frequency_rms: third-order Butterworth >3 Hz, SOS forward/backward, '
                'one second excluded at each filter edge. Require >=2 s scored TIME SPAN after 2 s startup.',
            low_band='Third-order Butterworth 0.5-3 Hz band-pass prototype (six poles), SOS forward/backward on '
                'actual TCP minus logged 50 Hz ZOH world command, same scored window. Boundary guard does not '
                'eliminate all 0.5 Hz transients. This is not a causal reward filter.',
            raw_error='Unfiltered world actual minus logged ZOH command, reported separately for initial episode and poststartup; '
                'startup-only diagnostics are NOT stationary jitter estimates.',
            touchdown='Measured positive world-Z net force >30 N after >=20 ms <=30 N; initial support excluded. '
                'Pre-onset downward speed over strictly preceding 20 ms; peak and trapezoid impulse over closed '
                '[onset,onset+50 ms]. Incomplete windows and startup onsets excluded, no cross-reset/gap/stop windows.',
            arm='Actual measured joint velocity RMS, not action variation or PD-estimated torque. Limits only from manifest.',
            aggregation='TCP RMS pooled by scored sample count/squared RMS; touchdown p95 pooled by events, not means of env p95.',
            legacy_metrics_source=str(Path(legacy_source).resolve()), legacy_metrics_sha256=sha256(legacy_source)),
        limitations=['No acceptance claim from a short pre-termination trace; no zero substituted for unavailable RMS or p95.',
            'Do not compare short-window diagnostics with historical long-window results. Coordinate frame, force threshold, '
                'startup, filter order/guards, geometry, course stage and reset mode must also match for a controlled comparison.',
            'Even a complete sampled window does not establish reward convergence, robust success or deployment readiness.',
            'World-Y is meaningful as left/right only for the chosen straight/yaw-aligned world path; inspect command/base orientation.',
            'Net foot/tool contact forces are not actual joint-drive torque. Fifty-ms impulse is not full-stance impulse.',
            'Negative geometric lower bounds mean conservative exclusion only, not physical penetration or force.',
            'Gun-axis error is an unsigned angle and cannot identify signed left/right angular motion.'])


def analyze_log_dir(log_dir, legacy_source=LEGACY_METRICS):
    log_dir = Path(log_dir).resolve(strict=True)
    manifest_path, summary_path = log_dir / 'manifest.json', log_dir / 'summary.json'
    trace_path = log_dir / 'physics_200hz.jsonl.gz'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = analyze(rows, manifest, summary, legacy_source=legacy_source)
    report['source_log_dir'] = str(log_dir)
    report['source_files'] = {path.name: dict(path=str(path), sha256=sha256(path))
                              for path in (manifest_path, summary_path, trace_path)}
    return report


def write_unique_analysis(report, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix='vd12_play_analysis_', dir=output_root))
    output = output_dir / 'analysis.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_dir', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--legacy-metrics-source', type=Path, default=LEGACY_METRICS)
    args = parser.parse_args()
    report = analyze_log_dir(args.log_dir, args.legacy_metrics_source)
    output = write_unique_analysis(report, args.output_root)
    print(report['assessment']['message'])
    print('analysis=' + str(output))
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
