#!/usr/bin/env python3
"""Compare reset-safe TCP jitter and measured touchdown forces in VD10 probes.

This is an offline diagnostic, not a reward implementation. World TCP vectors
are filtered before projecting into each sample's current base-yaw axes. The
target interpolation is explicitly diagnostic: it is not a logged continuous
target. Foot impulses cover only the first 50 ms after a measured touchdown.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


FEET = ("FL", "FR", "RL", "RR")
REQUIRED = {
    "env_tcp_world_m": (3,),
    "env_tcp_target_world_m": (3,),
    "env_base_quat_wxyz": (4,),
    "env_gun_axis_error_rad": (),
    "env_foot_force_world_n": (4, 3),
    "env_foot_velocity_world_m_s": (4, 3),
    "env_foot_height_m": (4,),
    "env_episode_actor_step": (),
}


@dataclass(frozen=True)
class Settings:
    physics_dt_s: float = 0.005
    control_dt_s: float = 0.02
    startup_s: float = 2.0
    highpass_cutoff_hz: float = 3.0
    highpass_order: int = 4
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 3.0
    bandpass_order: int = 4
    filter_edge_guard_s: float = 0.5
    touchdown_threshold_n: float = 10.0
    touchdown_rearm_s: float = 0.02
    touchdown_window_s: float = 0.05
    touchdown_pre_window_s: float = 0.02

    def validate(self):
        if self.physics_dt_s <= 0 or self.control_dt_s <= 0:
            raise ValueError("Both time steps must be positive")
        ratio = self.control_dt_s / self.physics_dt_s
        if not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
            raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")
        if not 0 < self.highpass_cutoff_hz < 0.5 / self.physics_dt_s:
            raise ValueError("High-pass cutoff must lie below Nyquist")
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < 0.5 / self.physics_dt_s:
            raise ValueError("Band-pass cutoffs must be ordered and lie below Nyquist")
        if min(self.highpass_order, self.bandpass_order) < 1 or self.startup_s < 0 or self.filter_edge_guard_s < 0:
            raise ValueError("Invalid filter order or exclusion duration")
        for name in ("touchdown_rearm_s", "touchdown_window_s", "touchdown_pre_window_s"):
            count = getattr(self, name) / self.physics_dt_s
            if count < 1 or not np.isclose(count, round(count), atol=1e-8, rtol=0):
                raise ValueError(f"{name} must contain a positive integer number of physics intervals")


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def angular_summary(values_rad, low_band_rad, high_band_rad):
    """An unsigned axis-error angle is not a signed left/right angular signal."""
    values_deg = np.degrees(values_rad)
    return {**distribution(values_deg),
            "std_deg": float(np.std(values_deg)) if len(values_deg) else None,
            "low_frequency_band_rms_deg": float(np.sqrt(np.mean(np.degrees(low_band_rad) ** 2)))
            if len(low_band_rad) else None,
            "high_frequency_rms_deg": float(np.sqrt(np.mean(np.degrees(high_band_rad) ** 2)))
            if len(high_band_rad) else None}


def rms_summary(vectors):
    """Vectors have longitudinal, lateral, vertical components, in metres."""
    if not len(vectors):
        return {"samples": 0, "longitudinal_rms_mm": None, "lateral_rms_mm": None,
                "vertical_rms_mm": None, "xyz_rms_mm": None}
    meansquare = np.mean(np.asarray(vectors) ** 2, axis=0)
    return {"samples": len(vectors), "longitudinal_rms_mm": float(1000 * np.sqrt(meansquare[0])),
            "lateral_rms_mm": float(1000 * np.sqrt(meansquare[1])),
            "vertical_rms_mm": float(1000 * np.sqrt(meansquare[2])),
            "xyz_rms_mm": float(1000 * np.sqrt(meansquare.sum()))}


def project_yaw(vectors, quaternions):
    """Project WORLD vectors using SAME-FRAME wxyz base yaw; z stays world-up."""
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Base quaternion must have nonzero norm")
    w, x, y, z = (quaternions / norm).T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.column_stack((cosine * vectors[:, 0] + sine * vectors[:, 1],
                            -sine * vectors[:, 0] + cosine * vectors[:, 1], vectors[:, 2]))


def interpolate_logged_target(target, actor_age):
    """Interpolate first logged target of each control step, holding both ends.

    The logged 50 Hz zero-order-held target remains the source of truth. This
    synthetic series only tests sensitivity to its stair steps; it cannot
    recover the continuous command or silently replace the logged target.
    """
    knots = np.r_[0, np.flatnonzero(np.diff(actor_age) != 0) + 1]
    samples = np.arange(len(target))
    return np.column_stack([np.interp(samples, knots, target[knots, axis]) for axis in range(3)])


def segments(physics_steps, actor_age, reset_events, env_id, decimation):
    """Return half-open segments, splitting at actual resets and trace gaps."""
    boundaries = {0, len(physics_steps)}
    boundaries.update((np.flatnonzero(np.diff(physics_steps) != 1) + 1).tolist())
    boundaries.update((np.flatnonzero(np.diff(actor_age) < 0) + 1).tolist())
    for event in reset_events:
        if env_id in event["env_ids"]:
            # Reset occurs after the actor step's final physics callback.
            boundary = np.searchsorted(physics_steps, int(event["actor_step"]) * decimation + 1)
            if 0 < boundary < len(physics_steps):
                boundaries.add(int(boundary))
    points = sorted(boundaries)
    return list(zip(points[:-1], points[1:]))


def touchdown_events(force, velocity, height, age_s, physics_steps, settings, env_id, episode_id):
    """Find genuine threshold onsets after >=20 ms continuously unloaded."""
    rearm = round(settings.touchdown_rearm_s / settings.physics_dt_s)
    post = round(settings.touchdown_window_s / settings.physics_dt_s)
    pre = round(settings.touchdown_pre_window_s / settings.physics_dt_s)
    records = []
    counts = {foot: {"initial_support_segments": 0, "not_rearmed_crossings": 0,
                     "excluded_startup": 0, "excluded_incomplete_pre_window": 0,
                     "excluded_incomplete_post_window": 0} for foot in FEET}
    for foot_id, foot in enumerate(FEET):
        fz = np.maximum(force[:, foot_id, 2], 0.)
        previous_unloaded, unloaded_samples, armed = False, 0, False
        counts[foot]["initial_support_segments"] += int(fz[0] > settings.touchdown_threshold_n)
        for index, current_force in enumerate(fz):
            unloaded = current_force <= settings.touchdown_threshold_n
            if unloaded:
                unloaded_samples += 1
                armed |= unloaded_samples >= rearm
            else:
                if previous_unloaded:
                    if not armed:
                        counts[foot]["not_rearmed_crossings"] += 1
                    elif age_s[index] < settings.startup_s:
                        counts[foot]["excluded_startup"] += 1
                    elif index < pre:
                        counts[foot]["excluded_incomplete_pre_window"] += 1
                    elif index + post >= len(fz):
                        counts[foot]["excluded_incomplete_post_window"] += 1
                    else:
                        window = fz[index:index + post + 1]
                        pre_velocity = velocity[index - pre:index, foot_id, 2]
                        records.append({
                            "env_id": int(env_id), "episode_segment": int(episode_id), "foot": foot,
                            "physics_step": int(physics_steps[index]),
                            "global_time_s": float(physics_steps[index] * settings.physics_dt_s),
                            "episode_age_s": float(age_s[index]),
                            "upward_force_peak_50ms_n": float(window.max()),
                            "upward_impulse_50ms_n_s": float(np.trapz(window, dx=settings.physics_dt_s)),
                            "vertical_velocity_immediately_before_m_s": float(pre_velocity[-1]),
                            "max_downward_speed_pre20ms_m_s": float(max(0., -pre_velocity.min())),
                            "foot_height_at_onset_m": float(height[index, foot_id]),
                        })
                armed, unloaded_samples = False, 0
            previous_unloaded = unloaded
    return records, counts


def analyze(trace, metadata=None, settings=None):
    """Analyze decoded schema-2 trace rows; no simulator or torch dependency."""
    metadata = metadata or {}
    if settings is None:
        settings = Settings(physics_dt_s=metadata.get("physics_dt_s", .005),
                            control_dt_s=metadata.get("control_dt_s", .02))
    settings.validate()
    if not trace:
        raise ValueError("Trace is empty")
    num_envs = len(trace[0]["env_episode_actor_step"])
    if metadata.get("num_envs", num_envs) != num_envs:
        raise ValueError("Probe metadata num_envs does not match trace")
    fields = {}
    for name, trailing in REQUIRED.items():
        try:
            fields[name] = np.asarray([row[name] for row in trace], dtype=float)
        except KeyError as error:
            raise ValueError(f"Trace schema lacks {name}; re-record a schema-2 motion probe") from error
        expected = (len(trace), num_envs, *trailing)
        if fields[name].shape != expected or not np.isfinite(fields[name]).all():
            raise ValueError(f"Invalid {name}: expected finite array with shape {expected}")
    physics_steps = np.asarray([row["physics_step"] for row in trace], dtype=int)
    if np.any(physics_steps < 1) or np.any(np.diff(physics_steps) <= 0):
        raise ValueError("physics_step must be strictly increasing and start at >=1")
    actor_age = fields["env_episode_actor_step"]
    if np.any(actor_age < 0) or np.any(actor_age != actor_age.astype(int)):
        raise ValueError("Episode actor steps must be nonnegative integers")
    decimation = round(settings.control_dt_s / settings.physics_dt_s)
    if metadata.get("decimation", decimation) != decimation:
        raise ValueError("Probe decimation disagrees with logged time steps")
    sos = butter(settings.highpass_order, settings.highpass_cutoff_hz, btype="highpass",
                 fs=1 / settings.physics_dt_s, output="sos")
    band_sos = butter(settings.bandpass_order, (settings.bandpass_low_hz, settings.bandpass_high_hz),
                      btype="bandpass", fs=1 / settings.physics_dt_s, output="sos")
    phase = ((physics_steps - 1) % decimation + 1) * settings.physics_dt_s
    all_signals = {name: [] for name in ("actual_tcp", "tracking_error_logged_zoh",
                                       "tracking_error_diagnostic_linear_target")}
    all_low_band_signals = {name: [] for name in all_signals}
    raw_errors, gun_errors, episode_results, events = [], [], [], []
    gun_low_band_errors, gun_high_band_errors, scored_episode_ages = [], [], []
    contact_counts = {foot: {} for foot in FEET}
    total_segments, excluded_segments, interpolated_intra_step_changes = 0, 0, 0
    reset_events = metadata.get("reset_events", [])
    for env_id in range(num_envs):
        slices = segments(physics_steps, actor_age[:, env_id], reset_events, env_id, decimation)
        for episode_id, (start, end) in enumerate(slices):
            total_segments += 1
            selection = slice(start, end)
            age = actor_age[selection, env_id]
            age_s = age * settings.control_dt_s + phase[selection]
            positions = fields["env_tcp_world_m"][selection, env_id]
            targets = fields["env_tcp_target_world_m"][selection, env_id]
            same_step = np.diff(age) == 0
            interpolated_intra_step_changes += int(np.count_nonzero(
                np.linalg.norm(np.diff(targets, axis=0), axis=1)[same_step] > 1e-9))
            found, counts = touchdown_events(
                fields["env_foot_force_world_n"][selection, env_id],
                fields["env_foot_velocity_world_m_s"][selection, env_id],
                fields["env_foot_height_m"][selection, env_id], age_s, physics_steps[selection],
                settings, env_id, episode_id)
            events.extend(found)
            for foot in FEET:
                for name, value in counts[foot].items():
                    contact_counts[foot][name] = contact_counts[foot].get(name, 0) + value
            after_startup = np.flatnonzero(age_s >= settings.startup_s)
            guard = round(settings.filter_edge_guard_s / settings.physics_dt_s)
            # Filtering never sees an adjacent episode. Apply only after startup,
            # and exclude both padded boundaries from every RMS statistic.
            if len(after_startup) <= max(2 * guard + 1, 3 * (settings.highpass_order + 1)):
                excluded_segments += 1
                episode_results.append({"env_id": env_id, "episode_segment": episode_id,
                                        "start_physics_step": int(physics_steps[start]),
                                        "physics_samples": end - start, "scored_samples": 0})
                continue
            initial = after_startup[0]
            score = slice(guard, -guard if guard else None)
            quaternions = fields["env_base_quat_wxyz"][selection, env_id][initial:][score]
            interpolated = interpolate_logged_target(targets, age)
            signals = {
                "actual_tcp": positions,
                "tracking_error_logged_zoh": positions - targets,
                "tracking_error_diagnostic_linear_target": positions - interpolated,
            }
            result = {"env_id": env_id, "episode_segment": episode_id,
                      "start_physics_step": int(physics_steps[start]), "physics_samples": end - start,
                      "scored_samples": len(quaternions),
                      "first_scored_age_s": float(age_s[initial:][score][0]),
                      "last_scored_age_s": float(age_s[initial:][score][-1])}
            for name, values in signals.items():
                # Remove a harmless constant to improve conditioning at distant
                # vector-environment origins; this does not alter a high-pass.
                series = values[initial:] - values[initial]
                filtered = sosfiltfilt(sos, series, axis=0)[score]
                projected = project_yaw(filtered, quaternions)
                all_signals[name].append(projected)
                result[name] = rms_summary(projected)
                band_projected = project_yaw(sosfiltfilt(band_sos, series, axis=0)[score], quaternions)
                all_low_band_signals[name].append(band_projected)
                result.setdefault("tcp_low_frequency_band", {})[name] = rms_summary(band_projected)
            raw_errors.append(project_yaw((positions - targets)[initial:][score], quaternions))
            angles = fields["env_gun_axis_error_rad"][selection, env_id][initial:]
            angle_series = angles - angles[0]
            angles_low_band = sosfiltfilt(band_sos, angle_series)[score]
            angles_high_band = sosfiltfilt(sos, angle_series)[score]
            gun_errors.append(angles[score])
            gun_low_band_errors.append(angles_low_band)
            gun_high_band_errors.append(angles_high_band)
            scored_episode_ages.append(age_s[initial:][score])
            result["gun_axis_error_deg"] = angular_summary(angles[score], angles_low_band, angles_high_band)
            episode_results.append(result)

    def combine(values, width=None):
        return np.concatenate(values, axis=0) if values else np.empty((0, width) if width else (0,))

    angles = combine(gun_errors)
    low_angles = combine(gun_low_band_errors)
    high_angles = combine(gun_high_band_errors)
    episode_ages = combine(scored_episode_ages)
    age_windows = []
    for index in range(int(np.ceil(episode_ages.max() / 5)) if len(episode_ages) else 0):
        selected = (episode_ages >= index * 5) & (episode_ages < (index + 1) * 5)
        age_windows.append({"episode_age_start_s": index * 5, "episode_age_end_s": (index + 1) * 5,
                            **angular_summary(angles[selected], low_angles[selected], high_angles[selected])})
    foot_reports = {}
    for foot in FEET:
        foot_events = [event for event in events if event["foot"] == foot]
        foot_reports[foot] = {"valid_event_count": len(foot_events), **contact_counts[foot]}
        for key in ("upward_force_peak_50ms_n", "upward_impulse_50ms_n_s",
                    "vertical_velocity_immediately_before_m_s", "max_downward_speed_pre20ms_m_s",
                    "foot_height_at_onset_m"):
            foot_reports[foot][key] = distribution([event[key] for event in foot_events])
    return {
        "schema_version": 1,
        "field_version": 2,
        "settings": asdict(settings), "num_envs": num_envs, "physics_steps": len(trace),
        "duration_s_per_env": float((physics_steps[-1] - physics_steps[0] + 1) * settings.physics_dt_s),
        "checkpoint": metadata.get("checkpoint"), "action_mode": metadata.get("action_mode"),
        "frozen_ground_blend": metadata.get("frozen_ground_blend"),
        "segmentation": {
            "environment_episode_segments": total_segments,
            "additional_segments_after_reset_or_gap": total_segments - num_envs,
            "excluded_short_segments_from_tcp_filter": excluded_segments,
            "metadata_reset_count": sum(len(event["env_ids"]) for event in reset_events),
            "trace_gap_count": int(np.count_nonzero(np.diff(physics_steps) != 1)),
        },
        "tcp_high_frequency": {name: rms_summary(combine(values, 3)) for name, values in all_signals.items()},
        "tcp_low_frequency_band": {
            name: rms_summary(combine(values, 3)) for name, values in all_low_band_signals.items()},
        "tcp_raw_tracking_error_logged_zoh": rms_summary(combine(raw_errors, 3)),
        "gun_axis_error_deg": angular_summary(angles, low_angles, high_angles),
        "gun_axis_error_by_episode_age_window": age_windows,
        "target_interpolation_intra_actor_step_changes": interpolated_intra_step_changes,
        "foot_contacts": {"total_valid_events": len(events), "by_foot": foot_reports, "events": events},
        "episodes": episode_results,
        "method": {
            "tcp_frame": "Filter WORLD vectors first; then rotate each by SAME-FRAME current base yaw: "
                         "+X longitudinal/forward, +Y lateral/left, +Z world-up. Not seam-tangent coordinates.",
            "highpass": "Fourth-order Butterworth by default, scipy SOS forward/backward zero-phase; "
                        "3 Hz is the single-pass cutoff (the offline double-pass response is squared). "
                        "This is not the causal training reward filter.",
            "low_frequency_band": "0.5-3 Hz fourth-order Butterworth band-pass prototype by default "
                                  "(8 poles after low-pass-to-band-pass transform), forward/backward SOS; "
                                  "the offline double-pass response is squared. Same startup and 0.5 s "
                                  "edge exclusions as high-pass. At 0.5 Hz this short edge guard does not "
                                  "guarantee removal of all low-frequency boundary transients. Actual "
                                  "world TCP band energy can include intended path motion; compare "
                                  "tracking-error band separately before calling it unwanted sway.",
            "gun_axis_error": "Scalar unsigned angle between actual and target gun axes. Mean/std and "
                              "band RMS are pooled over scored samples, per episode and 5 s episode-age "
                              "windows. A static bias raises the mean but not band RMS. This angle lacks "
                              "signed yaw/pitch direction, so its band RMS is NOT left/right angular sway.",
            "exclusions": "Filter each episode independently AFTER 2 s startup; discard another 0.5 s "
                          "at both ends for filter boundaries. Contact events only exclude 2 s startup.",
            "target": "Logged target is 50 Hz zero-order hold. Actual TCP high-pass is primary. "
                      "Raw-ZOH and linearly interpolated-target error high-pass are reported separately; "
                      "interpolation is a diagnostic estimate, NOT a measured continuous target.",
            "touchdown": "Positive measured world-Z net foot force crosses >10 N after >=20 ms <=10 N. "
                         "Initial support is not an onset. Integrate clipped-positive Fz by trapezoids "
                         "over [onset,onset+50ms], including both endpoints; not full-stance impulse. "
                         "Pre-onset speed uses the 20 ms strictly before onset. No window crosses reset/gap.",
            "foot_order": list(FEET), "logged_foot_names": metadata.get("foot_names"),
            "limitations": "Single deterministic seed/probe does not establish convergence or deployment "
                           "readiness. Net contact forces are not joint drive torques. CPU synthetic tests "
                           "verify analysis semantics, not physical simulator validity.",
        },
    }


def analyze_log_dir(log_dir, settings=None):
    log_dir = Path(log_dir)
    probe_file = log_dir / "vd10_policy_probe.json"
    if not probe_file.exists():
        probe_file = log_dir / "vd10_zero_action_probe.json"
    metadata = json.loads(probe_file.read_text(encoding="utf-8"))
    trace = json.loads((log_dir / "vd10_200hz_trace.json").read_text(encoding="utf-8"))
    result = analyze(trace, metadata, settings)
    result["source_log_dir"] = str(log_dir.resolve())
    return result


def compare_reports(reports):
    """Absolute deltas vs the first probe; never infer acceptance thresholds."""
    if len(reports) < 2:
        return []
    comparisons = []
    primary_keys = ("longitudinal_rms_mm", "lateral_rms_mm", "vertical_rms_mm", "xyz_rms_mm")
    baseline = reports[0]
    for candidate in reports[1:]:
        deltas = {}
        for key in primary_keys:
            initial = baseline["tcp_high_frequency"]["actual_tcp"][key]
            final = candidate["tcp_high_frequency"]["actual_tcp"][key]
            deltas[key] = None if initial is None or final is None else final - initial
        low_deltas = {}
        for key in primary_keys:
            initial = baseline["tcp_low_frequency_band"]["actual_tcp"][key]
            final = candidate["tcp_low_frequency_band"]["actual_tcp"][key]
            low_deltas[key] = None if initial is None or final is None else final - initial
        comparisons.append({"baseline": baseline["source_log_dir"], "candidate": candidate["source_log_dir"],
                            "actual_tcp_high_frequency_delta_mm": deltas,
                            "actual_tcp_low_frequency_band_delta_mm": low_deltas})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze_log_dir(path) for path in args.log_dirs]
    result = {"probes": reports, "comparisons": compare_reports(reports), "field_version": 2}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for report in reports:
        primary = report["tcp_high_frequency"]["actual_tcp"]
        print(f"{report['source_log_dir']}: actual TCP HP lateral={primary['lateral_rms_mm']} mm, "
              f"xyz={primary['xyz_rms_mm']} mm, touchdown events={report['foot_contacts']['total_valid_events']}")
        band = report["tcp_low_frequency_band"]["actual_tcp"]
        print(f"  actual TCP 0.5-3 Hz: lateral={band['lateral_rms_mm']} mm, xyz={band['xyz_rms_mm']} mm")
        for foot, data in report["foot_contacts"]["by_foot"].items():
            print(f"  {foot}: n={data['valid_event_count']}, "
                  f"50ms Fz peak p95={data['upward_force_peak_50ms_n']['p95']} N, "
                  f"50ms impulse p95={data['upward_impulse_50ms_n_s']['p95']} N s")


if __name__ == "__main__":
    main()
