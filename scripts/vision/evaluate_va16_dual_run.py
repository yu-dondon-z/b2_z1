#!/usr/bin/env python3
"""Report measured VA16 baseline tracking from one retained simulation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    array = np.asarray([float(value) for value in values if finite_number(value)], dtype=np.float64)
    if not len(array):
        return {"samples": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {"samples": len(array), "min": float(np.min(array)), "mean": float(np.mean(array)),
            "p50": float(np.quantile(array, .5)), "p95": float(np.quantile(array, .95)), "max": float(np.max(array))}


def read_jsonl(path):
    records, invalid = [], 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not a dictionary")
                records.append(value)
            except ValueError:
                invalid += 1
    return records, invalid


def scalar(records, key):
    return [record.get(key) for record in records]


def component(records, key, axis):
    return [record[key][axis] for record in records
            if isinstance(record.get(key), list) and len(record[key]) > axis]


def trace_numeric(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
            if math.isfinite(value):
                values.append(value)
        except (KeyError, TypeError, ValueError):
            continue
    return values


def trace_body_roll(rows):
    result = []
    for row in rows:
        try:
            q = np.asarray([float(row[f"base_q{axis}"]) for axis in ("w", "x", "y", "z")])
            if not np.isfinite(q).all() or np.linalg.norm(q) < .1:
                continue
            w, x, y, z = q / np.linalg.norm(q)
            result.append(abs(math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def body_angles(records):
    roll, tilt = [], []
    for record in records:
        gravity = np.asarray(record.get("base_projected_gravity_b", []), dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < .1:
            continue
        gravity /= np.linalg.norm(gravity)
        roll.append(abs(math.degrees(math.atan2(-gravity[1], -gravity[2]))))
        tilt.append(math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1)))))
    return {"abs_roll_deg": distribution(roll), "tilt_deg": distribution(tilt)}


def vector_speed_errors(records):
    errors = []
    for record in records:
        desired, actual = record.get("base_velocity_command_b"), record.get("base_linear_velocity_b")
        if isinstance(desired, list) and isinstance(actual, list) and len(desired) >= 1 and len(actual) >= 1:
            if finite_number(desired[0]) and finite_number(actual[0]):
                errors.append(abs(actual[0] - desired[0]))
    return errors


def effective_stream_rate(packets, stream):
    groups = {}
    for packet in packets:
        observation = packet.get(stream, {})
        if not isinstance(observation, dict):
            continue
        generation, sequence = observation.get("capture_generation"), observation.get("sequence")
        captured = observation.get("capture_timestamp")
        if isinstance(generation, int) and isinstance(sequence, int) and finite_number(captured):
            groups.setdefault(generation, {})[sequence] = float(captured)
    periods, exported_intervals, elapsed = 0, 0, 0.
    processed = sum(len(group) for group in groups.values())
    for observations in groups.values():
        ordered = sorted(observations.items())
        if len(ordered) >= 2 and ordered[-1][1] > ordered[0][1]:
            periods += len(ordered) - 1
            exported_intervals += ordered[-1][0] - ordered[0][0]
            elapsed += ordered[-1][1] - ordered[0][1]
    return {"processed_capture_count": processed, "capture_clock_span_s": elapsed,
            "processed_frames_per_wall_s": periods / elapsed if elapsed else None,
            "exported_sequence_intervals_per_wall_s": exported_intervals / elapsed if elapsed else None,
            "basis": "original monotonic capture timestamps; no configured-rate substitution"}


def motion_trace_report(path):
    if not path.is_file():
        return {"available": False, "samples": 0, "foot_gait_available": False,
                "foot_gait_reason": "no motion trace"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def values(key):
        result = []
        for row in rows:
            try:
                value = float(row.get(key, ""))
                if math.isfinite(value):
                    result.append(value)
            except (TypeError, ValueError):
                pass
        return result

    joint_rates = []
    for previous, current in zip(rows[:-1], rows[1:]):
        try:
            dt = float(current["time_s"]) - float(previous["time_s"])
            if dt <= 0 or float(current.get("reset", 0) or 0):
                continue
            for joint in range(1, 7):
                rate = abs(float(current[f"joint{joint}"]) - float(previous[f"joint{joint}"])) / dt
                if math.isfinite(rate):
                    joint_rates.append(rate)
        except (KeyError, ValueError, TypeError):
            continue
    contact_fields = [key for key in (rows[0] if rows else {})
                      if "foot" in key.lower() or "contact" in key.lower() and not key.startswith("gun_")]
    return {"available": True, "samples": len(rows), "recorded_reset_count": int(sum(values("reset"))),
            "time_s": distribution(values("time_s")), "body_tilt_deg": distribution(values("body_tilt_deg")),
            "base_height_w_m": distribution(values("base_z")),
            "tcp_height_w_m": distribution(values("actual_z")),
            "command_vx_mps": distribution(values("command_vx_b")),
            "actual_vx_mps": distribution(values("base_vx_actual")),
            "actual_vy_mps": distribution(values("base_vy_actual")),
            "actual_yaw_rate_rps": distribution(values("actual_yaw_rate_b")),
            "z1_abs_joint_rate_from_positions_rad_s": distribution(joint_rates),
            "foot_gait_available": bool(contact_fields),
            "foot_gait_fields": {key: distribution(values(key)) for key in contact_fields},
            "foot_gait_reason": "recorded fields only; no inferred foot contact" if contact_fields else
                "current trace does not record individual foot contact, swing height, or leg joint states"}


def high_rate_ground_projection(path, statuses):
    """Use recorded mesh parameters and the full-rate independent TCP trace."""
    if not path.is_file():
        return {"available": False, "reason": "motion trace unavailable"}
    records = [record for record in statuses if finite_number(record.get("simulation_time_s"))]
    if not records or not all(isinstance(record.get("evaluation_scene_geometry"), dict) for record in records):
        return {"available": False, "reason": "physical mesh parameters were not recorded in controller history"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"available": False, "reason": "motion trace is empty"}
    records.sort(key=lambda record: record["simulation_time_s"])
    status_times = np.asarray([record["simulation_time_s"] for record in records])
    try:
        row_times = np.asarray([float(row["time_s"]) for row in rows])
        if not np.isfinite(row_times).all():
            raise ValueError("nonfinite trace clock")
    except (KeyError, TypeError, ValueError) as error:
        return {"available": False, "invalid": True, "reason": f"invalid motion trace time: {error}"}
    differences = np.diff(row_times)
    step_dt = float(np.median(differences[differences > 0])) if np.any(differences > 0) else .02
    # play_amp records row time before incrementing its counter, after env.step.
    # Controller common_step_counter timestamps the resulting state.
    aligned_times = row_times + step_dt
    indices = np.searchsorted(status_times, aligned_times + 1e-8, side="right") - 1
    cadence = np.diff(status_times)
    cadence = float(np.median(cadence[cadence > 0])) if np.any(cadence > 0) else step_dt
    matched = (indices >= 0) & (aligned_times <= status_times[-1] + cadence + 1e-8)
    direct_active = "visual_target_active" in rows[0]
    active = np.zeros(len(rows), dtype=bool)
    tcp_errors = np.full(len(rows), np.nan)
    base_errors = np.full(len(rows), np.nan)
    geometry_groups = {}
    for row_index, row in enumerate(rows):
        # The physical mesh is static. Its first recorded metadata also
        # describes preparation samples before the first low-rate status.
        record = records[max(indices[row_index], 0)]
        if direct_active:
            try:
                active[row_index] = float(row["visual_target_active"]) == 1
            except (KeyError, TypeError, ValueError):
                return {"available": False, "invalid": True, "reason": "invalid full-rate visual active flag"}
        elif matched[row_index]:
            active[row_index] = record.get("tool_target_active") is True
        geometry = record["evaluation_scene_geometry"]
        geometry_groups.setdefault(json.dumps(geometry, sort_keys=True), []).append(row_index)
    from evaluate_vision_ground_trace import distances_to_polyline
    try:
        for encoded, selected in geometry_groups.items():
            geometry = json.loads(encoded)
            amplitude, wavelength = float(geometry["amplitude_m"]), float(geometry["wavelength_m"])
            half_curve, half_road, mesh_step = (float(geometry[key]) for key in
                ("curve_half_length_m", "road_half_length_m", "mesh_step_m"))
            origin = np.asarray(geometry["env_origin_w"], dtype=float)
            if (not np.isfinite([amplitude, wavelength, half_curve, half_road, mesh_step, *origin]).all()
                    or wavelength <= 0 or mesh_step <= 0 or not 0 < half_curve < half_road or origin.shape != (3,)):
                raise ValueError("recorded mesh parameters are invalid")
            x = np.linspace(-half_curve, half_curve, round(2 * half_curve / mesh_step) + 1)
            vertices = np.vstack(([-half_road, 0.],
                np.column_stack((x, amplitude * np.sin(2 * math.pi * x / wavelength)**3)), [half_road, 0.]))
            tcp = np.asarray([[float(rows[index]["actual_x"]), float(rows[index]["actual_y"])] for index in selected])
            base = np.asarray([[float(rows[index]["base_x"]), float(rows[index]["base_y"])] for index in selected])
            if not np.isfinite(tcp).all() or not np.isfinite(base).all():
                raise ValueError("motion trace contains nonfinite poses")
            tcp_errors[selected] = distances_to_polyline(tcp - origin[:2], vertices)
            base_errors[selected] = distances_to_polyline(base - origin[:2], vertices)
    except (KeyError, ValueError, TypeError) as error:
        return {"available": False, "invalid": True, "reason": f"cannot evaluate physical trace: {error}"}
    foot_report = physical_foot_report(rows, active, geometry_groups)
    active_rows = [rows[index] for index in np.flatnonzero(active)]
    return {"available": True, "samples": len(rows), "active_samples": int(active.sum()),
            "active_sample_source": "motion trace visual_target_active" if direct_active else "controller status held between samples",
            "active_ratio": float(np.mean(active)), "active_duration_s": float(active.sum() * step_dt),
            "trace_duration_s": float(row_times[-1] - row_times[0] + step_dt),
            "first_active_sim_time_s": float(aligned_times[np.flatnonzero(active)[0]]) if active.any() else None,
            "unmatched_active_alignment_samples": 0 if direct_active else int((~matched).sum()), "trace_step_dt_s": step_dt,
            "active": distribution(tcp_errors[active]), "all_including_preparation": distribution(tcp_errors),
            "base_distance_to_physical_crack_m": {"active": distribution(base_errors[active]),
                "all_including_preparation": distribution(base_errors)},
            "active_body_tilt_deg": distribution([float(rows[index]["body_tilt_deg"])
                for index in np.flatnonzero(active) if rows[index].get("body_tilt_deg")]),
            "body_motion_active": {"abs_roll_deg": distribution(trace_body_roll(active_rows)),
                "command_vx_mps": distribution(trace_numeric(active_rows, "command_vx_b")),
                "actual_vx_mps": distribution(trace_numeric(active_rows, "base_vx_actual")),
                "actual_vy_mps": distribution(trace_numeric(active_rows, "base_vy_actual"))},
            "abs_body_roll_deg_global": distribution(trace_body_roll(rows)),
            "foot_gait": foot_report,
            "source": "full-rate motion-trace TCP XY against mesh parameters recorded in controller status",
            "active_alignment": "exact per-policy trace flag" if direct_active else
                "preceding controller active flag held until next status; row time shifted +one policy step",
            "alignment_uncertainty_s": step_dt if direct_active else cadence,
            "note": "true physical mesh geometry is evaluation-only"}


def physical_foot_report(rows, active, geometry_groups, contact_threshold_n=5.):
    """Measure foot-centre clearance; this does not evaluate sole collision."""
    labels = ("FL", "FR", "RL", "RR")
    expected = [f"foot_{label}_{field}" for label in labels
                for field in ("x_w", "y_w", "z_w", "contact_force_n")]
    if not rows or any(field not in rows[0] for field in expected):
        return {"available": False, "reason": "individual foot positions/forces are absent from this trace"}
    from evaluate_vision_ground_trace import distances_to_polyline
    result = {"available": True, "support_contact_force_threshold_n": contact_threshold_n,
              "reference_training_margin_m": .07, "reference_margin_is_acceptance_criterion": False,
              "distance_definition": "XY distance of each recorded foot body centre to physical crack centreline; not sole clearance/collision",
              "feet": {}}
    all_active_support_distances = []
    for label in labels:
        points = np.asarray([[float(row[f"foot_{label}_{axis}_w"]) for axis in ("x", "y", "z")] for row in rows])
        forces = np.asarray([float(row[f"foot_{label}_contact_force_n"]) for row in rows])
        if not np.isfinite(points).all() or not np.isfinite(forces).all():
            result["feet"][label] = {"available": False, "reason": "nonfinite foot positions or contact forces"}
            continue
        distance = np.full(len(rows), np.nan)
        heights = np.full(len(rows), np.nan)
        for encoded, selected in geometry_groups.items():
            geometry = json.loads(encoded)
            half_curve = geometry["curve_half_length_m"]
            x = np.linspace(-half_curve, half_curve, round(2 * half_curve / geometry["mesh_step_m"]) + 1)
            line = np.vstack(([-geometry["road_half_length_m"], 0.],
                np.column_stack((x, geometry["amplitude_m"] * np.sin(2 * math.pi * x / geometry["wavelength_m"])**3)),
                [geometry["road_half_length_m"], 0.]))
            origin = np.asarray(geometry["env_origin_w"])
            distance[selected] = distances_to_polyline(points[selected, :2] - origin[:2], line)
            heights[selected] = points[selected, 2] - origin[2]
        support = forces > contact_threshold_n
        active_support, active_swing = active & support, active & ~support
        all_active_support_distances.extend(distance[active_support].tolist())
        lift_events = int(np.count_nonzero(support[:-1] & ~support[1:] & active[1:]))
        result["feet"][label] = {"available": True,
            "active_contact_ratio": float(np.mean(support[active])) if np.any(active) else None,
            "active_lift_events": lift_events,
            "contact_force_n": distribution(forces),
            "support_centre_to_crack_m_active": distribution(distance[active_support]),
            "support_centre_to_crack_m_global": distribution(distance[support]),
            "swing_foot_centre_height_above_road_m_active": distribution(heights[active_swing]),
            "foot_centre_height_above_road_m_global": distribution(heights)}
    result["all_support_centres_to_crack_m_active"] = distribution(all_active_support_distances)
    return result


def evaluate_run(directory, tcp_limit_m=.05):
    directory = Path(directory).expanduser().resolve()
    statuses, malformed_status = read_jsonl(directory / "controller_history.jsonl")
    packets, malformed_packets = read_jsonl(directory / "dual_vision_packets.jsonl")
    active = [record for record in statuses if record.get("tool_target_active") is True]
    inactive = [record for record in statuses if record.get("tool_target_active") is not True]
    times = [record.get("simulation_time_s") for record in statuses if finite_number(record.get("simulation_time_s"))]
    active_times = [record.get("simulation_time_s") for record in active if finite_number(record.get("simulation_time_s"))]
    intervals = [later - earlier for earlier, later in zip(times[:-1], times[1:]) if later > earlier]
    duration = max(times) - min(times) if len(times) > 1 else 0.
    active_duration = 0.
    for previous, current in zip(statuses[:-1], statuses[1:]):
        before, after = previous.get("simulation_time_s"), current.get("simulation_time_s")
        if previous.get("tool_target_active") is True and finite_number(before) and finite_number(after) and after > before:
            active_duration += after - before
    tcp_key = "evaluation_only_tcp_projection_to_physical_crack_m"
    status_active_error = distribution(scalar(active, tcp_key))
    global_error = distribution(scalar(statuses, tcp_key))
    trace = motion_trace_report(directory / "motion_trace.csv")
    high_rate = high_rate_ground_projection(directory / "motion_trace.csv", statuses)
    direct_active = high_rate.get("available") and high_rate.get("active_sample_source") == "motion trace visual_target_active"
    active_error = high_rate["active"] if high_rate.get("available") and (high_rate["active_samples"] or direct_active) else status_active_error
    active_count = high_rate["active_samples"] if direct_active else len(active)
    active_ratio = high_rate["active_ratio"] if direct_active else len(active) / len(statuses) if statuses else 0.
    if direct_active:
        active_duration = high_rate["active_duration_s"]
        duration = high_rate["trace_duration_s"]
        active_first = high_rate["first_active_sim_time_s"]
    else:
        active_first = min(active_times) if active_times else None
    failures = []
    if not active_count:
        failures.append("no actual visual tracking samples")
    elif not direct_active and status_active_error["samples"] != len(active):
        failures.append("active tracking lacks independent physical-crack TCP error samples")
    elif direct_active and high_rate["active"]["samples"] != active_count:
        failures.append("full-rate active tracking lacks independent physical-crack TCP error samples")
    if active_error["max"] is not None and active_error["max"] > tcp_limit_m:
        failures.append(f"active TCP ground-projection error exceeds {tcp_limit_m:.3f} m")
    if high_rate.get("invalid"):
        failures.append(high_rate["reason"])
    if trace.get("recorded_reset_count", 0) > 0:
        failures.append("motion trace records episode resets")
    active_ages = {stream: distribution([
        record.get("capture_age_s", {}).get(stream) for record in active
        if isinstance(record.get("capture_age_s"), dict)]) for stream in ("front", "rear")}
    streams = {}
    for stream in ("front", "rear"):
        observations = [packet.get(stream) for packet in packets if isinstance(packet.get(stream), dict)]
        streams[stream] = {"processed_samples": len(observations),
                           "valid_ratio": sum(record.get("valid") is True for record in observations) / len(observations)
                           if observations else None,
                           "effective_rate": effective_stream_rate(packets, stream)}
    first = min(times) if times else None
    started = [record["simulation_time_s"] for record in statuses
               if record.get("visual_tracking_started") is True and finite_number(record.get("simulation_time_s"))]
    session = {}
    try:
        session = json.loads((directory / "session.json").read_text())
    except (OSError, ValueError):
        pass
    review = None
    try:
        review = json.loads((directory / "review_notes.json").read_text())
    except (OSError, ValueError):
        pass
    return {"schema": "b2_z1_va16_dual_run_report_v1", "run_directory": str(directory),
            "scope": "local Isaac VA15 70D/18-action external-vision baseline; not real-robot validation",
            "pass": not failures, "failures": failures, "tcp_projection_limit_m": tcp_limit_m,
            "criterion_scope": "independent physical-crack error during recorded active tracking; resets also fail",
            "checkpoint": session.get("checkpoint"), "checkpoint_sha256": session.get("checkpoint_sha256"),
            "source_sha256_at_launch": session.get("source_sha256_at_launch"),
            "source_files_changed_during_session": session.get("source_files_changed_during_session"),
            "review_notes": review,
            "mode": session.get("vision_mode"), "controller_samples": len(statuses), "processed_packets": len(packets),
            "malformed_log_records": {"controller": malformed_status, "vision": malformed_packets},
            "simulation_duration_observed_s": duration,
            "controller_sampling_interval_s": distribution(intervals),
            "active_samples": active_count, "controller_active_samples": len(active),
            "active_sample_source": "motion trace visual_target_active" if direct_active else "controller status",
            "preparation_or_inactive_controller_samples": len(inactive),
            "active_ratio_by_recorded_samples": active_ratio,
            "active_duration_observed_s": active_duration,
            "active_ratio_by_recorded_intervals": active_duration / duration if duration > 0 else 0.,
            "tracking_first_active_sim_time_s": active_first,
            "tracking_start_delay_from_first_record_s": active_first - (0. if direct_active else first)
                if active_first is not None and (direct_active or first is not None) else None,
            "visual_started_sim_time_s": min(started) if started else None,
            "tcp_ground_projection_error_m": {"active": active_error,
                "all_including_preparation": high_rate["all_including_preparation"] if high_rate.get("available") else global_error,
                "controller_all_including_preparation": global_error,
                "controller_active": status_active_error, "full_rate_trace": high_rate,
                "source_field": tcp_key, "source": "scene physical mesh; independent of visual control target",
                "criterion_source": "full-rate motion trace" if high_rate.get("available") and high_rate["active_samples"] else "controller status",
                "note": "maximum covers recorded samples only; see trace active-time alignment uncertainty"},
            "visual_tcp_estimation_error_m": {
                "active": distribution(scalar(active, "evaluation_only_tcp_visual_position_error_m")),
                "all_recorded": distribution(scalar(statuses, "evaluation_only_tcp_visual_position_error_m")),
                "measurement_models": sorted({str(record["tcp_visual_measurement_model"]) for record in statuses
                                               if record.get("tcp_visual_measurement_model")}),
                "source": "diagnostic comparison with independent TCP truth; inspect measurement model/time propagation"},
            "body_motion": {"active": body_angles(active), "all_including_preparation": body_angles(statuses),
                "command_vx_active_mps": distribution(component(active, "base_velocity_command_b", 0)),
                "actual_vx_active_mps": distribution(component(active, "base_linear_velocity_b", 0)),
                "abs_vx_tracking_error_active_mps": distribution(vector_speed_errors(active)),
                "command_yaw_rate_active_rps": distribution(component(active, "base_velocity_command_b", 2)),
                "actual_vx_global_mps": distribution(component(statuses, "base_linear_velocity_b", 0))},
            "vision": {"streams": streams, "packet_processing_ms": distribution(scalar(packets, "processing_ms")),
                "packet_capture_age_s": distribution(scalar(packets, "capture_age_s")),
                "controller_active_capture_age_s": active_ages},
            "full_motion_trace": trace,
            "limitations": ["No real-robot qualification or new-policy training is implied.",
                "Recorded active intervals exclude preparation from tracking scores; global extrema remain reported.",
                "Configured camera cadence is not used as measured perception throughput."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--tcp-limit-m", type=float, default=.05)
    parser.add_argument("--output", type=Path, help="Default: RUN/run_report.json")
    args = parser.parse_args()
    if not math.isfinite(args.tcp_limit_m) or args.tcp_limit_m <= 0:
        parser.error("--tcp-limit-m must be finite and positive")
    if not args.run_directory.is_dir():
        parser.error("run directory does not exist")
    report = evaluate_run(args.run_directory, args.tcp_limit_m)
    destination = args.output or args.run_directory / "run_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    error = report["tcp_ground_projection_error_m"]["active"]
    print(json.dumps({"report": str(destination.resolve()), "pass": report["pass"], "failures": report["failures"],
        "active_samples": report["active_samples"], "active_ratio": report["active_ratio_by_recorded_samples"],
        "active_tcp_error_m": error}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
