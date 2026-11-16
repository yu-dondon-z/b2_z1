#!/usr/bin/env python3
"""Summarize existing reset-aware motion reports; never run a simulator.

This presentation layer does not change analyze_vd10_motion_probe.py or its
filters. Cohort RMS values are reconstructed exactly from per-episode sample
counts and RMS values, not from a mean of environment RMS values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FEET = ("FL", "FR", "RL", "RR")
SERIES = {
    "actual_tcp_lateral_highpass_rms_mm": ("tcp_high_frequency", "actual_tcp", "lateral_rms_mm"),
    "actual_tcp_xyz_highpass_rms_mm": ("tcp_high_frequency", "actual_tcp", "xyz_rms_mm"),
    "actual_tcp_lateral_low_band_rms_mm": ("tcp_low_frequency_band", "actual_tcp", "lateral_rms_mm"),
    "actual_tcp_xyz_low_band_rms_mm": ("tcp_low_frequency_band", "actual_tcp", "xyz_rms_mm"),
    "tracking_error_lateral_low_band_rms_mm": (
        "tcp_low_frequency_band", "tracking_error_logged_zoh", "lateral_rms_mm"),
    "tracking_error_xyz_low_band_rms_mm": (
        "tcp_low_frequency_band", "tracking_error_logged_zoh", "xyz_rms_mm"),
    "diagnostic_linear_target_error_lateral_low_band_rms_mm": (
        "tcp_low_frequency_band", "tracking_error_diagnostic_linear_target", "lateral_rms_mm"),
    "raw_tracking_error_lateral_rms_mm": ("tcp_raw_tracking_error_logged_zoh", "lateral_rms_mm"),
    "raw_tracking_error_xyz_rms_mm": ("tcp_raw_tracking_error_logged_zoh", "xyz_rms_mm"),
    "gun_axis_error_mean_deg": ("gun_axis_error_deg", "mean"),
    "gun_axis_error_std_deg": ("gun_axis_error_deg", "std_deg"),
    "gun_axis_error_low_band_rms_deg": ("gun_axis_error_deg", "low_frequency_band_rms_deg"),
    "gun_axis_error_highpass_rms_deg": ("gun_axis_error_deg", "high_frequency_rms_deg"),
}
FOOT_METRICS = {
    "force_peak_p95_n": "upward_force_peak_50ms_n",
    "impulse_p95_n_s": "upward_impulse_50ms_n_s",
    "pre_touchdown_downward_speed_p95_m_s": "max_downward_speed_pre20ms_m_s",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def value_at(mapping, keys):
    for key in keys:
        mapping = mapping[key]
    return mapping


def change(baseline, candidate):
    return {"baseline": float(baseline), "candidate": float(candidate),
            "delta": float(candidate - baseline),
            "percent_change": float(100 * (candidate / baseline - 1)) if baseline else None}


def cause_for(event, env_id):
    return [cause for cause, ids in event["attributed_causes"].items() if env_id in ids]


def reset_summary(metadata):
    horizon = metadata["actor_steps"]
    first = {}
    premature = []
    for event in metadata["reset_events"]:
        for env_id in event["env_ids"]:
            record = {"env_id": env_id, "actor_step": event["actor_step"],
                      "global_time_s": event["actor_step"] * metadata["control_dt_s"],
                      "causes": cause_for(event, env_id)}
            first.setdefault(env_id, record)
            if event["actor_step"] < horizon:
                premature.append(record)
    first_failures = [row for row in first.values() if row["actor_step"] < horizon]
    failed_ids = sorted({row["env_id"] for row in premature})
    return {
        "num_envs": metadata["num_envs"],
        "horizon_s": horizon * metadata["control_dt_s"],
        "all_window_real_reset_count": metadata["total_resets"],
        "all_window_attributed_causes": metadata["termination_events"],
        "all_window_premature_reset_count": len(premature),
        "distinct_premature_reset_env_ids": failed_ids,
        "first_episode_premature_failure_count": len(first_failures),
        "first_episode_premature_failure_fraction": len(first_failures) / metadata["num_envs"],
        "first_episode_timeout_count": sum("time_out" in row["causes"] for row in first.values()),
        "first_episode_incomplete_without_reset_count": metadata["num_envs"] - len(first),
        "premature_reset_events": premature,
        "cause_note": metadata["termination_events_note"],
    }


def cohort_rms(report, excluded_env_ids):
    episodes = [row for row in report["episodes"]
                if row["env_id"] not in excluded_env_ids and row["scored_samples"]]
    count = sum(row["scored_samples"] for row in episodes)
    paths = {
        "actual_tcp_lateral_highpass_rms_mm": ("actual_tcp", "lateral_rms_mm"),
        "actual_tcp_lateral_low_band_rms_mm": ("tcp_low_frequency_band", "actual_tcp", "lateral_rms_mm"),
        "tracking_error_lateral_low_band_rms_mm": (
            "tcp_low_frequency_band", "tracking_error_logged_zoh", "lateral_rms_mm"),
    }
    if not count:
        raise ValueError("No scored common-cohort samples")
    metrics = {name: math.sqrt(sum(row["scored_samples"] * value_at(row, path) ** 2
                                   for row in episodes) / count)
               for name, path in paths.items()}
    return {"num_envs": len({row["env_id"] for row in episodes}),
            "scored_samples": count, "metrics": metrics}


def build_summary(comparison_path):
    comparison = read_json(comparison_path)
    reports = comparison["probes"]
    if len(reports) != 2:
        raise ValueError("Provide exactly baseline and candidate reports, in that order")
    metadata = [read_json(Path(row["source_log_dir"]) / "vd10_policy_probe.json") for row in reports]
    if len({(row["num_envs"], row["actor_steps"], row["physics_dt_s"], row["action_mode"],
             row["frozen_ground_blend"]) for row in metadata}) != 1:
        raise ValueError("Probe environment counts, horizons, rates or modes differ")
    resets = [reset_summary(row) for row in metadata]
    excluded = set().union(*(row["distinct_premature_reset_env_ids"] for row in resets))
    cohorts = [cohort_rms(row, excluded) for row in reports]
    tcp = {key: change(*(value_at(row, path) for row in reports)) for key, path in SERIES.items()}
    feet = {}
    for foot in FEET:
        data = [row["foot_contacts"]["by_foot"][foot] for row in reports]
        feet[foot] = {"valid_events": {"baseline": data[0]["valid_event_count"],
                                       "candidate": data[1]["valid_event_count"]},
                      **{name: change(*(row[field]["p95"] for row in data))
                         for name, field in FOOT_METRICS.items()}}
    sampling = []
    for report in reports:
        total = report["physics_steps"] * report["num_envs"]
        scored = report["tcp_high_frequency"]["actual_tcp"]["samples"]
        sampling.append({
            "recorded_env_physics_samples": total, "scored_tcp_samples": scored,
            "fraction_scored": scored / total, "segmentation": report["segmentation"],
            "segments_with_under_1s_scored": [
                {key: row[key] for key in ("env_id", "episode_segment", "physics_samples", "scored_samples")}
                for row in report["episodes"]
                if row["scored_samples"] * report["settings"]["physics_dt_s"] < 1],
        })
    return {
        "schema_version": 1, "source_comparison": str(Path(comparison_path).resolve()),
        "baseline": {"label": "VD10 5998", "source_log_dir": reports[0]["source_log_dir"],
                     "checkpoint": metadata[0]["checkpoint"]},
        "candidate": {"label": "VD11 5999", "source_log_dir": reports[1]["source_log_dir"],
                      "checkpoint": metadata[1]["checkpoint"]},
        "percent_change_definition": "100 * (candidate / baseline - 1); delta=candidate-baseline",
        "settings": reports[0]["settings"], "tcp_and_gun_axis": tcp,
        "foot_contacts": feet,
        "total_valid_touchdown_events": change(*(row["foot_contacts"]["total_valid_events"] for row in reports)),
        "real_terminations": {"baseline": resets[0], "candidate": resets[1]},
        "sampling": {"baseline": sampling[0], "candidate": sampling[1]},
        "common_no_premature_reset_cohort": {
            "excluded_env_ids_union": sorted(excluded),
            "retained_env_ids": [env for env in range(metadata[0]["num_envs"]) if env not in excluded],
            "baseline_num_envs": cohorts[0]["num_envs"], "candidate_num_envs": cohorts[1]["num_envs"],
            "baseline_scored_samples": cohorts[0]["scored_samples"],
            "candidate_scored_samples": cohorts[1]["scored_samples"],
            "metrics": {key: change(*(row["metrics"][key] for row in cohorts)) for key in cohorts[0]["metrics"]},
            "interpretation": "Lateral tracking-error deterioration persists in the same 28 environment IDs. "
                              "This survivor-only sensitivity check does not replace the all-environment safety report.",
        },
        "limitations": [
            "One deterministic seed, 32 environments and 30 seconds do not establish convergence or deployment reliability.",
            "Filter episodes separately after 2 s startup and discard 0.5 s at both ends; every reset changes eligible "
            "exposure. Failure-adjacent transients can be excluded, even though post-reset episodes are retained.",
            "VD10 env7 first 1.2 s segment is excluded entirely; VD11 env14 final 3.06 s segment has only 13 scored "
            "samples (0.065 s), so its standalone band estimate is not a stable-duration metric.",
            "The 0.5 s filter edge guard does not guarantee elimination of all 0.5 Hz boundary transients.",
            "Actual TCP movement includes intended target motion. ZOH tracking error and diagnostic linear-target "
            "error must also be inspected; the linear target is an offline estimate, not measured truth.",
            "Gun-axis error is an unsigned angle; its band RMS cannot identify signed left/right rotation.",
            "Foot metrics use measured world-Z net force, not joint drive torque; 50 ms impulse is not full-stance impulse.",
            "Touchdown event counts and gait distribution change. A smaller peak does not imply a smaller impulse "
            "or pre-touchdown speed: left-rear impulse and downward speed both increase.",
            "These Isaac Lab results must not be numerically equated with historical MuJoCo probes using different gun geometry.",
        ],
        "assessment": {
            "local_isolated_mujoco_diagnostic_candidate": True, "formal_deployment_approved": False,
            "reason": "Peak foot forces improve, but lateral tracking errors worsen and gun-support failures occur "
                      "in 4/32 initial episodes (5 events). Test only an isolated MuJoCo candidate; retain default actor.",
        },
    }


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#64748b", "#0e7490")
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.3))

    def bars(ax, labels, pairs, title, unit):
        positions = np.arange(len(labels))
        for index, (label, color) in enumerate(zip(("VD10 5998", "VD11 5999"), colors)):
            values = [pair[("baseline", "candidate")[index]] for pair in pairs]
            artists = ax.bar(positions + (index - .5) * .34, values, width=.34, color=color, label=label)
            ax.bar_label(artists, fmt="%.2f" if max(values) < 20 else "%.0f", padding=3, fontsize=8)
        ax.set_xticks(positions, labels)
        ax.set_title(title, fontsize=10, loc="left", weight="bold")
        ax.set_ylabel(unit)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=.18)
        ax.set_axisbelow(True)

    signals = summary["tcp_and_gun_axis"]
    bars(axes[0, 0], ["0.5-3 Hz", ">3 Hz"],
         [signals["actual_tcp_lateral_low_band_rms_mm"], signals["actual_tcp_lateral_highpass_rms_mm"]],
         "A  Actual TCP: lateral motion", "RMS (mm)")
    bars(axes[0, 1], ["Raw error", "0.5-3 Hz error"],
         [signals["raw_tracking_error_lateral_rms_mm"], signals["tracking_error_lateral_low_band_rms_mm"]],
         "B  Lateral tracking error (logged target)", "RMS (mm)")
    bars(axes[0, 2], FEET, [summary["foot_contacts"][foot]["force_peak_p95_n"] for foot in FEET],
         "C  Touchdown: 50 ms peak force", "p95 (N)")
    bars(axes[1, 0], FEET, [summary["foot_contacts"][foot]["impulse_p95_n_s"] for foot in FEET],
         "D  Touchdown: first 50 ms impulse", "p95 (N s)")
    bars(axes[1, 1], FEET, [summary["foot_contacts"][foot]["pre_touchdown_downward_speed_p95_m_s"] for foot in FEET],
         "E  Maximum downward speed in pre-20 ms", "p95 (m/s)")
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("F  Safety outcomes: first 30 s episode", fontsize=10, loc="left", weight="bold")
    ax.text(0, .91, "VD10: 31/32 timeout; 1 base-contact failure\n"
            "VD11: 28/32 timeout; 4 gun-support failures\n"
            "        5 gun-support events over all reset episodes\n\n"
            "Common 28 non-failing environment IDs:\n"
            "lateral 0.5-3 Hz tracking error 1.96 -> 2.34 mm\n\n"
            "Isolated MuJoCo diagnostic candidate only.\n"
            "Not accepted as the default deployment actor.",
            va="top", fontsize=10, linespacing=1.45, transform=ax.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper right", frameon=False, bbox_to_anchor=(.99, .995))
    fig.suptitle("VD11 final review: smaller force peaks, unresolved lateral tracking and tool support", fontsize=14,
                 x=.035, ha="left", y=1.035)
    fig.text(.035, -.015, "Isaac Lab | 32 env x 30 s | deterministic | 200 Hz | full ground blend | "
             "reset-aware filters; single seed; force/impulse p95 pooled over touchdown events", fontsize=9)
    fig.tight_layout(pad=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(summary, args.figure)
    print(f"Summary: {args.output.resolve()}")
    print(f"Figure: {args.figure.resolve()}")


if __name__ == "__main__":
    main()
