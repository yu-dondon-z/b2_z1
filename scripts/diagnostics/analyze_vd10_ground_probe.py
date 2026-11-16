#!/usr/bin/env python3
"""Summarize actual VD10 probe resets and the matching 200 Hz contact trace."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def analyze(log_dir, control_dt=0.02, startup_s=2.0):
    probe = json.loads((log_dir / "vd10_policy_probe.json").read_text())
    trace = json.loads((log_dir / "vd10_200hz_trace.json").read_text())
    force = np.asarray([row["env_tool_up_force_n"] for row in trace])
    height = np.asarray([row["env_tube_surface_min_z_m"] for row in trace])
    age = np.asarray([row["env_episode_actor_step"] for row in trace]) * control_dt
    assert force.shape == height.shape == age.shape
    assert force.shape[1] == probe["num_envs"]

    previous_reset = np.zeros(probe["num_envs"], dtype=int)
    first_reset = {}
    events = []
    for reset in probe["reset_events"]:
        for env_id in reset["env_ids"]:
            causes = [name for name, ids in reset["attributed_causes"].items() if env_id in ids]
            event = {
                "env_id": env_id,
                "global_time_s": reset["actor_step"] * control_dt,
                "episode_age_s": float((reset["actor_step"] - previous_reset[env_id]) * control_dt),
                "attributed_causes": causes,
            }
            events.append(event)
            first_reset.setdefault(env_id, event)
            previous_reset[env_id] = reset["actor_step"]
    assert len(events) == probe["total_resets"]
    counted = Counter(cause for event in events for cause in event["attributed_causes"])
    assert dict(counted) == {k: v for k, v in probe["termination_events"].items() if v}

    def contact_stats(mask):
        selected_force = force[mask]
        selected_height = height[mask]
        return {
            "physics_env_samples": int(mask.sum()),
            "upward_force_over_1n_fraction": float((selected_force > 1).mean()),
            "upward_force_over_20n_fraction": float((selected_force > 20).mean()),
            "max_upward_force_n": float(selected_force.max()),
            "surface_height_quantiles_m": dict(zip(
                ("min", "p01", "median", "p99", "max"),
                np.quantile(selected_height, (0, 0.01, 0.5, 0.99, 1)).tolist(),
            )),
        }

    first_counts = Counter(cause for event in first_reset.values() for cause in event["attributed_causes"])
    return {
        "source_log_dir": str(log_dir.resolve()),
        "checkpoint": probe["checkpoint"],
        "action_mode": probe["action_mode"],
        "frozen_ground_blend": probe["frozen_ground_blend"],
        "num_envs": probe["num_envs"],
        "duration_s_per_env": probe["actor_steps"] * control_dt,
        "physics_steps": len(trace),
        "physics_rate_hz": len(trace) / (probe["actor_steps"] * control_dt),
        "total_resets": len(events),
        "all_episode_reset_causes": dict(counted),
        "first_episode_reset_causes": dict(first_counts),
        "first_episodes_still_running": probe["num_envs"] - len(first_reset),
        "first_episode_time_out_fraction": first_counts["time_out"] / probe["num_envs"],
        "startup_tool_resets": int(sum(
            "tool_support" in event["attributed_causes"] and event["episode_age_s"] <= startup_s
            for event in events
        )),
        "tool_reset_events": [event for event in events if "tool_support" in event["attributed_causes"]],
        "all_samples": contact_stats(np.ones(force.shape, dtype=bool)),
        "after_startup_samples": contact_stats(age >= startup_s),
        "limits": [
            probe["termination_events_note"],
            "A single fixed-seed 64-environment probe does not establish convergence or deployment readiness.",
            "Force fractions count physics/environment samples, not episodes or independent contact events.",
            "Net link06 upward contact force is measured; it is not actuator torque.",
            "Analytic cylinder height uses measured link/TCP poses; tiny negative values can occur at solver contact.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
