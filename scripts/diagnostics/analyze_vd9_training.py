#!/usr/bin/env python3
"""Summarize VD9 optimization trends without treating iteration count as convergence."""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    "Train/mean_reward", "Train/mean_episode_length",
    "Episode_Reward/track_lin_vel_xy",
    "Episode_Reward/end_effector_position_tracking_exp",
    "Episode_Reward/vd9_tcp_highpass_200hz",
    "Episode_Reward/vd9_foot_descent_200hz",
    "Episode_Termination/base_contact", "Episode_Termination/bad_orientation",
    "Metrics/ee_target/position_error", "Metrics/ee_target/base_inclination_rad",
    "Metrics/base_velocity/error_vel_xy", "Loss/learning_rate",
)


def analyze(log_dir: Path, window: int = 50):
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    available = set(events.Tags()["scalars"])
    missing = sorted(set(TAGS) - available)
    if missing:
        raise ValueError(f"Missing expected VD9 scalars: {missing}")
    series = {key: {point.step: point.value for point in events.Scalars(key)} for key in TAGS}
    steps = sorted(set.intersection(*(set(values) for values in series.values())))
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    stages = []
    for checkpoint in checkpoints:
        iteration = int(checkpoint.stem.split("_")[1])
        selected = [step for step in steps if iteration - window < step <= iteration]
        if not selected:
            continue
        stages.append({
            "checkpoint": str(checkpoint), "iteration": iteration,
            "first_step": selected[0], "last_step": selected[-1], "samples": len(selected),
            "trailing": {
                key: {"mean": float(np.mean([values[step] for step in selected])),
                      "median": float(np.median([values[step] for step in selected]))}
                for key, values in series.items()
            },
        })
    return {
        "log_dir": str(log_dir), "window_iterations": window, "stages": stages,
        "note": "Training scalars indicate optimization trends, not reward convergence or sim2sim quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for stage in result["stages"]:
        metrics = stage["trailing"]
        print(stage["iteration"], "n=", stage["samples"],
              "length=", round(metrics["Train/mean_episode_length"]["median"], 1),
              "tcp=", round(metrics["Metrics/ee_target/position_error"]["median"], 4),
              "tcp_hp=", round(metrics["Episode_Reward/vd9_tcp_highpass_200hz"]["median"], 5),
              "foot=", round(metrics["Episode_Reward/vd9_foot_descent_200hz"]["median"], 5))


if __name__ == "__main__":
    main()
