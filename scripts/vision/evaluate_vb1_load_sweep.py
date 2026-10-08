#!/usr/bin/env python3
"""Run fixed-load VB1 synthetic-curve checks and summarize motion traces."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize(trace_path):
    with trace_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Empty motion trace: {trace_path}")
    metrics_to_check = (
        "base_x", "base_y", "base_z", "target_x", "target_y", "target_z",
        "actual_x", "actual_y", "actual_z", "path_lateral_error_b",
        "hose_mass_kg", "hose_force_n", "hose_torque_nm", "hose_tension_n", "body_tilt_deg",
    )
    for index, row in enumerate(rows):
        if any(not math.isfinite(float(row[key])) for key in metrics_to_check):
            raise RuntimeError(f"Non-finite motion or hose data at row {index + 2} of {trace_path}")
    tcp_errors = []
    lateral_errors = []
    travel_m = 0.0
    for index, row in enumerate(rows):
        tcp_errors.append(math.sqrt(sum(
            (float(row[f"target_{axis}"]) - float(row[f"actual_{axis}"])) ** 2
            for axis in ("x", "y", "z")
        )))
        lateral_errors.append(abs(float(row["path_lateral_error_b"])))
        if index:
            prev = rows[index - 1]
            travel_m += math.hypot(
                float(row["base_x"]) - float(prev["base_x"]),
                float(row["base_y"]) - float(prev["base_y"]),
            )
    return {
        "steps": len(rows),
        "sim_time_s": float(rows[-1]["time_s"]),
        "base_travel_xy_m": travel_m,
        "resets": sum(int(row["reset"]) for row in rows),
        "tcp_error_p95_m": percentile(tcp_errors, 0.95),
        "lateral_error_p95_m": percentile(lateral_errors, 0.95),
        "max_body_tilt_deg": max(float(row["body_tilt_deg"]) for row in rows),
        "mean_hose_force_n": sum(float(row["hose_force_n"]) for row in rows) / len(rows),
        "max_hose_torque_nm": max(float(row["hose_torque_nm"]) for row in rows),
        "max_hose_tension_n": max(float(row["hose_tension_n"]) for row in rows),
    }


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--masses", type=float, nargs="+", default=[1, 3, 5, 7, 10])
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.max_steps <= 0 or any(not 1.0 <= mass <= 10.0 for mass in args.masses):
        parser.error("Need positive --max-steps and hose masses within [1, 10] kg")
    output_dir = args.output_dir or repo / "outputs" / f"vb1_load_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"checkpoint": str(checkpoint), "results": {}}
    for mass in args.masses:
        label = f"{mass:g}kg"
        trace = output_dir / f"{label}.csv"
        log = output_dir / f"{label}.log"
        command = [
            sys.executable, str(repo / "scripts/rsl_rl/play_amp.py"),
            "--task", "Unitree-b2z1-AMP-VB1-Loaded-Gun-Play",
            "--checkpoint", str(checkpoint), "--num_envs", "1", "--headless",
            "--max-steps", str(args.max_steps), "--motion-trace-csv", str(trace),
        ]
        environment = os.environ.copy()
        environment["B2_Z1_HOSE_MASS_KG"] = str(mass)
        print(f"[VB1] Checking {label}; log: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != 0 or not trace.is_file():
            raise RuntimeError(f"{label} playback failed; inspect {log}")
        summary["results"][label] = summarize(trace)
        print(f"[VB1] {label}: {summary['results'][label]}", flush=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[VB1] Completed fixed-load sweep: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
