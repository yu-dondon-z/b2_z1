"""Measure tool support and base motion before automatic resets, without rendering."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", required=True, help="Exported deterministic policy.pt from play.")
parser.add_argument("--output", required=True, help="Output filename stem for JSON and CSV.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--legacy-play", action="store_true", help="Reproduce the old permissive Play guards.")
parser.add_argument("--with-push", action="store_true", help="Keep interval velocity disturbances.")
parser.add_argument("--fixed-target", action="store_true", help="Use nominal TCP endpoint (-1.02, 0, -0.30) m.")
parser.add_argument("--settle-seconds", type=float, default=10.0, help="Exclude this much of each episode from settled statistics.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import hashlib
import json
import math

import gymnasium as gym
import numpy as np
import torch

import b2_z1.tasks  # noqa: F401
from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import link_origin_velocity_w, tcp_pose_w, tcp_pose_b
from b2_z1.tasks.manager_based.b2_z1.mdp.vd3_arm_rewards import (
    _flange_state_and_target, flange_velocity_errors,
)
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg


def main():
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num_envs must be positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        raise ValueError("settle-seconds must be finite and nonnegative")
    cfg = parse_env_cfg("Unitree-b2z1-Velocity", device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.reference_state_initialization = False
    cfg.observations.policy.enable_corruption = False
    cfg.scene.base_camera = None
    cfg.scene.rear_camera = None
    cfg.scene.contact_forces.history_length = 20
    cfg.commands.base_velocity.curriculum_enabled = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.turn_angle_range = (math.pi / 4, math.pi / 2)
    cfg.commands.base_velocity.turn_count_range = (3, 3)
    cfg.commands.base_velocity.max_curvature = 1.5
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.ee_target.curriculum_enabled = False
    cfg.commands.ee_target.debug_vis = False
    if args.fixed_target:
        for ranges in (cfg.commands.ee_target.ranges, cfg.commands.ee_target.limit_ranges):
            ranges.pos_x = (-1.02, -1.02)
            ranges.pos_y = (0.0, 0.0)
            ranges.pos_z = (-0.30, -0.30)
    if not args.with_push:
        cfg.events.push_robot = None
    if args.legacy_play:
        cfg.terminations.base_contact = None
        if hasattr(cfg.terminations, "tool_support"):
            cfg.terminations.tool_support = None
        cfg.terminations.bad_orientation.params["limit_angle"] = 1.0
    env = gym.make("Unitree-b2z1-Velocity", cfg=cfg)
    raw = env.unwrapped
    policy_path = Path(args.policy).resolve()
    policy = torch.jit.load(str(policy_path), map_location=raw.device).eval()
    robot = raw.scene["robot"]
    sensor = raw.scene.sensors["contact_forces"]
    tool_id = sensor.body_names.index("link06")
    foot_ids = [i for i, name in enumerate(sensor.body_names) if name.endswith("_foot")]
    arm_ids = [i for i, name in enumerate(sensor.body_names) if name.startswith("link0")]
    base_id = robot.body_names.index("base_link")
    arm_joint_ids, _ = robot.find_joints([f"joint{i}" for i in range(1, 7)])
    mass = robot.root_physx_view.get_masses().sum(dim=-1).to(raw.device)
    samples, forces, terminal_counts = [], [], {name: 0 for name in raw.termination_manager.active_terms}
    columns = ["time_s", "env", "episode_step", "command_vx", "vx_body", "vy_body",
               "roll_deg", "pitch_deg", "tilt_deg", "base_z", "tcp_z", "tool_fz",
               "tool_peak_fz", "tool_min_100ms_fz", "feet_fz", "arm_fz", "tool_weight_fraction",
               "terminated", "timeout", "tcp_error", "tcp_orientation_error_rad", "flange_error",
               "flange_linear_velocity_error", "flange_angular_velocity_error", "arm_joint_velocity_rms",
               "tcp_error_x", "tcp_error_y", "tcp_error_z",
               "flange_error_x", "flange_error_y", "flange_error_z"]
    original_compute = raw.termination_manager.compute

    def capture_before_reset():
        done = original_compute()
        history = sensor.data.net_forces_w_history
        tool_history = history[:, :, tool_id, 2].clamp_min(0)
        # Only the newest four physics samples belong to this policy step.
        tool_recent = tool_history[:, :cfg.decimation]
        q = robot.data.body_link_quat_w[:, base_id]
        vel, _ = link_origin_velocity_w(robot, base_id)
        vel_b = quat_apply_inverse(q, vel)
        roll, pitch, _ = euler_xyz_from_quat(q)
        roll = (roll + math.pi) % (2 * math.pi) - math.pi
        pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        tcp_pos, _ = tcp_pose_w(robot)
        actual_tcp_b, actual_tcp_quat_b = tcp_pose_b(robot)
        desired_tcp = raw.command_manager.get_command("ee_target")
        tcp_error = actual_tcp_b - desired_tcp[:, :3]
        orientation_dot = (actual_tcp_quat_b * desired_tcp[:, 3:7]).sum(dim=-1).abs().clamp(0, 1)
        orientation_error = 2 * torch.acos(orientation_dot)
        _, _, _, _, flange_error, _ = _flange_state_and_target(raw, "ee_target", "base_link", "link06")
        flange_linear, flange_angular = flange_velocity_errors(raw)
        arm_rms = robot.data.joint_vel[:, arm_joint_ids].square().mean(dim=-1).sqrt()
        tool_mean = tool_recent.mean(dim=1)
        values = torch.stack((
            torch.full_like(tool_mean, raw.common_step_counter * raw.step_dt),
            torch.arange(raw.num_envs, device=raw.device), raw.episode_length_buf,
            raw.command_manager.get_command("base_velocity")[:, 0], vel_b[:, 0], vel_b[:, 1],
            roll.rad2deg(), pitch.rad2deg(), tilt.rad2deg(), robot.data.body_link_pos_w[:, base_id, 2],
            tcp_pos[:, 2], tool_mean, tool_recent.amax(dim=1), tool_history.amin(dim=1),
            history[:, :cfg.decimation, foot_ids, 2].clamp_min(0).sum(dim=-1).mean(dim=1),
            history[:, :cfg.decimation, arm_ids, 2].clamp_min(0).sum(dim=-1).mean(dim=1),
            tool_mean / (mass * 9.81), raw.termination_manager.terminated, raw.termination_manager.time_outs,
            tcp_error.norm(dim=-1), orientation_error, flange_error.norm(dim=-1),
            flange_linear.norm(dim=-1), flange_angular.norm(dim=-1), arm_rms,
            *tcp_error.unbind(dim=-1), *flange_error.unbind(dim=-1),
        ), dim=-1)
        samples.append(values.cpu().numpy())
        forces.append(tool_recent.cpu().numpy())
        for name in terminal_counts:
            terminal_counts[name] += int((raw.termination_manager.get_term(name) & done).sum().item())
        return done

    raw.termination_manager.compute = capture_before_reset
    try:
        obs, _ = env.reset(seed=args.seed)
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, _, _, _, _ = env.step(policy(obs["policy"]))
        values = np.concatenate(samples)
        trace = np.stack(forces)
        col = {name: values[:, i] for i, name in enumerate(columns)}
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(values)
        # These are time samples, rather than reset-time training summaries.
        summary = {
            "policy": str(policy_path), "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "num_envs": args.num_envs, "steps_per_env": args.steps, "seconds_per_env": args.steps * raw.step_dt,
            "seed": args.seed, "legacy_play_guards": args.legacy_play, "push_enabled": args.with_push,
            "policy_observation_corruption": cfg.observations.policy.enable_corruption,
            "fixed_target": args.fixed_target, "settle_seconds": args.settle_seconds,
            "endpoint_ranges": {axis: list(getattr(raw.command_manager.get_term("ee_target").cfg.ranges, axis)) for axis in ("pos_x", "pos_y", "pos_z")},
            "body_names": robot.body_names, "mass_kg": mass.cpu().tolist(),
            "termination_counts": terminal_counts,
            "mean_forward_velocity_m_s": float(col["vx_body"].mean()),
            "backward_time_fraction": float((col["vx_body"] < -.02).mean()),
            "mean_base_height_m": float(col["base_z"].mean()),
            "pitch_deg_mean_p05_p95": [float(x) for x in (col["pitch_deg"].mean(), *np.quantile(col["pitch_deg"], [.05, .95]))],
            "tilt_deg_mean_p95_max": [float(x) for x in (col["tilt_deg"].mean(), np.quantile(col["tilt_deg"], .95), col["tilt_deg"].max())],
            "tool_fz_mean_p95_max_N": [float(trace.mean()), float(np.quantile(trace, .95)), float(trace.max())],
            "tool_force_gt_20N_fraction": float((trace > 20).mean()),
            "tool_force_gt_40N_fraction": float((trace > 40).mean()),
            "tool_force_gt_150N_fraction": float((trace > 150).mean()),
            "tool_sustained_100ms_gt_40N_fraction": float((col["tool_min_100ms_fz"] > 40).mean()),
            "tool_sustained_100ms_gt_20N_fraction": float((col["tool_min_100ms_fz"] > 20).mean()),
            "mean_tool_weight_fraction": float(col["tool_weight_fraction"].mean()),
            "terminated_episodes": int(col["terminated"].sum()), "timeouts": int(col["timeout"].sum()),
            "note": "Four physics force samples per policy step; state captured before automatic reset. Termination counts use framework final attribution, not independent simultaneous causes. Legacy mode is diagnostic only. No training.",
        }
        for label, mask in (
            ("all_time", np.ones(len(values), dtype=bool)),
            ("after_settle", col["episode_step"] * raw.step_dt >= args.settle_seconds),
        ):
            stats = {"samples": int(mask.sum()), "aggregate_env_seconds": float(mask.sum() * raw.step_dt)}
            if mask.any():
                for name in ("tcp_error", "tcp_orientation_error_rad", "flange_error", "flange_linear_velocity_error",
                             "flange_angular_velocity_error", "arm_joint_velocity_rms", "vx_body", "tilt_deg", "tool_fz"):
                    arr = col[name][mask]
                    stats[name] = {"mean": float(arr.mean()), "p95": float(np.quantile(arr, .95)),
                                   "rms": float(np.sqrt(np.mean(arr ** 2)))}
                # De-mean each individual episode, avoiding between-environment bias as apparent oscillation.
                for prefix in ("tcp_error", "flange_error"):
                    errors = np.stack([col[prefix + "_" + axis] for axis in "xyz"], axis=-1)
                    centered_squared, bias_vectors = [], []
                    for env_id in range(raw.num_envs):
                        ids = np.flatnonzero(col["env"] == env_id)
                        ends = np.flatnonzero((col["terminated"][ids] > 0) | (col["timeout"][ids] > 0)) + 1
                        segments = np.split(ids, ends)
                        for segment in segments:
                            segment = segment[mask[segment]]
                            if len(segment) < 2:
                                continue
                            bias = errors[segment].mean(axis=0)
                            bias_vectors.append(bias.tolist())
                            centered_squared.extend(np.sum((errors[segment] - bias) ** 2, axis=1))
                    stats[prefix + "_episode_bias_vectors_m"] = bias_vectors
                    stats[prefix + "_demeaned_rms_m"] = float(np.sqrt(np.mean(centered_squared))) if centered_squared else None
            summary[label] = stats
        summary["episode_records"] = []
        for env_id in range(raw.num_envs):
            ids = np.flatnonzero(col["env"] == env_id)
            for idx in ids[(col["terminated"][ids] > 0) | (col["timeout"][ids] > 0)]:
                summary["episode_records"].append({"env": env_id, "duration_s": float(col["episode_step"][idx] * raw.step_dt),
                                                   "failed": bool(col["terminated"][idx]), "timeout": bool(col["timeout"][idx])})
        summary["tracking_note"] = "Continuous pre-reset samples against the command actually used for the step. Fixed endpoints still use the original Bezier transition from the measured reset/resample pose. De-meaned error RMS is computed separately per episode; it is not a full oscillation frequency analysis."
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print("[SUPPORT PROBE] " + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
