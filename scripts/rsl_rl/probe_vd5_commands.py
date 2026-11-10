"""Finite VD3/VD5 actor rollout; report physical state before automatic reset."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", choices=("VD3", "VD5", "VD5Startup"), default="VD5")
parser.add_argument("--reset-mode", choices=("nominal", "reference", "mixed"), default="nominal")
parser.add_argument("--reference-init-prob", type=float, default=None,
                    help="Reference-state fraction for mixed resets; default is 0.5.")
parser.add_argument("--restore-curve", action="store_true", help="Freeze the checkpoint's saved curve level.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--nominal-base-height", type=float, default=None,
                    help="For VD5Startup, replace the nominal reset base height in meters.")
parser.add_argument("--ground-blend", type=float, default=None,
                    help="Freeze the ground-target curriculum at a value in [0, 1].")
parser.add_argument("--seam-metrics", action="store_true",
                    help="Measure world-ground seam distance and actual base speed; does not change control.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import json
import torch
import gymnasium as gym

import b2_z1.tasks  # noqa: F401
from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import tcp_pose_b, tcp_pose_w
from isaaclab_tasks.utils import parse_env_cfg


def ground_path_distance(points_w, positions_w, *, extend_start=True):
    """XY distance to the polyline, optionally extending its first tangent."""
    starts = points_w[:, :-1, :2]
    segments = points_w[:, 1:, :2] - starts
    fraction = ((positions_w[:, None, :2] - starts) * segments).sum(-1)
    fraction = fraction / segments.square().sum(-1).clamp_min(1.0e-12)
    bounded = fraction.clamp(0.0, 1.0)
    if extend_start:
        bounded[:, 0] = fraction[:, 0].clamp(max=1.0)
    closest = starts + bounded[..., None] * segments
    return (closest - positions_w[:, None, :2]).norm(dim=-1).min(dim=1).values


def main():
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError("steps and num_envs must be positive")
    if args.reference_init_prob is not None and not 0.0 <= args.reference_init_prob <= 1.0:
        raise ValueError("reference-init-prob must be in [0, 1]")
    if args.reference_init_prob is not None and args.reset_mode != "mixed":
        raise ValueError("reference-init-prob applies only to mixed reset mode")
    if args.ground_blend is not None and not 0.0 <= args.ground_blend <= 1.0:
        raise ValueError("ground-blend must be in [0, 1]")
    task_id = {"VD3": "Unitree-b2z1-Velocity", "VD5": "Unitree-b2z1-VD5-Velocity",
               "VD5Startup": "Unitree-b2z1-VD5-Startup-Velocity"}[args.task]
    cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.reference_state_initialization = args.reset_mode != "nominal"
    if args.reset_mode == "mixed":
        cfg.reference_state_initialization_prob = (
            0.5 if args.reference_init_prob is None else args.reference_init_prob
        )
    if args.nominal_base_height is not None:
        if args.task != "VD5Startup" or not 0.0 < args.nominal_base_height < 1.0:
            raise ValueError("--nominal-base-height applies only to VD5Startup and must be in (0, 1)")
        cfg.vd5_nominal_base_height_m = args.nominal_base_height
        cfg.scene.robot.init_state.pos = (*cfg.scene.robot.init_state.pos[:2], args.nominal_base_height)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.ee_target.debug_vis = False
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if args.restore_curve:
        saved = checkpoint.get("vd5_curriculum_state")
        if saved is None:
            raise ValueError("--restore-curve requires a VD5 checkpoint with saved curriculum")
        level = saved["common_curriculum"]["curve_level"]
        cfg.commands.base_velocity.curriculum_enabled = False
        cfg.commands.base_velocity.trajectory_curve_offset_scale = level
        cfg.commands.base_velocity.limit_trajectory_curve_offset_scale = level
    if "obs_norm_state_dict" in checkpoint:
        raise ValueError("This probe requires a checkpoint without observation normalization")
    # Same ELU actor layout as the VD3 checkpoint; do not load optimizer/AMP training state.
    actor = torch.nn.Sequential(
        torch.nn.Linear(70, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
        torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 18),
    ).to(args.device).eval()
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items()
                           if k.startswith("actor.")}, strict=True)
    env = gym.make(task_id, cfg=cfg)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    ee = raw.command_manager.get_term("ee_target")
    base_command = raw.command_manager.get_term("base_velocity")
    if args.ground_blend is not None:
        if args.task not in ("VD5", "VD5Startup"):
            raise ValueError("ground-blend applies only to VD5")
        ee._vd5_ground_blend = args.ground_blend
        ee._vd5_ground_transition_steps = 0
    sensor = raw.scene.sensors["contact_forces"]
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    if args.seam_metrics:
        foot_ids, _ = robot.find_bodies(foot_names, preserve_order=True)
        foot_sensor_ids = [sensor.body_names.index(name) for name in foot_names]
    illegal_cfg = raw.termination_manager.get_term_cfg("base_contact").params["sensor_cfg"]
    illegal_ids = illegal_cfg.body_ids
    illegal_names = [sensor.body_names[i] for i in illegal_ids]
    contact_counts = {name: 0 for name in illegal_names}
    tool_id = sensor.body_names.index("link06")
    samples = []
    previous_world = None
    term_counts = {name: 0 for name in raw.termination_manager.active_terms}
    original_compute = raw.termination_manager.compute

    def capture():
        nonlocal previous_world
        done = original_compute()
        target = raw.command_manager.get_command("ee_target")[:, :3]
        actual, _ = tcp_pose_b(robot)
        error = actual - target
        world = ee._get_target_pos_w().clone()
        seam_values = {}
        foot_values = None
        if args.seam_metrics:
            tcp_w, _ = tcp_pose_w(robot)
            path = base_command.curve_points_w
            base_w = robot.data.body_pos_w[:, base_command.base_idx]
            seam_values = {
                "tcp_seam_projection_error_m": ground_path_distance(path, tcp_w),
                "base_cross_track_m": ground_path_distance(path, base_w),
                "tcp_world_target_xy_error_m": (tcp_w[:, :2] - world[:, :2]).norm(dim=-1),
                "tcp_height_above_seam_m": tcp_w[:, 2] - path[:, 0, 2],
                "actual_forward_speed_mps": robot.data.root_lin_vel_b[:, 0],
                "actual_lateral_speed_mps": robot.data.root_lin_vel_b[:, 1],
                "command_forward_speed_mps": base_command.command[:, 0],
                "actual_yaw_rate_rps": robot.data.root_ang_vel_b[:, 2],
                "command_yaw_rate_rps": base_command.command[:, 2],
            }
            foot_positions = getattr(robot.data, "body_link_pos_w", robot.data.body_pos_w)[:, foot_ids]
            foot_distances = torch.stack(
                [ground_path_distance(path, foot_positions[:, j], extend_start=False) for j in range(4)], dim=1
            )
            foot_contact = sensor.data.net_forces_w[:, foot_sensor_ids, 2] > 10.0
            foot_values = (foot_distances, foot_contact)
        jump = torch.zeros(raw.num_envs, device=raw.device)
        if previous_world is not None:
            jump = (world - previous_world).norm(dim=-1)
            jump = torch.where(raw.episode_length_buf > 2, jump, torch.zeros_like(jump))
        previous_world = world
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)).rad2deg()
        history = sensor.data.net_forces_w_history
        illegal_force = history[:, :, illegal_ids].norm(dim=-1).amax(dim=1)
        peak_force, peak_body = illegal_force.max(dim=-1)
        for body_index, name in enumerate(illegal_names):
            contact_counts[name] += int((illegal_force[:, body_index] > 1.0).sum())
        for name in term_counts:
            # get_term retains the last episode's attribution between resets.
            term_counts[name] += int((raw.termination_manager.get_term(name) & done).sum())
        for i in range(raw.num_envs):
            samples.append({"step": raw.common_step_counter, "env": i,
                            "episode_step": int(raw.episode_length_buf[i]),
                            "target_b": target[i].tolist(), "tcp_error_m": float(error[i].norm()),
                            "tcp_horizontal_target_error_m": float(error[i, :2].norm()),
                            "tcp_vertical_target_error_m": float(error[i, 2].abs()),
                            "target_world_step_m": float(jump[i]), "tilt_deg": float(tilt[i]),
                            "base_height_m": float(robot.data.root_pos_w[i, 2]),
                            "peak_illegal_contact_body": illegal_names[int(peak_body[i])],
                            "peak_illegal_contact_n": float(peak_force[i]),
                            "tool_upward_force_n": float(history[i, :, tool_id, 2].max()),
                            "ground_blend": getattr(ee, "_vd5_ground_blend", None),
                            "terminated": bool(done[i]),
                            **{key: float(value[i]) for key, value in seam_values.items()}})
            if foot_values is not None:
                samples[-1]["foot_seam_center_distances_m"] = foot_values[0][i].tolist()
                samples[-1]["foot_ground_contact_gt10n"] = foot_values[1][i].tolist()
        return done

    raw.termination_manager.compute = capture
    try:
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, _, _, _, _ = env.step(actor(obs["policy"]))
        def stats(key):
            values = torch.tensor([s[key] for s in samples])
            return {"mean": float(values.mean()), "p95": float(values.quantile(.95)), "max": float(values.max())}
        report = {"checkpoint": str(Path(args.checkpoint).resolve()), "task": args.task, "steps": args.steps,
                  "reset_mode": args.reset_mode, "seed": args.seed,
                  "reference_state_initialization": cfg.reference_state_initialization,
                  "reference_state_initialization_prob": cfg.reference_state_initialization_prob,
                  "nominal_base_height_m": getattr(cfg, "vd5_nominal_base_height_m", None),
                  "curve_level": raw.command_manager.get_term("base_velocity").cfg.trajectory_curve_offset_scale,
                  "startup_duration_s": getattr(ee.cfg, "vd5_startup_duration_s", 0.0),
                  "num_envs": args.num_envs, "camera_assets_retained": True,
                  "ground_blend": getattr(ee, "_vd5_ground_blend", None),
                  "trailing_distance_m": getattr(ee, "trailing_distance", None),
                  "image_input": False, "termination_counts": term_counts,
                  "illegal_contact_body_counts": contact_counts,
                  "reset_count": sum(s["terminated"] for s in samples),
                  "termination_count_note": "Framework final attribution at actual resets; simultaneous causes may be overwritten.",
                  "max_episode_steps": max(s["episode_step"] for s in samples),
                  "base_height_m": stats("base_height_m"),
                  "tcp_horizontal_target_error_m": stats("tcp_horizontal_target_error_m"),
                  "tcp_vertical_target_error_m": stats("tcp_vertical_target_error_m"),
                  "tcp_error_m": stats("tcp_error_m"), "target_world_step_m": stats("target_world_step_m"),
                  "tilt_deg": stats("tilt_deg"), "samples": samples}
        if args.seam_metrics:
            report["world_seam_metrics"] = {
                "definition": "World XY distance to the actual path polyline, extended along its first tangent before the start.",
                "settled_after_s": 2.0,
                "note": "Use first-episode samples after 2 s for steady walking; startup remains in raw samples.",
                "foot_names": foot_names,
                "foot_measurement": "XY distance from foot link origin (sphere center) to the finite seam polyline, without backward extension; contact is measured upward force >10 N.",
                "seam_width_m": base_command.cfg.ground_curve_width,
            }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
