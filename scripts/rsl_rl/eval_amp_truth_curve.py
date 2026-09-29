"""Compare AMP checkpoints on the final VA truth-curve distribution."""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--foot_radius", type=float, default=0.032)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from b2_z1.rsl_rl import AMP_RslRlOnPolicyRunnerCfg, AMPRslRlVecEnvWrapper
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import b2_z1.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: AMP_RslRlOnPolicyRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.reference_state_initialization_prob = 0.0
    env_cfg.episode_length_s = 60.0

    # Evaluate every checkpoint on the hardest VA1 envelope.
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False
    path_cfg = env_cfg.commands.ee_target
    path_cfg.debug_vis = False
    path_cfg.synthetic_straight_path_fraction = 0.40
    path_cfg.synthetic_arc_radius_range_m = (6.0, 12.0)
    path_cfg.synthetic_speed_range_mps = (0.20, 0.38)
    path_cfg.max_yaw_rate_rps = 0.18
    path_cfg.cross_track_gain = 0.50
    path_cfg.path_heading_gain = 0.90

    # Remove random disturbances so checkpoint differences reflect policy and
    # curve following rather than a different mass/push draw.
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.actuator_gains = None
    env_cfg.events.joint_dynamics = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = AMPRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    contact = raw.scene.sensors["contact_forces"]
    base_command = raw.command_manager.get_term("base_velocity")
    path_command = raw.command_manager.get_term("ee_target")
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    sensor_foot_ids, sensor_foot_names = contact.find_bodies(".*_foot")
    if robot_foot_names != sensor_foot_names:
        raise RuntimeError("robot and contact sensor foot ordering differs")
    nonfoot_sensor_ids = [
        index
        for index, name in enumerate(contact.body_names)
        if name not in sensor_foot_names
    ]
    hip_joint_ids, _ = robot.find_joints(".*_hip_joint")
    thigh_joint_ids, _ = robot.find_joints(".*_thigh_joint")
    calf_joint_ids, _ = robot.find_joints(".*_calf_joint")
    arm_joint_ids, _ = robot.find_joints(
        [f"joint{index}" for index in range(1, 7)], preserve_order=True
    )
    calf_body_ids, _ = robot.find_bodies(".*_calf")
    thigh_body_ids, _ = robot.find_bodies(".*_thigh")

    runner = AMP_OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    reports = []
    for checkpoint in args_cli.checkpoints:
        path = retrieve_file_path(checkpoint)
        runner.load(path, load_optimizer=False)
        policy = runner.get_inference_policy(device=raw.device)

        # A timeout reset gives every checkpoint a fresh nominal pose and a
        # newly sampled truth curve. Re-seeding makes the curve draw paired
        # across checkpoints rather than merely statistically similar.
        torch.manual_seed(agent_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(agent_cfg.seed)
        raw.episode_length_buf[:] = raw.max_episode_length - 1
        with torch.inference_mode():
            obs, _, _, _ = env.step(
                torch.zeros((raw.num_envs, env.num_actions), device=raw.device)
            )

        totals = {
            "vel_error_sq": 0.0,
            "target_speed": 0.0,
            "actual_along_path": 0.0,
            "actual_cross_path": 0.0,
            "yaw_error_sq": 0.0,
            "tcp_error": 0.0,
            "path_lateral": 0.0,
            "path_heading": 0.0,
            "tilt": 0.0,
            "base_height": 0.0,
            "minimum_base_height": float("inf"),
            "nonfoot_contact": 0.0,
            "hip_abs": 0.0,
            "thigh_abs": 0.0,
            "calf_abs": 0.0,
            "leg_default_deviation": 0.0,
            "minimum_calf_height": float("inf"),
            "minimum_thigh_height": float("inf"),
            "front_clearance": 0.0,
            "rear_clearance": 0.0,
            "front_clearance_count": 0,
            "rear_clearance_count": 0,
            "diagonal_mismatch": 0.0,
            "long_contact": 0.0,
            "resets": 0,
            "arm_velocity_sq": 0.0,
        }
        total_steps = args_cli.warmup_steps + args_cli.eval_steps
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            if step < args_cli.warmup_steps:
                continue

            actual_xy = robot.data.root_lin_vel_b[:, :2]
            target_xy = base_command.command[:, :2]
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            target_yaw = base_command.command[:, 2]
            totals["vel_error_sq"] += torch.square(
                actual_xy - target_xy
            ).sum(dim=1).sum().item()
            target_speed = torch.linalg.vector_norm(target_xy, dim=1)
            target_direction = target_xy / target_speed.unsqueeze(1).clamp_min(1.0e-6)
            actual_along = torch.sum(actual_xy * target_direction, dim=1)
            actual_cross = (
                -actual_xy[:, 0] * target_direction[:, 1]
                + actual_xy[:, 1] * target_direction[:, 0]
            )
            totals["target_speed"] += target_speed.sum().item()
            totals["actual_along_path"] += actual_along.sum().item()
            totals["actual_cross_path"] += torch.abs(actual_cross).sum().item()
            totals["yaw_error_sq"] += torch.square(
                actual_yaw - target_yaw
            ).sum().item()
            totals["tcp_error"] += path_command.metrics["position_error"].sum().item()
            totals["path_lateral"] += path_command.metrics[
                "path_lateral_error"
            ].sum().item()
            totals["path_heading"] += path_command.metrics[
                "path_heading_error"
            ].sum().item()
            gravity_z = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
            totals["tilt"] += torch.acos(-gravity_z).sum().item()
            base_height = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            totals["base_height"] += base_height.sum().item()
            totals["minimum_base_height"] = min(
                totals["minimum_base_height"], float(base_height.min().item())
            )
            if nonfoot_sensor_ids:
                nonfoot_force = torch.linalg.vector_norm(
                    contact.data.net_forces_w[:, nonfoot_sensor_ids], dim=-1
                )
                totals["nonfoot_contact"] += (
                    nonfoot_force > 10.0
                ).any(dim=1).float().sum().item()
            joint_pos = robot.data.joint_pos
            totals["arm_velocity_sq"] += torch.square(
                robot.data.joint_vel[:, arm_joint_ids]
            ).sum().item()
            default_joint_pos = robot.data.default_joint_pos
            totals["hip_abs"] += torch.abs(
                joint_pos[:, hip_joint_ids]
            ).mean(dim=1).sum().item()
            totals["thigh_abs"] += torch.abs(
                joint_pos[:, thigh_joint_ids]
            ).mean(dim=1).sum().item()
            totals["calf_abs"] += torch.abs(
                joint_pos[:, calf_joint_ids]
            ).mean(dim=1).sum().item()
            leg_joint_ids = hip_joint_ids + thigh_joint_ids + calf_joint_ids
            totals["leg_default_deviation"] += torch.abs(
                joint_pos[:, leg_joint_ids] - default_joint_pos[:, leg_joint_ids]
            ).mean(dim=1).sum().item()
            ground_height = raw.scene.env_origins[:, 2].unsqueeze(1)
            calf_height = robot.data.body_pos_w[:, calf_body_ids, 2] - ground_height
            thigh_height = robot.data.body_pos_w[:, thigh_body_ids, 2] - ground_height
            totals["minimum_calf_height"] = min(
                totals["minimum_calf_height"], float(calf_height.min().item())
            )
            totals["minimum_thigh_height"] = min(
                totals["minimum_thigh_height"], float(thigh_height.min().item())
            )
            totals["resets"] += int(dones.sum().item())

            ground_z = raw.scene.env_origins[:, 2].unsqueeze(1)
            clearance = (
                robot.data.body_pos_w[:, robot_foot_ids, 2]
                - ground_z
                - args_cli.foot_radius
            )
            air_time = contact.data.current_air_time[:, sensor_foot_ids]
            contact_time = contact.data.current_contact_time[:, sensor_foot_ids]
            swing = (contact_time <= 0.0) & (air_time >= 0.04)
            front_mask = torch.tensor(
                [name.startswith("F") for name in robot_foot_names],
                device=raw.device,
            )
            rear_mask = ~front_mask
            for region, foot_mask in (("front", front_mask), ("rear", rear_mask)):
                selected = clearance[:, foot_mask][swing[:, foot_mask]]
                if selected.numel():
                    totals[f"{region}_clearance"] += selected.sum().item()
                    totals[f"{region}_clearance_count"] += selected.numel()

            active_contact = contact_time > 0.0
            names = {name: index for index, name in enumerate(sensor_foot_names)}
            mismatch = (
                active_contact[:, names["FL_foot"]]
                != active_contact[:, names["RR_foot"]]
            ).float() + (
                active_contact[:, names["FR_foot"]]
                != active_contact[:, names["RL_foot"]]
            ).float()
            totals["diagonal_mismatch"] += mismatch.sum().item()
            totals["long_contact"] += (contact_time > 0.32).float().sum().item()

        samples = raw.num_envs * args_cli.eval_steps
        reports.append(
            {
                "checkpoint": path,
                "velocity_rmse": math.sqrt(totals["vel_error_sq"] / samples),
                "target_speed": totals["target_speed"] / samples,
                "actual_along_path": totals["actual_along_path"] / samples,
                "actual_cross_path": totals["actual_cross_path"] / samples,
                "yaw_rmse": math.sqrt(totals["yaw_error_sq"] / samples),
                "tcp_error_m": totals["tcp_error"] / samples,
                "path_lateral_m": totals["path_lateral"] / samples,
                "path_heading_deg": math.degrees(totals["path_heading"] / samples),
                "tilt_deg": math.degrees(totals["tilt"] / samples),
                "base_height_m": totals["base_height"] / samples,
                "minimum_base_height_m": totals["minimum_base_height"],
                "nonfoot_contact": totals["nonfoot_contact"] / samples,
                "mean_abs_hip_rad": totals["hip_abs"] / samples,
                "mean_abs_thigh_rad": totals["thigh_abs"] / samples,
                "mean_abs_calf_rad": totals["calf_abs"] / samples,
                "leg_default_deviation_rad": totals["leg_default_deviation"] / samples,
                "minimum_calf_height_m": totals["minimum_calf_height"],
                "minimum_thigh_height_m": totals["minimum_thigh_height"],
                "front_clearance_cm": 100.0
                * totals["front_clearance"]
                / max(totals["front_clearance_count"], 1),
                "rear_clearance_cm": 100.0
                * totals["rear_clearance"]
                / max(totals["rear_clearance_count"], 1),
                "diagonal_mismatch": totals["diagonal_mismatch"] / (2.0 * samples),
                "long_contact": totals["long_contact"] / (4.0 * samples),
                "resets": totals["resets"],
                "arm_velocity_rms": math.sqrt(
                    totals["arm_velocity_sq"] / (6.0 * samples)
                ),
            }
        )

    print("\n========== VA truth-curve checkpoint gate ==========")
    for report in reports:
        print(f"\ncheckpoint: {report['checkpoint']}")
        print(
            "target/actual_along={target_speed:.3f}/{actual_along_path:.3f} m/s  "
            "cross_speed={actual_cross_path:.3f} m/s  velocity_rmse={velocity_rmse:.3f} m/s  "
            "yaw_rmse={yaw_rmse:.3f} rad/s  "
            "tcp_error={tcp_error_m:.3f} m  path_lateral={path_lateral_m:.3f} m  "
            "path_heading={path_heading_deg:.2f} deg".format(**report)
        )
        print(
            "tilt={tilt_deg:.2f} deg  base_height(mean/min)="
            "{base_height_m:.3f}/{minimum_base_height_m:.3f} m  "
            "nonfoot_contact={nonfoot_contact:.3f}  front/rear_clearance="
            "{front_clearance_cm:.2f}/{rear_clearance_cm:.2f} cm  "
            "diagonal_mismatch={diagonal_mismatch:.3f}  "
            "long_contact={long_contact:.3f}  resets={resets}".format(**report)
        )
        print(
            "mean_abs_joint(hip/thigh/calf)="
            "{mean_abs_hip_rad:.3f}/{mean_abs_thigh_rad:.3f}/"
            "{mean_abs_calf_rad:.3f} rad  default_leg_deviation="
            "{leg_default_deviation_rad:.3f} rad".format(**report)
        )
        print(
            "minimum_body_height(calf/thigh)="
            "{minimum_calf_height_m:.3f}/{minimum_thigh_height_m:.3f} m".format(
                **report
            )
        )
        print("arm_joint_velocity_rms={arm_velocity_rms:.3f} rad/s".format(**report))
    print("====================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
