"""Evaluate forward/backward AMP gait tracking and physical swing clearance.

This is intentionally a deterministic, headless diagnostic rather than a
training job.  Half of the environments receive a fixed forward command and
the other half receive a fixed reverse command.  The report uses the sole of
the B2 foot (link origin minus the spherical foot radius), so clearance is
printed directly in centimetres instead of as a shaped reward value.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an AMP locomotion checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Gym registry entry point for the RSL-RL agent configuration.",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--forward_speed", type=float, default=0.25)
parser.add_argument("--backward_speed", type=float, default=-0.12)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--foot_radius", type=float, default=0.032)
parser.add_argument(
    "--force_amp_reset",
    action="store_true",
    help="Force one timeout reset so AMP reference-state initialization is actually applied.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from collections import defaultdict

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from b2_z1.rsl_rl import AMP_RslRlOnPolicyRunnerCfg, AMPRslRlVecEnvWrapper
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import b2_z1.tasks  # noqa: F401


def _mean(total: float, count: int) -> float:
    return total / max(count, 1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: AMP_RslRlOnPolicyRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # The deterministic evaluator is headless and does not need command arrows.
    # Disabling them also avoids an unnecessary online USD dependency.
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.ee_target.debug_vis = False

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = AMPRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = AMP_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    device = raw.device
    num_envs = raw.num_envs
    if num_envs < 2:
        raise ValueError("Use at least two environments so forward and backward are evaluated together.")

    command = raw.command_manager.get_term("base_velocity")
    robot = raw.scene["robot"]
    contact = raw.scene.sensors["contact_forces"]
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    sensor_foot_ids, sensor_foot_names = contact.find_bodies(".*_foot")
    if robot_foot_names != sensor_foot_names:
        raise RuntimeError(
            f"Robot/contact foot order differs: {robot_foot_names} vs {sensor_foot_names}"
        )

    env_ids = torch.arange(num_envs, device=device)
    forward_env = env_ids < (num_envs // 2)
    backward_env = ~forward_env

    command.vel_command_b[:] = 0.0
    command.vel_command_b[forward_env, 0] = args_cli.forward_speed
    command.vel_command_b[backward_env, 0] = args_cli.backward_speed
    command.is_longitudinal_env[:] = True
    command.is_lateral_env[:] = False
    command.is_yaw_env[:] = False
    command.is_mixed_env[:] = False
    command.is_standing_env[:] = False
    command.is_heading_hold_env[:] = True
    command.is_forward_env[:] = forward_env
    command.is_backward_env[:] = backward_env
    command.straight_heading_target[:] = robot.data.heading_w
    # Prevent the normal 8--12 s sampler from replacing the fixed diagnostic commands.
    if hasattr(command, "time_left"):
        command.time_left[:] = 1.0e6

    obs, _ = env.get_observations()
    if args_cli.force_amp_reset:
        # B2Z1ManagerAMPBasedRLEnv applies the AMP frame override inside its
        # custom step-time reset path, not during the wrapper's initial reset.
        raw.episode_length_buf[:] = raw.max_episode_length - 1
        with torch.inference_mode():
            obs, _, _, _ = env.step(torch.zeros((num_envs, env.num_actions), device=device))

        # The reset resamples command terms; restore the deterministic split.
        command.vel_command_b[:] = 0.0
        command.vel_command_b[forward_env, 0] = args_cli.forward_speed
        command.vel_command_b[backward_env, 0] = args_cli.backward_speed
        command.is_longitudinal_env[:] = True
        command.is_lateral_env[:] = False
        command.is_yaw_env[:] = False
        command.is_mixed_env[:] = False
        command.is_standing_env[:] = False
        command.is_heading_hold_env[:] = True
        command.is_forward_env[:] = forward_env
        command.is_backward_env[:] = backward_env
        command.straight_heading_target[:] = robot.data.heading_w
        if hasattr(command, "time_left"):
            command.time_left[:] = 1.0e6
        obs, _ = env.get_observations()
    totals = defaultdict(float)
    counts = defaultdict(int)

    total_steps = args_cli.warmup_steps + args_cli.eval_steps
    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # A reset should not change the requested motion mode or its reference heading.
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            command.straight_heading_target[reset_ids] = robot.data.heading_w[reset_ids]
            if hasattr(command, "time_left"):
                command.time_left[reset_ids] = 1.0e6
        command.vel_command_b[forward_env, 0] = args_cli.forward_speed
        command.vel_command_b[backward_env, 0] = args_cli.backward_speed
        command.vel_command_b[:, 1] = 0.0

        if step < args_cli.warmup_steps:
            continue

        body_velocity = robot.data.root_lin_vel_b
        yaw_rate = robot.data.root_ang_vel_b[:, 2]
        ground_z = raw.scene.env_origins[:, 2].unsqueeze(1)
        clearance = robot.data.body_pos_w[:, robot_foot_ids, 2] - ground_z - args_cli.foot_radius
        air_time = contact.data.current_air_time[:, sensor_foot_ids]
        contact_time = contact.data.current_contact_time[:, sensor_foot_ids]
        swing = (contact_time <= 0.0) & (air_time >= 0.04)

        for label, mask, target in (
            ("forward", forward_env, args_cli.forward_speed),
            ("backward", backward_env, args_cli.backward_speed),
        ):
            vx = body_velocity[mask, 0]
            vy = body_velocity[mask, 1]
            wz = yaw_rate[mask]
            totals[f"{label}_vx_error_sq"] += torch.square(vx - target).sum().item()
            totals[f"{label}_vx"] += vx.sum().item()
            totals[f"{label}_abs_vy"] += torch.abs(vy).sum().item()
            totals[f"{label}_abs_wz"] += torch.abs(wz).sum().item()
            totals[f"{label}_command_x"] += command.vel_command_b[mask, 0].sum().item()
            totals[f"{label}_action_abs"] += torch.abs(actions[mask]).mean(dim=1).sum().item()
            counts[f"{label}_samples"] += int(mask.sum().item())
            counts[f"{label}_resets"] += int((dones & mask).sum().item())

            local_contact_time = contact_time[mask]
            local_contact = local_contact_time > 0.0
            contact_count = local_contact.sum(dim=1)
            totals[f"{label}_single_support"] += (contact_count == 1).sum().item()
            totals[f"{label}_two_foot_support"] += (contact_count == 2).sum().item()
            for foot_index, foot_name in enumerate(sensor_foot_names):
                foot_contact_time = local_contact_time[:, foot_index]
                totals[f"{label}_{foot_name}_contact"] += local_contact[:, foot_index].sum().item()
                totals[f"{label}_{foot_name}_long_contact_032"] += (
                    foot_contact_time > 0.32
                ).sum().item()
                totals[f"{label}_{foot_name}_long_contact_040"] += (
                    foot_contact_time > 0.40
                ).sum().item()
                totals[f"{label}_{foot_name}_max_contact"] = max(
                    totals[f"{label}_{foot_name}_max_contact"],
                    foot_contact_time.max().item(),
                )

            foot_index = {name: index for index, name in enumerate(sensor_foot_names)}
            first_pair_mismatch = (
                local_contact[:, foot_index["FL_foot"]]
                != local_contact[:, foot_index["RR_foot"]]
            )
            second_pair_mismatch = (
                local_contact[:, foot_index["FR_foot"]]
                != local_contact[:, foot_index["RL_foot"]]
            )
            totals[f"{label}_diagonal_pair_mismatch"] += (
                first_pair_mismatch.float() + second_pair_mismatch.float()
            ).sum().item()

            local_clearance = clearance[mask]
            local_swing = swing[mask]
            for region, foot_mask in (
                ("front", torch.tensor([name.startswith("F") for name in robot_foot_names], device=device)),
                ("rear", torch.tensor([name.startswith("R") for name in robot_foot_names], device=device)),
            ):
                selected = local_clearance[:, foot_mask][local_swing[:, foot_mask]]
                if selected.numel() > 0:
                    totals[f"{label}_{region}_clearance"] += selected.sum().item()
                    counts[f"{label}_{region}_clearance"] += selected.numel()

    print("\n========== deterministic AMP gait report ==========")
    print(f"checkpoint: {resume_path}")
    print(f"environments: {num_envs}, measured steps: {args_cli.eval_steps}")
    for label, target in (("forward", args_cli.forward_speed), ("backward", args_cli.backward_speed)):
        n = counts[f"{label}_samples"]
        rmse = (_mean(totals[f"{label}_vx_error_sq"], n)) ** 0.5
        print(
            f"{label:8s} target={target:+.3f} m/s  "
            f"observed_cmd={_mean(totals[f'{label}_command_x'], n):+.3f}  "
            f"mean_vx={_mean(totals[f'{label}_vx'], n):+.3f}  "
            f"vx_rmse={rmse:.3f}  "
            f"|vy|={_mean(totals[f'{label}_abs_vy'], n):.3f}  "
            f"|yaw_rate|={_mean(totals[f'{label}_abs_wz'], n):.3f}  "
            f"|action|={_mean(totals[f'{label}_action_abs'], n):.3f}  "
            f"resets={counts[f'{label}_resets']}"
        )
        for region in ("front", "rear"):
            key = f"{label}_{region}_clearance"
            clearance_cm = 100.0 * _mean(totals[key], counts[key])
            print(f"  {region:5s} conditional swing sole clearance: {clearance_cm:.2f} cm")
        print(
            "  support fractions: "
            f"one-foot={_mean(totals[f'{label}_single_support'], n):.3f}, "
            f"two-foot={_mean(totals[f'{label}_two_foot_support'], n):.3f}, "
            "diagonal-pair mismatch="
            f"{_mean(totals[f'{label}_diagonal_pair_mismatch'], 2 * n):.3f}"
        )
        for foot_name in sensor_foot_names:
            print(
                f"  {foot_name:7s} duty="
                f"{_mean(totals[f'{label}_{foot_name}_contact'], n):.3f}, "
                f">0.32s={_mean(totals[f'{label}_{foot_name}_long_contact_032'], n):.3f}, "
                f">0.40s={_mean(totals[f'{label}_{foot_name}_long_contact_040'], n):.3f}, "
                f"max_contact={totals[f'{label}_{foot_name}_max_contact']:.3f}s"
            )
    print("===================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
