"""Supervise the AMP actor on recorded backward-motion joint targets.

PPO continuation from the mature forward policy repeatedly converged to the
stationary local optimum for negative velocity commands.  This script uses the
existing slow backward motion prior to initialize only that missing command
branch, while distilling the original actor on positive and zero commands so
the established forward/standing behavior is retained.
"""

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Backward behavior-cloning warm start for AMP.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=400)
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--forward_anchor_weight", type=float, default=1.0)
parser.add_argument("--standing_anchor_weight", type=float, default=0.5)
parser.add_argument("--arm_anchor_weight", type=float, default=1.0)
parser.add_argument("--output", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from b2_z1.rsl_rl import AMP_RslRlOnPolicyRunnerCfg, AMPRslRlVecEnvWrapper
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import b2_z1.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: AMP_RslRlOnPolicyRunnerCfg,
):
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.ee_target.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = AMPRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = AMP_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.alg.policy
    actor = policy.actor
    teacher = copy.deepcopy(actor).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    actor.train()

    raw = env.unwrapped
    device = raw.device
    robot = raw.scene["robot"]
    loader = raw.amp_loader
    if len(loader.trajectory_idxs) != 1 or not loader.trajectory_names[0].endswith("b2_backward"):
        raise RuntimeError(
            "Use the V13 expert-start task so the environment loader contains only b2_backward.txt."
        )

    command = raw.command_manager.get_term("base_velocity")
    env_ids = torch.arange(raw.num_envs, device=device)
    data_to_lab = torch.as_tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], device=device)
    default_joint_pos = robot.data.default_joint_pos
    action_scale = 0.25

    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.learning_rate)
    trajectory_ids = np.zeros(raw.num_envs, dtype=np.int64)

    print(f"[INFO] Backward BC source checkpoint: {checkpoint}")
    print(f"[INFO] Expert motion: {loader.trajectory_names[0]}")
    print(f"[INFO] Batch={raw.num_envs}, iterations={args_cli.iterations}")

    for iteration in range(args_cli.iterations):
        times = loader.traj_time_sample_batch(trajectory_ids)
        frames = loader.get_full_frame_at_time_batch(trajectory_ids, times)
        next_frames = loader.get_full_frame_at_time_batch(
            trajectory_ids, times + raw.step_dt
        )

        with torch.no_grad():
            raw._reset_dofs_amp(env_ids, frames, robot)
            raw._reset_root_states_amp(env_ids, frames, robot)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(dt=raw.physics_dt)

            current_q_data = loader.get_joint_pose_batch(frames)
            next_q_data = loader.get_joint_pose_batch(next_frames)
            current_q_lab = current_q_data[:, data_to_lab]
            next_q_lab = next_q_data[:, data_to_lab]

            current_actions = torch.zeros((raw.num_envs, env.num_actions), device=device)
            target_actions = torch.zeros_like(current_actions)
            current_actions[:, :12] = (
                current_q_lab - default_joint_pos[:, :12]
            ) / action_scale
            target_actions[:, :12] = (
                next_q_lab - default_joint_pos[:, :12]
            ) / action_scale
            raw.action_manager.process_action(current_actions)

            command.vel_command_b[:] = 0.0
            command.vel_command_b[:, :2] = loader.get_linear_vel_batch(frames)[:, :2]
            command.vel_command_b[:, 2] = loader.get_angular_vel_batch(frames)[:, 2]
            command.is_longitudinal_env[:] = True
            command.is_lateral_env[:] = False
            command.is_yaw_env[:] = False
            command.is_mixed_env[:] = False
            command.is_standing_env[:] = False
            command.is_heading_hold_env[:] = True
            command.is_forward_env[:] = False
            command.is_backward_env[:] = True
            command.straight_heading_target[:] = robot.data.heading_w

            observations = raw.observation_manager.compute()["policy"]
            forward_observations = observations.clone()
            forward_observations[:, 0] = torch.abs(forward_observations[:, 0])
            standing_observations = observations.clone()
            standing_observations[:, :3] = 0.0
            teacher_backward = teacher(observations)
            teacher_forward = teacher(forward_observations)
            teacher_standing = teacher(standing_observations)
            target_actions[:, 12:] = teacher_backward[:, 12:]

        predicted_backward = actor(observations)
        predicted_forward = actor(forward_observations)
        predicted_standing = actor(standing_observations)

        leg_loss = F.mse_loss(predicted_backward[:, :12], target_actions[:, :12])
        arm_loss = F.mse_loss(predicted_backward[:, 12:], target_actions[:, 12:])
        forward_anchor = F.mse_loss(predicted_forward, teacher_forward)
        standing_anchor = F.mse_loss(predicted_standing, teacher_standing)
        loss = (
            leg_loss
            + args_cli.arm_anchor_weight * arm_loss
            + args_cli.forward_anchor_weight * forward_anchor
            + args_cli.standing_anchor_weight * standing_anchor
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration == 0 or (iteration + 1) % 25 == 0:
            print(
                f"[BC {iteration + 1:04d}/{args_cli.iterations}] "
                f"total={loss.item():.6f} leg={leg_loss.item():.6f} "
                f"arm={arm_loss.item():.6f} forward_anchor={forward_anchor.item():.6f} "
                f"standing_anchor={standing_anchor.item():.6f}"
            )

    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    runner.save(
        output_path,
        infos={
            "source_checkpoint": checkpoint,
            "backward_bc_iterations": args_cli.iterations,
            "backward_bc_learning_rate": args_cli.learning_rate,
        },
    )
    print(f"[INFO] Saved backward-BC checkpoint: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
