"""Warm-start missing cardinal locomotion branches from the recorded AMP priors.

The selected V19 actor already has a good forward gait and visual/TCP branch,
but PPO cannot reliably leave its stationary local optimum for reverse,
lateral, or yaw-only commands.  This script supervises only those missing leg
branches from the slow motion priors.  The original actor remains the teacher
for forward, standing, and every arm output.

The environment uses an EMA action filter.  Expert joint targets are therefore
converted to the *pre-filter* action that reaches the next expert pose after
one control step instead of being attenuated by ``alpha``.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Omnidirectional AMP behavior-cloning warm start.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=1200)
parser.add_argument("--learning_rate", type=float, default=5.0e-5)
parser.add_argument("--forward_anchor_weight", type=float, default=1.5)
parser.add_argument("--standing_anchor_weight", type=float, default=1.0)
parser.add_argument("--arm_anchor_weight", type=float, default=2.0)
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


# Only the slow priors are used.  Their gait shape is easier to retain during
# the subsequent low-speed PPO stage than the fast/faster recordings.
MOTION_COMMANDS = {
    "b2_backward": (-0.12, 0.0, 0.0),
    "b2_left": (0.0, 0.10, 0.0),
    "b2_right": (0.0, -0.10, 0.0),
    "b2_turn_left": (0.0, 0.0, 0.12),
    "b2_turn_right": (0.0, 0.0, -0.12),
}


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
    command = raw.command_manager.get_term("base_velocity")
    action_term = raw.action_manager.get_term("JointPositionAction")
    alpha = float(getattr(action_term, "_alpha", 1.0))
    if not 0.0 < alpha <= 1.0:
        raise RuntimeError(f"Invalid action filter alpha: {alpha}")

    # Resolve exact slow-motion names without accidentally selecting *_fast.
    motion_indices: dict[str, int] = {}
    for index, path in enumerate(loader.trajectory_names):
        stem = os.path.basename(path)
        if stem in MOTION_COMMANDS:
            motion_indices[stem] = index
    missing = sorted(set(MOTION_COMMANDS) - set(motion_indices))
    if missing:
        raise RuntimeError(f"Missing slow AMP priors: {missing}")

    labels = tuple(MOTION_COMMANDS)
    env_ids = torch.arange(raw.num_envs, device=device)
    data_to_lab = torch.as_tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], device=device)
    default_joint_pos = robot.data.default_joint_pos
    action_scale = 0.25
    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.learning_rate)

    print(f"[INFO] Omni BC teacher: {checkpoint}")
    print(f"[INFO] Motions: {', '.join(labels)}")
    print(f"[INFO] Batch={raw.num_envs}, iterations={args_cli.iterations}, action alpha={alpha:.3f}")

    for iteration in range(args_cli.iterations):
        sampled_labels = np.random.choice(labels, size=raw.num_envs)
        trajectory_ids = np.asarray([motion_indices[label] for label in sampled_labels], dtype=np.int64)
        times = loader.traj_time_sample_batch(trajectory_ids)
        frames = loader.get_full_frame_at_time_batch(trajectory_ids, times)
        next_frames = loader.get_full_frame_at_time_batch(trajectory_ids, times + raw.step_dt)

        with torch.no_grad():
            raw._reset_dofs_amp(env_ids, frames, robot)
            raw._reset_root_states_amp(env_ids, frames, robot)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(dt=raw.physics_dt)

            current_q = loader.get_joint_pose_batch(frames)[:, data_to_lab]
            next_q = loader.get_joint_pose_batch(next_frames)[:, data_to_lab]
            current_action = torch.zeros((raw.num_envs, env.num_actions), device=device)
            current_action[:, :12] = (current_q - default_joint_pos[:, :12]) / action_scale
            next_action = (next_q - default_joint_pos[:, :12]) / action_scale

            # Seed the observation's last-action term with the action that
            # corresponds to the current expert pose.  Invert the EMA for the
            # supervised output so applying it once reaches ``next_action``.
            action_term._raw_actions.copy_(current_action)
            target_actions = current_action.clone()
            target_actions[:, :12] = current_action[:, :12] + (
                next_action - current_action[:, :12]
            ) / alpha

            command.vel_command_b.zero_()
            for label, values in MOTION_COMMANDS.items():
                mask_np = sampled_labels == label
                if np.any(mask_np):
                    mask = torch.as_tensor(mask_np, device=device)
                    command.vel_command_b[mask] = torch.as_tensor(values, device=device)
            command.is_standing_env.zero_()
            command.is_longitudinal_env[:] = command.vel_command_b[:, 0] != 0.0
            command.is_lateral_env[:] = command.vel_command_b[:, 1] != 0.0
            command.is_yaw_env[:] = command.vel_command_b[:, 2] != 0.0
            command.is_mixed_env.zero_()
            command.is_heading_hold_env[:] = command.is_longitudinal_env | command.is_lateral_env
            command.is_forward_env.zero_()
            command.is_backward_env[:] = command.vel_command_b[:, 0] < 0.0
            command.straight_heading_target[:] = robot.data.heading_w

            observations = raw.observation_manager.compute()["policy"]
            teacher_missing = teacher(observations)
            target_actions[:, 12:] = teacher_missing[:, 12:]

            # Preserve the mature V19 branches under identical proprioceptive
            # and TCP observations; only the velocity command is changed.
            forward_observations = observations.clone()
            forward_observations[:, :3] = torch.tensor((0.25, 0.0, 0.0), device=device)
            standing_observations = observations.clone()
            standing_observations[:, :3] = 0.0
            teacher_forward = teacher(forward_observations)
            teacher_standing = teacher(standing_observations)

        predicted_missing = actor(observations)
        predicted_forward = actor(forward_observations)
        predicted_standing = actor(standing_observations)

        leg_loss = F.mse_loss(predicted_missing[:, :12], target_actions[:, :12])
        arm_loss = F.mse_loss(predicted_missing[:, 12:], target_actions[:, 12:])
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

        if iteration == 0 or (iteration + 1) % 50 == 0:
            print(
                f"[BC {iteration + 1:04d}/{args_cli.iterations}] total={loss.item():.6f} "
                f"leg={leg_loss.item():.6f} arm={arm_loss.item():.6f} "
                f"forward_anchor={forward_anchor.item():.6f} "
                f"standing_anchor={standing_anchor.item():.6f}"
            )

    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # ``log_dir=None`` intentionally disables external logging for this
    # offline warm start.  Older runner versions do not initialize these two
    # attributes in that mode even though ``save()`` reads them.
    runner.logger_type = ""
    runner.disable_logs = True
    runner.save(
        output_path,
        infos={
            "source_checkpoint": checkpoint,
            "omni_bc_iterations": args_cli.iterations,
            "omni_bc_learning_rate": args_cli.learning_rate,
            "action_filter_alpha": alpha,
            "motion_commands": MOTION_COMMANDS,
        },
    )
    print(f"[INFO] Saved omni-BC checkpoint: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
