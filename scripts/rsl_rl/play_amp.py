# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# Complete PyTorch's native-module initialization before Kit starts extensions.
# Under debugpy, an extension can otherwise re-enter torch while torch._C is
# still loading and observe a partially initialized module without Tensor.
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video-output-dir",
    type=str,
    default=None,
    help="Directory for this playback video; defaults to the checkpoint's videos/play folder.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max-steps", type=int, default=None, help="Stop playback after this many policy steps.")
parser.add_argument(
    "--motion-trace-csv", type=str, default=None,
    help="Write one-environment base, TCP target/actual and Z1 joint motion to CSV for diagnosis.",
)
parser.add_argument(
    "--tcp-z-lower-limit", type=float, default=None,
    help="Playback-only lower bound for the rear TCP command in the base frame (metres).",
)
parser.add_argument(
    "--hold-z1-nominal", action="store_true",
    help="Diagnostic only: hold the six Z1 actions at their nominal joint offsets.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--arm-action-filter-alpha",
    type=float,
    default=None,
    help=(
        "Optional extra EMA for the six Z1 joint actions during playback. "
        "Smaller values suppress gait-synchronous arm motion more strongly."
    ),
)
parser.add_argument(
    "--arm-action-max-delta",
    type=float,
    default=None,
    help="Optional per-policy-step limit for each filtered Z1 action.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import os
import time

# from rsl_rl.runners import OnPolicyRunner
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from b2_z1.rsl_rl import (
    AMP_RslRlOnPolicyRunnerCfg, 
    AMPRslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import b2_z1.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: AMP_RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.arm_action_filter_alpha is not None and not (
        0.0 < args_cli.arm_action_filter_alpha <= 1.0
    ):
        raise ValueError("--arm-action-filter-alpha must be in (0, 1]")
    if args_cli.arm_action_max_delta is not None and args_cli.arm_action_max_delta <= 0.0:
        raise ValueError("--arm-action-max-delta must be positive")
    if args_cli.arm_action_max_delta is not None and args_cli.arm_action_filter_alpha is None:
        raise ValueError("--arm-action-max-delta requires --arm-action-filter-alpha")
    if args_cli.hold_z1_nominal and args_cli.arm_action_filter_alpha is not None:
        raise ValueError("--hold-z1-nominal cannot be combined with --arm-action-filter-alpha")
    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.tcp_z_lower_limit is not None:
        current_range = env_cfg.commands.ee_target.limit_ranges.pos_z
        if args_cli.tcp_z_lower_limit >= current_range[1]:
            raise ValueError("--tcp-z-lower-limit must be below the configured upper bound")
        env_cfg.commands.ee_target.limit_ranges.pos_z = (
            args_cli.tcp_z_lower_limit, current_range[1]
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_output_dir or os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = AMPRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = AMP_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    arm_action_indices = None
    arm_joint_indices = None
    if (
        args_cli.arm_action_filter_alpha is not None
        or args_cli.motion_trace_csv
        or args_cli.hold_z1_nominal
    ):
        action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
        joint_names = action_term._joint_names
        expected_arm_joints = [f"joint{index}" for index in range(1, 7)]
        if any(joint not in joint_names for joint in expected_arm_joints):
            raise RuntimeError(
                f"Cannot identify Z1 actions in resolved joint order: {joint_names}"
            )
        arm_action_indices = [joint_names.index(joint) for joint in expected_arm_joints]
        arm_joint_indices, _ = env.unwrapped.scene["robot"].find_joints(
            expected_arm_joints, preserve_order=True
        )
        print(
            "[INFO] Z1 action filter indices: "
            f"{arm_action_indices} ({', '.join(expected_arm_joints)})"
        )

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    arm_action_state = None
    previous_raw_actions = None
    previous_applied_actions = None
    raw_arm_delta_sq = 0.0
    applied_arm_delta_sq = 0.0
    action_delta_samples = 0
    tcp_error_sum = 0.0
    tcp_error_samples = 0
    arm_joint_vel_sq = 0.0
    arm_joint_vel_samples = 0
    reset_count = 0
    path_command = (
        env.unwrapped.command_manager.get_term("ee_target")
        if arm_action_indices is not None or args_cli.motion_trace_csv else None
    )
    motion_trace = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            raw_actions = actions
            if args_cli.hold_z1_nominal:
                actions = actions.clone()
                actions[:, arm_action_indices] = 0.0
            if args_cli.arm_action_filter_alpha is not None:
                # Resolve action coordinates from joint names.  Zero action
                # is the configured nominal work pose, so starting the filter
                # from zero provides a smooth policy takeover.
                if arm_action_state is None:
                    arm_action_state = torch.zeros_like(actions[:, arm_action_indices])
                desired_arm = actions[:, arm_action_indices]
                filtered_delta = args_cli.arm_action_filter_alpha * (
                    desired_arm - arm_action_state
                )
                if args_cli.arm_action_max_delta is not None:
                    filtered_delta = torch.clamp(
                        filtered_delta,
                        -args_cli.arm_action_max_delta,
                        args_cli.arm_action_max_delta,
                    )
                arm_action_state = arm_action_state + filtered_delta
                actions = actions.clone()
                actions[:, arm_action_indices] = arm_action_state

            if arm_action_indices is not None and previous_raw_actions is not None:
                raw_arm_delta_sq += torch.square(
                    raw_actions[:, arm_action_indices] - previous_raw_actions[:, arm_action_indices]
                ).sum().item()
                applied_arm_delta_sq += torch.square(
                    actions[:, arm_action_indices] - previous_applied_actions[:, arm_action_indices]
                ).sum().item()
                action_delta_samples += actions.shape[0] * len(arm_action_indices)
            previous_raw_actions = raw_actions.clone()
            previous_applied_actions = actions.clone()
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if path_command is not None:
                # CommandTerm metrics are replaced on each update, so fetch
                # the current tensor after stepping rather than retaining it.
                tcp_metric = path_command.metrics["position_error"]
                tcp_error_sum += tcp_metric.sum().item()
                tcp_error_samples += tcp_metric.numel()
                arm_vel = env.unwrapped.scene["robot"].data.joint_vel[:, arm_joint_indices]
                arm_joint_vel_sq += torch.square(arm_vel).sum().item()
                arm_joint_vel_samples += arm_vel.numel()
                if args_cli.motion_trace_csv:
                    if env.unwrapped.num_envs != 1:
                        raise ValueError("--motion-trace-csv requires --num_envs 1")
                    robot = env.unwrapped.scene["robot"]
                    base = robot.data.body_pos_w[0, path_command.base_idx]
                    base_quat = robot.data.body_state_w[0, path_command.base_idx, 3:7]
                    actual = robot.data.body_pos_w[0, path_command.body_idx]
                    target = path_command.tracking_pose_w[0, :3]
                    command = path_command.pose_command[0, :3]
                    base_command = env.unwrapped.command_manager.get_command("base_velocity")[0]
                    motion_trace.append(
                        [timestep * dt, *base.tolist(), *base_quat.tolist(), *target.tolist(), *actual.tolist(),
                         *command.tolist(), *robot.data.joint_pos[0, arm_joint_indices].tolist(),
                         base_command[0].item(), base_command[1].item(), base_command[2].item(),
                         robot.data.root_ang_vel_b[0, 2].item(),
                         path_command._path_lateral_error_b[0].item(),
                         path_command._path_heading_error_b[0].item()]
                    )
            if arm_action_state is not None:
                reset_mask = dones.to(dtype=torch.bool).reshape(-1)
                if torch.any(reset_mask):
                    reset_count += int(reset_mask.sum().item())
                    arm_action_state[reset_mask] = 0.0
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if action_delta_samples:
        raw_rms = (raw_arm_delta_sq / action_delta_samples) ** 0.5
        applied_rms = (applied_arm_delta_sq / action_delta_samples) ** 0.5
        print(
            "[INFO] Z1 action-delta RMS: "
            f"policy={raw_rms:.6f}, playback_applied={applied_rms:.6f}"
        )
    if tcp_error_samples:
        print(f"[INFO] Mean TCP position error: {tcp_error_sum / tcp_error_samples:.4f} m")
    if arm_joint_vel_samples:
        print(
            "[INFO] Z1 measured joint velocity RMS: "
            f"{(arm_joint_vel_sq / arm_joint_vel_samples) ** 0.5:.4f} rad/s; "
            f"episode resets: {reset_count}"
        )
    if args_cli.motion_trace_csv:
        robot = env.unwrapped.scene["robot"]
        jacobian_body_idx = path_command.body_idx - 1 if robot.is_fixed_base else path_command.body_idx
        jacobian_joint_ids = (
            arm_joint_indices if robot.is_fixed_base else [joint_id + 6 for joint_id in arm_joint_indices]
        )
        jacobian = robot.root_physx_view.get_jacobians()[0, jacobian_body_idx, :, jacobian_joint_ids]
        elbow_column = jacobian[:, 2]
        wrist_column = jacobian[:, 3]
        column_cosine = torch.dot(elbow_column, wrist_column) / (
            torch.linalg.norm(elbow_column) * torch.linalg.norm(wrist_column)
        ).clamp_min(1.0e-8)
        print(
            "[INFO] Z1 Jacobian: "
            f"joint3/joint4 column cosine={column_cosine.item():.3f}, "
            f"singular values={[round(value, 4) for value in torch.linalg.svdvals(jacobian).tolist()]}"
        )
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.motion_trace_csv)), exist_ok=True)
        with open(args_cli.motion_trace_csv, "w", newline="") as trace_file:
            writer = csv.writer(trace_file)
            writer.writerow(
                ["time_s", "base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz",
                 "target_x", "target_y", "target_z",
                 "actual_x", "actual_y", "actual_z", "command_x_b", "command_y_b", "command_z_b",
                 *[f"joint{index}" for index in range(1, 7)],
                 "command_vx_b", "command_vy_b", "command_yaw_rate_b", "actual_yaw_rate_b",
                 "path_lateral_error_b", "path_heading_error_b"]
            )
            writer.writerows(motion_trace)
        print(f"[INFO] Wrote motion trace: {args_cli.motion_trace_csv}")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
