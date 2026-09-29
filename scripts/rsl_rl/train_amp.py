# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--reset_leg_policy_std",
    type=float,
    default=None,
    help="After loading a checkpoint, reset only the 12 leg-action exploration standard deviations.",
)
parser.add_argument(
    "--freeze-arm-actor",
    action="store_true",
    help="Fine-tune only B2 action-head rows; freeze the actor trunk, six Z1 action rows and their noise.",
)
parser.add_argument(
    "--arm-teacher-weight", type=float, default=0.0,
    help="Distill six Z1 actor means from the resumed checkpoint while training the full-body actor.",
)
parser.add_argument(
    "--leg-teacher-weight", type=float, default=0.0,
    help="Preserve the resumed B2 trot, releasing most of this constraint for yaw commands.",
)
parser.add_argument(
    "--teacher-checkpoint", type=str, default=None,
    help="Load an independent frozen actor teacher without initializing the student from it.",
)
parser.add_argument(
    "--initial-arm-policy-std", type=float, default=None,
    help="Initial Z1 exploration std for a fresh student with an independent teacher.",
)
parser.add_argument(
    "--leg-teacher-yaw-release", type=float, default=0.10,
    help="Absolute yaw command where B2 teacher strength reaches its 20%% floor (rad/s).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.teacher_checkpoint and not (args_cli.arm_teacher_weight or args_cli.leg_teacher_weight):
    parser.error("--teacher-checkpoint requires a nonzero teacher loss weight")
if args_cli.initial_arm_policy_std is not None:
    if (not args_cli.teacher_checkpoint or args_cli.arm_teacher_weight <= 0.0
            or not math.isfinite(args_cli.initial_arm_policy_std) or args_cli.initial_arm_policy_std <= 0.0):
        parser.error("--initial-arm-policy-std requires an independent arm teacher and finite positive std")

if args_cli.reset_leg_policy_std is not None:
    if not math.isfinite(args_cli.reset_leg_policy_std) or args_cli.reset_leg_policy_std <= 0.0:
        parser.error("--reset_leg_policy_std must be finite and positive")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)
    
    

''' compress the target files'''
import os
import tarfile
PATH = os.path.dirname(os.path.realpath(__file__))

def make_tar_gz(source_dir, output_tar_gz):
    """
    将 source_dir 目录打包并压缩成 output_tar_gz
    """
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        
        

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

import omni
# from rsl_rl.runners import OnPolicyRunner
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from b2_z1.rsl_rl import AMP_RslRlOnPolicyRunnerCfg, AMPRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import b2_z1.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: AMP_RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    
    
    # tar source dir 
    souce_dir = PATH + "/../../source"
    # target dir check 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    make_tar_gz(souce_dir, os.path.join(log_dir, "source.tar.gz"))
    print("[INFO] source code is saved to ", os.path.join(log_dir, "source.tar.gz"))
    
    

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = AMPRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.reset_leg_policy_std is not None and not agent_cfg.resume:
        raise ValueError("--reset_leg_policy_std requires resuming a checkpoint")

    # create runner from rsl-rl
    runner = AMP_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)
        if args_cli.reset_leg_policy_std is not None:
            if args_cli.reset_leg_policy_std <= 0.0:
                raise ValueError("--reset_leg_policy_std must be positive")
            action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
            joint_names = action_term._joint_names
            arm_names = {f"joint{index}" for index in range(1, 7)}
            if not arm_names.issubset(joint_names) or runner.alg.policy.std.numel() != len(joint_names):
                raise RuntimeError(f"Cannot resolve the 12 B2 action rows from {joint_names}")
            leg_indices = [index for index, name in enumerate(joint_names) if name not in arm_names]
            if len(leg_indices) != 12:
                raise RuntimeError(f"Expected 12 B2 action rows, got {leg_indices}")
            with torch.no_grad():
                runner.alg.policy.std[leg_indices] = args_cli.reset_leg_policy_std
            print(
                "[INFO]: Reset leg exploration std to "
                f"{args_cli.reset_leg_policy_std:.3f} on action rows {leg_indices}; "
                "arm std and actor means are unchanged"
            )

    if args_cli.freeze_arm_actor:
        if not agent_cfg.resume:
            raise ValueError("--freeze-arm-actor requires --resume from an existing 18-action checkpoint")
        policy = runner.alg.policy
        actor = policy.actor
        if not isinstance(actor[-1], torch.nn.Linear):
            raise RuntimeError("--freeze-arm-actor requires a final Linear actor layer")
        action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
        joint_names = action_term._joint_names
        arm_names = [f"joint{index}" for index in range(1, 7)]
        if actor[-1].out_features != len(joint_names) or any(name not in joint_names for name in arm_names):
            raise RuntimeError(f"Cannot resolve all six Z1 actions from {joint_names}")
        arm_indices = [joint_names.index(name) for name in arm_names]
        leg_mask = torch.ones(actor[-1].out_features, device=actor[-1].weight.device)
        leg_mask[arm_indices] = 0.0
        for layer in list(actor.children())[:-1]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        actor[-1].weight.register_hook(lambda grad: grad * leg_mask[:, None])
        actor[-1].bias.register_hook(lambda grad: grad * leg_mask)
        noise_parameter = getattr(policy, "std", None)
        if noise_parameter is None:
            noise_parameter = getattr(policy, "log_std", None)
        if noise_parameter is None or noise_parameter.numel() != leg_mask.numel():
            raise RuntimeError("Cannot resolve per-action policy noise for Z1 freezing")
        noise_parameter.register_hook(lambda grad: grad * leg_mask)
        print(
            "[INFO]: Frozen actor trunk and Z1 output/noise rows; "
            f"trainable B2 action rows={[i for i, value in enumerate(leg_mask.tolist()) if value > 0]}"
        )

    if args_cli.arm_teacher_weight or args_cli.leg_teacher_weight:
        if (
            not all(math.isfinite(value) for value in (
                args_cli.arm_teacher_weight, args_cli.leg_teacher_weight, args_cli.leg_teacher_yaw_release
            ))
            or args_cli.arm_teacher_weight < 0.0
            or args_cli.leg_teacher_weight < 0.0
            or args_cli.leg_teacher_yaw_release <= 0.0
            or (not agent_cfg.resume and not args_cli.teacher_checkpoint)
            or args_cli.freeze_arm_actor
        ):
            raise ValueError("teacher weights require resume or --teacher-checkpoint, nonnegative weights, and no --freeze-arm-actor")
        import copy

        policy = runner.alg.policy
        action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
        joint_names = action_term._joint_names
        arm_names = [f"joint{index}" for index in range(1, 7)]
        if any(name not in joint_names for name in arm_names):
            raise RuntimeError(f"Cannot resolve all six Z1 actions from {joint_names}")
        if not isinstance(policy.actor[-1], torch.nn.Linear) or policy.actor[-1].out_features != len(joint_names):
            raise RuntimeError("Teacher action dimension does not match the action-manager joint order")
        arm_indices = [joint_names.index(name) for name in arm_names]
        teacher = copy.deepcopy(policy.actor).eval()
        if args_cli.teacher_checkpoint:
            teacher_path = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
            teacher_state = torch.load(teacher_path, map_location=runner.device, weights_only=False)
            if runner.empirical_normalization or "obs_norm_state_dict" in teacher_state:
                raise ValueError("Independent teacher currently requires unnormalized student and teacher observations")
            actor_state = {
                key[len("actor."):]: value
                for key, value in teacher_state["model_state_dict"].items()
                if key.startswith("actor.")
            }
            teacher.load_state_dict(actor_state, strict=True)
            if not all(torch.isfinite(parameter).all() for parameter in teacher.parameters()):
                raise ValueError("Teacher checkpoint has non-finite actor parameters")
            print(f"[INFO]: Independent frozen teacher: {teacher_path}; student weights are not loaded from teacher")
        if args_cli.initial_arm_policy_std is not None:
            value = args_cli.initial_arm_policy_std
            if agent_cfg.resume or not args_cli.teacher_checkpoint or not math.isfinite(value) or value <= 0:
                raise ValueError("--initial-arm-policy-std requires a fresh student, independent teacher, and finite positive std")
            with torch.no_grad():
                policy.std[arm_indices] = value
            print(f"[INFO]: Initial arm std={value} on action rows {arm_indices}")
        teacher.requires_grad_(False)
        runner.alg.arm_teacher_actor = teacher
        runner.alg.arm_teacher_action_ids = arm_indices
        runner.alg.arm_teacher_weight = args_cli.arm_teacher_weight
        runner.alg.leg_teacher_action_ids = [
            index for index in range(len(joint_names)) if index not in arm_indices
        ]
        runner.alg.leg_teacher_weight = args_cli.leg_teacher_weight
        runner.alg.leg_teacher_yaw_release = args_cli.leg_teacher_yaw_release
        if args_cli.arm_teacher_weight > 0.0 and agent_cfg.resume and not args_cli.teacher_checkpoint:
            # Do not let exploration noise on the Z1 joints drift during tuning.
            arm_mask = torch.ones_like(policy.std)
            arm_mask[arm_indices] = 0.0
            policy.std.register_hook(lambda grad: grad * arm_mask)
        print(
            "[INFO]: Distilling reference actor actions; "
            f"Z1 weight={args_cli.arm_teacher_weight:.3f}, B2 weight={args_cli.leg_teacher_weight:.3f}, "
            f"yaw release={args_cli.leg_teacher_yaw_release:.3f}, Z1 rows={arm_indices}"
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "training_cli.yaml"), vars(args_cli))
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
