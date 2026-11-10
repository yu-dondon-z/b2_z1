# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from training_options import load_training_checkpoint, persistent_log_root


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
parser.add_argument("--vd2_warm_start", type=str, default=None, help="Warm-start model and AMP state without optimizer.")
parser.add_argument("--vd3-warm-start", "--vd3_warm_start", dest="vd3_warm_start", type=str,
                    default=None, help="Load policy and AMP into fresh VD3 optimizer, iteration and curricula.")
parser.add_argument("--vd4-warm-start", "--vd4_warm_start", dest="vd4_warm_start", type=str,
                    default=None, help="Load policy and AMP into fresh VD4 optimizer, iteration and curricula.")
parser.add_argument("--vd5-warm-start", "--vd5_warm_start", dest="vd5_warm_start", type=str,
                    default=None, help="Load policy and AMP into fresh VD5 optimizer, iteration and curricula.")
parser.add_argument("--vd5-startup-from", type=str, default=None,
                    help="Adapt an original VD5 checkpoint; retain optimizer, iteration and ground curriculum.")
parser.add_argument("--log-root", "--log_root", dest="log_root", type=str, default=None,
                    help="Persistent parent of experiment directories; default: logs/rsl_rl.")
parser.add_argument("--keep-last-checkpoints", "--keep_last_checkpoints", dest="keep_last_checkpoints",
                    type=int, default=None, help="Keep this runner's latest N checkpoints; 0 disables pruning.")
parser.add_argument("--vd2_legacy_curve_level", type=float, default=None, help="Approximate curve level for a legacy VD2 checkpoint.")
parser.add_argument("--vd2_legacy_position_steps", type=int, default=None, help="Approximate arm curriculum expansions for a legacy VD2 checkpoint.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if (args_cli.vd2_legacy_curve_level is None) != (args_cli.vd2_legacy_position_steps is None):
    parser.error("Specify both VD2 legacy curriculum options together")
if sum(bool(value) for value in (args_cli.vd2_warm_start, args_cli.vd3_warm_start,
                                  args_cli.vd4_warm_start, args_cli.vd5_warm_start)) > 1:
    parser.error("Choose one warm-start option")
if args_cli.resume and (args_cli.vd2_warm_start or args_cli.vd3_warm_start or args_cli.vd4_warm_start
                        or args_cli.vd5_warm_start):
    parser.error("--resume and warm start are mutually exclusive")
if args_cli.vd5_startup_from and (args_cli.resume or args_cli.vd2_warm_start or args_cli.vd3_warm_start
                                or args_cli.vd4_warm_start or args_cli.vd5_warm_start):
    parser.error("--vd5-startup-from is mutually exclusive with resume and warm start")
if args_cli.keep_last_checkpoints is not None and args_cli.keep_last_checkpoints < 0:
    parser.error("--keep-last-checkpoints must be nonnegative")
if args_cli.log_root:
    try:
        args_cli.log_root = str(persistent_log_root(args_cli.log_root))
    except ValueError as error:
        parser.error(str(error))

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
from rsl_rl.runners import OnPolicyRunner
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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from b2_z1.rsl_rl.wrappers import AMPRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import b2_z1.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.keep_last_checkpoints is not None:
        agent_cfg.keep_last_checkpoints = args_cli.keep_last_checkpoints
    if args_cli.vd3_warm_start and agent_cfg.experiment_name != "b2_z1_vd3_curve_amp":
        raise ValueError("--vd3-warm-start requires experiment_name=b2_z1_vd3_curve_amp")
    if args_cli.vd4_warm_start and agent_cfg.experiment_name != "b2_z1_vd4_front_calibrated_amp":
        raise ValueError("--vd4-warm-start requires experiment_name=b2_z1_vd4_front_calibrated_amp")
    if args_cli.vd5_warm_start and agent_cfg.experiment_name != "b2_z1_vd5_world_seam_amp":
        raise ValueError("--vd5-warm-start requires experiment_name=b2_z1_vd5_world_seam_amp")
    if args_cli.vd5_startup_from and agent_cfg.experiment_name != "b2_z1_vd5_startup_amp":
        raise ValueError("--vd5-startup-from requires experiment_name=b2_z1_vd5_startup_amp")
    if args_cli.vd2_legacy_curve_level is not None and agent_cfg.experiment_name != "b2_z1_vd2_curve_amp":
        raise ValueError("Legacy VD2 curriculum approximations cannot be applied outside VD2")
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
    log_parent = persistent_log_root(args_cli.log_root or os.path.join("logs", "rsl_rl"))
    log_root_path = str(log_parent / agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Resolve the old run before making a new directory: .* must not select the new empty run.
    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
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
    os.makedirs(log_dir, exist_ok=False)
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
    if agent_cfg.experiment_name == "b2_z1_vd3_curve_amp":
        from b2_z1.assets.vd3_runtime_contract import audit_vd3_runtime_asset

        runtime_asset = audit_vd3_runtime_asset(env.unwrapped.scene["robot"])
        with open(os.path.join(log_dir, "vd3_runtime_asset.json"), "w", encoding="utf-8") as stream:
            json.dump(runtime_asset, stream, indent=2, ensure_ascii=False)
        print("[INFO] VD3 runtime asset audit passed; contract saved to vd3_runtime_asset.json")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    use_amp = hasattr(agent_cfg, "amp_cfg")
    wrapper_class = AMPRslRlVecEnvWrapper if use_amp else RslRlVecEnvWrapper
    env = wrapper_class(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner_class = AMP_OnPolicyRunner if use_amp else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    warm_start = (args_cli.vd5_warm_start or args_cli.vd4_warm_start or
                  args_cli.vd3_warm_start or args_cli.vd2_warm_start)
    if warm_start:
        print(f"[INFO]: Warm-starting {agent_cfg.experiment_name} from: {warm_start}; optimizer/iteration/curricula reset")
    if args_cli.vd5_startup_from:
        print(f"[INFO]: Adapting VD5 startup from: {args_cli.vd5_startup_from}; optimizer/iteration/curricula retained")
    legacy = None
    if args_cli.vd2_legacy_curve_level is not None:
        legacy = (args_cli.vd2_legacy_curve_level, args_cli.vd2_legacy_position_steps)
    load_training_checkpoint(
        runner, env, agent_cfg.experiment_name if use_amp else "", resume_path=resume_path,
        warm_start=warm_start, legacy_curriculum=legacy,
        startup_from=args_cli.vd5_startup_from,
    )
    if agent_cfg.experiment_name == "b2_z1_vd3_curve_amp":
        from b2_z1.rsl_rl.utils.curriculum_state import capture_curriculum_state

        with open(os.path.join(log_dir, "vd3_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_curriculum_state(env), stream, indent=2, ensure_ascii=False)
    if agent_cfg.experiment_name == "b2_z1_vd4_front_calibrated_amp":
        from b2_z1.rsl_rl.utils.curriculum_state import capture_curriculum_state

        with open(os.path.join(log_dir, "vd4_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_curriculum_state(env), stream, indent=2, ensure_ascii=False)
    if agent_cfg.experiment_name in {"b2_z1_vd5_world_seam_amp", "b2_z1_vd5_startup_amp"}:
        from b2_z1.rsl_rl.utils.vd5_curriculum_state import capture_vd5_curriculum_state

        with open(os.path.join(log_dir, "vd5_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_vd5_curriculum_state(env), stream, indent=2, ensure_ascii=False)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # Startup phase uses physical episode time; randomizing that clock would skip
    # the handoff immediately after reset while keeping the nominal physical pose.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=agent_cfg.experiment_name != "b2_z1_vd5_startup_amp")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
