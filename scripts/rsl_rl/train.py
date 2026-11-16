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


STARTUP_EXPERIMENTS = {
    "b2_z1_vd5_startup_amp", "b2_z1_vd6_smooth_amp", "b2_z1_vd7_tcp_amp",
    "b2_z1_vd8_roll_landing_amp", "b2_z1_vd9_physics_rate_amp",
    "b2_z1_vd10_hollow_gun_amp",
    "b2_z1_vd11_motion_quality_amp",
    "b2_z1_vd12_geometry_guard_amp",
    "b2_z1_vd13_mount75_amp",
    "b2_z1_vd14_posture_amp",
}
TUBE_EXPERIMENTS = {"b2_z1_vd10_hollow_gun_amp", "b2_z1_vd11_motion_quality_amp",
                    "b2_z1_vd12_geometry_guard_amp", "b2_z1_vd13_mount75_amp", "b2_z1_vd14_posture_amp"}


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
parser.add_argument('--vd6-from', type=str, default=None,
                    help='Continue VD5 Startup in VD6 with optimizer, AMP and curriculum retained.')
parser.add_argument('--vd7-from', type=str, default=None,
                    help='Continue VD5 Startup in VD7 with optimizer, AMP and curriculum retained.')
parser.add_argument('--vd8-from', type=str, default=None,
                    help='Continue VD5 Startup in VD8 with optimizer, AMP and curriculum retained.')
parser.add_argument('--vd9-from', type=str, default=None,
                    help='Warm-start VD9 from a VD5 Startup checkpoint with optimizer reset.')
parser.add_argument('--vd9-warm-start', type=str, default=None,
                    help='Warm-start VD9 from a VD9 checkpoint with optimizer and curriculum reset.')
parser.add_argument('--vd10-from', type=str, default=None,
                    help='Start VD10 with an old actor/AMP; reset optimizer, iteration and old-gun curriculum.')
parser.add_argument('--vd11-from', type=str, default=None,
                    help='Adapt VD10 to VD11; retain actor/AMP and ground curriculum, reset optimizer/iteration.')
parser.add_argument('--vd12-from', type=str, default=None,
                    help='Adapt only a VD11 measured-tube checkpoint to VD12; retain actor/AMP, '
                         'reset optimizer/iteration/ground curriculum. An empty value starts fresh.')
parser.add_argument('--vd13-from', type=str, default=None,
                    help='Transfer VD12 policy/AMP into isolated mount75 VD13; reset optimizer/iteration/curricula.')
parser.add_argument('--vd14-from', type=str, default=None,
                    help='Transfer VD13 policy/AMP and curriculum into posture VD14; reset optimizer/iteration.')
parser.add_argument('--vd10-trace', action='store_true',
                    help='Save the first 120 actual 200 Hz VD10 tool/arm physics samples for local diagnosis.')
parser.add_argument('--vd10-zero-action-probe-steps', type=int, default=0,
                    help='VD10 only: run N 50 Hz zero-action steps without optimizer updates, with real contact/terminations.')
parser.add_argument('--vd10-policy-probe-steps', type=int, default=0,
                    help='VD10 only: run N deterministic 50 Hz actor steps, without learning, with real contact/terminations.')
parser.add_argument('--vd10-probe-nominal-reset', action='store_true',
                    help='With a VD10 probe only: reset without AMP reference-state initialization.')
parser.add_argument('--vd10-probe-ground-blend', type=float, default=None,
                    help='With a VD10 probe only: hold its ground curriculum at the specified blend in [0,1].')
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
if args_cli.vd14_from is not None and (args_cli.vd12_from is not None or args_cli.vd13_from is not None or any((
        args_cli.resume, args_cli.vd2_warm_start, args_cli.vd3_warm_start, args_cli.vd4_warm_start,
        args_cli.vd5_warm_start, args_cli.vd5_startup_from, args_cli.vd6_from, args_cli.vd7_from,
        args_cli.vd8_from, args_cli.vd9_from, args_cli.vd9_warm_start, args_cli.vd10_from, args_cli.vd11_from))):
    parser.error('--vd14-from is exclusive with all other load modes, including fresh start')
if args_cli.vd13_from is not None and (args_cli.vd12_from is not None or any((
        args_cli.resume, args_cli.vd2_warm_start, args_cli.vd3_warm_start, args_cli.vd4_warm_start,
        args_cli.vd5_warm_start, args_cli.vd5_startup_from, args_cli.vd6_from, args_cli.vd7_from,
        args_cli.vd8_from, args_cli.vd9_from, args_cli.vd9_warm_start, args_cli.vd10_from, args_cli.vd11_from))):
    parser.error('--vd13-from is exclusive with all other load modes, including fresh start')
vd10_probe_steps = args_cli.vd10_zero_action_probe_steps or args_cli.vd10_policy_probe_steps
if args_cli.vd10_zero_action_probe_steps and args_cli.vd10_policy_probe_steps:
    parser.error('Choose one VD10 probe action mode')
if args_cli.vd10_probe_ground_blend is not None:
    if not vd10_probe_steps or not 0.0 <= args_cli.vd10_probe_ground_blend <= 1.0:
        parser.error('--vd10-probe-ground-blend needs a probe and a value in [0,1]')
if (args_cli.vd2_legacy_curve_level is None) != (args_cli.vd2_legacy_position_steps is None):
    parser.error("Specify both VD2 legacy curriculum options together")
if sum(bool(value) for value in (args_cli.vd2_warm_start, args_cli.vd3_warm_start,
                                  args_cli.vd4_warm_start, args_cli.vd5_warm_start,
                                  args_cli.vd9_warm_start)) > 1:
    parser.error("Choose one warm-start option")
if args_cli.resume and (args_cli.vd2_warm_start or args_cli.vd3_warm_start or args_cli.vd4_warm_start
                        or args_cli.vd5_warm_start or args_cli.vd9_warm_start):
    parser.error("--resume and warm start are mutually exclusive")
if args_cli.vd5_startup_from and (args_cli.resume or args_cli.vd2_warm_start or args_cli.vd3_warm_start
                                or args_cli.vd4_warm_start or args_cli.vd5_warm_start
                                or args_cli.vd9_warm_start):
    parser.error("--vd5-startup-from is mutually exclusive with resume and warm start")
if args_cli.vd6_from and (args_cli.resume or args_cli.vd5_startup_from or args_cli.vd2_warm_start
                        or args_cli.vd3_warm_start or args_cli.vd4_warm_start or args_cli.vd5_warm_start
                        or args_cli.vd9_warm_start):
    parser.error('--vd6-from is mutually exclusive with other checkpoint loading modes')
if args_cli.keep_last_checkpoints is not None and args_cli.keep_last_checkpoints < 0:
    parser.error("--keep-last-checkpoints must be nonnegative")
if args_cli.vd7_from and any((args_cli.resume,args_cli.vd6_from,args_cli.vd5_startup_from,
                             args_cli.vd2_warm_start,args_cli.vd3_warm_start,args_cli.vd4_warm_start,
                             args_cli.vd5_warm_start,args_cli.vd8_from,args_cli.vd9_from,
                             args_cli.vd9_warm_start)):
    parser.error('--vd7-from is mutually exclusive with other loading modes')
if args_cli.vd8_from and any((args_cli.resume,args_cli.vd6_from,args_cli.vd5_startup_from,
                             args_cli.vd2_warm_start,args_cli.vd3_warm_start,args_cli.vd4_warm_start,
                             args_cli.vd5_warm_start,args_cli.vd7_from,args_cli.vd9_from,
                             args_cli.vd9_warm_start)):
    parser.error('--vd8-from is mutually exclusive with other loading modes')
if args_cli.vd9_from and any((args_cli.resume,args_cli.vd6_from,args_cli.vd5_startup_from,
                             args_cli.vd2_warm_start,args_cli.vd3_warm_start,args_cli.vd4_warm_start,
                             args_cli.vd5_warm_start,args_cli.vd7_from,args_cli.vd8_from,
                             args_cli.vd9_warm_start)):
    parser.error('--vd9-from is mutually exclusive with other loading modes')
if args_cli.vd9_warm_start and any((args_cli.resume, args_cli.vd5_startup_from, args_cli.vd6_from,
                                  args_cli.vd7_from, args_cli.vd8_from, args_cli.vd9_from)):
    parser.error('--vd9-warm-start is mutually exclusive with other checkpoint loading modes')
if args_cli.vd10_from and any((args_cli.resume, args_cli.vd2_warm_start, args_cli.vd3_warm_start,
                              args_cli.vd4_warm_start, args_cli.vd5_warm_start, args_cli.vd5_startup_from,
                              args_cli.vd6_from, args_cli.vd7_from, args_cli.vd8_from, args_cli.vd9_from,
                              args_cli.vd9_warm_start)):
    parser.error('--vd10-from is mutually exclusive with resume and all other checkpoint loading modes')
if args_cli.vd11_from and any((args_cli.resume, args_cli.vd2_warm_start, args_cli.vd3_warm_start,
                              args_cli.vd4_warm_start, args_cli.vd5_warm_start, args_cli.vd5_startup_from,
                              args_cli.vd6_from, args_cli.vd7_from, args_cli.vd8_from, args_cli.vd9_from,
                              args_cli.vd9_warm_start, args_cli.vd10_from)):
    parser.error('--vd11-from is mutually exclusive with resume and all other checkpoint loading modes')
if args_cli.vd12_from is not None and any((args_cli.resume, args_cli.vd2_warm_start,
                                          args_cli.vd3_warm_start, args_cli.vd4_warm_start,
                                          args_cli.vd5_warm_start, args_cli.vd5_startup_from,
                                          args_cli.vd6_from, args_cli.vd7_from, args_cli.vd8_from,
                                          args_cli.vd9_from, args_cli.vd9_warm_start,
                                          args_cli.vd10_from, args_cli.vd11_from)):
    parser.error('--vd12-from (including fresh start) is mutually exclusive with all other loading modes')
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
    if args_cli.vd6_from and agent_cfg.experiment_name != 'b2_z1_vd6_smooth_amp':
        raise ValueError('--vd6-from requires experiment_name=b2_z1_vd6_smooth_amp')
    if args_cli.vd7_from and agent_cfg.experiment_name != 'b2_z1_vd7_tcp_amp':
        raise ValueError('--vd7-from requires experiment_name=b2_z1_vd7_tcp_amp')
    if args_cli.vd8_from and agent_cfg.experiment_name != 'b2_z1_vd8_roll_landing_amp':
        raise ValueError('--vd8-from requires experiment_name=b2_z1_vd8_roll_landing_amp')
    if args_cli.vd9_from and agent_cfg.experiment_name != 'b2_z1_vd9_physics_rate_amp':
        raise ValueError('--vd9-from requires experiment_name=b2_z1_vd9_physics_rate_amp')
    if args_cli.vd9_warm_start and agent_cfg.experiment_name != 'b2_z1_vd9_physics_rate_amp':
        raise ValueError('--vd9-warm-start requires experiment_name=b2_z1_vd9_physics_rate_amp')
    if args_cli.vd10_from and agent_cfg.experiment_name != 'b2_z1_vd10_hollow_gun_amp':
        raise ValueError('--vd10-from requires experiment_name=b2_z1_vd10_hollow_gun_amp')
    if args_cli.vd11_from and agent_cfg.experiment_name != 'b2_z1_vd11_motion_quality_amp':
        raise ValueError('--vd11-from requires experiment_name=b2_z1_vd11_motion_quality_amp')
    if args_cli.vd12_from is not None and agent_cfg.experiment_name != 'b2_z1_vd12_geometry_guard_amp':
        raise ValueError('--vd12-from requires experiment_name=b2_z1_vd12_geometry_guard_amp')
    if args_cli.vd13_from is not None and agent_cfg.experiment_name != 'b2_z1_vd13_mount75_amp':
        raise ValueError('--vd13-from requires experiment_name=b2_z1_vd13_mount75_amp')
    if args_cli.vd14_from is not None and agent_cfg.experiment_name != 'b2_z1_vd14_posture_amp':
        raise ValueError('--vd14-from requires experiment_name=b2_z1_vd14_posture_amp')
    is_vd14 = agent_cfg.experiment_name == 'b2_z1_vd14_posture_amp'
    if is_vd14 != bool(getattr(env_cfg, 'vd14_posture_profile', None)):
        raise ValueError('VD14 posture environment and experiment must be used together')
    if (agent_cfg.experiment_name in {'b2_z1_vd13_mount75_amp', 'b2_z1_vd14_posture_amp'}) != bool(
            getattr(env_cfg, 'vd13_mount_profile', None)):
        raise ValueError('VD13 mount environment requires a VD13 or VD14 experiment')
    if is_vd14:
        from b2_z1.assets.vd14_contract import validate_environment_profile
        validate_environment_profile(env_cfg)
    elif agent_cfg.experiment_name == 'b2_z1_vd13_mount75_amp':
        from b2_z1.assets.vd13_contract import validate_environment_profile
        validate_environment_profile(env_cfg)
    if (agent_cfg.experiment_name in {'b2_z1_vd12_geometry_guard_amp', 'b2_z1_vd13_mount75_amp', 'b2_z1_vd14_posture_amp'}) != bool(
            getattr(env_cfg, 'vd12_geometry_profile', None)):
        raise ValueError('VD12 geometry environment and VD12 experiment name must be used together')
    if args_cli.vd10_probe_nominal_reset:
        if not vd10_probe_steps or agent_cfg.experiment_name not in TUBE_EXPERIMENTS:
            raise ValueError('--vd10-probe-nominal-reset requires a VD10 probe')
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
    if agent_cfg.experiment_name in TUBE_EXPERIMENTS:
        if agent_cfg.experiment_name in {'b2_z1_vd13_mount75_amp', 'b2_z1_vd14_posture_amp'}:
            from b2_z1.assets.vd13_runtime_contract import audit_vd13_runtime_asset as audit_vd10_runtime_asset
        else:
            from b2_z1.assets.vd10_runtime_contract import audit_vd10_runtime_asset

        runtime_asset = audit_vd10_runtime_asset(env.unwrapped.scene['robot'])
        with open(os.path.join(log_dir, 'vd10_runtime_asset.json'), 'w', encoding='utf-8') as stream:
            json.dump(runtime_asset, stream, indent=2, ensure_ascii=False)
        print(f'[INFO] {agent_cfg.experiment_name} actual PhysX asset audit passed; saved to vd10_runtime_asset.json')

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
    if args_cli.vd6_from:
        print(f'[INFO]: Continuing VD5 Startup as VD6 from {args_cli.vd6_from}; optimizer/AMP/iteration/curricula retained')
    if args_cli.vd9_from:
        print(f'[INFO]: Warm-starting VD9 from {args_cli.vd9_from}; optimizer/iteration reset, AMP and VD5 curriculum retained')
    if args_cli.vd9_warm_start:
        print(f'[INFO]: Warm-starting VD9 from {args_cli.vd9_warm_start}; optimizer/iteration/curriculum reset')
    if args_cli.vd10_from:
        print(f'[INFO]: Warm-starting VD10 from {args_cli.vd10_from}; optimizer/iteration/old-gun curriculum reset')
    if args_cli.vd11_from:
        print(f'[INFO]: Adapting VD11 from {args_cli.vd11_from}; optimizer/iteration reset, AMP and tube curriculum retained')
    if args_cli.vd12_from:
        print(f'[INFO]: Adapting VD12 from VD11 {args_cli.vd12_from}; actor/AMP retained, '
              'optimizer/iteration/ground curriculum reset; conservative exclusion zones, not contact forces')
    if args_cli.vd13_from:
        print(f'[INFO]: VD13 warm start from VD12 {args_cli.vd13_from}; reset optimizer/iteration/curricula')
    if args_cli.vd14_from:
        print(f'[INFO]: VD14 posture adaptation from VD13 {args_cli.vd14_from}; '
              'retain policy/AMP/curriculum, reset optimizer/iteration')
    legacy = None
    if args_cli.vd2_legacy_curve_level is not None:
        legacy = (args_cli.vd2_legacy_curve_level, args_cli.vd2_legacy_position_steps)
    load_training_checkpoint(
        runner, env, agent_cfg.experiment_name if use_amp else "", resume_path=resume_path,
        warm_start=warm_start, legacy_curriculum=legacy,
        startup_from=args_cli.vd5_startup_from,
        vd6_from=args_cli.vd6_from,
        vd7_from=args_cli.vd7_from,
        vd8_from=args_cli.vd8_from,
        vd9_from=args_cli.vd9_from,
        vd9_warm_start=args_cli.vd9_warm_start,
        vd10_from=args_cli.vd10_from,
        vd11_from=args_cli.vd11_from,
        vd12_from=args_cli.vd12_from,
        vd13_from=args_cli.vd13_from,
        vd14_from=args_cli.vd14_from,
    )
    if args_cli.vd10_probe_nominal_reset:
        # Validate/load the checkpoint against its original training reset
        # distribution first. Nominal reset is a declared diagnostic override,
        # not a change to the saved curriculum's physical startup contract.
        env.unwrapped.cfg.reference_state_initialization_prob = 0.0
        env.reset()
    if args_cli.vd10_probe_ground_blend is not None:
        if agent_cfg.experiment_name not in TUBE_EXPERIMENTS:
            raise ValueError('--vd10-probe-ground-blend requires VD10, VD11 or VD12')
        goal = env.unwrapped.command_manager.get_term('ee_target')
        goal._vd5_ground_blend = args_cli.vd10_probe_ground_blend
        # Explicit diagnostic override, not a training or checkpoint mutation.
        goal._vd5_ground_transition_steps = 0
        env.reset()
    if agent_cfg.experiment_name in TUBE_EXPERIMENTS:
        from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import link_pose_w, quat_apply, tcp_pose_w
        from b2_z1.tasks.manager_based.b2_z1.mdp.vd10_geometry import cylinder_surface_min_z

        robot = env.unwrapped.scene['robot']
        joint_ids, _ = robot.find_joints([f'joint{i}' for i in range(1, 7)], preserve_order=True)
        tcp_w, _ = tcp_pose_w(robot)
        link_id = robot.body_names.index('link06')
        link_w, link_quat_w = link_pose_w(robot, link_id)
        axis = quat_apply(link_quat_w, link_w.new_tensor((1., 0., 0.)).expand_as(link_w))
        tube_min_z = cylinder_surface_min_z(tcp_w, axis)
        base_z = robot.data.root_pos_w[:, 2]
        with open(os.path.join(log_dir, 'vd10_reset_diagnostics.json'), 'w', encoding='utf-8') as stream:
            json.dump({
                'sample_arm_joint_rad': robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
                'sample_tcp_relative_to_root_m': (tcp_w[0] - robot.data.root_pos_w[0]).detach().cpu().tolist(),
                'min_tube_surface_z_m': float(tube_min_z.min().item()),
                'max_tube_surface_z_m': float(tube_min_z.max().item()),
                'min_base_z_m': float(base_z.min().item()),
                'max_base_z_m': float(base_z.max().item()),
                'tube_axis_z_range': [float(axis[:, 2].min().item()), float(axis[:, 2].max().item())],
                'reset_reference_fraction': float(env.unwrapped.cfg.reference_state_initialization_prob),
                'clearance_geometry': 'Finite 1.1 m cylinder, radius 0.028 m; analytic surface minimum, not capsule.',
            }, stream, indent=2)
        print('[INFO] VD10 post-reset gun clearance and joint pose saved to vd10_reset_diagnostics.json')
    vd10_trace = []
    if args_cli.vd10_trace:
        if agent_cfg.experiment_name not in TUBE_EXPERIMENTS:
            raise ValueError('--vd10-trace requires VD10, VD11 or VD12')
        robot = env.unwrapped.scene['robot']
        contact = env.unwrapped.scene.sensors['contact_forces']
        contact_id = contact.body_names.index('link06')
        trace_foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        trace_foot_ids, _ = robot.find_bodies(trace_foot_names, preserve_order=True)
        trace_foot_contact_ids = [contact.body_names.index(name) for name in trace_foot_names]

        def record_vd10_physics():
            if len(vd10_trace) >= max(120, vd10_probe_steps * env.unwrapped.cfg.decimation):
                return
            from b2_z1.tasks.manager_based.b2_z1.mdp.tcp_frame import link_pose_w, quat_apply, tcp_pose_w
            from b2_z1.tasks.manager_based.b2_z1.mdp.vd10_geometry import cylinder_surface_min_z

            tcp_w, tcp_quat_w = tcp_pose_w(robot)
            link_w, link_quat_w = link_pose_w(robot, robot.body_names.index('link06'))
            axis = quat_apply(link_quat_w, link_w.new_tensor((1., 0., 0.)).expand_as(link_w))
            tube_min_z = cylinder_surface_min_z(tcp_w, axis)
            force_w = contact.data.net_forces_w[:, contact_id]
            goal = env.unwrapped.command_manager.get_term('ee_target')
            target_quat_w = goal._get_target_quat_w()
            tcp_axis = quat_apply(tcp_quat_w, tcp_w.new_tensor((1., 0., 0.)).expand_as(tcp_w))
            target_axis = quat_apply(target_quat_w, tcp_w.new_tensor((1., 0., 0.)).expand_as(tcp_w))
            axis_error = torch.acos((tcp_axis * target_axis).sum(dim=-1).clamp(-1., 1.))
            vd10_trace.append({
                'physics_step': len(vd10_trace) + 1,
                'sample_tcp_z_m': float(tcp_w[0, 2].item()),
                'sample_tube_min_z_m': float(tube_min_z[0].item()),
                'all_env_min_tube_z_m': float(tube_min_z.min().item()),
                'sample_tool_force_world_n': force_w[0].detach().cpu().tolist(),
                'max_tool_up_force_n': float(force_w[:, 2].max().item()),
                'sample_arm_joint_rad': robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
                'sample_arm_joint_velocity_rad_s': robot.data.joint_vel[0, joint_ids].detach().cpu().tolist(),
                'sample_arm_joint_target_rad': robot.data.joint_pos_target[0, joint_ids].detach().cpu().tolist(),
                'sample_root_height_m': float(robot.data.root_pos_w[0, 2].item()),
                'sample_root_quat_wxyz': robot.data.root_quat_w[0].detach().cpu().tolist(),
                'sample_episode_actor_step': int(env.unwrapped.episode_length_buf[0].item()),
                'sample_tube_axis_world': axis[0].detach().cpu().tolist(),
                'sample_tcp_target_world_m': goal._get_target_pos_w()[0].detach().cpu().tolist(),
                'ground_blend': float(goal._vd5_ground_blend),
                'env_tube_surface_min_z_m': tube_min_z.detach().cpu().tolist(),
                'env_tool_up_force_n': force_w[:, 2].detach().cpu().tolist(),
                'env_episode_actor_step': env.unwrapped.episode_length_buf.detach().cpu().tolist(),
                'env_tcp_world_m': tcp_w.detach().cpu().tolist(),
                'env_tcp_target_world_m': goal._get_target_pos_w().detach().cpu().tolist(),
                'env_base_quat_wxyz': robot.data.root_quat_w.detach().cpu().tolist(),
                'env_gun_axis_error_rad': axis_error.detach().cpu().tolist(),
                'env_foot_force_world_n': contact.data.net_forces_w[:, trace_foot_contact_ids].detach().cpu().tolist(),
                'env_foot_velocity_world_m_s': robot.data.body_link_lin_vel_w[:, trace_foot_ids].detach().cpu().tolist(),
                'env_foot_height_m': (robot.data.body_link_pos_w[:, trace_foot_ids, 2]
                                      - env.unwrapped.scene.env_origins[:, 2:3] - 0.032).detach().cpu().tolist(),
            })
            geometry = getattr(env.unwrapped, '_vd12_gun_guard', None)
            if geometry is not None:
                # Read the guard's pre-reset physics window; negative lower
                # bounds denote exclusion-zone violations, not measured contact.
                vd10_trace[-1]['vd12_geometry'] = {
                    'env_window_minimum_clearance_lower_bound_m': geometry.window.minimum_m.detach().cpu().tolist(),
                    'env_last_clearance_lower_bound_m': geometry.window.last_m.detach().cpu().tolist(),
                    'env_exclusion_violated': geometry.window.violated.detach().cpu().tolist(),
                    'env_first_bad_physics_step': geometry.window.first_bad_physics_step.detach().cpu().tolist(),
                }

        env.unwrapped._physics_step_callbacks.append(record_vd10_physics)
    if agent_cfg.experiment_name == "b2_z1_vd3_curve_amp":
        from b2_z1.rsl_rl.utils.curriculum_state import capture_curriculum_state

        with open(os.path.join(log_dir, "vd3_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_curriculum_state(env), stream, indent=2, ensure_ascii=False)
    if agent_cfg.experiment_name == "b2_z1_vd4_front_calibrated_amp":
        from b2_z1.rsl_rl.utils.curriculum_state import capture_curriculum_state

        with open(os.path.join(log_dir, "vd4_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_curriculum_state(env), stream, indent=2, ensure_ascii=False)
    if agent_cfg.experiment_name == "b2_z1_vd5_world_seam_amp" or agent_cfg.experiment_name in STARTUP_EXPERIMENTS:
        from b2_z1.rsl_rl.utils.vd5_curriculum_state import capture_vd5_curriculum_state

        with open(os.path.join(log_dir, "vd5_initial_curriculum.json"), "w", encoding="utf-8") as stream:
            json.dump(capture_vd5_curriculum_state(env), stream, indent=2, ensure_ascii=False)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if vd10_probe_steps:
        if (agent_cfg.experiment_name not in TUBE_EXPERIMENTS
                or not args_cli.vd10_trace or not 1 <= vd10_probe_steps <= 2000
                or args_cli.vd10_zero_action_probe_steps > 250):
            raise ValueError('VD10 probe needs --vd10-trace and 1..250 zero-action or 1..2000 policy steps')
        if args_cli.vd10_policy_probe_steps and not (
                args_cli.vd10_from or args_cli.vd11_from or args_cli.vd12_from or args_cli.vd13_from
                or args_cli.vd14_from or resume_path):
            raise ValueError('VD10 policy probe requires an explicitly loaded checkpoint')
        # Compare actual transient with actor-driven traces. All contacts and
        # terminations remain active; no optimizer is called in either mode.
        observations, _ = env.reset()
        end_counts = {name: 0 for name in env.unwrapped.termination_manager.active_terms}
        no_action = torch.zeros((env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
                                device=env.unwrapped.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device) if args_cli.vd10_policy_probe_steps else None
        total_resets = 0
        reset_events = []
        for actor_step in range(vd10_probe_steps):
            with torch.inference_mode():
                action = no_action if policy is None else policy(observations)
                observations, _, dones, _ = env.step(action)
            # Isaac Lab's get_term() retains a previous episode's event on
            # subsequent non-terminal steps. Only attribute it to this step
            # when the same environment actually terminated on this step.
            reset_this_step = dones.to(dtype=torch.bool)
            total_resets += int(reset_this_step.sum().item())
            for term in end_counts:
                end_counts[term] += int(
                    (env.unwrapped.termination_manager.get_term(term) & reset_this_step).sum().item()
                )
            if reset_this_step.any():
                reset_events.append({
                    'actor_step': actor_step + 1,
                    'env_ids': reset_this_step.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
                    'attributed_causes': {term: (
                        env.unwrapped.termination_manager.get_term(term) & reset_this_step
                    ).nonzero(as_tuple=True)[0].detach().cpu().tolist() for term in end_counts},
                })
        probe_filename = 'vd10_policy_probe.json' if policy is not None else 'vd10_zero_action_probe.json'
        with open(os.path.join(log_dir, probe_filename), 'w', encoding='utf-8') as stream:
            json.dump({'num_envs': int(env.unwrapped.num_envs),
                       'actor_steps': vd10_probe_steps,
                       'control_dt_s': float(env.unwrapped.step_dt),
                       'physics_dt_s': float(env.unwrapped.physics_dt),
                       'decimation': int(env.unwrapped.cfg.decimation),
                       'foot_names': trace_foot_names,
                       'trace_schema_version': 2,
                       'action_mode': 'deterministic_actor' if policy is not None else 'zero_action',
                       'checkpoint': args_cli.vd10_from or args_cli.vd11_from or args_cli.vd12_from
                                     or args_cli.vd13_from or args_cli.vd14_from or resume_path,
                       'vd12_geometry_profile': getattr(env.unwrapped.cfg, 'vd12_geometry_profile', None),
                       'vd12_geometry_note': 'Conservative exclusion zones, not physical contact measurements.',
                       'frozen_ground_blend': args_cli.vd10_probe_ground_blend,
                       'nominal_reset_override': args_cli.vd10_probe_nominal_reset,
                       'total_resets': total_resets,
                       'termination_events': end_counts,
                       'termination_events_note': 'Terminations attributed only on actual reset steps; '
                                                  'Isaac Lab retains old values otherwise and attributes '
                                                  'simultaneous causes to the last matching term.',
                       'reset_events': reset_events}, stream, indent=2)
        with open(os.path.join(log_dir, 'vd10_200hz_trace.json'), 'w', encoding='utf-8') as stream:
            json.dump(vd10_trace, stream, indent=2)
        env.close()
        return

    # run training
    # Startup phase uses physical episode time; randomizing that clock would skip
    # the handoff immediately after reset while keeping the nominal physical pose.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=agent_cfg.experiment_name not in STARTUP_EXPERIMENTS)
    if args_cli.vd10_trace:
        with open(os.path.join(log_dir, 'vd10_200hz_trace.json'), 'w', encoding='utf-8') as stream:
            json.dump(vd10_trace, stream, indent=2)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
