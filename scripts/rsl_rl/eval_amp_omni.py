"""Deterministically compare AMP checkpoints on six cardinal body commands."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument(
    "--agent",
    default="rsl_rl_cfg_entry_point",
    help="Gym registry entry point for the RSL-RL agent configuration.",
)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=60)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=300)
parser.add_argument("--forward_speed", type=float, default=0.25)
parser.add_argument("--cardinal_speed", type=float, default=0.10)
parser.add_argument("--yaw_speed", type=float, default=0.12)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from b2_z1.rsl_rl import AMP_RslRlOnPolicyRunnerCfg, AMPRslRlVecEnvWrapper
from b2_z1.rsl_rl.runners import AMP_OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import b2_z1.tasks  # noqa: F401


COMMANDS = (
    ("forward", (args_cli.forward_speed, 0.0, 0.0), 0, args_cli.forward_speed),
    ("backward", (-args_cli.cardinal_speed, 0.0, 0.0), 0, -args_cli.cardinal_speed),
    ("left", (0.0, args_cli.cardinal_speed, 0.0), 1, args_cli.cardinal_speed),
    ("right", (0.0, -args_cli.cardinal_speed, 0.0), 1, -args_cli.cardinal_speed),
    ("yaw_left", (0.0, 0.0, args_cli.yaw_speed), 2, args_cli.yaw_speed),
    ("yaw_right", (0.0, 0.0, -args_cli.yaw_speed), 2, -args_cli.yaw_speed),
)


def _install_commands(command, robot, masks):
    command.vel_command_b.zero_()
    command.is_longitudinal_env.zero_()
    command.is_lateral_env.zero_()
    command.is_yaw_env.zero_()
    command.is_mixed_env.zero_()
    command.is_standing_env.zero_()
    command.is_heading_hold_env.zero_()
    command.is_forward_env.zero_()
    command.is_backward_env.zero_()

    for name, values, _, _ in COMMANDS:
        mask = masks[name]
        command.vel_command_b[mask] = torch.tensor(
            values, device=command.device, dtype=command.vel_command_b.dtype
        )
        if name in ("forward", "backward"):
            command.is_longitudinal_env[mask] = True
            command.is_heading_hold_env[mask] = True
        elif name in ("left", "right"):
            command.is_lateral_env[mask] = True
            command.is_heading_hold_env[mask] = True
        else:
            command.is_yaw_env[mask] = True
        if name == "forward":
            command.is_forward_env[mask] = True
        elif name == "backward":
            command.is_backward_env[mask] = True
    command.straight_heading_target[:] = robot.data.heading_w
    if hasattr(command, "time_left"):
        command.time_left[:] = 1.0e6


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: AMP_RslRlOnPolicyRunnerCfg,
):
    if args_cli.num_envs % len(COMMANDS):
        raise ValueError("--num_envs must be divisible by 6")
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
    raw = env.unwrapped
    command = raw.command_manager.get_term("base_velocity")
    ee_command = raw.command_manager.get_term("ee_target")
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    group_size = raw.num_envs // len(COMMANDS)
    masks = {
        spec[0]: (ids >= index * group_size) & (ids < (index + 1) * group_size)
        for index, spec in enumerate(COMMANDS)
    }

    runner = AMP_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    reports = []
    for checkpoint in args_cli.checkpoints:
        path = retrieve_file_path(checkpoint)
        runner.load(path)
        policy = runner.get_inference_policy(device=raw.device)
        # A second ordinary reset after policy creation can attempt to update
        # Isaac buffers allocated under inference mode.  Trigger the standard
        # step-time timeout reset inside inference mode instead.
        raw.episode_length_buf[:] = raw.max_episode_length - 1
        with torch.inference_mode():
            obs, _, _, _ = env.step(
                torch.zeros((raw.num_envs, env.num_actions), device=raw.device)
            )
        _install_commands(command, robot, masks)

        sums = {
            name: {"error_sq": 0.0, "actual": 0.0, "cross": 0.0, "resets": 0}
            for name, *_ in COMMANDS
        }
        ee_error = 0.0
        tilt = 0.0
        samples = group_size * args_cli.eval_steps
        total_steps = args_cli.warmup_steps + args_cli.eval_steps
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            if step < args_cli.warmup_steps:
                continue

            linear = robot.data.root_lin_vel_b
            yaw = robot.data.root_ang_vel_b[:, 2]
            for name, _, axis, target in COMMANDS:
                mask = masks[name]
                actual = yaw[mask] if axis == 2 else linear[mask, axis]
                if axis == 0:
                    cross = torch.abs(linear[mask, 1]) + torch.abs(yaw[mask])
                elif axis == 1:
                    cross = torch.abs(linear[mask, 0]) + torch.abs(yaw[mask])
                else:
                    cross = torch.linalg.vector_norm(linear[mask, :2], dim=1)
                sums[name]["error_sq"] += torch.square(actual - target).sum().item()
                sums[name]["actual"] += actual.sum().item()
                sums[name]["cross"] += cross.sum().item()
                sums[name]["resets"] += int(dones[mask].sum().item())
            ee_error += float(ee_command.metrics["position_error"].mean().item())
            gravity_z = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
            tilt += float(torch.acos(-gravity_z).mean().item())

        rows = []
        for name, _, _, target in COMMANDS:
            values = sums[name]
            rows.append(
                {
                    "name": name,
                    "target": target,
                    "actual": values["actual"] / samples,
                    "rmse": math.sqrt(values["error_sq"] / samples),
                    "cross": values["cross"] / samples,
                    "resets": values["resets"],
                }
            )
        reports.append((path, rows, ee_error / args_cli.eval_steps, math.degrees(tilt / args_cli.eval_steps)))

    print("\n========== V20 deterministic cardinal-command gate ==========")
    for path, rows, ee_error, tilt_deg in reports:
        print(f"\ncheckpoint: {path}")
        print(f"mean TCP position error={ee_error:.4f} m, mean tilt={tilt_deg:.3f} deg")
        for row in rows:
            print(
                f"{row['name']:9s} target={row['target']:+.3f} "
                f"actual={row['actual']:+.3f} rmse={row['rmse']:.3f} "
                f"cross={row['cross']:.3f} resets={row['resets']}"
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
