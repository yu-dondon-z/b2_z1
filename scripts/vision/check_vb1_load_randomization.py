#!/usr/bin/env python3
"""Verify VB1 load sampling is per-env, bounded, and constant within an episode."""

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks  # noqa: F401


def main():
    task = "Unitree-b2z1-AMP-VB1-Loaded-Gun"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=16)
    cfg.scene.sky_light = None
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        env.reset()
        first = env._hose_mass_kg.clone()
        first_anchor = env._hose_anchor_w.clone()
        first_slack = env._hose_slack_m.clone()
        low, high = cfg.hose_mass_range_kg
        assert bool(torch.all((first >= low) & (first <= high)))
        anchor_center = env.scene.env_origins + torch.tensor(cfg.hose_truck_anchor_env_m, device=env.device)
        anchor_jitter = torch.tensor(cfg.hose_anchor_jitter_m, device=env.device)
        assert bool(torch.all(torch.abs(first_anchor - anchor_center) <= anchor_jitter + 1e-5))
        assert bool(torch.all(torch.abs(first_slack - cfg.hose_slack_length_m) <= cfg.hose_slack_jitter_m + 1e-5))
        assert float(first.max() - first.min()) > 0.05, "Environments share one load"

        action = torch.zeros(env.action_space.shape, device=env.device)
        with torch.inference_mode():
            for _ in range(3):
                env.step(action)
        assert torch.equal(first, env._hose_mass_kg), "Load changed within an episode"
        assert torch.equal(first_anchor, env._hose_anchor_w), "Anchor changed within an episode"
        assert torch.equal(first_slack, env._hose_slack_m), "Slack changed within an episode"

        selected = torch.tensor([0, 1, 2, 3], device=env.device)
        with torch.inference_mode():
            env._reset_idx(selected)
        assert torch.equal(first[4:], env._hose_mass_kg[4:]), "Unreset environments changed load"
        assert torch.equal(first_anchor[4:], env._hose_anchor_w[4:]), "Unreset anchor changed"
        assert torch.equal(first_slack[4:], env._hose_slack_m[4:]), "Unreset slack changed"
        assert bool(torch.all((env._hose_mass_kg[selected] >= low) & (env._hose_mass_kg[selected] <= high)))
        assert not torch.equal(first[selected], env._hose_mass_kg[selected]), "Reset failed to resample load"
        if cfg.hose_slack_jitter_m:
            assert not torch.equal(first_slack[selected], env._hose_slack_m[selected]), "Slack was not resampled"
            assert not torch.equal(first_anchor[selected], env._hose_anchor_w[selected]), "Anchor was not resampled"
        assert torch.isfinite(env.hose_force_w).all()
        assert torch.isfinite(env.hose_torque_w).all()
        print(
            f"PASS: VB1 stage range {low:g}-{high:g} kg; 16 independent envs; "
            f"load held for episode and resampled only on selected reset; "
            f"mean force={env._hose_force_sum_n.mean().item() / 3.0:.2f} N",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        launcher.app.close()
