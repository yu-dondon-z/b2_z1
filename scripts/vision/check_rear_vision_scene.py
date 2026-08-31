#!/usr/bin/env python3
"""Validate B2 front/rear RGB-D and optional D435i camera outputs."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="outputs/rear_vision_camera_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
import b2_z1.tasks  # noqa: F401


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    image = tensor[0].detach().cpu().numpy()
    if image.shape[-1] == 4:
        image = image[..., :3]
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_depth(path: Path, tensor: torch.Tensor) -> tuple[float, float]:
    depth = tensor[0].detach().cpu().numpy().squeeze()
    finite = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[finite] = np.clip(depth[finite] * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)
    if not np.any(finite):
        return float("nan"), float("nan")
    return float(depth[finite].min()), float(depth[finite].max())


def main() -> None:
    output = Path(args_cli.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(
        "Unitree-b2z1-AMP-Rear-Vision-Play",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env = gym.make("Unitree-b2z1-AMP-Rear-Vision-Play", cfg=cfg)
    env.reset()
    try:
        for _ in range(20):
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

        scene = env.unwrapped.scene
        robot = scene["robot"]
        base_id = robot.find_bodies("base_link")[0][0]
        arm_base_id = robot.find_bodies("link00")[0][0]
        tcp_id = robot.find_bodies("dummy_target")[0][0]
        base_pos_w = robot.data.body_pos_w[:, base_id]
        base_quat_w = robot.data.body_quat_w[:, base_id]
        arm_quat_w = robot.data.body_quat_w[:, arm_base_id]
        arm_quat_b = quat_mul(quat_conjugate(base_quat_w), arm_quat_w)
        tcp_pos_b = quat_apply_inverse(
            base_quat_w,
            robot.data.body_pos_w[:, tcp_id] - base_pos_w,
        )
        print(f"arm base relative quaternion (wxyz): {arm_quat_b[0].cpu().tolist()}")
        print(f"current TCP relative position (xyz m): {tcp_pos_b[0].cpu().tolist()}")
        if abs(float(arm_quat_b[0, 3])) < 0.95:
            raise RuntimeError("Z1 base is not rotated approximately 180 degrees behind B2")

        for name in ("front_rgb", "rear_rgb"):
            rgb = scene[name].data.output["rgb"]
            _save_rgb(output / f"{name}.png", rgb)
            print(f"{name}: rgb={tuple(rgb.shape)}")
        for name in ("front_depth", "rear_depth"):
            depth = scene[name].data.output["distance_to_image_plane"]
            depth_range = _save_depth(output / f"{name}_mm.png", depth)
            print(f"{name}: depth={tuple(depth.shape)}, valid_range_m={depth_range}")
        d435i = scene["d435i"].data.output
        _save_rgb(output / "d435i_rgb.png", d435i["rgb"])
        depth_range = _save_depth(output / "d435i_depth_mm.png", d435i["distance_to_image_plane"])
        print(
            f"d435i: rgb={tuple(d435i['rgb'].shape)}, "
            f"depth={tuple(d435i['distance_to_image_plane'].shape)}, valid_range_m={depth_range}"
        )
        print(f"Saved camera diagnostics to: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
