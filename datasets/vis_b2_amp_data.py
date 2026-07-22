# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Go2 AMP motion datasets in Isaac Lab (no trained policy needed).

This script kinematically replays the mocap frames stored in the ``go2_motion``
directory directly onto a Go2 articulation, without running any RL policy.

.. code-block:: bash

    # Usage
    python datasets/vis_go2_amp_data.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize Go2 AMP motion datasets.")
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Directory containing the AMP motion files. Defaults to the 'b2_motion' folder next to this script.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_quat, quat_apply

from legged_amp_lab.assets.unitree import UNITREE_GO2_CFG, UNITREE_B2_CFG
from legged_amp_lab.rsl_rl.datasets import AMPLoader


@configclass
class Go2MotionSceneCfg(InteractiveSceneCfg):
    """Configuration for a minimal scene used to visualize Go2 AMP motions."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot -- disable the PD actuator gains so the joints follow the mocap frames exactly
    # (this mirrors zeroing out `control.stiffness` / `control.damping` in the legged_gym visualizer).
    robot: ArticulationCfg = UNITREE_B2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "M107-24-2": UNITREE_B2_CFG.actuators["M107-24-2"].replace(stiffness=0.0, damping=0.0),
            "2": UNITREE_B2_CFG.actuators["2"].replace(stiffness=0.0, damping=0.0),
            
        },
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene, loader: AMPLoader):
    """Kinematically replays every trajectory in the loader onto the robot in env 0."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    env_origin = scene.env_origins[0:1]

    for traj_idx in range(loader.num_motions):
        traj_name = loader.trajectory_names[traj_idx]
        traj_len = loader.trajectory_lens[traj_idx]
        print(f"[{traj_idx + 1}/{loader.num_motions}] Playing: {traj_name} ({traj_len:.2f}s)")

        t = 0.0
        while t + loader.time_between_frames + sim_dt < traj_len and simulation_app.is_running():
            frame = loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))

            # root pose (position + quaternion). Motion files store quaternions in xyzw order.
            root_pos = AMPLoader.get_root_pos_batch(frame) + env_origin
            root_rot_wxyz = convert_quat(AMPLoader.get_root_rot_batch(frame), to="wxyz")
            root_pose = torch.cat([root_pos, root_rot_wxyz], dim=-1)
            robot.write_root_pose_to_sim(root_pose)

            # root velocities are stored in the base's local frame -- rotate into world frame.
            lin_vel_w = quat_apply(root_rot_wxyz, AMPLoader.get_linear_vel_batch(frame))
            ang_vel_w = quat_apply(root_rot_wxyz, AMPLoader.get_angular_vel_batch(frame))
            robot.write_root_velocity_to_sim(torch.cat([lin_vel_w, ang_vel_w], dim=-1))

            # joint state
            joint_pos = AMPLoader.get_joint_pose_batch(frame)
            joint_vel = AMPLoader.get_joint_vel_batch(frame)

            # joint reindex
            # data order:  FL(hip,thigh,calf), FR(hip,thigh,calf), RL(hip,thigh,calf), RR(hip,thigh,calf)
            # IsaacLab order: all hips (FL,FR,RL,RR), all thighs, all calves
            DATA_TO_LAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
            joint_pos = joint_pos[:, DATA_TO_LAB]
            joint_vel = joint_vel[:, DATA_TO_LAB]

            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # # follow the robot with the camera
            # look_at = root_pose[0, :3].cpu().numpy().astype(np.float64)
            # cam_offset = np.array([-2.0, 0.0, 0.9])
            # sim.set_camera_view((look_at + cam_offset).tolist(), look_at.tolist())

            t += sim_dt

        print(f"  Done: {traj_name}")

    print("All trajectories played.")


def main():
    """Main function."""
    # motion files to load
    motion_dir = args_cli.motion_dir or os.path.join(os.path.dirname(__file__), "b2_motion")
    motion_files = sorted(glob.glob(os.path.join(motion_dir, "*")))
    if not motion_files:
        print(f"No motion files found in {motion_dir}")
        return

    # load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 0.0])

    # design scene: a single Go2 robot
    scene_cfg = Go2MotionSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")

    # load the AMP motion dataset
    loader = AMPLoader(device=sim.device, time_between_frames=sim.get_physics_dt(), motion_files=motion_files)

    # run playback
    run_simulator(sim, scene, loader)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
