#!/usr/bin/env python3
"""Check the playback gun's geometry, TCP alignment, mass and physical contact."""

import argparse
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)

import gymnasium as gym
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks  # noqa: F401


def main():
    task = "Unitree-b2z1-AMP-Rear-Curve-Gait-Heading-V5-Finetune-Play"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    for name in ("front_rgb", "rear_rgb", "front_depth", "rear_depth", "d435i", "sky_light"):
        setattr(cfg.scene, name, None)
    cfg.commands.ee_target.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    # Isolate the contact test from falling/locomotion and from the live bridge.
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    with tempfile.TemporaryDirectory(prefix="b2_z1_gun_check_") as exchange:
        cfg.commands.ee_target.vision_exchange_directory = exchange
        cfg.commands.ee_target.trajectory_file = str(Path(exchange) / "latest_trajectory.json")
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            gun_id = robot.find_bodies("link_gun")[0][0]
            tcp_id = robot.find_bodies("dummy_target")[0][0]
            stage = env.sim.stage
            gun = stage.GetPrimAtPath("/World/envs/env_0/Robot/link_gun")
            shape_root = gun.GetChild("welding_gun")
            cylinders = [p for p in Usd.PrimRange(shape_root) if p.IsA(UsdGeom.Cylinder)]
            assert len(cylinders) == 1, "Expected one visible cylinder"
            cylinder = UsdGeom.Cylinder(cylinders[0])
            assert abs(cylinder.GetHeightAttr().Get() - 1.10) < 1e-6
            assert abs(cylinder.GetRadiusAttr().Get() - 0.028) < 1e-6
            assert cylinder.GetAxisAttr().Get() == "Z"
            assert UsdPhysics.CollisionAPI(cylinders[0]).GetCollisionEnabledAttr().Get()
            assert gun.HasAPI(UsdPhysics.RigidBodyAPI)
            assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(shape_root))
            assert not any(p.HasAPI(UsdPhysics.MassAPI) for p in Usd.PrimRange(shape_root))
            mass = float(robot.root_physx_view.get_masses()[0, gun_id])
            assert abs(mass - 0.001) < 1e-6, f"Unexpected gun-link mass: {mass} kg"

            transform = UsdGeom.XformCache().ComputeRelativeTransform(cylinders[0], gun)[0]
            tip_b = transform.Transform(Gf.Vec3d(0.0, 0.0, -0.55))
            offset = torch.tensor([list(tip_b)], device=env.device, dtype=torch.float32)
            gun_quat = robot.data.body_quat_w[:, gun_id]
            tip_w = robot.data.body_pos_w[:, gun_id] + quat_apply(gun_quat, offset)
            tip_error = float(torch.linalg.vector_norm(tip_w - robot.data.body_pos_w[:, tcp_id]))
            assert tip_error < 1e-5, f"Cylinder tip misses TCP by {tip_error} m"

            sensor = env.scene["contact_forces"]
            sensor_id = sensor.find_bodies("link_gun")[0][0]
            action = torch.zeros(env.action_space.shape, device=env.device)
            with torch.inference_mode():
                env.step(action)
            baseline_force = float(torch.linalg.vector_norm(sensor.data.net_forces_w[0, sensor_id]))
            assert baseline_force < 1e-4, "Gun already contacts the environment before the test"

            # Intersect just the gun tip with a small static sphere. No contact
            # is faked: the existing contact sensor must receive a PhysX force.
            gun_quat = robot.data.body_quat_w[:, gun_id]
            axis_w = quat_apply(gun_quat, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))
            tip_w = robot.data.body_pos_w[:, gun_id] + quat_apply(gun_quat, offset)
            obstacle_position = (tip_w - 0.02 * axis_w)[0].cpu().tolist()
            obstacle_cfg = sim_utils.SphereCfg(
                radius=0.04,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
                ),
            )
            obstacle_cfg.func("/World/GunContactProbe", obstacle_cfg, translation=obstacle_position)
            peak_force = 0.0
            with torch.inference_mode():
                for _ in range(10):
                    env.step(action)
                    peak_force = max(
                        peak_force,
                        float(torch.linalg.vector_norm(sensor.data.net_forces_w[0, sensor_id])),
                    )
            assert peak_force > 0.01, "Gun collider generated no physical contact force"
            print(
                f"PASS: cylinder 1.10 m x radius 0.028 m; link mass={mass:.6f} kg; "
                f"TCP error={tip_error:.3g} m; baseline force={baseline_force:.3g} N; "
                f"probe contact force={peak_force:.3f} N",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        launcher.app.close()
