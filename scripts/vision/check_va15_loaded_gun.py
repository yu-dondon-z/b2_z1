#!/usr/bin/env python3
"""Check VA15 gun mass/collision and the applied hose force without starting vision."""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)

import gymnasium as gym
import torch
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks  # noqa: F401


def main():
    task = "Unitree-b2z1-AMP-VA15-Loaded-Gun-Vision-Play"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    for name in ("front_rgb", "rear_rgb", "front_depth", "rear_depth", "d435i", "sky_light"):
        setattr(cfg.scene, name, None)
    cfg.commands.ee_target.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    with tempfile.TemporaryDirectory(prefix="b2_z1_loaded_gun_check_") as exchange:
        cfg.commands.ee_target.vision_exchange_directory = exchange
        cfg.commands.ee_target.trajectory_file = str(Path(exchange) / "latest_trajectory.json")
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            gun_id = robot.find_bodies("link_gun")[0][0]
            mass = float(robot.root_physx_view.get_masses()[0, gun_id])
            assert abs(mass - 3.25904698235386) < 0.01, f"Gun mass is {mass} kg"

            gun_prim = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/link_gun")
            visuals_prim = gun_prim.GetChild("visuals")
            assert any(
                prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(visuals_prim.GetPrototype())
            ), "Gun visual mesh did not import"
            collisions_prim = gun_prim.GetChild("collisions")
            collider_shapes = [prim for prim in Usd.PrimRange(collisions_prim.GetPrototype()) if prim.HasAPI(UsdPhysics.CollisionAPI)]
            assert len(collider_shapes) == 1, f"Expected one gun collider, got {len(collider_shapes)}"
            # The Isaac URDF importer converts the URDF cylinder to a capsule.
            assert collider_shapes[0].IsA(UsdGeom.Capsule)
            capsule = UsdGeom.Capsule(collider_shapes[0])
            assert abs(capsule.GetHeightAttr().Get() - 1.1) < 1e-5
            assert abs(capsule.GetRadiusAttr().Get() - 0.028) < 1e-5
            assert UsdPhysics.CollisionAPI(collider_shapes[0]).GetCollisionEnabledAttr().Get()

            with torch.inference_mode():
                env._update_hose_wrench()
            force = env.hose_force_w[0].cpu()
            torque = env.hose_torque_w[0].cpu()
            expected_weight = cfg.hose_effective_mass_kg * abs(cfg.sim.gravity[2])
            assert abs(float(force[2]) + expected_weight) < 1e-3, (force, expected_weight)
            assert float(torch.linalg.vector_norm(torque)) > 0.01, f"Missing connector torque: {torque}"
            assert torch.isfinite(force).all() and torch.isfinite(torque).all()

            # A deliberately short cable must pull toward the truck, rather than push.
            original_slack_m = env.cfg.hose_slack_length_m
            env.cfg.hose_slack_length_m = 0.5
            with torch.inference_mode():
                env._update_hose_wrench()
            tension_n = float(env.hose_tension_n[0])
            assert tension_n > 0.0, f"Truck-roof tether did not become taut: {tension_n} N"
            env.cfg.hose_slack_length_m = original_slack_m
            with torch.inference_mode():
                env._update_hose_wrench()

            # Verify actual solver contact, not just a CollisionAPI flag.
            sensor = env.scene["contact_forces"]
            sensor_id = sensor.find_bodies("link_gun")[0][0]
            offset_b = torch.tensor([[0.003, -0.032, -0.05]], device=env.device)
            collider_center_w = robot.data.body_link_pos_w[:, gun_id] + quat_apply(
                robot.data.body_link_quat_w[:, gun_id], offset_b
            )
            probe_cfg = sim_utils.SphereCfg(
                radius=0.035,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            probe_cfg.func("/World/GunContactProbe", probe_cfg, translation=collider_center_w[0].cpu().tolist())
            action = torch.zeros(env.action_space.shape, device=env.device)
            peak_contact_n = 0.0
            with torch.inference_mode():
                for _ in range(10):
                    env.step(action)
                    peak_contact_n = max(
                        peak_contact_n,
                        float(torch.linalg.vector_norm(sensor.data.net_forces_w[0, sensor_id])),
                    )
            assert peak_contact_n > 0.01, f"Gun collider produced no solver contact: {peak_contact_n} N"
            print(
                f"PASS: gun visual mesh, mass={mass:.3f} kg, collider=capsule 1.10m x radius 0.028m, "
                f"hose mass={cfg.hose_effective_mass_kg:g} kg, "
                f"static hose force={force.tolist()} N, torque={torque.tolist()} Nm, "
                f"taut-cable tension={tension_n:.2f} N, probe contact={peak_contact_n:.2f} N",
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
