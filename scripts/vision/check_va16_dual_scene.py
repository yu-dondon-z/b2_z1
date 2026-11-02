#!/usr/bin/env python3
"""Render VA16's two cameras and independently check tool measurement.

No policy or hardware is started.  The simulated pose is held without
gravity while optical calibration, visible markers and collision metadata
are checked.  Robot TCP truth is read only after image processing, for an
independent measurement-error report.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, help="New directory for images and report")
parser.add_argument("--task", default="Unitree-b2z1-AMP-VA16-DualVision-Play")
parser.add_argument("--usd-cache-dir", help="Override Isaac's hard-coded /tmp USD conversion cache directory")
parser.add_argument("--hold", action="store_true", help="Keep the GUI open at the checked default pose")
parser.add_argument("--mount-sweep", action="store_true", help="Render alternative rear mounts for occlusion diagnosis")
parser.add_argument("--mount-only", action="store_true",
                    help="Validate the requested centre/45-degree mount and current visible TCP; report full workspace separately")
parser.add_argument("--viewport-screenshot", action="store_true",
                    help="Save a GUI screenshot; respects the configured frustum visibility")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# Use the same standard Kit shutdown mode as playback. Full plugin teardown
# (fast_shutdown=False) crashes this installation after saving valid reports.
# Validation still raises on failure; no exit status is overridden here.
launcher = AppLauncher(args_cli)

import cv2
import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
import b2_z1.tasks  # noqa: F401

from dual_camera_perception import DualCameraPerception, PerceptionConfig, camera_geometry


def _camera_data(camera, robot, base_id: int, name: str):
    camera.update(0.0, force_recompute=True)
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
    camera_position_b = quat_apply_inverse(
        robot.data.body_quat_w[:, base_id],
        camera.data.pos_w - robot.data.body_pos_w[:, base_id],
    )
    quaternion_b = quat_mul(
        quat_conjugate(robot.data.body_quat_w[:, base_id]), camera.data.quat_w_ros
    )
    metadata = {
        "camera_name": name,
        "intrinsic_matrix": camera.data.intrinsic_matrices[0].cpu().tolist(),
        "camera_position_b": camera_position_b[0].cpu().tolist(),
        "camera_quaternion_b_ros_wxyz": quaternion_b[0].cpu().tolist(),
        "depth_unit_m": 1.0,
        "depth_aligned_to_rgb": True,
        "marker_radius_m": 0.035,
        "marker_axial_spacing_m": 0.10,
        "marker_tip_extension_m": 0.10,
    }
    return metadata, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    output = Path(args_cli.output).expanduser().resolve()
    if args_cli.viewport_screenshot and args_cli.headless:
        raise ValueError("--viewport-screenshot requires a GUI (omit --headless)")
    if args_cli.hold and args_cli.headless:
        raise ValueError("--hold requires a GUI")
    output.mkdir(parents=True, exist_ok=False)
    task = args_cli.task
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    if args_cli.usd_cache_dir:
        cfg.scene.robot.spawn.usd_dir = str(Path(args_cli.usd_cache_dir).expanduser().resolve())
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.reference_state_initialization = False
    cfg.commands.ee_target.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    with tempfile.TemporaryDirectory(prefix="va16_scene_check_") as exchange:
        cfg.commands.ee_target.vision_exchange_directory = exchange
        cfg.commands.ee_target.trajectory_file = str(Path(exchange) / "latest_dual_trajectory.json")
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            # Propagate initialized articulation poses through PhysX/Fabric
            # before reading RTX buffers. Rendering alone can still display
            # the imported zero-joint mesh while body tensors have reset.
            with torch.inference_mode():
                for _ in range(20):
                    env.step(torch.zeros(env.action_space.shape, device=env.device))
            for _ in range(12):
                env.sim.render()
            robot = env.scene["robot"]
            base_id = robot.find_bodies("base_link")[0][0]
            camera_results = {}
            data = {}
            for name in ("front_rgb", "rear_d435i"):
                camera = env.scene[name]
                metadata, bgr = _camera_data(camera, robot, base_id, name)
                cv2.imwrite(str(output / f"{name}_rgb.png"), bgr)
                (output / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                camera_results[name] = {
                    "rgb_shape": list(bgr.shape),
                    "period_s": camera.cfg.update_period,
                    "data_types": camera.cfg.data_types,
                }
                data[name] = (metadata, bgr)
            rear_camera = env.scene["rear_d435i"]
            depth = rear_camera.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
            np.save(output / "rear_d435i_depth_m.npy", depth)
            valid_depth = np.isfinite(depth) & (depth > 0)
            camera_results["rear_d435i"].update(
                depth_shape=list(depth.shape), depth_valid_ratio=float(np.mean(valid_depth)),
            )
            processor = DualCameraPerception(PerceptionConfig(mode="classical"))
            rear_metadata, rear_bgr = data["rear_d435i"]
            rear_result = processor.process_rear(rear_metadata, rear_bgr, depth)
            # Evaluate against the same capture pose before any optional
            # mount sweep advances physics or changes the rendering camera.
            tcp_id = robot.find_bodies("dummy_target")[0][0]
            gun_id = robot.find_bodies("link_gun")[0][0]
            tcp_truth_b = quat_apply_inverse(
                robot.data.body_quat_w[:, base_id],
                robot.data.body_pos_w[:, tcp_id] - robot.data.body_pos_w[:, base_id],
            )[0].cpu().numpy().copy()
            capture_base_z = float(robot.data.body_pos_w[0, base_id, 2])
            if args_cli.mount_sweep:
                from scipy.spatial.transform import Rotation
                sweep_results = {}
                for name, position_b, yaw_deg, pitch_deg in (
                    ("rear_left", (-0.35, 0.28, 0.34), 225.0, 55.0),
                    ("rear_left_far", (-0.48, 0.30, 0.34), 225.0, 55.0),
                    ("rear_left_high", (-0.45, 0.25, 0.50), 225.0, 65.0),
                    ("rear_center_far", (-0.68, 0.0, 0.34), 180.0, 75.0),
                ):
                    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
                    forward = np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), -np.sin(pitch)])
                    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
                    down = np.cross(forward, right)
                    xyzw = Rotation.from_matrix(np.column_stack((right, down, forward))).as_quat()
                    quaternion_b = torch.tensor([[xyzw[3], *xyzw[:3]]], device=env.device, dtype=torch.float32)
                    position = robot.data.body_pos_w[:, base_id] + quat_apply(
                        robot.data.body_quat_w[:, base_id], torch.tensor([position_b], device=env.device)
                    )
                    orientation = quat_mul(robot.data.body_quat_w[:, base_id], quaternion_b)
                    rear_camera.set_world_poses(position, orientation, convention="ros")
                    for _ in range(12):
                        env.sim.step(render=True)
                        env.scene.update(cfg.sim.dt)
                    meta, image = _camera_data(rear_camera, robot, base_id, name)
                    candidate_depth = rear_camera.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
                    cv2.imwrite(str(output / f"{name}_rgb.png"), image)
                    np.save(output / f"{name}_depth_m.npy", candidate_depth)
                    (output / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
                    sweep_results[name] = processor.process_rear(meta, image, candidate_depth)
                (output / "mount_sweep.json").write_text(json.dumps(sweep_results, indent=2) + "\n")
            front_metadata, front_bgr = data["front_rgb"]
            if "ground_plane_b" in rear_result:
                front_metadata["ground_plane_b"] = rear_result["ground_plane_b"]
            front_result = processor.process_front(front_metadata, front_bgr)

            # Truth is deliberately outside the processor API, and is used
            # only for this independent scene/sensor validation.
            measured = rear_result.get("tcp_measured_b")
            measurement_error_m = (
                None if measured is None else float(np.linalg.norm(np.asarray(measured) - tcp_truth_b))
            )
            gun = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/link_gun")
            gun_shape = gun.GetChild("welding_gun")
            cylinder = [prim for prim in Usd.PrimRange(gun_shape) if prim.IsA(UsdGeom.Cylinder)]
            assert len(cylinder) == 1
            collision_enabled = bool(UsdPhysics.CollisionAPI(cylinder[0]).GetCollisionEnabledAttr().Get())
            gun_mass = float(robot.root_physx_view.get_masses()[0, gun_id])
            axial_grip = None
            if "right_joint7" in robot.joint_names:
                wrist_id = robot.find_bodies("right_link7")[0][0]
                base_quat = robot.data.body_quat_w[:, base_id]
                x = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
                minus_z = torch.tensor([[0.0, 0.0, -1.0]], device=env.device)
                flange_axis_b = quat_apply_inverse(base_quat, quat_apply(
                    robot.data.body_quat_w[:, wrist_id], x))[0]
                gun_axis_b = quat_apply_inverse(base_quat, quat_apply(
                    robot.data.body_quat_w[:, gun_id], minus_z))[0]
                dot = float(torch.dot(flange_axis_b, gun_axis_b).clamp(-1.0, 1.0))
                axial_grip = {
                    "flange_forward_axis_b": flange_axis_b.cpu().tolist(),
                    "gun_tip_axis_b": gun_axis_b.cpu().tolist(),
                    "flange_gun_axis_difference_deg": math.degrees(math.acos(dot)),
                    "gun_tilt_from_vertical_deg": math.degrees(math.acos(
                        float((-gun_axis_b[2]).clamp(-1.0, 1.0)))),
                    "cylinder_length_m": cfg.scene.welding_gun.spawn.height,
                    "coaxial_top_grip": dot > 0.99999 and float(gun_axis_b[2]) < -0.95,
                }
            marker_results = {}
            for name in ("tool_marker_cyan", "tool_marker_magenta"):
                marker = gun.GetChild(name)
                prims = list(Usd.PrimRange(marker))
                marker_results[name] = {
                    "has_extra_rigid_body": any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
                    "has_extra_mass": any(p.HasAPI(UsdPhysics.MassAPI) for p in prims),
                    "has_enabled_collision": any(
                        p.HasAPI(UsdPhysics.CollisionAPI)
                        and bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
                        for p in prims
                    ),
                }
            intrinsic, transform = camera_geometry(rear_metadata)
            axis = transform[:3, 2]
            pitch_down_deg = math.degrees(math.atan2(-axis[2], np.linalg.norm(axis[:2])))
            housing = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link/rear_d435i_model")
            housing_physics = {
                "has_rigid_body": False, "has_mass": False, "has_enabled_collision": False,
            }
            if housing:
                for prim in Usd.PrimRange(housing):
                    housing_physics["has_rigid_body"] |= prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    housing_physics["has_mass"] |= prim.HasAPI(UsdPhysics.MassAPI)
                    housing_physics["has_enabled_collision"] |= (
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        and bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()))
            mount = {
                "position_b": transform[:3, 3].tolist(), "optical_axis_b": axis.tolist(),
                "pitch_down_deg": pitch_down_deg,
                "rear_centre_45_deg": bool(abs(transform[1, 3]) < 1e-5 and axis[0] < 0
                                           and abs(pitch_down_deg - 45.0) < 1e-3),
                "housing_exists": bool(housing), "housing_physics": housing_physics,
            }
            coverage = []
            base_z = capture_base_z
            for x in cfg.commands.ee_target.rear_work_x_range_m:
                for y in (-0.15, 0.0, 0.15):
                    point_b = np.array([x, y, cfg.commands.ee_target.tool_height_m - base_z])
                    point_c = transform[:3, :3].T @ (point_b - transform[:3, 3])
                    pixel = intrinsic @ point_c
                    uv = pixel[:2] / pixel[2]
                    coverage.append({"point_b": point_b.tolist(), "pixel_uv": uv.tolist(),
                                     "inside_image": bool(point_c[2] > 0 and 0 <= uv[0] < 640 and 0 <= uv[1] < 480)})
            common_passed = (
                collision_enabled and abs(gun_mass - 0.001) < 1e-6
                and measurement_error_m is not None and measurement_error_m < 0.01
                and all(not any(item.values()) for item in marker_results.values())
                and all(abs(item["period_s"] - 0.04) < 1e-8 for item in camera_results.values())
                and front_result["valid"] and rear_result["valid"]
                and (axial_grip is None or axial_grip["coaxial_top_grip"])
            )
            workspace_visible = all(point["inside_image"] for point in coverage)
            mount_passed = (common_passed and mount["rear_centre_45_deg"] and bool(housing)
                            and not any(housing_physics.values()))
            passed = mount_passed if args_cli.mount_only else common_passed and workspace_visible
            visual = getattr(env.command_manager.get_term("ee_target"), "_rear_camera_visual", None)
            if args_cli.viewport_screenshot:
                if cfg.commands.ee_target.show_rear_camera_frustum and (visual is None or not visual.update()):
                    raise RuntimeError("rear camera field-of-view overlay did not attach to the GUI viewport")
                env.viewport_camera_controller.update_view_to_asset_root("robot")
                import omni.kit.app
                omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                    "omni.kit.renderer.capture", True)
                import omni.renderer_capture
                for _ in range(10):
                    env.sim.render()
                screenshot = output / "viewport_robot.png"
                omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(screenshot))
                for _ in range(40):
                    launcher.app.update()
                    if screenshot.is_file():
                        break
                if not screenshot.is_file():
                    raise RuntimeError("GUI camera/frustum screenshot was not saved")
            report = {
                "pass": bool(passed),
                "task": task,
                "robot_asset": robot.cfg.spawn.asset_path,
                "joint_names": list(robot.joint_names),
                "scope": ("rear-centre 45-degree mount and current visible TCP; classical seam diagnostic"
                          if args_cli.mount_only else "rendered stationary full-workspace scene diagnostic; classical seam detector"),
                "rear_camera_mount": mount, "rear_camera_mount_pass": bool(mount_passed),
                "full_workspace_visible": workspace_visible,
                "viewport_frustum_available": bool(visual is not None and visual.available),
                "policy_observation_dimensions": env.observation_manager.group_obs_dim["policy"],
                "action_dimensions": env.action_manager.total_action_dim,
                "cameras": camera_results, "gun_mass_kg": gun_mass,
                "gun_collision_enabled": collision_enabled, "markers": marker_results,
                "axial_grip": axial_grip,
                "tcp_truth_b_for_evaluation_only": tcp_truth_b.tolist(),
                "image_tcp_measurement_error_m": measurement_error_m,
                "rear_workspace_projection": coverage,
                "rear_perception": rear_result, "front_perception": front_result,
            }
            (output / "scene_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            if not passed:
                raise RuntimeError(f"VA16 scene check did not pass; inspect {output}/scene_report.json")
            if args_cli.hold:
                env.viewport_camera_controller.update_view_to_asset_root("robot")
                print("Default-pose model inspection: no policy loaded; gravity disabled. Close the window or Ctrl+C to exit.", flush=True)
                while launcher.app.is_running():
                    with torch.inference_mode():
                        env.step(torch.zeros(env.action_space.shape, device=env.device))
        finally:
            # Detach UI overlays while their viewport and sensor are alive,
            # rather than retaining a render-product camera until app exit.
            command_manager = getattr(env, "command_manager", None)
            if command_manager is not None:
                visual = getattr(command_manager.get_term("ee_target"), "_rear_camera_visual", None)
                if visual is not None:
                    visual.close()
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
