"""Rear Z1 mounting and EE frame-transform regressions, without Isaac Sim."""

import ast
import importlib.util
import math
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_curve_tracking import COMMANDS, geometry, quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

PACKAGE = COMMANDS.parents[4]
URDF = ET.parse(PACKAGE / "assets/b2_z1/b2_z1_gun.urdf").getroot()
MOUNT = URDF.find("joint[@name='joint0']")
MOUNT_RPY = tuple(float(value) for value in MOUNT.find("origin").get("rpy").split())


def rotation_rpy(rpy):
    quaternion = quat_from_euler_xyz(*torch.tensor(rpy, dtype=torch.float64).unbind())
    return quat_apply(quaternion.expand(3, -1), torch.eye(3, dtype=torch.float64)).numpy().T


class CommandFixture:
    def __init__(self, cfg, env):
        self.cfg, self.num_envs, self.device = cfg, env.num_envs, env.device
        self.metrics = {}
        self.time_left = torch.full((env.num_envs,), 8.0)


math_fixture = types.SimpleNamespace(
    quat_apply=quat_apply,
    quat_apply_inverse=quat_apply_inverse,
    quat_from_euler_xyz=quat_from_euler_xyz,
    quat_mul=quat_mul,
    quat_conjugate=lambda quat: quat * quat.new_tensor([1, -1, -1, -1]),
    quat_unique=lambda quat: torch.where(quat[:, :1] < 0, -quat, quat),
    axis_angle_from_quat=lambda quat: torch.nn.functional.normalize(quat[:, 1:], dim=-1)
    * (2 * torch.atan2(torch.linalg.vector_norm(quat[:, 1:], dim=-1, keepdim=True), quat[:, :1])),
)
package = types.ModuleType("rear_arm_test")
package.__path__ = [str(COMMANDS)]
with patch.dict(sys.modules, {
    "rear_arm_test": package,
    "rear_arm_test.curve_geometry": geometry,
    "isaaclab": types.ModuleType("isaaclab"),
    "isaaclab.sim": types.ModuleType("isaaclab.sim"),
    "isaaclab.assets": types.SimpleNamespace(Articulation=object),
    "isaaclab.managers": types.SimpleNamespace(CommandTerm=CommandFixture),
    "isaaclab.markers": types.SimpleNamespace(VisualizationMarkers=object),
    "isaaclab.markers.config": types.SimpleNamespace(FRAME_MARKER_CFG=None, POSITION_GOAL_MARKER_CFG=None),
    "isaaclab.utils.math": math_fixture,
}):
    spec = importlib.util.spec_from_file_location("rear_arm_test.pose_command_wbc", COMMANDS / "pose_command_wbc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    UniformPoseWBCCommand = module.UniformPoseWBCCommand


def make_command():
    # Read the actual training ranges so a double rotation/sign change is caught.
    tree = ast.parse((COMMANDS.parents[1] / "b2_z1_env_cfg.py").read_text())
    commands = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CommandsCfg")
    ee = next(node.value for node in commands.body if isinstance(node, ast.Assign) and node.targets[0].id == "ee_target")
    ranges = next(keyword.value for keyword in ee.keywords if keyword.arg == "ranges")
    values = {
        keyword.arg: eval(compile(ast.Expression(keyword.value), "<ee ranges>", "eval"), {"math": math, "__builtins__": {}})
        for keyword in ranges.keywords
    }
    cfg = types.SimpleNamespace(
        asset_name="robot", body_name="dummy_target", link_name="link00", base_name="base_link",
        ranges=types.SimpleNamespace(**values), make_quat_unique=True,
        trajectory_num_points=20, trajectory_curve_offset_scale=0.0,
    )
    base_pos = torch.tensor([[0.0, 0.0, 0.7], [10.0, 2.0, 0.7], [-4.0, 1.0, 0.7]])
    zeros = torch.zeros(3)
    base_quat = quat_from_euler_xyz(zeros, zeros, torch.tensor([0.0, 0.7, -1.3]))
    mount_quat = quat_from_euler_xyz(*torch.tensor(MOUNT_RPY).expand(3, -1).unbind(-1))
    mount_offset = torch.tensor([float(value) for value in MOUNT.find("origin").get("xyz").split()]).expand(3, -1)
    body_state = torch.zeros(3, 3, 13)
    body_state[:, 0, :3], body_state[:, 0, 3:7] = base_pos, base_quat
    body_state[:, 1, :3] = base_pos + quat_apply(base_quat, mount_offset)
    body_state[:, 1, 3:7] = quat_mul(base_quat, mount_quat)
    body_state[:, 2, :3] = base_pos + quat_apply(base_quat, torch.tensor([[-0.8, 0.03, -0.4]]).expand(3, -1))
    body_state[:, 2, 3:7] = body_state[:, 1, 3:7]
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(body_pos_w=body_state[..., :3], body_state_w=body_state),
        find_bodies=lambda name: ([{"base_link": 0, "link00": 1, "dummy_target": 2}[name]], [name]),
    )
    env = types.SimpleNamespace(num_envs=3, device="cpu", step_dt=0.02, scene={"robot": robot})
    return UniformPoseWBCCommand(cfg, env)


class RearArmTests(unittest.TestCase):
    def test_default_posture_points_backwards_within_joint_limits(self):
        tree = ast.parse((PACKAGE / "assets/unitree.py").read_text())
        config = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "B2_Z1_GUN_CFG" for target in node.targets))
        initial = next(keyword.value for keyword in config.keywords if keyword.arg == "init_state")
        positions = ast.literal_eval(next(keyword.value for keyword in initial.keywords if keyword.arg == "joint_pos"))
        transforms = {"base_link": np.eye(4)}
        for joint in URDF.findall("joint"):
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = rotation_rpy([float(value) for value in origin.get("rpy", "0 0 0").split()])
            transform[:3, 3] = [float(value) for value in origin.get("xyz", "0 0 0").split()]
            if joint.get("type") == "revolute" and joint.get("name") in positions:
                angle = positions[joint.get("name")]
                limit = joint.find("limit")
                self.assertLessEqual(float(limit.get("lower")), angle)
                self.assertLessEqual(angle, float(limit.get("upper")))
                x, y, z = [float(value) for value in joint.find("axis").get("xyz").split()]
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                motion = np.eye(4)
                motion[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
                transform = transform @ motion
            transforms[joint.find("child").get("link")] = transforms[parent] @ transform
        for name in ("link06", "link_gun", "dummy_target"):
            self.assertLess(transforms[name][0, 3], -0.5, name)

    def test_target_orientation_and_curve_follow_rear_mount_at_different_headings(self):
        command = make_command()
        command._resample_command(torch.arange(3))
        samples = command._evaluate_curve(torch.linspace(0, 1, 20).view(1, -1, 1))
        torch.testing.assert_close(samples[:, 0], command.robot.data.body_pos_w[:, 2])
        torch.testing.assert_close(samples[:, -1], command._get_target_pos_w())
        base_pos = command.robot.data.body_pos_w[:, 0]
        base_quat = command.robot.data.body_state_w[:, 0, 3:7]
        samples_b = quat_apply_inverse(base_quat[:, None].expand(-1, 20, -1), samples - base_pos[:, None])
        self.assertTrue((samples_b[..., 0] < 0.0).all())
        command.time_left[:] = 0.0
        command._update_command()
        torch.testing.assert_close(command.command[:, 0], -torch.ones(3))
        torch.testing.assert_close(command._get_target_pos_w()[:, 2], torch.full((3,), 0.2))
        # Target local +Z (tool direction) points toward the rear in base coordinates.
        tool_axis_b = quat_apply(command.command[:, 3:], torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1))
        torch.testing.assert_close(tool_axis_b, torch.tensor([[-1.0, 0.0, 0.0]]).expand(3, -1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(command._get_tracked_trajectory_points_w()[:, 0], command.robot.data.body_pos_w[:, 2])
        command._update_trajectory_visualization(
            command.robot.data.body_pos_w[:, 1], command.robot.data.body_state_w[:, 1, 3:7]
        )
        torch.testing.assert_close(command.interpolation_curve_positions_w, samples)
        torch.testing.assert_close(
            command.interpolation_curve_orientations_w,
            command._get_target_quat_w()[:, None].expand(-1, 20, -1),
        )

    def test_partial_resampling_writes_rear_command_immediately(self):
        command = make_command()
        command.pose_command[0, :3] = torch.tensor([4.0, 5.0, 6.0])
        untouched = command.command[0].clone()
        command._resample_command(torch.tensor([1]))
        torch.testing.assert_close(command.command[0], untouched)
        torch.testing.assert_close(command.command[1, :3], torch.tensor([-0.8, 0.03, -0.4]), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
