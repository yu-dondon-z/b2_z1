"""Rear Z1 mounting, base-frame EE curves, and position curriculum regressions."""

import ast
from copy import deepcopy
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

from test_curve_tracking import (
    COMMANDS,
    curriculums,
    geometry,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
)

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


def make_command(curriculum_enabled=True, **options):
    # Read the actual training ranges so coordinate/sign or workspace changes are caught.
    tree = ast.parse((COMMANDS.parents[1] / "b2_z1_env_cfg.py").read_text())
    commands = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CommandsCfg")
    ee = next(node.value for node in commands.body if isinstance(node, ast.Assign) and node.targets[0].id == "ee_target")
    range_configs = {}
    for name in ("ranges", "limit_ranges"):
        ranges = next(keyword.value for keyword in ee.keywords if keyword.arg == name)
        values = {
            keyword.arg: eval(
                compile(ast.Expression(keyword.value), "<ee ranges>", "eval"), {"math": math, "__builtins__": {}}
            )
            for keyword in ranges.keywords
        }
        range_configs[name] = types.SimpleNamespace(**values)
    cfg = types.SimpleNamespace(
        asset_name="robot", body_name="dummy_target", base_name="base_link",
        **range_configs, make_quat_unique=True, curriculum_enabled=curriculum_enabled,
        trajectory_num_points=20, trajectory_curve_offset_scale=0.0,
    )
    cfg.__dict__.update(options)
    base_pos = torch.tensor([[0.0, 0.0, 0.7], [10.0, 2.0, 0.55], [-4.0, 1.0, 0.9]])
    base_quat = quat_from_euler_xyz(
        torch.tensor([0.0, 0.1, -0.15]), torch.tensor([0.15, -0.2, 0.0]), torch.tensor([0.0, 0.7, -1.3])
    )
    mount_quat = quat_from_euler_xyz(*torch.tensor(MOUNT_RPY).expand(3, -1).unbind(-1))
    mount_offset = torch.tensor([float(value) for value in MOUNT.find("origin").get("xyz").split()]).expand(3, -1)
    body_state = torch.zeros(3, 3, 13)
    body_state[:, 0, :3], body_state[:, 0, 3:7] = base_pos, base_quat
    body_state[:, 1, :3] = base_pos + quat_apply(base_quat, mount_offset)
    body_state[:, 1, 3:7] = quat_mul(base_quat, mount_quat)
    body_state[:, 2, :3] = base_pos + quat_apply(base_quat, torch.tensor([[-1.02, 0.03, -0.4]]).expand(3, -1))
    target_rpy = torch.tensor([[0.0, math.pi / 2 - 0.45, math.pi]]).expand(3, -1)
    body_state[:, 2, 3:7] = quat_mul(base_quat, quat_from_euler_xyz(*target_rpy.unbind(-1)))
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

    def test_xyz_orientation_and_curve_are_sampled_in_full_base_frame(self):
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
        torch.testing.assert_close(command.command, command.pose_target, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(samples_b[:, -1], command.pose_target[:, :3], atol=1e-6, rtol=1e-6)
        for index, axis in enumerate(("pos_x", "pos_y", "pos_z")):
            low, high = getattr(command.cfg.ranges, axis)
            self.assertTrue((command.command[:, index] >= low - 1e-6).all())
            self.assertTrue((command.command[:, index] <= high + 1e-6).all())
        # Orientation is already in base coordinates; no rear-mount rotation is applied.
        tool_axis_b = quat_apply(command.command[:, 3:], torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1))
        expected_axis = torch.tensor([[-math.cos(0.45), 0.0, math.sin(0.45)]]).expand(3, -1)
        torch.testing.assert_close(tool_axis_b, expected_axis, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(command._get_tracked_trajectory_points_w()[:, 0], command.robot.data.body_pos_w[:, 2])
        command._update_trajectory_visualization()
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
        torch.testing.assert_close(command.command[1, :3], torch.tensor([-1.02, 0.03, -0.4]), atol=1e-6, rtol=1e-6)

    def test_base_translation_roll_pitch_and_height_move_the_entire_ee_curve(self):
        command = make_command()
        command._resample_command(torch.arange(3))
        command.time_left[:] = 4.0
        command._update_command()
        before = command.command.clone()
        tracked_base = command.tracked_trajectory_base_positions.clone()
        samples = torch.linspace(0, 1, 20).view(1, -1, 1)
        old_world = command._evaluate_curve(samples).clone()
        state = command.robot.data.body_state_w
        state[:, 0, :3] += torch.tensor([0.4, -0.3, 0.2])
        state[:, 0, 3:7] = quat_from_euler_xyz(torch.full((3,), 0.2), torch.full((3,), -0.25), torch.full((3,), 0.8))
        command._update_command()
        torch.testing.assert_close(command.command, before, atol=1e-6, rtol=1e-6)
        world = command._evaluate_curve(samples)
        self.assertFalse(torch.equal(world, old_world))
        base_quat = state[:, 0, 3:7]
        tracked_world = command._get_tracked_trajectory_points_w()
        recovered = quat_apply_inverse(base_quat[:, None].expand(-1, 20, -1), tracked_world - state[:, 0:1, :3])
        torch.testing.assert_close(recovered, tracked_base, atol=1e-6, rtol=1e-6)

    def test_control_points_respect_current_workspace_and_start_is_continuous(self):
        command = make_command(trajectory_curve_offset_scale=0.4)
        command._resample_command(torch.arange(3))
        for index, axis in enumerate(("pos_x", "pos_y", "pos_z")):
            low, high = getattr(command.cfg.ranges, axis)
            self.assertTrue((command._curve_control_points[..., index] >= low - 1e-6).all())
            self.assertTrue((command._curve_control_points[..., index] <= high + 1e-6).all())
        torch.testing.assert_close(
            command.command[:, :3], torch.tensor([[-1.02, 0.03, -0.4]]).expand(3, -1), atol=1e-6, rtol=1e-6
        )

    def test_disabled_curriculum_uses_full_limits_before_first_sample(self):
        command = make_command(curriculum_enabled=False)
        self.assertIsNot(command.cfg.ranges, command.cfg.limit_ranges)
        self.assertEqual(vars(command.cfg.ranges), vars(command.cfg.limit_ranges))
        command._resample_command(torch.arange(3))
        for index, axis in enumerate(("pos_x", "pos_y", "pos_z")):
            low, high = getattr(command.cfg.limit_ranges, axis)
            self.assertTrue((command.pose_target[:, index] >= low).all())
            self.assertTrue((command.pose_target[:, index] <= high).all())


class EEPositionCurriculumTests(unittest.TestCase):
    def make_env(self):
        command = make_command()
        env = command.env
        env.common_step_counter, env.max_episode_length, env.max_episode_length_s = 1507, 1500, 30.0
        env.command_manager = types.SimpleNamespace(get_term=lambda name: command)
        env.reward_manager = types.SimpleNamespace(
            _episode_sums={"end_effector_position_tracking_exp": torch.full((3,), 5.0 * 0.9 * 30)},
            get_term_cfg=lambda name: types.SimpleNamespace(weight=5.0),
        )
        return env, command

    def test_position_ranges_expand_and_stop_at_limits_without_rotating_targets(self):
        env, command = self.make_env()
        limits = deepcopy(command.cfg.limit_ranges)
        orientation = {axis: getattr(command.cfg.ranges, axis) for axis in ("roll", "pitch", "yaw")}
        for step in range(16):
            env.common_step_counter = 1507 + step * 1500
            curriculums.pos_cmd_levels(env, [0, 1, 2])
        for axis in ("pos_x", "pos_y", "pos_z"):
            torch.testing.assert_close(
                torch.tensor(getattr(command.cfg.ranges, axis)), torch.tensor(getattr(limits, axis))
            )
        self.assertEqual(vars(command.cfg.limit_ranges), vars(limits))
        self.assertEqual({axis: getattr(command.cfg.ranges, axis) for axis in orientation}, orientation)

    def test_reward_gate_async_cadence_and_partial_reset_selection(self):
        env, command = self.make_env()
        original = deepcopy(command.cfg.ranges)
        env.common_step_counter = 0
        curriculums.pos_cmd_levels(env, [1])
        self.assertEqual(vars(command.cfg.ranges), vars(original))
        env.common_step_counter = 1507
        curriculums.pos_cmd_levels(env, [])
        curriculums.pos_cmd_levels(env, slice(0, 0))
        self.assertEqual(command._last_pos_curriculum_step, 0)
        env.reward_manager._episode_sums["end_effector_position_tracking_exp"][[0, 2]] = 0.0
        curriculums.pos_cmd_levels(env, torch.tensor([1]))
        self.assertAlmostEqual(command.cfg.ranges.pos_y[1], 0.03)
        curriculums.pos_cmd_levels(env, [1])
        self.assertAlmostEqual(command.cfg.ranges.pos_y[1], 0.03)
        env.common_step_counter += 1500
        env.reward_manager._episode_sums["end_effector_position_tracking_exp"].fill_(5.0 * 0.8 * 30)
        curriculums.pos_cmd_levels(env, None)
        self.assertAlmostEqual(command.cfg.ranges.pos_y[1], 0.03)

    def test_promotion_does_not_change_existing_curve(self):
        env, command = self.make_env()
        command._resample_command(torch.arange(3))
        original = command._evaluate_curve(torch.linspace(0, 1, 20).view(1, -1, 1)).clone()
        curriculums.pos_cmd_levels(env, [0, 1, 2])
        torch.testing.assert_close(command._evaluate_curve(torch.linspace(0, 1, 20).view(1, -1, 1)), original)


if __name__ == "__main__":
    unittest.main()
