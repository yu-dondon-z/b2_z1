# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU geometry and command checks; run with a Python environment containing torch.

Isaac Lab's simulator imports are replaced with small fixtures. These tests execute
the production geometry and command methods but do not exercise PhysX or rendering.
"""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

COMMANDS = Path(__file__).resolve().parents[1] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/commands"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, COMMANDS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_module("curve_geometry", "curve_geometry.py")


def quat_apply(quat, vector):
    cross = 2.0 * torch.linalg.cross(quat[..., 1:], vector)
    return vector + quat[..., :1] * cross + torch.linalg.cross(quat[..., 1:], cross)


def quat_apply_inverse(quat, vector):
    return quat_apply(torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1), vector)


def quat_mul(first, second):
    a, b = first[..., 1:], second[..., 1:]
    return torch.cat(
        (
            first[..., :1] * second[..., :1] - (a * b).sum(dim=-1, keepdim=True),
            first[..., :1] * b + second[..., :1] * a + torch.linalg.cross(a, b),
        ),
        dim=-1,
    )


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, cp, cy = torch.cos(roll / 2), torch.cos(pitch / 2), torch.cos(yaw / 2)
    sr, sp, sy = torch.sin(roll / 2), torch.sin(pitch / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def yaw_quat(quat):
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def convert_camera_frame_orientation_convention(quat, origin, target):
    assert origin == "world" and target == "opengl"
    # OpenGL right/up/back axes expressed in the x-forward/y-left/z-up camera frame.
    conversion = torch.tensor([0.5, 0.5, -0.5, -0.5]).expand_as(quat)
    return quat_mul(quat, conversion)


class VelocityCommandFixture:
    """Provide the inherited buffers used by CurveVelocityCommand, without an app."""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.vel_command_b = torch.zeros(env.num_envs, 3)
        self.heading_target = torch.zeros(env.num_envs)
        self.time_left = torch.full((env.num_envs,), 10.0)
        self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool)
        self.is_heading_env = torch.zeros_like(self.is_standing_env)
        self.metrics = {}


package = types.ModuleType("curve_tracking_test")
package.__path__ = [str(COMMANDS)]
math_fixture = types.ModuleType("isaaclab.utils.math")
for function in (
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    quat_from_euler_xyz,
    yaw_quat,
    convert_camera_frame_orientation_convention,
):
    setattr(math_fixture, function.__name__, function)
with patch.dict(
    sys.modules,
    {
        "curve_tracking_test": package,
        "curve_tracking_test.curve_geometry": geometry,
        "isaaclab.envs.mdp.commands": types.SimpleNamespace(UniformVelocityCommand=VelocityCommandFixture),
        "isaaclab.markers": types.SimpleNamespace(VisualizationMarkers=object),
        "isaaclab.utils.math": math_fixture,
    },
):
    CurveVelocityCommand = load_module(
        "curve_tracking_test.curve_velocity_command", "curve_velocity_command.py"
    ).CurveVelocityCommand


class ConfigurationFixture:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class AssetCfgFixture(ConfigurationFixture):
    InitialStateCfg = ConfigurationFixture


class CameraCfgFixture(ConfigurationFixture):
    OffsetCfg = ConfigurationFixture


class PinholeCfgFixture:
    @classmethod
    def from_intrinsic_matrix(cls, intrinsic_matrix, **kwargs):
        return ConfigurationFixture(intrinsic_matrix=intrinsic_matrix, **kwargs)


with patch.dict(
    sys.modules,
    {
        "isaaclab.assets": types.SimpleNamespace(AssetBaseCfg=AssetCfgFixture),
        "isaaclab.sensors": types.SimpleNamespace(CameraCfg=CameraCfgFixture),
        "isaaclab.sim": types.SimpleNamespace(PinholeCameraCfg=PinholeCfgFixture),
        "isaaclab.utils.math": math_fixture,
    },
):
    make_curve_camera_cfg = load_module("curve_camera", "curve_camera.py").make_curve_camera_cfg


def make_command(count=3, **options):
    cfg = types.SimpleNamespace(
        forward_speed=0.4,
        lookahead_distance=0.6,
        heading_control_stiffness=1.5,
        trajectory_num_points=96,
        trajectory_curve_offset_scale=0.1,
        ground_curve_width=0.04,
        ground_curve_thickness=0.002,
        ground_curve_z_offset=0.002,
        camera_position_b=(0.41251, 0.024997, 0.04765),
        camera_rpy_b=(0.0, math.radians(45), 0.0),
        camera_intrinsics=(320.0, 320.0, 320.0, 240.0),
        camera_image_size=(640, 480),
        camera_depth_range=(0.05, 5.0),
        camera_update_period=0.1,
        ground_height=0.0,
        ranges=types.SimpleNamespace(ang_vel_z=(-0.8, 0.8)),
        base_name="base_link",
    )
    cfg.__dict__.update(options)
    positions = torch.zeros(count, 1, 3)
    positions[:, :, 2] = 0.5
    quaternions = torch.zeros(count, 1, 4)
    quaternions[:, :, 0] = 1.0
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(body_pos_w=positions, body_quat_w=quaternions),
        find_bodies=lambda name: ([0], [name]),
    )

    class Scene(dict):
        env_origins = torch.zeros(count, 3)

    env = types.SimpleNamespace(num_envs=count, device="cpu", scene=Scene(robot=robot))
    command = CurveVelocityCommand(cfg, env)
    command._resample_command(torch.arange(count))
    return command


class GeometryTests(unittest.TestCase):
    def test_bezier_endpoints_and_batch_broadcasting(self):
        points = torch.randn(4, 4, 3)
        progress = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1)
        sampled = geometry.cubic_bezier(*(points[:, i : i + 1] for i in range(4)), progress)
        torch.testing.assert_close(sampled[:, 0], points[:, 0])
        torch.testing.assert_close(sampled[:, -1], points[:, -1])
        self.assertEqual(sampled.shape, (4, 3, 3))

    def test_parallel_offset_turns_towards_curve(self):
        points = torch.zeros(3, 12, 3)
        points[:, :, 0] = torch.linspace(0.5, 3, 12)
        points[:, :, 1] = torch.tensor([-0.2, 0.0, 0.2]).unsqueeze(-1)
        lateral, heading, curvature, valid = geometry.curve_tracking_errors(
            points, torch.ones(3, 12, dtype=torch.bool), 0.6
        )
        torch.testing.assert_close(lateral, torch.tensor([0.2, 0.0, -0.2]))
        torch.testing.assert_close(heading, -torch.atan(lateral / 0.6))
        torch.testing.assert_close(curvature, torch.zeros(3))
        self.assertTrue(valid.all())

    def test_circle_curvature_and_turn_direction(self):
        angle = torch.linspace(0.0, 0.8, 40, dtype=torch.float64)
        left = torch.stack((2 * torch.sin(angle), 2 * (1 - torch.cos(angle)), torch.zeros_like(angle)), -1)
        right = left * torch.tensor([1.0, -1.0, 1.0])
        _, heading, curvature, valid = geometry.curve_tracking_errors(
            torch.stack((left, right)), torch.ones(2, 40, dtype=torch.bool), 0.6
        )
        torch.testing.assert_close(curvature, torch.tensor([0.5, -0.5], dtype=torch.float64))
        self.assertGreater(heading[0], 0)
        self.assertLess(heading[1], 0)
        self.assertTrue(valid.all())

    def test_hidden_and_degenerate_segments_have_no_target(self):
        points = torch.zeros(3, 6, 3)
        points[0, :, 0] = torch.arange(6)
        points[1, :, 0] = torch.arange(6)
        visible = torch.zeros(3, 6, dtype=torch.bool)
        visible[1, ::2] = True  # Isolated detections must not bridge invisible gaps.
        visible[2] = True  # Coincident points have no usable tangent.
        *errors, valid = geometry.curve_tracking_errors(points, visible, 0.6)
        self.assertFalse(valid.any())
        for error in errors:
            torch.testing.assert_close(error, torch.zeros(3))

    def test_curvature_at_visibility_boundary(self):
        angle = torch.linspace(0.0, 0.8, 40, dtype=torch.float64)
        points = torch.stack((2 * torch.sin(angle), 2 * (1 - torch.cos(angle)), torch.zeros_like(angle)), -1)
        visible = torch.ones(1, 40, dtype=torch.bool)
        visible[:, :10] = False
        _, _, curvature, valid = geometry.curve_tracking_errors(points.unsqueeze(0), visible, 0.6)
        self.assertTrue(valid.item())
        torch.testing.assert_close(curvature, torch.tensor([0.5], dtype=torch.float64))

    def test_ground_camera_round_trip_with_tilt_and_height(self):
        command = make_command(1)
        base_quat = quat_from_euler_xyz(torch.tensor([0.12]), torch.tensor([-0.08]), torch.tensor([0.7]))
        camera_quat = quat_mul(base_quat, command._camera_quat_b)
        camera_pos = torch.tensor([[0.5, -0.4, 0.8]])
        local = torch.zeros(1, 20, 3)
        local[:, :, 0] = torch.linspace(0.8, 3.0, 20)
        points_w = quat_apply(yaw_quat(base_quat).unsqueeze(1).expand(-1, 20, -1), local)
        points_w += torch.tensor([[[0.5, -0.4, 0.0]]])
        points_c = quat_apply_inverse(camera_quat.unsqueeze(1).expand(-1, 20, -1), points_w - camera_pos[:, None])
        recovered, _, visible = geometry.perceive_ground_curve(
            points_c,
            quat_apply_inverse(camera_quat, command._up),
            camera_pos[:, 2],
            command.cfg.camera_intrinsics,
            command.cfg.camera_image_size,
            command.cfg.camera_depth_range,
        )
        self.assertTrue(visible.any())
        torch.testing.assert_close(recovered[visible], points_c[visible], atol=1e-6, rtol=1e-5)

    def test_camera_rejects_behind_outside_and_above_horizon(self):
        points = torch.tensor([[[-1.0, 0.0, -1.0], [1.0, 10.0, -1.0], [1.0, 0.0, 1.0], [1.0, 0.0, -0.2]]])
        recovered, _, visible = geometry.perceive_ground_curve(
            points,
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([1.0]),
            (320.0, 320.0, 320.0, 240.0),
            (640, 480),
            (0.05, 5.0),
        )
        torch.testing.assert_close(visible, torch.tensor([[False, False, False, True]]))
        self.assertTrue(torch.isfinite(recovered).all())


class CameraConfigurationTests(unittest.TestCase):
    def test_training_spawns_only_usd_camera_and_play_enables_rgb(self):
        cfg = make_command(1).cfg
        training = make_curve_camera_cfg(cfg, render=False)
        play = make_curve_camera_cfg(cfg, render=True)
        self.assertIsInstance(training, AssetCfgFixture)
        self.assertIsInstance(play, CameraCfgFixture)
        self.assertEqual(training.prim_path, "{ENV_REGEX_NS}/Robot/base_link/BaseCamera")
        self.assertEqual(training.prim_path, play.prim_path)
        self.assertEqual(training.spawn.__dict__, play.spawn.__dict__)
        self.assertEqual(training.init_state.pos, play.offset.pos)
        self.assertEqual(play.data_types, ["rgb"])
        self.assertEqual(play.offset.convention, "world")
        self.assertEqual(play.update_period, 0.1)
        self.assertFalse(hasattr(training, "data_types"))

    def test_usd_camera_mount_matches_geometric_forward_and_up_axes(self):
        cfg = make_command(1).cfg
        training = make_curve_camera_cfg(cfg, render=False)
        play = make_curve_camera_cfg(cfg, render=True)
        usd_rotation = torch.tensor([training.init_state.rot])
        geometry_rotation = torch.tensor([play.offset.rot])
        torch.testing.assert_close(
            quat_apply(usd_rotation, torch.tensor([[0.0, 0.0, -1.0]])),
            quat_apply(geometry_rotation, torch.tensor([[1.0, 0.0, 0.0]])),
        )
        torch.testing.assert_close(
            quat_apply(usd_rotation, torch.tensor([[0.0, 1.0, 0.0]])),
            quat_apply(geometry_rotation, torch.tensor([[0.0, 0.0, 1.0]])),
        )

    def test_calibration_changes_reach_both_camera_modes(self):
        cfg = make_command(
            1,
            camera_image_size=(800, 600),
            camera_intrinsics=(400.0, 400.0, 400.0, 300.0),
            camera_position_b=(0.3, 0.1, 0.2),
        ).cfg
        for render in (False, True):
            camera = make_curve_camera_cfg(cfg, render=render)
            self.assertEqual(camera.spawn.intrinsic_matrix, [400.0, 0.0, 400.0, 0.0, 400.0, 300.0, 0.0, 0.0, 1.0])
            self.assertEqual(camera.spawn.clipping_range, cfg.camera_depth_range)
            self.assertEqual(camera.spawn.width, 800)
            self.assertEqual(camera.spawn.height, 600)
            offset = camera.offset if render else camera.init_state
            self.assertEqual(offset.pos, cfg.camera_position_b)

    def test_unsupported_render_calibration_is_rejected(self):
        for intrinsics in ((320.0, 300.0, 320.0, 240.0), (320.0, 320.0, 300.0, 240.0)):
            cfg = make_command(1, camera_intrinsics=intrinsics).cfg
            for render in (False, True):
                with self.subTest(intrinsics=intrinsics, render=render), self.assertRaises(ValueError):
                    make_curve_camera_cfg(cfg, render=render)


class CommandTests(unittest.TestCase):
    def test_ground_strip_stays_on_world_curve_and_points_along_segments(self):
        command = make_command(2)
        command.ground_curve_visualizer = types.SimpleNamespace()

        def capture(positions, orientations, scales):
            command.ground_curve_visualizer.data = (positions, orientations, scales)

        command.ground_curve_visualizer.visualize = capture
        original = command.curve_points_w.clone()
        command._visualize_ground_curve()
        positions, orientations, scales = command.ground_curve_visualizer.data
        self.assertEqual(positions.shape, (2 * 95, 3))
        torch.testing.assert_close(positions[:, 2], torch.full((190,), 0.002))
        torch.testing.assert_close(scales[:, 1], torch.full((190,), 0.04))
        torch.testing.assert_close(scales[:, 2], torch.full((190,), 0.002))
        directions = quat_apply(orientations, torch.tensor([[1.0, 0.0, 0.0]]).expand(190, -1))
        segments = (original[:, 1:] - original[:, :-1]).reshape(-1, 3)
        torch.testing.assert_close(directions, torch.nn.functional.normalize(segments, dim=-1))
        torch.testing.assert_close(command.curve_points_w, original)

    def test_constant_speed_and_yaw_limits(self):
        torch.manual_seed(7)
        command = make_command(64)
        self.assertTrue(command.path_visible.all())
        torch.testing.assert_close(command.vel_command_b[:, 0], torch.full((64,), 0.4))
        torch.testing.assert_close(command.vel_command_b[:, 1], torch.zeros(64))
        self.assertTrue((command.vel_command_b[:, 2].abs() <= 0.8).all())
        self.assertFalse(command.is_standing_env.any())

    def test_curve_stays_fixed_when_base_moves_and_turns(self):
        command = make_command()
        points = command.curve_points_w.clone()
        command.robot.data.body_pos_w[:, 0, 0] += 0.3
        zeros = torch.zeros(3)
        command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zeros, zeros, torch.full((3,), 0.15))
        command._update_command()
        torch.testing.assert_close(command.curve_points_w, points)
        self.assertTrue(command.path_visible.all())
        mask = command.camera_visible
        torch.testing.assert_close(command.perceived_curve_points_w[mask], points[mask], atol=1e-5, rtol=1e-5)

    def test_partial_resample_preserves_other_environments(self):
        command = make_command()
        points = command.curve_points_w.clone()
        velocity = command.vel_command_b.clone()
        command.robot.data.body_pos_w[1, 0, 0] += 1
        command._resample_command([1])
        torch.testing.assert_close(command.curve_points_w[[0, 2]], points[[0, 2]])
        torch.testing.assert_close(command.vel_command_b[[0, 2]], velocity[[0, 2]])
        self.assertFalse(torch.equal(command.curve_points_w[1], points[1]))

    def test_lost_curve_requests_straight_travel(self):
        command = make_command()
        command.robot.data.body_pos_w[:, 0, 0] += 20
        command._update_command()
        self.assertFalse(command.path_visible.any())
        torch.testing.assert_close(command.vel_command_b, torch.tensor([[0.4, 0.0, 0.0]]).expand(3, -1))
        self.assertTrue(torch.isfinite(command.vel_command_b).all())

    def test_global_rotation_and_translation_preserve_command(self):
        command = make_command(1)
        original = command.vel_command_b.clone()
        yaw = torch.tensor([2.9])
        zero = torch.zeros(1)
        rotation = quat_from_euler_xyz(zero, zero, yaw)
        command.curve_points_w[:] = quat_apply(rotation[:, None].expand(-1, 96, -1), command.curve_points_w)
        command.curve_points_w[:, :, :2] += torch.tensor([8.0, -3.0])
        command.robot.data.body_pos_w[:, 0, :2] = torch.tensor([8.0, -3.0])
        command.robot.data.body_quat_w[:, 0] = rotation
        command._update_command()
        torch.testing.assert_close(command.vel_command_b, original, atol=2e-4, rtol=1e-4)

    def test_closed_loop_reduces_lateral_error_using_projected_curve(self):
        command = make_command(1)
        command.curve_points_w.zero_()
        command.curve_points_w[0, :, 0] = torch.linspace(0, 7, 96)
        command.robot.data.body_pos_w[0, 0, 1] = 0.2
        heading = torch.zeros(1)
        zero = torch.zeros(1)
        for _ in range(300):
            command._update_command()
            self.assertTrue(command.path_visible.item())
            heading += 0.02 * command.vel_command_b[:, 2]
            command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zero, zero, heading)
            command.robot.data.body_pos_w[:, 0, 0] += 0.02 * 0.4 * torch.cos(heading)
            command.robot.data.body_pos_w[:, 0, 1] += 0.02 * 0.4 * torch.sin(heading)
        self.assertLess(command.robot.data.body_pos_w[0, 0, 1].abs(), 0.02)

    def test_invalid_configuration_is_rejected(self):
        for options in (
            {"forward_speed": 0},
            {"lookahead_distance": 0},
            {"trajectory_num_points": 2},
            {"camera_depth_range": (0.0, 1.0)},
            {"camera_intrinsics": (0.0, 320.0, 320.0, 240.0)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_command(**options)


if __name__ == "__main__":
    unittest.main()
