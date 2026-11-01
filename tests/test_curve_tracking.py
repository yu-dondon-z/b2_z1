# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU geometry and command checks; run with a Python environment containing torch.

Isaac Lab's simulator imports are replaced with small fixtures. These tests execute
the production geometry and command methods but do not exercise PhysX or rendering.
"""

import ast
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
curriculums = load_module("curve_curriculums", "../curriculums.py")


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
        self.time_left = torch.full((env.num_envs,), float(cfg.resampling_time_range[0]))
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


class TiledCameraCfgFixture(CameraCfgFixture):
    pass


class PinholeCfgFixture:
    @classmethod
    def from_intrinsic_matrix(cls, intrinsic_matrix, **kwargs):
        return ConfigurationFixture(intrinsic_matrix=intrinsic_matrix, **kwargs)


with patch.dict(
    sys.modules,
    {
        "isaaclab.assets": types.SimpleNamespace(AssetBaseCfg=AssetCfgFixture),
        "isaaclab.sensors": types.SimpleNamespace(TiledCameraCfg=TiledCameraCfgFixture),
        "isaaclab.sim": types.SimpleNamespace(PinholeCameraCfg=PinholeCfgFixture),
        "isaaclab.utils.math": math_fixture,
    },
):
    camera_builders = load_module("curve_camera", "curve_camera.py")
    make_curve_camera_cfg = camera_builders.make_curve_camera_cfg
    make_rear_camera_cfg = camera_builders.make_rear_camera_cfg


def rear_camera_command_cfg():
    """Read the production EE camera defaults without importing the simulator."""
    tree = ast.parse((COMMANDS / "command_cfg.py").read_text())
    config = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UniformPoseWBCCommandCfg"
    )
    values = {
        node.target.id: eval(
            compile(ast.Expression(node.value), "<rear camera>", "eval"), {"math": math, "__builtins__": {}}
        )
        for node in config.body
        if isinstance(node, ast.AnnAssign)
        and (node.target.id == "base_name" or node.target.id.startswith("camera_"))
    }
    return types.SimpleNamespace(**values)


def make_command(count=3, **options):
    cfg = types.SimpleNamespace(
        forward_speed=0.4,
        resampling_time_range=(10.0, 10.0),
        lookahead_distance=0.6,
        heading_control_stiffness=1.5,
        trajectory_num_points=96,
        path_type="bezier",
        turn_count_range=(1, 1),
        turn_angle_range=(math.pi / 4, math.pi / 2),
        turn_curvature=1.0,
        turn_start_distance=0.8,
        turn_gap_distance=1.0,
        trajectory_curve_offset_scale=0.1,
        limit_trajectory_curve_offset_scale=0.1,
        max_curvature=None,
        curriculum_enabled=False,
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


def analytic_bezier_curvatures(control_points, num_samples=1025):
    """Evaluate geometric curvature independently from the conservative limiter."""
    points = control_points.to(torch.float64)
    t = torch.linspace(0.0, 1.0, num_samples, dtype=points.dtype).view(1, -1, 1)
    p0, p1, p2, p3 = (points[:, i : i + 1, :2] for i in range(4))
    velocity = 3.0 * ((1 - t) ** 2 * (p1 - p0) + 2 * (1 - t) * t * (p2 - p1) + t**2 * (p3 - p2))
    acceleration = 6.0 * ((1 - t) * (p2 - 2 * p1 + p0) + t * (p3 - 2 * p2 + p1))
    cross = velocity[..., 0] * acceleration[..., 1] - velocity[..., 1] * acceleration[..., 0]
    return cross / torch.linalg.vector_norm(velocity, dim=-1).pow(3)


def polyline_curvatures(points):
    points = points.to(torch.float64)[..., :2]
    before, after = points[:, 1:-1] - points[:, :-2], points[:, 2:] - points[:, 1:-1]
    cross = before[..., 0] * after[..., 1] - before[..., 1] * after[..., 0]
    denominator = before.norm(dim=-1) * after.norm(dim=-1) * (before + after).norm(dim=-1)
    return 2.0 * cross / denominator


class GeometryTests(unittest.TestCase):
    def test_three_turn_path_preserves_angles_length_and_curvature(self):
        lengths = torch.full((2,), 11.2, dtype=torch.float64)
        angles = torch.tensor([[math.pi / 2, -math.pi / 2, math.pi / 2]], dtype=torch.float64).repeat(2, 1)
        angles[1] *= -1
        points, actual = geometry.multi_turn_path(
            lengths, angles, 1.0, 0.8, 1.0, 0.6, torch.linspace(0, 1, 8193, dtype=torch.float64)
        )
        torch.testing.assert_close(actual, angles)
        torch.testing.assert_close(points[:, -1, 0], torch.full((2,), 4.8, dtype=torch.float64))
        torch.testing.assert_close(points[:, -1, 1], angles[:, 0].sign() * (4 + 11.2 - 0.8 - 2 - 3 * math.pi / 2))
        segments = points[:, 1:] - points[:, :-1]
        torch.testing.assert_close(segments.norm(dim=-1).sum(dim=-1), lengths, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.atan2(segments[:, -1, 1], segments[:, -1, 0]), actual.sum(dim=-1))
        self.assertLessEqual(polyline_curvatures(points).abs().max().item(), 1.0 + 1e-7)

    def test_multiple_turns_share_length_budget_without_exceeding_curvature_limit(self):
        lengths = torch.tensor([1.2, 5.2, 11.2], dtype=torch.float64)
        angles = torch.tensor([[math.pi / 2, -math.pi / 2, math.pi / 2]], dtype=torch.float64).repeat(3, 1)
        samples = torch.linspace(0, 1, 4097, dtype=torch.float64)
        points, actual = geometry.multi_turn_path(lengths, angles, 0.2, 0.8, 1.0, 0.6, samples)
        torch.testing.assert_close(actual.abs().sum(dim=-1), torch.tensor([0.03, 0.38, 1.56], dtype=torch.float64))
        torch.testing.assert_close(actual[:, 0], -actual[:, 1])
        self.assertLessEqual(polyline_curvatures(points).abs().max().item(), 0.2 + 1e-7)
        for curvature, requested in ((0.0, angles), (1.0, torch.zeros_like(angles))):
            straight, actual = geometry.multi_turn_path(lengths, requested, curvature, 0.8, 1.0, 0.6, samples)
            torch.testing.assert_close(actual, torch.zeros_like(angles))
            torch.testing.assert_close(straight[..., 0], lengths[:, None] * samples)
            torch.testing.assert_close(straight[..., 1:], torch.zeros(3, 4097, 2, dtype=torch.float64))

    def test_straight_arc_straight_has_large_turn_and_tangent_exit(self):
        angles = torch.tensor([math.pi / 2, -math.pi / 2], dtype=torch.float64)
        lengths = torch.full((2,), 5.2, dtype=torch.float64)
        points, actual = geometry.straight_turn_straight(
            lengths, angles, 1.0, 0.8, 0.6, torch.linspace(0, 1, 4097, dtype=torch.float64)
        )
        torch.testing.assert_close(actual, angles)
        torch.testing.assert_close(points[:, 0], torch.zeros(2, 3, dtype=torch.float64))
        torch.testing.assert_close(points[:, -1, 0], torch.full((2,), 1.8, dtype=torch.float64))
        torch.testing.assert_close(points[:, -1, 1], angles.sign() * (1.0 + 5.2 - 0.8 - math.pi / 2))
        segments = points[:, 1:] - points[:, :-1]
        heading = torch.atan2(segments[:, -1, 1], segments[:, -1, 0])
        torch.testing.assert_close(heading, actual)
        torch.testing.assert_close(segments.norm(dim=-1).sum(dim=-1), lengths, atol=1e-6, rtol=1e-6)
        self.assertLessEqual(polyline_curvatures(points).abs().max().item(), 1.0 + 1e-7)

    def test_turn_path_shortens_angle_to_honor_curvature_and_length(self):
        lengths = torch.tensor([1.2, 5.2], dtype=torch.float64)
        angles = torch.full((2,), math.pi / 2, dtype=torch.float64)
        samples = torch.linspace(0, 1, 2049, dtype=torch.float64)
        points, actual = geometry.straight_turn_straight(lengths, angles, 0.2, 0.8, 0.6, samples)
        torch.testing.assert_close(actual, torch.tensor([0.06, 0.76], dtype=torch.float64))
        self.assertLessEqual(polyline_curvatures(points).abs().max().item(), 0.2 + 1e-7)
        straight, actual = geometry.straight_turn_straight(lengths, angles, 0.0, 0.8, 0.6, samples)
        torch.testing.assert_close(actual, torch.zeros_like(angles))
        torch.testing.assert_close(straight[:, :, 0], lengths[:, None] * samples)
        torch.testing.assert_close(straight[:, :, 1:], torch.zeros(2, 2049, 2, dtype=torch.float64))

    def test_turn_path_tangent_is_continuous_at_both_joins(self):
        length = torch.tensor([5.2], dtype=torch.float64)
        epsilon = 1e-5
        for join in (0.8, 0.8 + math.pi / 2):
            samples = torch.tensor([join - epsilon, join, join + epsilon], dtype=torch.float64) / length
            points, _ = geometry.straight_turn_straight(
                length, torch.tensor([math.pi / 2]), 1.0, 0.8, 0.6, samples
            )
            tangent = torch.nn.functional.normalize(points[:, 1:] - points[:, :-1], dim=-1)
            torch.testing.assert_close(tangent[:, 0], tangent[:, 1], atol=1e-5, rtol=0.0)

    def test_bezier_endpoints_and_batch_broadcasting(self):
        points = torch.randn(4, 4, 3)
        progress = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1)
        sampled = geometry.cubic_bezier(*(points[:, i : i + 1] for i in range(4)), progress)
        torch.testing.assert_close(sampled[:, 0], points[:, 0])
        torch.testing.assert_close(sampled[:, -1], points[:, -1])
        self.assertEqual(sampled.shape, (4, 3, 3))

    def test_curvature_limit_covers_entire_curve_and_both_turn_directions(self):
        points = torch.zeros(4, 4, 3, dtype=torch.float64)
        lengths = torch.tensor([0.6, 2.0, 5.2, 9.0], dtype=torch.float64)
        points[:, :, 0] = lengths[:, None] * torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        points[:, :, 1] = lengths[:, None] * torch.tensor([0.0, 0.8, -0.8, 0.8])
        points[1::2, :, 1] *= -1
        original = points.clone()
        self.assertTrue((analytic_bezier_curvatures(points).abs().amax(dim=-1) > 0.2).all())
        limited = geometry.limit_ground_curve_curvature(points, 0.2)
        self.assertLessEqual(analytic_bezier_curvatures(limited, 4097).abs().max().item(), 0.2 + 1e-7)
        torch.testing.assert_close(limited[:, :, (0, 2)], points[:, :, (0, 2)])
        torch.testing.assert_close(limited[:, 0], points[:, 0])
        torch.testing.assert_close(points, original)

    def test_curvature_limiter_preserves_gentle_curves_and_accepts_zero(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.01, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        torch.testing.assert_close(geometry.limit_ground_curve_curvature(points, 0.2), points)
        straight = geometry.limit_ground_curve_curvature(points, 0.0)
        torch.testing.assert_close(analytic_bezier_curvatures(straight), torch.zeros(1, 1025, dtype=torch.float64))
        torch.testing.assert_close(geometry.limit_ground_curve_curvature(straight, 0.0), straight)

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
    def test_rear_camera_has_distinct_prim_and_looks_backwards_and_down(self):
        cfg = rear_camera_command_cfg()
        front = make_curve_camera_cfg(make_command(1).cfg, render=True)
        for render in (False, True):
            with self.subTest(render=render):
                rear = make_rear_camera_cfg(cfg, render=render)
                self.assertIs(type(rear), TiledCameraCfgFixture if render else AssetCfgFixture)
                self.assertEqual(rear.prim_path, "{ENV_REGEX_NS}/Robot/base_link/RearCamera")
                self.assertNotEqual(rear.prim_path, front.prim_path)
                mount = rear.offset if render else rear.init_state
                self.assertEqual(mount.pos, cfg.camera_position_b)
                optical_forward = torch.tensor([[1.0, 0.0, 0.0]] if render else [[0.0, 0.0, -1.0]])
                direction = quat_apply(torch.tensor([mount.rot]), optical_forward)
                torch.testing.assert_close(
                    direction, torch.tensor([[-math.sqrt(0.5), 0.0, -math.sqrt(0.5)]]), atol=1e-6, rtol=1e-6
                )
                self.assertEqual(rear.spawn.width, 640)
                self.assertEqual(rear.spawn.height, 480)
                self.assertEqual(rear.spawn.clipping_range, cfg.camera_depth_range)
                if render:
                    self.assertEqual(rear.data_types, ["rgb"])
                    self.assertEqual(rear.update_period, 0.1)

    def test_training_spawns_only_usd_camera_and_play_enables_tiled_rgb(self):
        cfg = make_command(1).cfg
        training = make_curve_camera_cfg(cfg, render=False)
        play = make_curve_camera_cfg(cfg, render=True)
        self.assertIsInstance(training, AssetCfgFixture)
        self.assertIs(type(play), TiledCameraCfgFixture)
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

    def test_environment_rebuild_preserves_tiled_mode_after_calibration_override(self):
        class EnvFixture:
            def __init__(self, cfg, **kwargs):
                self.cfg = cfg
                self.num_envs, self.device = cfg.scene.num_envs, "cpu"
                self.action_manager = types.SimpleNamespace(total_action_dim=18)

        with patch.dict(
            sys.modules,
            {
                "isaaclab.envs": types.SimpleNamespace(ManagerBasedRLEnv=EnvFixture),
                "isaaclab.sensors": types.SimpleNamespace(CameraCfg=CameraCfgFixture),
                "b2_z1.tasks.manager_based.b2_z1.mdp.commands.curve_camera": types.SimpleNamespace(
                    make_curve_camera_cfg=make_curve_camera_cfg, make_rear_camera_cfg=make_rear_camera_cfg
                ),
            },
        ):
            env_class = load_module(
                "curve_camera_env_test", COMMANDS.parents[4] / "envs/b2z1_rl_env.py"
            ).B2Z1ManagerBasedRLEnv

        for render, expected_type in ((False, AssetCfgFixture), (True, TiledCameraCfgFixture)):
            with self.subTest(render=render):
                command_cfg = make_command(1).cfg
                rear_cfg = rear_camera_command_cfg()
                scene_cfg = types.SimpleNamespace(
                    num_envs=4,
                    base_camera=make_curve_camera_cfg(command_cfg, render=render),
                    rear_camera=make_rear_camera_cfg(rear_cfg, render=render),
                )
                # Override calibration after the scene config was initially built, as Hydra does.
                command_cfg.camera_image_size = (800, 600)
                command_cfg.camera_intrinsics = (400.0, 400.0, 400.0, 300.0)
                rear_cfg.camera_position_b = (-0.4, -0.03, 0.1)
                rear_cfg.camera_image_size = (320, 240)
                rear_cfg.camera_intrinsics = (160.0, 160.0, 160.0, 120.0)
                rear_cfg.camera_update_period = 0.2
                cfg = types.SimpleNamespace(
                    scene=scene_cfg, commands=types.SimpleNamespace(base_velocity=command_cfg, ee_target=rear_cfg)
                )
                env = env_class(cfg)
                camera = env.cfg.scene.base_camera
                self.assertIs(type(camera), expected_type)
                self.assertEqual(camera.spawn.width, 800)
                self.assertEqual(camera.spawn.height, 600)
                self.assertEqual(camera.spawn.intrinsic_matrix, [400.0, 0.0, 400.0, 0.0, 400.0, 300.0, 0.0, 0.0, 1.0])
                rear = env.cfg.scene.rear_camera
                self.assertIs(type(rear), expected_type)
                self.assertEqual(rear.spawn.width, 320)
                self.assertEqual(rear.spawn.height, 240)
                self.assertEqual(rear.spawn.intrinsic_matrix, [160.0, 0.0, 160.0, 0.0, 160.0, 120.0, 0.0, 0.0, 1.0])
                self.assertEqual((rear.offset if render else rear.init_state).pos, rear_cfg.camera_position_b)
                if render:
                    self.assertEqual(rear.update_period, 0.2)

        # Setting a scene camera to None must still disable its reconstruction.
        scene_cfg.base_camera = scene_cfg.rear_camera = None
        env = env_class(cfg)
        self.assertIsNone(env.cfg.scene.base_camera)
        self.assertIsNone(env.cfg.scene.rear_camera)

    def test_unsupported_render_calibration_is_rejected(self):
        for intrinsics in ((320.0, 300.0, 320.0, 240.0), (320.0, 320.0, 300.0, 240.0)):
            cfg = make_command(1, camera_intrinsics=intrinsics).cfg
            for render in (False, True):
                with self.subTest(intrinsics=intrinsics, render=render), self.assertRaises(ValueError):
                    make_curve_camera_cfg(cfg, render=render)


class CurveCurriculumTests(unittest.TestCase):
    def make_env(self, **options):
        command = make_command(curriculum_enabled=True, trajectory_curve_offset_scale=0.0, **options)
        env = command._env
        env.max_episode_length = 1000
        env.max_episode_length_s = 20.0
        env.common_step_counter = 1000
        env.command_manager = types.SimpleNamespace(get_term=lambda name: command)
        weights = {"track_lin_vel_xy": 4.5, "track_ang_vel_z": 2.5}
        env.reward_manager = types.SimpleNamespace(
            _episode_sums={name: torch.full((3,), 0.9 * weight * 20.0) for name, weight in weights.items()},
            get_term_cfg=lambda name: types.SimpleNamespace(weight=weights[name]),
        )
        return env, command

    def test_course_increases_corner_count_and_angles_then_stops_at_limits(self):
        env, command = self.make_env(
            path_type="turn", turn_count_range=(2, 3), turn_angle_range=(math.pi / 2, math.pi / 2),
            resampling_time_range=(25.0, 25.0), trajectory_num_points=192,
        )
        torch.testing.assert_close(command.curve_turn_count, torch.zeros(3, dtype=torch.long))
        for interval in range(1, 13):
            env.common_step_counter = interval * 1000
            level = curriculums.curve_curvature_cmd_levels(env, [0, 1, 2]).item()
            command._resample_command([0, 1, 2])
            expected_count = 2 if interval < 5 else 3
            torch.testing.assert_close(command.curve_turn_count, torch.full((3,), expected_count, dtype=torch.long))
            expected_angle = min(interval / 10, 1.0) * math.pi / 2
            torch.testing.assert_close(command.curve_turn_angles.abs().amax(dim=-1), torch.full((3,), expected_angle))
            self.assertLessEqual(level, 0.1 + 1e-7)
            self.assertTrue((command.curve_turn_angles.abs() <= math.pi / 2).all())

    def test_turn_curriculum_scales_angle_and_changes_only_future_paths(self):
        env, command = self.make_env(path_type="turn", turn_angle_range=(math.pi / 2, math.pi / 2))
        original = command.curve_points_w.clone()
        torch.testing.assert_close(original[..., 1], torch.zeros(3, 96))
        torch.testing.assert_close(command.curve_turn_angle, torch.zeros(3))
        curriculums.curve_curvature_cmd_levels(env, [0, 1, 2])
        torch.testing.assert_close(command.curve_points_w, original)
        command._resample_command([1])
        self.assertAlmostEqual(command.curve_turn_angle[1].abs().item(), math.pi / 20)
        torch.testing.assert_close(command.curve_points_w[[0, 2]], original[[0, 2]])

    def test_promotion_preserves_paths_until_partial_resampling(self):
        env, command = self.make_env()
        original = command.curve_points_w.clone()
        torch.testing.assert_close(original[:, :, 1], torch.zeros(3, 96))
        level = curriculums.curve_curvature_cmd_levels(env, [0, 1, 2])
        self.assertAlmostEqual(level.item(), 0.01)
        torch.testing.assert_close(command.curve_points_w, original)
        torch.manual_seed(7)
        command._resample_command([1])
        self.assertGreater(command.curve_points_w[1, :, 1].abs().max().item(), 0.0)
        torch.testing.assert_close(command.curve_points_w[[0, 2]], original[[0, 2]])
        torch.testing.assert_close(command.vel_command_b[:, 0], torch.full((3,), 0.4))
        self.assertTrue((command.vel_command_b[:, 2].abs() <= 0.8).all())

    def test_requires_both_tracking_rewards_above_threshold(self):
        for name in ("track_lin_vel_xy", "track_ang_vel_z"):
            for fraction in (0.0, 0.8, float("nan")):
                with self.subTest(name=name, fraction=fraction):
                    env, command = self.make_env()
                    weight = env.reward_manager.get_term_cfg(name).weight
                    env.reward_manager._episode_sums[name].fill_(fraction * weight * 20.0)
                    level = curriculums.curve_curvature_cmd_levels(env, [0, 1, 2])
                    self.assertEqual(level.item(), 0.0)
                    self.assertEqual(command.cfg.trajectory_curve_offset_scale, 0.0)

    def test_only_reset_environments_contribute_to_reward(self):
        env, _ = self.make_env()
        for sums in env.reward_manager._episode_sums.values():
            sums[[0, 2]] = 0.0
        level = curriculums.curve_curvature_cmd_levels(env, torch.tensor([1]))
        self.assertAlmostEqual(level.item(), 0.01)

    def test_asynchronous_resets_update_at_most_once_per_interval(self):
        env, _ = self.make_env()
        for step, expected in ((0, 0.0), (999, 0.0), (1007, 0.01), (1007, 0.01), (2006, 0.01), (2007, 0.02)):
            env.common_step_counter = step
            level = curriculums.curve_curvature_cmd_levels(env, [0, 1, 2])
            self.assertAlmostEqual(level.item(), expected)

    def test_failed_evaluation_waits_an_interval_before_retrying(self):
        env, _ = self.make_env()
        sums = env.reward_manager._episode_sums["track_ang_vel_z"]
        sums.zero_()
        curriculums.curve_curvature_cmd_levels(env, [0, 1, 2])
        sums.fill_(0.9 * 2.5 * 20.0)
        for step, expected in ((1001, 0.0), (2000, 0.01)):
            env.common_step_counter = step
            self.assertAlmostEqual(curriculums.curve_curvature_cmd_levels(env, [0, 1, 2]).item(), expected)

    def test_empty_resets_do_not_consume_evaluation_interval(self):
        env, command = self.make_env()
        for ids in ([], torch.empty(0, dtype=torch.long), slice(0, 0)):
            level = curriculums.curve_curvature_cmd_levels(env, ids)
            self.assertEqual(level.item(), 0.0)
        self.assertEqual(command._last_curve_curriculum_step, 0)
        self.assertAlmostEqual(curriculums.curve_curvature_cmd_levels(env, slice(None)).item(), 0.01)

    def test_difficulty_stops_at_configured_limit(self):
        env, command = self.make_env(limit_trajectory_curve_offset_scale=0.025)
        for step, expected in ((1000, 0.01), (2000, 0.02), (3000, 0.025), (4000, 0.025)):
            env.common_step_counter = step
            self.assertAlmostEqual(curriculums.curve_curvature_cmd_levels(env, None).item(), expected)
        self.assertEqual(command.cfg.trajectory_curve_offset_scale, 0.025)

    def test_disabled_curriculum_uses_final_difficulty_from_first_sample(self):
        command = make_command(trajectory_curve_offset_scale=0.0, limit_trajectory_curve_offset_scale=0.15)
        self.assertEqual(command.cfg.trajectory_curve_offset_scale, 0.15)
        self.assertGreater(command.curve_points_w[:, :, 1].abs().max().item(), 0.0)
        env = command._env
        env.command_manager = types.SimpleNamespace(get_term=lambda name: command)
        self.assertAlmostEqual(curriculums.curve_curvature_cmd_levels(env, []).item(), 0.15)

    def test_missing_or_nonpositive_reward_does_not_promote(self):
        for weight in (None, 0.0, -1.0):
            with self.subTest(weight=weight):
                env, _ = self.make_env()
                if weight is None:
                    del env.reward_manager._episode_sums["track_ang_vel_z"]
                else:
                    env.reward_manager.get_term_cfg = lambda name: types.SimpleNamespace(weight=weight)
                self.assertEqual(curriculums.curve_curvature_cmd_levels(env, [0, 1, 2]).item(), 0.0)

    def test_invalid_curriculum_parameters_are_rejected(self):
        for options in (
            {"reward_term_names": ()},
            {"reward_threshold": 0.0},
            {"reward_threshold": 1.1},
            {"offset_scale_step": 0.0},
            {"offset_scale_step": float("inf")},
        ):
            env, _ = self.make_env()
            with self.subTest(options=options), self.assertRaises(ValueError):
                curriculums.curve_curvature_cmd_levels(env, [0, 1, 2], **options)


class CommandTests(unittest.TestCase):
    def test_full_difficulty_starts_with_three_maximum_angle_corners(self):
        command = make_command(
            path_type="turn", turn_count_range=(2, 3), turn_angle_range=(math.pi / 2, math.pi / 2),
            trajectory_curve_offset_scale=0.0, resampling_time_range=(25.0, 25.0),
            trajectory_num_points=192, max_curvature=1.5,
        )
        self.assertEqual(command.cfg.trajectory_curve_offset_scale, 0.1)
        torch.testing.assert_close(command.curve_turn_count, torch.full((3,), 3, dtype=torch.long))
        torch.testing.assert_close(command.curve_turn_angles.abs(), torch.full((3, 3), math.pi / 2))
        torch.testing.assert_close(command.curve_path_length, torch.full((3,), 11.2))
        torch.testing.assert_close(command.metrics["turn_count"], torch.full((3,), 3.0))
        torch.testing.assert_close(command.metrics["turn_angle"], torch.full((3,), math.pi / 2))
        self.assertTrue(command.path_visible.all())
        self.assertLessEqual(polyline_curvatures(command.curve_points_w).abs().max().item(), 1.001)

    def test_multi_corner_partial_resampling_preserves_other_path_metadata(self):
        command = make_command(path_type="turn", turn_count_range=(2, 3), resampling_time_range=(25.0, 25.0))
        fields = ("curve_points_w", "curve_turn_angles", "curve_turn_count", "curve_turn_angle", "curve_path_length")
        original = {name: getattr(command, name).clone() for name in fields}
        command.time_left[1] = 1.0
        command.robot.data.body_pos_w[1, 0, 0] += 0.5
        command._resample_command([1])
        for name in fields:
            torch.testing.assert_close(getattr(command, name)[[0, 2]], original[name][[0, 2]])
        self.assertLess(command.curve_path_length[1].item(), original["curve_path_length"][1].item())

    def test_kinematic_closed_loop_follows_three_maximum_corners_in_both_directions(self):
        with patch.object(torch, "rand", return_value=torch.tensor([[0.5, 0.5, 0.5, 0.75], [0.5, 0.5, 0.5, 0.25]])):
            command = make_command(
                2, path_type="turn", turn_count_range=(2, 3), turn_angle_range=(math.pi / 2, math.pi / 2),
                resampling_time_range=(25.0, 25.0), trajectory_num_points=192, max_curvature=1.5,
            )
        heading, zero = torch.zeros(2), torch.zeros(2)
        for _ in range(1150):
            command._update_command()
            self.assertTrue(command.path_visible.all())
            self.assertTrue((command.vel_command_b[:, 2].abs() <= 0.8).all())
            heading += 0.02 * command.vel_command_b[:, 2]
            command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zero, zero, heading)
            command.robot.data.body_pos_w[:, 0, 0] += 0.02 * 0.4 * torch.cos(heading)
            command.robot.data.body_pos_w[:, 0, 1] += 0.02 * 0.4 * torch.sin(heading)
        torch.testing.assert_close(heading, command.curve_turn_angle, atol=math.radians(2.0), rtol=0.0)
        self.assertTrue((command.lateral_error.abs() < 0.02).all())
        self.assertTrue((command.robot.data.body_pos_w[:, 0, 0] > 4.5).all())

    def test_turn_sampler_generates_45_to_90_degree_left_and_right_turns(self):
        torch.manual_seed(7)
        command = make_command(64, path_type="turn", max_curvature=1.5)
        angles = command.curve_turn_angle
        self.assertTrue((angles.abs() >= math.pi / 4).all())
        self.assertTrue((angles.abs() <= math.pi / 2).all())
        self.assertTrue((angles > 0).any() and (angles < 0).any())
        self.assertTrue(command.path_visible.all())
        self.assertIsNone(command.curve_control_points_w)
        self.assertLessEqual(polyline_curvatures(command.curve_points_w).abs().max().item(), 1.001)
        torch.testing.assert_close(command.metrics["turn_angle"], angles.abs())
        segments = command.curve_points_w[:, -1] - command.curve_points_w[:, -2]
        torch.testing.assert_close(torch.atan2(segments[:, 1], segments[:, 0]), angles, atol=1e-5, rtol=1e-5)

    def test_turn_curvature_cap_and_zero_difficulty(self):
        command = make_command(path_type="turn", max_curvature=0.2)
        self.assertTrue((command.curve_turn_angle.abs() <= 0.76 + 1e-6).all())
        self.assertLessEqual(polyline_curvatures(command.curve_points_w).abs().max().item(), 0.201)
        for options in (
            {"max_curvature": 0.0},
            {"trajectory_curve_offset_scale": 0.0, "limit_trajectory_curve_offset_scale": 0.0},
        ):
            command = make_command(path_type="turn", **options)
            torch.testing.assert_close(command.curve_points_w[..., 1], torch.zeros(3, 96))
            torch.testing.assert_close(command.curve_turn_angle, torch.zeros(3))

    def test_turn_paths_stay_world_fixed_and_partial_resets_preserve_other_environments(self):
        command = make_command(path_type="turn")
        points, angles = command.curve_points_w.clone(), command.curve_turn_angle.clone()
        command.robot.data.body_pos_w[1, 0, 0] += 0.5
        zero = torch.zeros(1)
        command.robot.data.body_quat_w[1:2, 0] = quat_from_euler_xyz(zero, zero, torch.tensor([0.7]))
        command._update_command()
        torch.testing.assert_close(command.curve_points_w, points)
        command._resample_command([1])
        torch.testing.assert_close(command.curve_points_w[[0, 2]], points[[0, 2]])
        torch.testing.assert_close(command.curve_turn_angle[[0, 2]], angles[[0, 2]])
        self.assertFalse(torch.equal(command.curve_points_w[1], points[1]))

    def test_kinematic_closed_loop_follows_both_90_degree_turns(self):
        with patch.object(torch, "rand", return_value=torch.tensor([[0.5, 0.75], [0.5, 0.25]])):
            command = make_command(2, path_type="turn", turn_angle_range=(math.pi / 2, math.pi / 2))
        heading, zero = torch.zeros(2), torch.zeros(2)
        for _ in range(450):
            command._update_command()
            self.assertTrue(command.path_visible.all())
            self.assertTrue((command.vel_command_b[:, 2].abs() <= 0.8).all())
            heading += 0.02 * command.vel_command_b[:, 2]
            command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zero, zero, heading)
            command.robot.data.body_pos_w[:, 0, 0] += 0.02 * 0.4 * torch.cos(heading)
            command.robot.data.body_pos_w[:, 0, 1] += 0.02 * 0.4 * torch.sin(heading)
        torch.testing.assert_close(heading, command.curve_turn_angle, atol=math.radians(2.0), rtol=0.0)
        self.assertTrue((command.lateral_error.abs() < 0.02).all())

    def test_curvature_cap_handles_different_lengths_and_world_poses(self):
        torch.manual_seed(7)
        command = make_command(16, max_curvature=0.1)
        command.time_left = torch.linspace(0.0, 10.0, 16)
        zeros = torch.zeros(16)
        command.robot.data.body_quat_w[:, 0] = quat_from_euler_xyz(zeros, zeros, torch.linspace(-2.0, 2.0, 16))
        command.robot.data.body_pos_w[:, 0, :2] = torch.randn(16, 2) * 3.0
        command._resample_command(torch.arange(16))
        curvatures = analytic_bezier_curvatures(command.curve_control_points_w)
        self.assertTrue(torch.isfinite(curvatures).all())
        self.assertLessEqual(curvatures.abs().max().item(), 0.1 + 1e-5)
        torch.testing.assert_close(command.vel_command_b[:, 0], torch.full((16,), 0.4))
        self.assertTrue((command.vel_command_b[:, 2].abs() <= 0.8).all())

    def test_curvature_cap_preserves_other_environments_on_partial_resampling(self):
        command = make_command(max_curvature=0.2)
        original_points = command.curve_points_w.clone()
        original_controls = command.curve_control_points_w.clone()
        command.cfg.max_curvature = 0.0
        command._resample_command([1])
        torch.testing.assert_close(command.curve_points_w[[0, 2]], original_points[[0, 2]])
        torch.testing.assert_close(command.curve_control_points_w[[0, 2]], original_controls[[0, 2]])
        self.assertEqual(analytic_bezier_curvatures(command.curve_control_points_w[1:2]).abs().max().item(), 0.0)

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
            {"trajectory_curve_offset_scale": -0.01},
            {"trajectory_curve_offset_scale": 0.2},
            {"limit_trajectory_curve_offset_scale": -0.1},
            {"limit_trajectory_curve_offset_scale": float("inf")},
            {"trajectory_curve_offset_scale": float("nan")},
            {"max_curvature": -0.1},
            {"max_curvature": float("nan")},
            {"max_curvature": float("inf")},
            {"max_curvature": 2.1},
            {"max_curvature": 0.2, "ranges": types.SimpleNamespace(ang_vel_z=(-0.04, 0.8))},
            {"path_type": "unknown"},
            {"turn_angle_range": (-0.1, 1.0)},
            {"turn_angle_range": (0.0, 1.0, 2.0)},
            {"turn_angle_range": (1.0, 0.5)},
            {"turn_angle_range": (0.0, math.pi + 0.1)},
            {"turn_curvature": 0.0},
            {"turn_curvature": float("inf")},
            {"turn_start_distance": -0.1},
            {"path_type": "turn", "turn_curvature": 2.1},
            {"turn_count_range": (0, 3)},
            {"turn_count_range": (2, 4)},
            {"turn_count_range": (3, 2)},
            {"turn_count_range": (2.0, 3.0)},
            {"turn_count_range": (True, 3)},
            {"turn_gap_distance": -0.1},
            {"turn_gap_distance": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_command(**options)


if __name__ == "__main__":
    unittest.main()
