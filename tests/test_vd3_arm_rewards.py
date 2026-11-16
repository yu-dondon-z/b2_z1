"""CPU checks of VD3 stabilization geometry, command derivatives and joint IDs."""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch


MDP = Path(__file__).resolve().parents[1] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"
PACKAGE_NAME = "vd3_arm_reward_cpu"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(MDP)]
sys.modules[PACKAGE_NAME] = package
spec = importlib.util.spec_from_file_location(PACKAGE_NAME + ".vd3_arm_rewards", MDP / "vd3_arm_rewards.py")
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)
tcp_frame = sys.modules[PACKAGE_NAME + ".tcp_frame"]


def quaternion(roll, pitch, yaw):
    cr, cp, cy = math.cos(roll / 2), math.cos(pitch / 2), math.cos(yaw / 2)
    sr, sp, sy = math.sin(roll / 2), math.sin(pitch / 2), math.sin(yaw / 2)
    return torch.tensor([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=torch.float64)


def make_env(offset_tcp=False):
    """Nontrivial moving-base fixture with deliberately different COM velocities."""
    count, bodies = 2, 2 if offset_tcp else 3
    quat = torch.stack((quaternion(.2, -.3, 1.1), quaternion(-.4, .1, -.8)))
    base = torch.tensor([[7., -2., 1.2], [-4., 8., .7]], dtype=torch.float64)
    data = types.SimpleNamespace(
        body_link_pos_w=base[:, None].expand(-1, bodies, -1).clone(),
        body_link_quat_w=quat[:, None].expand(-1, bodies, -1).clone(),
        body_link_lin_vel_w=torch.zeros(count, bodies, 3, dtype=torch.float64),
        body_link_ang_vel_w=torch.zeros(count, bodies, 3, dtype=torch.float64),
        # If the stabilization accidentally reads COM velocities, tests fail.
        body_lin_vel_w=torch.full((count, bodies, 3), 99.0, dtype=torch.float64),
        body_ang_vel_w=torch.full((count, bodies, 3), 99.0, dtype=torch.float64),
        joint_vel=torch.zeros(count, 18, dtype=torch.float64),
    )
    data.body_link_pos_w[:, 1] += tcp_frame.quat_apply(
        quat, torch.tensor([[-.8, .05, .2]], dtype=torch.float64).expand(count, -1)
    )
    if not offset_tcp:
        data.body_link_pos_w[:, 2] += tcp_frame.quat_apply(
            quat, torch.tensor([[-1.02, .0, -.3]], dtype=torch.float64).expand(count, -1)
        )
    names = {"base_link": 0, "link06": 1}
    if not offset_tcp:
        names["dummy_target"] = 2
    asset = types.SimpleNamespace(
        data=data,
        find_bodies=lambda name: ([names[name]], [name]) if name in names else ([], []),
    )
    start, _ = tcp_frame.tcp_pose_b(asset)
    delta = torch.tensor([[.4, -.2, .1]], dtype=torch.float64).expand(count, -1).clone()
    command = types.SimpleNamespace(
        cfg=types.SimpleNamespace(asset_name="robot"),
        metrics={},
        pose_start=torch.cat((start.clone(), quat.clone()), dim=-1),
        pose_target=torch.cat((start + delta, quat.clone()), dim=-1),
        _curve_control_points=torch.stack((start + delta / 3, start + 2 * delta / 3), dim=1),
        _command_duration=torch.full((count,), 4.0, dtype=torch.float64),
        time_left=torch.full((count,), 2.4, dtype=torch.float64),
    )
    env = types.SimpleNamespace(
        num_envs=count,
        step_dt=.02,
        scene={"robot": asset},
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
    )
    set_rigid_motion(env)
    return env


def set_rigid_motion(env):
    asset = env.scene["robot"]
    data = asset.data
    base_velocity = torch.tensor([[.4, -.2, .1], [-.2, .5, -.1]], dtype=torch.float64)
    angular = torch.tensor([[.7, -.4, .6], [-.3, .2, .5]], dtype=torch.float64)
    offsets = data.body_link_pos_w - data.body_link_pos_w[:, :1]
    data.body_link_lin_vel_w[:] = base_velocity[:, None] + torch.linalg.cross(
        angular[:, None].expand_as(offsets), offsets, dim=-1
    )
    data.body_link_ang_vel_w[:] = angular[:, None]


def set_target_constant(env):
    command = env.command_manager.get_term("ee_target")
    start = command.pose_start[:, :3]
    command.pose_target[:, :3] = start
    command._curve_control_points[:] = start[:, None]


def add_relative_velocity(env, linear_b, angular_b=None):
    asset = env.scene["robot"]
    data = asset.data
    index, _ = tcp_frame.tcp_parent(asset)
    quat = data.body_link_quat_w[:, 0]
    data.body_link_lin_vel_w[:, index] += tcp_frame.quat_apply(quat, linear_b)
    if angular_b is not None:
        data.body_link_ang_vel_w[:, index] += tcp_frame.quat_apply(quat, angular_b)


class VD3ArmRewardTests(unittest.TestCase):
    def test_rigid_translation_and_rotation_have_zero_relative_twist(self):
        env = make_env()
        set_target_constant(env)
        linear, angular = rewards.tcp_velocity_errors(env)
        torch.testing.assert_close(linear, torch.zeros_like(linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(angular, torch.zeros_like(angular), atol=1e-14, rtol=0)
        torch.testing.assert_close(rewards.tcp_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))

    def test_massless_offset_tcp_has_zero_residual_in_rigid_motion(self):
        env = make_env(offset_tcp=True)
        set_target_constant(env)
        linear, angular = rewards.tcp_velocity_errors(env)
        torch.testing.assert_close(linear, torch.zeros_like(linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(angular, torch.zeros_like(angular), atol=1e-14, rtol=0)

    def test_correctly_following_a_moving_command_is_not_penalized(self):
        env = make_env()
        velocity = rewards.command_linear_velocity_b(env.command_manager.get_term("ee_target"), env.step_dt)
        add_relative_velocity(env, velocity)
        linear, angular = rewards.tcp_velocity_errors(env)
        torch.testing.assert_close(linear, torch.zeros_like(linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(angular, torch.zeros_like(angular), atol=1e-14, rtol=0)

    def test_unnecessary_gait_frequency_oscillation_has_positive_mean_penalty(self):
        env = make_env()
        set_target_constant(env)
        penalties = []
        for phase in torch.linspace(0., 2 * math.pi, 41, dtype=torch.float64):
            set_rigid_motion(env)
            velocity = torch.zeros(2, 3, dtype=torch.float64)
            velocity[:, 2] = .15 * torch.cos(phase)
            add_relative_velocity(env, velocity)
            penalties.append(rewards.tcp_linear_velocity_residual(env))
        self.assertGreater(torch.stack(penalties).mean().item(), .5)

    def test_relative_rotation_is_penalized_without_penalizing_base_rotation(self):
        env = make_env()
        set_target_constant(env)
        zeros = torch.zeros(2, 3, dtype=torch.float64)
        add_relative_velocity(env, zeros, torch.tensor([[0., .3, 0.], [0., .3, 0.]], dtype=torch.float64))
        expected = torch.full((2,), ((.3 - .05) / .5) ** 2, dtype=torch.float64)
        torch.testing.assert_close(rewards.tcp_angular_velocity_residual(env), expected)

    def test_bezier_derivative_matches_finite_difference(self):
        generator = torch.Generator().manual_seed(11)
        points = [torch.randn(7, 3, generator=generator, dtype=torch.float64) for _ in range(4)]
        progress = torch.linspace(.1, .9, 7, dtype=torch.float64)
        duration = torch.linspace(1., 5., 7, dtype=torch.float64)
        def position(u):
            u = u[:, None]
            return (1-u)**3*points[0] + 3*(1-u)**2*u*points[1] + 3*(1-u)*u**2*points[2] + u**3*points[3]
        epsilon = 1e-6
        expected = (position(progress + epsilon) - position(progress - epsilon)) / (2 * epsilon * duration[:, None])
        actual = rewards.cubic_bezier_velocity(*points, progress, duration)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)

    def test_bezier_endpoint_derivatives_and_duration_scaling(self):
        points = [torch.tensor([value], dtype=torch.float64) for value in
                  ((0., 1., 0.), (1., 0., 2.), (-1., 2., 1.), (3., 1., -2.))]
        first = rewards.cubic_bezier_velocity(*points, torch.tensor([0.]), torch.tensor([2.]))
        last = rewards.cubic_bezier_velocity(*points, torch.tensor([1.]), torch.tensor([2.]))
        torch.testing.assert_close(first, 3 * (points[1] - points[0]) / 2)
        torch.testing.assert_close(last, 3 * (points[3] - points[2]) / 2)
        for progress in (-.1, 1.1):
            torch.testing.assert_close(
                rewards.cubic_bezier_velocity(*points, torch.tensor([progress]), torch.tensor([2.])),
                torch.zeros_like(first),
            )

    def test_zero_duration_is_finite_and_inactive(self):
        start = torch.zeros(2, 3, dtype=torch.float64)
        value = rewards.cubic_bezier_velocity(start, start+1, start+2, start+3,
                                             torch.tensor([.4, .7]), torch.tensor([0., -1.]))
        torch.testing.assert_close(value, torch.zeros_like(value))

    def test_deadzone_and_upper_bound_are_in_physical_speed_units(self):
        velocities = torch.tensor([[.005, 0., 0.], [.13, 0., 0.], [1000., 0., 0.]])
        result = rewards.bounded_deadzone_quadratic(velocities, .12, .01, 4.)
        torch.testing.assert_close(result, torch.tensor([0., 1., 4.]))
        for options in ((0., .01, 4.), (.12, -.01, 4.), (.12, .01, 0.)):
            with self.assertRaises(ValueError):
                rewards.bounded_deadzone_quadratic(velocities, *options)

    def test_joint_selection_uses_interleaved_ids_and_does_not_penalize_legs(self):
        env = make_env()
        indices = [1, 4, 7, 10, 13, 16]
        asset_cfg = types.SimpleNamespace(name="robot", joint_ids=indices)
        env.scene["robot"].data.joint_vel[:] = 10.
        env.scene["robot"].data.joint_vel[:, indices] = .2
        actual = rewards.arm_joint_velocity_l2(env, asset_cfg)
        torch.testing.assert_close(actual, torch.full((2,), .24, dtype=torch.float64))
        metrics = env.command_manager.get_term("ee_target").metrics
        torch.testing.assert_close(metrics["arm_joint_velocity_rms"], torch.full((2,), .2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            rewards.arm_joint_velocity_l2(env, types.SimpleNamespace(name="robot", joint_ids=slice(None)))

    def test_command_resample_is_stateless_without_an_endpoint_difference_spike(self):
        env = make_env()
        command = env.command_manager.get_term("ee_target")
        target_velocity = rewards.command_linear_velocity_b(command, env.step_dt)
        add_relative_velocity(env, target_velocity)
        before = rewards.tcp_linear_velocity_residual(env)
        # A new curve can have a distant start; identical derivatives still mean
        # identical requested speed. Cross-resample position differences are irrelevant.
        command.pose_start[:, :3] += 100.
        command.pose_target[:, :3] += 100.
        command._curve_control_points[:] += 100.
        command.time_left[:] = command._command_duration
        after = rewards.tcp_linear_velocity_residual(env)
        torch.testing.assert_close(before, after, atol=1e-14, rtol=0)

    def test_partial_episode_reset_does_not_leave_or_transfer_velocity_history(self):
        env = make_env()
        set_target_constant(env)
        add_relative_velocity(env, torch.tensor([[0., .2, 0.], [0., .1, 0.]], dtype=torch.float64))
        before = rewards.tcp_linear_velocity_residual(env)
        env.scene["robot"].data.body_link_lin_vel_w[0, 2] -= tcp_frame.quat_apply(
            env.scene["robot"].data.body_link_quat_w[0:1, 0],
            torch.tensor([[0., .2, 0.]], dtype=torch.float64),
        )[0]
        after = rewards.tcp_linear_velocity_residual(env)
        self.assertEqual(after[0].item(), 0.)
        torch.testing.assert_close(after[1], before[1])

    def test_metrics_report_unclipped_velocity_error_and_relative_speed(self):
        env = make_env()
        set_target_constant(env)
        add_relative_velocity(env, torch.tensor([[0., .2, 0.], [0., .1, 0.]], dtype=torch.float64))
        rewards.tcp_linear_velocity_residual(env)
        metrics = env.command_manager.get_term("ee_target").metrics
        torch.testing.assert_close(metrics["tcp_linear_velocity_error"], torch.tensor([.2, .1], dtype=torch.float64))
        torch.testing.assert_close(metrics["tcp_relative_linear_speed"], metrics["tcp_linear_velocity_error"])
        torch.testing.assert_close(metrics["tcp_angular_velocity_error"], torch.zeros(2, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
