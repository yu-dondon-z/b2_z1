"""CPU regressions for flange motion hidden by a nearly stationary gun TCP."""

import math
import unittest

import torch

from test_vd3_arm_rewards import (
    make_env,
    quaternion,
    rewards,
    set_rigid_motion,
    tcp_frame,
)


def make_flange_env():
    """The VD3 asset has link06 plus a virtual TCP, without dummy_target."""
    env = make_env(offset_tcp=True)
    command = env.command_manager.get_term("ee_target")
    position, orientation = tcp_frame.tcp_pose_b(env.scene["robot"])
    command.command = torch.cat((position.clone(), orientation.clone()), dim=-1)
    command.pose_start = command.command.clone()
    command.pose_target = command.command.clone()
    command._curve_control_points[:] = position[:, None]
    return env


def set_translating_command(env, velocity_b):
    """The current target pose is unchanged, with an exact linear Bezier speed."""
    command = env.command_manager.get_term("ee_target")
    duration = command._command_duration[:, None]
    elapsed = duration - command.time_left[:, None]
    start = command.command[:, :3] - velocity_b * elapsed
    displacement = velocity_b * duration
    command.pose_start[:, :3] = start
    command.pose_target[:, :3] = start + displacement
    command._curve_control_points[:, 0] = start + displacement / 3
    command._curve_control_points[:, 1] = start + 2 * displacement / 3
    base_quat = env.scene["robot"].data.body_link_quat_w[:, 0]
    env.scene["robot"].data.body_link_lin_vel_w[:, 1] += tcp_frame.quat_apply(base_quat, velocity_b)


def pivot_about_fixed_tcp(env, angle, angular_speed):
    """Rotate the flange around the target TCP, preserving exact TCP position."""
    robot = env.scene["robot"]
    data = robot.data
    target = env.command_manager.get_term("ee_target").command
    base_pos, base_quat = tcp_frame.link_pose_w(robot, 0)
    tcp_pos = base_pos + tcp_frame.quat_apply(base_quat, target[:, :3])
    rotation_b = quaternion(0., angle, 0.).expand(env.num_envs, -1)
    flange_quat = tcp_frame.quat_mul(base_quat, rotation_b)
    offset_w = tcp_frame.quat_apply(
        flange_quat,
        data.body_link_pos_w.new_tensor(tcp_frame.TCP_POS_IN_LINK06).expand(env.num_envs, -1),
    )
    data.body_link_pos_w[:, 1] = tcp_pos - offset_w
    data.body_link_quat_w[:, 1] = flange_quat
    set_rigid_motion(env)
    angular_b = torch.zeros(env.num_envs, 3, dtype=torch.float64)
    angular_b[:, 1] = angular_speed
    angular_w = tcp_frame.quat_apply(base_quat, angular_b)
    data.body_link_ang_vel_w[:, 1] += angular_w
    data.body_link_lin_vel_w[:, 1] -= torch.linalg.cross(angular_w, offset_w, dim=-1)


class FlangeTargetTransformTests(unittest.TestCase):
    def test_inverse_pose_and_twist_reconstruct_arbitrary_desired_tcp(self):
        position = torch.tensor([[1., -2., .7], [-.8, .1, -.3]], dtype=torch.float64)
        orientation = torch.stack((quaternion(.4, -.7, 1.2), quaternion(-.8, .2, -.3)))
        linear = torch.tensor([[.2, -.1, .3], [-.4, .1, .2]], dtype=torch.float64)
        angular = torch.tensor([[.3, .2, -.1], [-.2, .4, .1]], dtype=torch.float64)
        flange_pos, flange_quat, flange_linear, flange_angular = rewards.flange_target_from_tcp_pose(
            position, orientation, linear, angular
        )
        offset = tcp_frame.quat_apply(
            flange_quat, position.new_tensor(tcp_frame.TCP_POS_IN_LINK06).expand_as(position)
        )
        fixed_quat = orientation.new_tensor(tcp_frame.TCP_QUAT_IN_LINK06).expand_as(orientation)
        torch.testing.assert_close(flange_pos + offset, position, atol=1e-12, rtol=0)
        torch.testing.assert_close(tcp_frame.quat_mul(flange_quat, fixed_quat), orientation, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            flange_linear + torch.linalg.cross(flange_angular, offset, dim=-1),
            linear, atol=1e-12, rtol=0,
        )
        torch.testing.assert_close(flange_angular, angular)

    def test_quaternion_sign_does_not_change_target_position_or_velocity(self):
        position = torch.tensor([[-1.02, .0, -.3]], dtype=torch.float64)
        orientation = quaternion(.2, -.3, .8).unsqueeze(0)
        velocity = torch.tensor([[.1, .2, .3]], dtype=torch.float64)
        omega = torch.tensor([[.3, -.2, .1]], dtype=torch.float64)
        positive = rewards.flange_target_from_tcp_pose(position, orientation, velocity, omega)
        negative = rewards.flange_target_from_tcp_pose(position, -orientation, velocity, omega)
        for index in (0, 2, 3):
            torch.testing.assert_close(positive[index], negative[index], atol=1e-12, rtol=0)
        torch.testing.assert_close(positive[1], -negative[1], atol=1e-12, rtol=0)


class VD3FlangeRewardTests(unittest.TestCase):
    def test_rigid_base_translation_and_rotation_have_zero_flange_error(self):
        env = make_flange_env()
        linear, angular = rewards.flange_velocity_errors(env)
        torch.testing.assert_close(linear, torch.zeros_like(linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(angular, torch.zeros_like(angular), atol=1e-14, rtol=0)
        torch.testing.assert_close(rewards.flange_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(rewards.flange_angular_velocity_residual(env), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(rewards.flange_position_residual(env), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(
            env.command_manager.get_term("ee_target").metrics["flange_position_error"],
            torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0,
        )
        self.assertTrue(torch.all(
            env.command_manager.get_term("ee_target").metrics["flange_world_linear_speed"] > .1
        ))

    def test_correct_target_translation_is_not_penalized(self):
        env = make_flange_env()
        velocity = torch.tensor([[.10, -.05, .025], [-.03, .02, -.01]], dtype=torch.float64)
        set_translating_command(env, velocity)
        linear, angular = rewards.flange_velocity_errors(env)
        torch.testing.assert_close(linear, torch.zeros_like(linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(angular, torch.zeros_like(angular), atol=1e-14, rtol=0)
        torch.testing.assert_close(rewards.flange_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))

    def test_pivot_has_zero_tcp_translation_but_positive_flange_penalty(self):
        env = make_flange_env()
        pivot_about_fixed_tcp(env, .05, .15)
        tcp_linear, _ = rewards.tcp_velocity_errors(env)
        torch.testing.assert_close(tcp_linear, torch.zeros_like(tcp_linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(rewards.tcp_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))
        flange_linear, flange_angular = rewards.flange_velocity_errors(env)
        self.assertTrue(torch.all(torch.linalg.vector_norm(flange_linear, dim=-1) > .12))
        torch.testing.assert_close(
            flange_angular, torch.tensor([[0., .15, 0.], [0., .15, 0.]], dtype=torch.float64),
            atol=1e-14, rtol=0,
        )
        self.assertTrue(torch.all(rewards.flange_linear_velocity_residual(env) > .8))

    def test_fixed_tcp_position_still_exposes_flange_displacement_at_zero_speed(self):
        env = make_flange_env()
        pivot_about_fixed_tcp(env, .10, 0.)
        desired = env.command_manager.get_term("ee_target").command[:, :3]
        position, _ = tcp_frame.tcp_pose_b(env.scene["robot"])
        torch.testing.assert_close(position, desired, atol=1e-12, rtol=0)
        torch.testing.assert_close(rewards.flange_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))
        self.assertTrue(torch.all(rewards.flange_position_residual(env) > 2.))

    def test_rotation_around_gun_offset_is_caught_by_angular_term(self):
        env = make_flange_env()
        robot = env.scene["robot"]
        base_quat = robot.data.body_link_quat_w[:, 0]
        offset_b = robot.data.body_link_pos_w.new_tensor(tcp_frame.TCP_POS_IN_LINK06)
        omega_b = (.3 * offset_b / torch.linalg.vector_norm(offset_b)).expand(env.num_envs, -1)
        # Angular velocity parallel to flange->TCP leaves both origins stationary.
        robot.data.body_link_ang_vel_w[:, 1] += tcp_frame.quat_apply(base_quat, omega_b)
        tcp_linear, _ = rewards.tcp_velocity_errors(env)
        torch.testing.assert_close(tcp_linear, torch.zeros_like(tcp_linear), atol=1e-14, rtol=0)
        torch.testing.assert_close(rewards.flange_position_residual(env), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(rewards.flange_linear_velocity_residual(env), torch.zeros(2, dtype=torch.float64))
        expected = torch.full((2,), ((.3 - .05) / .5) ** 2, dtype=torch.float64)
        torch.testing.assert_close(rewards.flange_angular_velocity_residual(env), expected, atol=1e-14, rtol=0)

    def test_gait_frequency_pivot_is_detected_over_a_cycle_with_fixed_tcp(self):
        env = make_flange_env()
        desired = env.command_manager.get_term("ee_target").command[:, :3].clone()
        penalties, position_errors = [], []
        for phase in torch.linspace(0., 2 * math.pi, 61, dtype=torch.float64):
            angle = .08 * torch.sin(phase).item()
            angular_speed = .08 * 2 * math.pi * 1.5 * torch.cos(phase).item()
            pivot_about_fixed_tcp(env, angle, angular_speed)
            position, _ = tcp_frame.tcp_pose_b(env.scene["robot"])
            torch.testing.assert_close(position, desired, atol=1e-12, rtol=0)
            tcp_linear, _ = rewards.tcp_velocity_errors(env)
            torch.testing.assert_close(tcp_linear, torch.zeros_like(tcp_linear), atol=1e-14, rtol=0)
            penalties.append(rewards.flange_linear_velocity_residual(env))
            position_errors.append(env.command_manager.get_term("ee_target").metrics["flange_position_error"])
        self.assertGreater(torch.stack(penalties).mean().item(), 1.)
        self.assertGreater(torch.stack(position_errors).max().item(), .06)

    def test_current_target_orientation_determines_flange_reference(self):
        env = make_flange_env()
        command = env.command_manager.get_term("ee_target")
        rotation = quaternion(0., .12, 0.).expand(env.num_envs, -1)
        command.command[:, 3:7] = tcp_frame.quat_mul(rotation, command.command[:, 3:7])
        # Deliberately leave pose_target unchanged: the actor sees command.command.
        rewards.flange_velocity_errors(env)
        error = command.metrics["flange_position_error"]
        self.assertTrue(torch.all(error > .09))

    def test_metrics_report_unclipped_flange_motion_even_when_penalty_caps(self):
        env = make_flange_env()
        pivot_about_fixed_tcp(env, .1, 10.)
        penalty = rewards.flange_linear_velocity_residual(env)
        torch.testing.assert_close(penalty, torch.full((2,), 4., dtype=torch.float64))
        command = env.command_manager.get_term("ee_target")
        self.assertTrue(torch.all(command.metrics["flange_linear_velocity_error"] > 8.))
        torch.testing.assert_close(
            command.metrics["flange_relative_linear_speed"], command.metrics["flange_linear_velocity_error"]
        )
        torch.testing.assert_close(
            command.metrics["flange_angular_velocity_error"], torch.full((2,), 10., dtype=torch.float64),
            atol=1e-12, rtol=0,
        )

    def test_command_resample_does_not_introduce_velocity_difference_history(self):
        env = make_flange_env()
        velocity = torch.tensor([[.10, -.05, .025], [-.03, .02, -.01]], dtype=torch.float64)
        set_translating_command(env, velocity)
        command = env.command_manager.get_term("ee_target")
        before = rewards.flange_linear_velocity_residual(env)
        for value in (command.pose_start[:, :3], command.pose_target[:, :3],
                      command._curve_control_points):
            value += 100.
        command.time_left[:] = command._command_duration
        after = rewards.flange_linear_velocity_residual(env)
        torch.testing.assert_close(before, after, atol=1e-14, rtol=0)

    def test_partial_reset_does_not_transfer_flange_velocity_history(self):
        env = make_flange_env()
        pivot_about_fixed_tcp(env, .05, .15)
        before = rewards.flange_linear_velocity_residual(env)
        reset = make_flange_env()
        robot, reset_robot = env.scene["robot"], reset.scene["robot"]
        for name in ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "body_link_ang_vel_w"):
            getattr(robot.data, name)[0] = getattr(reset_robot.data, name)[0]
        after = rewards.flange_linear_velocity_residual(env)
        self.assertEqual(after[0].item(), 0.)
        torch.testing.assert_close(after[1], before[1])


if __name__ == "__main__":
    unittest.main()
