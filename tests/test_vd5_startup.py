"""Check VD5 startup handoff endpoints, asynchronous reset and profile migration."""

import copy
import math
import sys
import types
import unittest
import ast
from pathlib import Path
from unittest.mock import patch

import torch

from test_vd5_commands import MDP, command_class, curriculum_env, load, state


class CurveParent:
    def _update_command(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = torch.tensor([.2, 0., .1])


curve_module = types.ModuleType("vd5_cpu.commands.curve_velocity_command")
curve_module.CurveVelocityCommand = CurveParent
seam_module = types.ModuleType("vd5_cpu.commands.vd5_world_seam_command")
seam_module.VD5WorldSeamCommand = command_class
with patch.dict(sys.modules, {curve_module.__name__: curve_module, seam_module.__name__: seam_module}):
    startup = load("vd5_cpu.commands.vd5_startup_command", MDP / "commands/vd5_startup_command.py")


class StartupTests(unittest.TestCase):
    def test_ramp_endpoints_midpoint_and_shortest_quaternion(self):
        steps = torch.tensor([-1., 0., 50., 100., 200.])
        torch.testing.assert_close(startup.startup_fraction(steps, .02, 2.),
                                   torch.tensor([0., 0., .5, 1., 1.]))
        for duration in (0., -1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                startup.startup_fraction(steps, .02, duration)
        quat = torch.tensor([[1., 0., 0., 0.]])
        torch.testing.assert_close(startup.interpolate_quaternion(quat, -quat, torch.tensor([.5])), quat)

    def test_velocity_ramp_does_not_recursively_reduce_commands(self):
        command = startup.VD5StartupCurveCommand.__new__(startup.VD5StartupCurveCommand)
        command._env = types.SimpleNamespace(episode_length_buf=torch.tensor([0, 50, 100]), step_dt=.02)
        command.cfg = types.SimpleNamespace(vd5_startup_duration_s=2.)
        command.vel_command_b = torch.zeros(3, 3)
        command._update_command()
        expected = torch.tensor([[0., 0., 0.], [.1, 0., .05], [.2, 0., .1]])
        torch.testing.assert_close(command.vel_command_b, expected)
        command._update_command([1])
        torch.testing.assert_close(command.vel_command_b, expected)

    def command(self):
        command = startup.VD5StartupSeamCommand.__new__(startup.VD5StartupSeamCommand)
        command.num_envs, command.device, command.base_idx = 2, "cpu", 0
        command.cfg = types.SimpleNamespace(vd5_startup_duration_s=2.)
        data = types.SimpleNamespace(body_pos_w=torch.tensor([[[0., 0., .7]], [[2., 0., .7]]]),
            body_state_w=torch.tensor([[[0., 0., .7, 1., 0., 0., 0.]], [[2., 0., .7, 1., 0., 0., 0.]]]))
        path = types.SimpleNamespace(curve_points_w=torch.tensor([
            [[0., 0., 0.], [3., 0., 0.]], [[2., 0., 0.], [5., 0., 0.]]]))
        command.env = types.SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long), step_dt=.02,
            command_manager=types.SimpleNamespace(get_term=lambda _: path))
        command.robot = types.SimpleNamespace(data=data)
        command.pose_command = torch.tensor([[-1., 0., -.3, 1., 0., 0., 0.]]).repeat(2, 1)
        command.pose_target = command.pose_command.clone()
        command._vd5_target_pos_w = torch.zeros(2, 3)
        command.pose_target_pos_w = torch.zeros(2, 3)
        command.trailing_distance, command._vd5_ground_blend = .7, 1.
        command._startup_pos_w = torch.zeros(2, 3)
        command._startup_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
        command._startup_initialized = torch.zeros(2, dtype=torch.bool)
        command.metrics = {"startup_fraction": torch.zeros(2)}
        return command

    def test_handoff_captures_physical_tcp_and_only_restarts_for_reset_environment(self):
        command = self.command()
        position = torch.tensor([[-1., 0., .4], [1., 0., .4]])
        orientation = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
        with patch.object(startup, "tcp_pose_w", return_value=(position, orientation)):
            command._set_seam_target(slice(None))
            torch.testing.assert_close(command._vd5_target_pos_w, position)
            command.env.episode_length_buf[:] = 50
            command._set_seam_target(slice(None))
            torch.testing.assert_close(command._vd5_target_pos_w,
                                       torch.tensor([[-.85, 0., .2], [1.15, 0., .2]]))
            command.env.episode_length_buf[:] = 100
            command._set_seam_target(slice(None))
            torch.testing.assert_close(command._vd5_target_pos_w,
                                       torch.tensor([[-.7, 0., 0.], [1.3, 0., 0.]]))
            # A normal command resample cannot restart the per-episode handoff.
            command._set_seam_target(slice(None))
            torch.testing.assert_close(command.metrics["startup_fraction"], torch.ones(2))
            command.env.episode_length_buf[0] = 0
            position[0] = torch.tensor([-.9, .1, .3])
            command._set_seam_target(torch.tensor([0]))
            torch.testing.assert_close(command._vd5_target_pos_w[0], position[0])
            torch.testing.assert_close(command._vd5_target_pos_w[1], torch.tensor([1.3, 0., 0.]))
            torch.testing.assert_close(command.metrics["startup_fraction"], torch.tensor([0., 1.]))


class StartupProfileTests(unittest.TestCase):
    def test_training_profile_uses_eighty_percent_reference_initialization(self):
        config_path = MDP.parent / "b2_z1_vd5_startup_env_cfg.py"
        tree = ast.parse(config_path.read_text())
        configure = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "_configure_startup")
        probability = next(node.value for node in configure.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Attribute)
                                   and isinstance(target.value, ast.Name)
                                   and target.value.id == "cfg"
                                   and target.attr == "reference_state_initialization_prob"
                                   for target in node.targets))
        self.assertEqual(ast.literal_eval(probability), 0.8)
        self.assertIn("--run_name vd5_startup_80ref_20stand", Path(__file__).resolve().parents[1]
                      .joinpath("scripts/train_vd5_startup.sh").read_text())

    def env(self):
        env, base, arm = curriculum_env()
        event = types.SimpleNamespace(params={"velocity_range": (0.0, 0.0)})
        robot = types.SimpleNamespace(init_state=types.SimpleNamespace(pos=(0.0, 0.0, .50)))
        env.cfg = types.SimpleNamespace(reference_state_initialization_prob=.5,
            vd5_nominal_base_height_m=.50,
            scene=types.SimpleNamespace(robot=robot), events=types.SimpleNamespace(reset_robot_joints=event))
        base.cfg.vd5_startup_duration_s = arm.cfg.vd5_startup_duration_s = 2.
        return env

    def test_original_curriculum_requires_explicit_migration_and_retains_progress(self):
        original, _, arm = curriculum_env()
        arm._vd5_ground_blend = .78048
        saved = state.capture_vd5_curriculum_state(original)
        target = self.env()
        before = state.capture_vd5_curriculum_state(target)
        with self.assertRaisesRegex(ValueError, "startup profile"):
            state.restore_vd5_curriculum_state(target, saved)
        self.assertEqual(state.capture_vd5_curriculum_state(target), before)
        state.restore_vd5_curriculum_state(target, saved, allow_startup_migration=True)
        current = state.capture_vd5_curriculum_state(target)
        self.assertEqual(current["ground_blend"], .78048)
        self.assertEqual(current["common_curriculum"], saved["common_curriculum"])
        self.assertEqual(current["startup_profile"]["duration_s"], 2.)
        self.assertEqual(current["startup_profile"]["nominal_base_height_m"], .50)

    def test_changed_startup_profile_is_rejected_even_during_migration(self):
        env = self.env()
        saved = state.capture_vd5_curriculum_state(env)
        for key, value in (("duration_s", 4.), ("reference_initialization_prob", 1.),
                           ("nominal_base_height_m", .55),
                           ("nominal_joint_velocity_range", [-1., 1.])):
            bad = copy.deepcopy(saved)
            bad["startup_profile"][key] = value
            with self.assertRaisesRegex(ValueError, "startup profile"):
                state.restore_vd5_curriculum_state(env, bad, allow_startup_migration=True)
            self.assertEqual(state.capture_vd5_curriculum_state(env), saved)


if __name__ == "__main__":
    unittest.main()
