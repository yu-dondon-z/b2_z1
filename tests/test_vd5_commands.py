"""CPU checks for VD5 geometry, curriculum and gun-axis reward semantics."""

import ast
import copy
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_vd2_curriculum_state import make_env


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"
UTILS = ROOT / "source/b2_z1/b2_z1/rsl_rl/utils"


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package("vd5_cpu", MDP)
package("vd5_cpu.commands", MDP / "commands")
package("vd5_utils_cpu", UTILS)
geometry = load("vd5_cpu.commands.vd5_geometry", MDP / "commands/vd5_geometry.py")
rewards = load("vd5_cpu.vd5_rewards", MDP / "vd5_rewards.py")
state = load("vd5_utils_cpu.vd5_curriculum_state", UTILS / "vd5_curriculum_state.py")


class NominalCommand:
    def _update_command(self):
        self.pose_command[:, :3] = self.nominal


parent = types.ModuleType("vd5_cpu.commands.pose_command_wbc")
parent.UniformPoseWBCCommand = NominalCommand
with patch.dict(sys.modules, {parent.__name__: parent}):
    command_class = load("vd5_cpu.commands.vd5_world_seam_command",
                         MDP / "commands/vd5_world_seam_command.py").VD5WorldSeamCommand


def curriculum_env(counter=0):
    env, base, arm = make_env(counter)
    arm.cfg.vd5_ground_transition_steps = 100_000
    arm.cfg.vd5_seam_trailing_distance = 0.70
    arm._vd5_ground_blend = 0.25
    arm._vd5_last_ground_curriculum_step = counter
    return env, base, arm


class VD5GeometryTests(unittest.TestCase):
    def test_world_base_roundtrip_with_full_rotation(self):
        quat = torch.tensor([[.5, .5, .5, .5]], dtype=torch.float64)
        base = torch.tensor([[5., -3., .7]], dtype=torch.float64)
        point = torch.tensor([[-.7, .1, -.7]], dtype=torch.float64)
        world = geometry.base_position_to_world(point, base, quat)
        torch.testing.assert_close(geometry.world_position_to_base(world, base, quat), point)

    def test_trailing_point_follows_polyline_arc_and_extends_start(self):
        path = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]])
        point = geometry.sample_polyline_behind_position(path, torch.tensor([[1., .5, .7]]), .7)
        torch.testing.assert_close(point, torch.tensor([[.8, 0., 0.]]))
        point = geometry.sample_polyline_behind_position(path, torch.tensor([[0., 0., .7]]), .7)
        torch.testing.assert_close(point, torch.tensor([[-.7, 0., 0.]]))

    def test_blend_endpoints_and_invalid_inputs(self):
        nominal, seam = torch.tensor([[-1., 0., -.3]]), torch.tensor([[-.7, 0., -.7]])
        for blend in (0., .25, 1.):
            torch.testing.assert_close(geometry.blend_target_position(nominal, seam, blend),
                                       (1-blend)*nominal + blend*seam)
        for blend in (-.01, 1.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                geometry.blend_target_position(nominal, seam, blend)

    def test_real_command_blends_from_nominal_without_recursive_blending(self):
        command = command_class.__new__(command_class)
        data = types.SimpleNamespace(body_pos_w=torch.tensor([[[0., 0., .7]]]),
                                     body_state_w=torch.tensor([[[0., 0., .7, 1., 0., 0., 0.]]]))
        path = types.SimpleNamespace(curve_points_w=torch.tensor([[[0., 0., 0.], [3., 0., 0.]]]))
        command.env = types.SimpleNamespace(common_step_counter=0,
            command_manager=types.SimpleNamespace(get_term=lambda _: path))
        command.robot = types.SimpleNamespace(data=data)
        command.base_idx = 0
        command.pose_command = torch.tensor([[-1., 0., -.3, 1., 0., 0., 0.]])
        command.pose_target = command.pose_command.clone()
        command.nominal = command.pose_command[:, :3].clone()
        command._vd5_target_pos_w = torch.zeros(1, 3)
        command.pose_target_pos_w = torch.zeros(1, 3)
        command.trailing_distance = .7
        command._vd5_ground_transition_steps = 100_000
        command._vd5_last_ground_curriculum_step = 0
        command._vd5_ground_blend = 0.
        command.env.common_step_counter = 50_000
        command._update_command()
        expected = torch.tensor([[-.85, 0., -.5]])
        torch.testing.assert_close(command.pose_command[:, :3], expected)
        command._update_command()
        torch.testing.assert_close(command.pose_command[:, :3], expected)
        torch.testing.assert_close(command._vd5_target_pos_w, torch.tensor([[-.85, 0., .2]]))
        command.env.common_step_counter = 150_000
        command._update_command()
        self.assertEqual(command._vd5_ground_blend, 1.)
        torch.testing.assert_close(command.pose_command[:, :3], torch.tensor([[-.7, 0., -.7]]))


class VD5RewardTests(unittest.TestCase):
    def test_roll_about_gun_axis_is_free_but_axis_tilt_is_penalized(self):
        current = torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        desired = torch.tensor([[math.sqrt(.5), math.sqrt(.5), 0., 0.], [1., 0., 0., 0.]])
        command = types.SimpleNamespace(metrics={})
        env = types.SimpleNamespace(scene={"robot": object()}, command_manager=types.SimpleNamespace(
            get_command=lambda _: torch.cat((torch.zeros(2, 3), desired), dim=-1),
            get_term=lambda _: command))
        with patch.object(rewards, "tcp_pose_b", return_value=(torch.zeros(2, 3), current)):
            result = rewards.tcp_gun_axis_command_error(env)
        torch.testing.assert_close(result, torch.tensor([0., math.pi/2]))
        torch.testing.assert_close(command.metrics["gun_axis_error"], result)


class VD5CurriculumTests(unittest.TestCase):
    def test_runner_keeps_leg_mirror_profile_under_separate_experiment(self):
        tree = ast.parse((MDP.parent / "agents/rsl_rl_ppo_cfg.py").read_text())
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VD5PPORunnerCfg")
        self.assertEqual([n.id for n in runner.bases], ["VD3LegSymmetryRunnerCfg"])
        fields = {n.targets[0].id: ast.literal_eval(n.value) for n in runner.body if isinstance(n, ast.Assign)}
        self.assertEqual(fields["experiment_name"], "b2_z1_vd5_world_seam_amp")
        self.assertEqual(fields["max_iterations"], 20000)

    def test_resume_keeps_blend_and_rebases_step_counter(self):
        source, _, arm = curriculum_env(123)
        arm._vd5_ground_blend = .43
        saved = state.capture_vd5_curriculum_state(source)
        target, _, restored = curriculum_env(4)
        state.restore_vd5_curriculum_state(target, saved)
        self.assertEqual(restored._vd5_ground_blend, .43)
        self.assertEqual(restored._vd5_last_ground_curriculum_step, 4)
        self.assertEqual(state.capture_vd5_curriculum_state(target), saved)

    def test_invalid_or_incompatible_state_is_rejected_without_mutation(self):
        env, _, _ = curriculum_env()
        before = state.capture_vd5_curriculum_state(env)
        for key, value in (("ground_blend", math.nan), ("ground_blend", True),
                           ("ground_blend", -1), ("ground_blend", 1.1),
                           ("ground_transition_steps", 500), ("seam_trailing_distance_m", 1.0),
                           ("common_curriculum", None)):
            bad = copy.deepcopy(before)
            bad[key] = value
            with self.assertRaises((ValueError, TypeError)):
                state.restore_vd5_curriculum_state(env, bad)
            self.assertEqual(state.capture_vd5_curriculum_state(env), before)

    def test_train_and_play_configuration_are_vd5_scoped(self):
        tree = ast.parse((MDP.parent / "b2_z1_vd5_env_cfg.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_vd5")
        namespace = {"VD5WorldSeamCommand": command_class, "mdp": rewards}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "vd5_config", "exec"), namespace)
        def config():
            return types.SimpleNamespace(episode_length_s=30.0,
                commands=types.SimpleNamespace(ee_target=types.SimpleNamespace(),
                                              base_velocity=types.SimpleNamespace()),
                rewards=types.SimpleNamespace(**{name: types.SimpleNamespace(weight=-.2) for name in (
                    "end_effector_orientation_error", "flange_position_residual",
                    "flange_linear_velocity_residual", "flange_angular_velocity_residual")}))
        train, play = config(), config()
        namespace["_configure_vd5"](train)
        namespace["_configure_vd5"](play, play=True)
        self.assertEqual(train.commands.ee_target.vd5_ground_transition_steps, 100_000)
        self.assertEqual(play.commands.ee_target.vd5_ground_transition_steps, 0)
        self.assertEqual(train.commands.ee_target.vd5_seam_trailing_distance, .7)
        self.assertEqual(train.commands.base_velocity.resampling_time_range, (31., 31.))
        self.assertEqual(play.commands.base_velocity.resampling_time_range, (31., 31.))
        self.assertEqual(train.rewards.flange_linear_velocity_residual.weight, 0.)


if __name__ == "__main__":
    unittest.main()
