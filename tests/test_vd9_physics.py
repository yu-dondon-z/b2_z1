"""Physics-rate reward and warm-start contracts for VD9."""

import ast
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


physics = load(
    "vd9_physics_test",
    ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vd9_physics.py",
)
options = load("vd9_options_test", ROOT / "scripts/rsl_rl/training_options.py")


class VD9PhysicsTests(unittest.TestCase):
    def test_vd9_keeps_physical_startup_time_and_records_initial_curriculum(self):
        source = (ROOT / "scripts/rsl_rl/train.py").read_text()
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "STARTUP_EXPERIMENTS"
                                  for target in node.targets))
        self.assertIn("b2_z1_vd9_physics_rate_amp", ast.literal_eval(assignment.value))
        self.assertIn("init_at_random_ep_len=agent_cfg.experiment_name not in STARTUP_EXPERIMENTS", source)
        self.assertIn('agent_cfg.experiment_name in STARTUP_EXPERIMENTS:', source)

    def test_highpass_is_causal_and_responds_at_physics_rate(self):
        lowpass = torch.zeros(1, 3)
        valid = torch.zeros(1, dtype=torch.bool)
        active = torch.ones(1, dtype=torch.bool)
        penalties = []
        for tick in range(40):
            error = torch.tensor([[0.008 if tick % 2 else -0.008, 0.0, 0.0]])
            penalty, lowpass, valid = physics.highpass_tracking_penalty(
                error, lowpass, valid, active, 0.005
            )
            penalties.append(penalty.item())
        self.assertGreater(sum(penalties[1:]) / len(penalties[1:]), 1.0)
        steady = torch.full((1, 3), 0.01)
        for _ in range(1000):
            penalty, lowpass, valid = physics.highpass_tracking_penalty(
                steady, lowpass, valid, active, 0.005
            )
        self.assertLess(penalty.item(), 1e-6)

    def test_highpass_seeds_history_before_active_window(self):
        lowpass = torch.zeros(1, 3)
        valid = torch.zeros(1, dtype=torch.bool)
        error = torch.full((1, 3), 0.02)
        inactive = torch.zeros(1, dtype=torch.bool)
        active = torch.ones(1, dtype=torch.bool)
        penalty, lowpass, valid = physics.highpass_tracking_penalty(
            error, lowpass, valid, inactive, 0.005
        )
        self.assertEqual(penalty.item(), 0.0)
        penalty, _, _ = physics.highpass_tracking_penalty(
            error, lowpass, valid, active, 0.005
        )
        self.assertEqual(penalty.item(), 0.0)

    def test_descent_penalty_is_near_ground_airborne_and_velocity_conditioned(self):
        height = torch.tensor([[0.08, 0.08, 0.20, 0.08]])
        velocity = torch.tensor([[-1.0, -1.0, -1.0, 0.1]])
        contact = torch.tensor([[False, True, False, False]])
        result = physics.descending_foot_penalty(height, velocity, contact)
        self.assertAlmostEqual(result[0, 0].item(), 1.69, places=6)
        self.assertEqual(result[0, 1].item(), 0.0)
        self.assertEqual(result[0, 2].item(), 0.0)
        self.assertEqual(result[0, 3].item(), 0.0)

    def test_warm_start_resets_optimizer_iteration_but_restores_startup_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 4498
        options.load_training_checkpoint(
            runner, env, "b2_z1_vd9_physics_rate_amp", vd9_from="vd5_4498.pt"
        )
        runner.load.assert_called_once_with(
            "vd5_4498.pt", load_optimizer=False, restore_curriculum=True, migrate_vd9=True
        )
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once()
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(
                runner, env, "b2_z1_vd8_roll_landing_amp", vd9_from="vd5_4498.pt"
            )
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(
                runner, env, "b2_z1_vd9_physics_rate_amp",
                vd9_from="vd5_4498.pt", resume_path="vd9.pt",
            )

    def test_vd9_checkpoint_warm_start_resets_optimizer_iteration_and_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 1500
        options.load_training_checkpoint(
            runner, env, "b2_z1_vd9_physics_rate_amp", vd9_warm_start="vd9_1500.pt"
        )
        runner.load.assert_called_once_with(
            "vd9_1500.pt", load_optimizer=False, restore_curriculum=False
        )
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once()
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(
                runner, env, "b2_z1_vd8_roll_landing_amp", vd9_warm_start="vd9_1500.pt"
            )
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(
                runner, env, "b2_z1_vd9_physics_rate_amp",
                vd9_warm_start="vd9_1500.pt", resume_path="vd9.pt",
            )

    def test_physics_callback_runs_after_each_scene_update(self):
        path = ROOT / "source/b2_z1/b2_z1/envs/b2z1_rl_env.py"
        tree = ast.parse(path.read_text())
        step = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "step")
        updates = [node for node in ast.walk(step)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "update"]
        self.assertTrue(updates)
        source = ast.get_source_segment(path.read_text(), step)
        self.assertIn("for callback in tuple(self._physics_step_callbacks)", source)


if __name__ == "__main__":
    unittest.main()
