"""CPU regression checks for curriculum admission, without launching Isaac Sim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

source = Path(__file__).resolve().parents[2] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/curriculums.py"
spec = importlib.util.spec_from_file_location("curve_curriculum", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CurriculumAdmissionTests(unittest.TestCase):
    def make_env(self, failures=6, length=0.93, penalty=-0.02):
        command = NS(cfg=NS(mode="synthetic", synthetic_world_fixed_path=True, synthetic_path_family="straight_arc"))
        env = NS(
            device="cpu", common_step_counter=1, max_episode_length_s=60,
            max_episode_length=3000,
            command_manager=NS(get_term=lambda name: command),
            reward_manager=NS(
                _episode_sums={"track": torch.full((100,), 0.9 * 60), "penalty": torch.full((100,), penalty * 60)},
                get_term_cfg=lambda name: NS(weight=1.0 if name == "track" else -1.0),
            ),
            termination_manager=NS(terminated=torch.arange(100) < failures),
            episode_length_buf=torch.full((100,), length * 3000),
        )
        return env, command

    def update(self, env, staged=True, **extra):
        params = dict(
            reward_term_names=("track", "penalty"), reward_thresholds=(0.7, 0.1),
            minimum_evaluation_episodes=100, successful_evaluations_required=2,
            max_terminated_fraction=0.01, min_mean_episode_length_fraction=0.95,
            emit_diagnostics=True,
        )
        if staged:
            params.update(max_terminated_fractions=(0.08, 0.06, 0.04, 0.03),
                          min_mean_episode_length_fractions=(0.90, 0.92, 0.94, 0.95))
        params.update(extra)
        return module.truth_curve_levels(env, list(range(100)), **params)

    def test_old_gate_blocks_observed_quality(self):
        env, _ = self.make_env()
        result = self.update(env, staged=False)
        self.assertEqual(result["level"], 0)
        self.assertEqual(result["survival_passed"], 0)
        self.assertEqual(result["length_passed"], 0)
        self.assertEqual(result["blocked_reward_count"], 0)

    def test_two_batches_promote_and_later_stage_is_stricter(self):
        env, command = self.make_env()
        self.assertEqual(self.update(env)["level"], 0)
        self.assertEqual(self.update(env)["level"], 1)
        self.assertEqual(command.cfg.synthetic_arc_radius_range_m, (12.0, 20.0))
        env.termination_manager.terminated = torch.arange(100) < 7
        result = self.update(env)
        self.assertEqual(result["level"], 1)
        self.assertEqual(result["survival_passed"], 0)
        self.assertEqual(result["success_streak"], 0)

    def test_contact_penalty_still_blocks(self):
        env, _ = self.make_env(penalty=-0.2)
        result = self.update(env)
        self.assertEqual(result["blocked_reward_count"], 1)
        self.assertLess(result["margin_penalty"], 0)
        self.assertEqual(result["batch_passed"], 0)

    def test_initial_reset_not_counted(self):
        env, _ = self.make_env()
        env.common_step_counter = 0
        self.assertEqual(self.update(env)["pending_episodes"], 0)

    def test_legacy_scalar_and_invalid_schedule(self):
        env, _ = self.make_env()
        self.assertIsInstance(self.update(env, staged=False, emit_diagnostics=False), torch.Tensor)
        with self.assertRaises(ValueError):
            self.update(env, max_terminated_fractions=(0.08,))


if __name__ == "__main__":
    unittest.main()
