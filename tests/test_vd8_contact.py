import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contact = load('vd8_contact_test', ROOT / 'source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vd8_contact.py')
options = load('vd8_options_test', ROOT / 'scripts/rsl_rl/training_options.py')


class VD8ContactTests(unittest.TestCase):
    def test_contact_onset_and_next_actor_interval_only(self):
        history = torch.tensor([[[80., 390., 0., 0.], [120., 370., 0., 0.],
                                 [500., 100., 0., 0.], [0., 0., 0., 0.],
                                 [0., 0., 0., 0.]]])
        penalty, onset = contact.onset_force_penalty(history, torch.zeros(1, 4, dtype=torch.bool))
        self.assertTrue(onset[0, 0])
        self.assertTrue(onset[0, 1])
        self.assertAlmostEqual(penalty.item(), 1.)
        stance = torch.full((1, 5, 4), 500.)
        p_next, no_new = contact.onset_force_penalty(stance, onset)
        self.assertAlmostEqual(p_next.item(), 1.)
        p_later, _ = contact.onset_force_penalty(stance, no_new)
        self.assertEqual(p_later.item(), 0.)

    def test_no_false_onset_from_reset_or_weak_contact(self):
        history = torch.full((2, 5, 4), 400.)
        history[1] = 8.
        penalty, onset = contact.onset_force_penalty(history, torch.zeros(2, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(penalty, torch.zeros(2)))
        self.assertFalse(onset.any())

    def test_migration_and_resume_are_exclusive(self):
        runner, env = Mock(), Mock()
        options.load_training_checkpoint(runner, env, 'b2_z1_vd8_roll_landing_amp', vd8_from='vd5.pt')
        runner.load.assert_called_once_with('vd5.pt', restore_curriculum=True, migrate_vd8=True)
        env.reset.assert_called_once()
        runner.reset_mock()
        options.load_training_checkpoint(runner, env, 'b2_z1_vd8_roll_landing_amp', resume_path='vd8.pt')
        runner.load.assert_called_once_with('vd8.pt', restore_curriculum=True, legacy_curriculum=None)
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner, env, 'b2_z1_vd8_roll_landing_amp',
                                             vd8_from='vd5.pt', resume_path='vd8.pt')


if __name__ == '__main__':
    unittest.main()
