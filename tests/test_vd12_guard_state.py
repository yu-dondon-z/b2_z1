"""CPU-only temporal regression tests; never starts the simulator."""

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vd12_guard_state", ROOT / "source/b2_z1/b2_z1/assets/vd12_guard_state.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GeometryWindow = MODULE.GeometryWindow


class GuardStateTests(unittest.TestCase):
    def feed(self, state, values, window=0, start=1):
        for step, value in enumerate(values, start):
            state.sample(torch.tensor(value), torch.ones(len(value), dtype=torch.bool),
                         window_id=window, physics_step=step)

    def test_transient_substep_violation_is_latched(self):
        state = GeometryWindow(2)
        self.feed(state, [[.05, .05], [-.001, .05], [.05, .05], [.05, .05]])
        self.assertEqual(state.violated.tolist(), [True, False])
        self.assertEqual(state.first_bad_physics_step.tolist(), [2, 0])
        state.require_complete(4)

    def test_cost_uses_all_substeps_and_readers_do_not_clear(self):
        state = GeometryWindow(1)
        self.feed(state, [[.03], [.0], [.03], [.0]])
        torch.testing.assert_close(state.mean_cost(4), torch.tensor([.5]))
        torch.testing.assert_close(state.mean_cost(4), torch.tensor([.5]))

    def test_new_window_clears_old_failure(self):
        state = GeometryWindow(1)
        self.feed(state, [[-.1]] * 4)
        self.feed(state, [[.05]] * 4, window=1, start=5)
        self.assertEqual(state.violated.tolist(), [False])
        self.assertEqual(state.first_bad_physics_step.tolist(), [0])
        self.assertEqual(state.mean_cost(8).tolist(), [0.])

    def test_partial_episode_reset_does_not_clear_other_environment(self):
        state = GeometryWindow(2)
        self.feed(state, [[-.01, -.02]] * 4)
        state.reset(torch.tensor([0]))
        self.assertEqual(state.violated.tolist(), [False, True])
        self.assertEqual(state.sample_count, 4)
        self.assertEqual(state.first_bad_physics_step.tolist(), [0, 1])

    def test_missing_duplicate_or_extra_sample_rejected(self):
        state = GeometryWindow(1)
        self.feed(state, [[.05]])
        with self.assertRaises(RuntimeError):
            state.require_complete(1)
        with self.assertRaises(RuntimeError):
            self.feed(state, [[.05]], start=3)
        self.feed(state, [[.05]] * 3, start=2)
        with self.assertRaises(RuntimeError):
            self.feed(state, [[.05]], start=5)

    def test_nonfinite_or_invalid_input_fails_closed_without_nan_reward(self):
        state = GeometryWindow(2)
        state.sample(torch.tensor([float("nan"), .05]), torch.tensor([True, False]),
                     window_id=0, physics_step=1)
        self.assertEqual(state.violated.tolist(), [True, True])
        self.assertTrue(torch.isfinite(state.total_cost).all())

    def test_margin_is_not_invented_physical_contact(self):
        state = GeometryWindow(1)
        self.feed(state, [[.015]] * 4)
        self.assertFalse(state.violated.item())
        torch.testing.assert_close(state.mean_cost(4), torch.tensor([.25]))

    def test_invalid_contract_rejected(self):
        for kwargs in ({"soft_margin_m": 0}, {"hard_threshold_m": .04},
                       {"cost_cap": float("nan")}, {"decimation": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeometryWindow(1, **kwargs)


if __name__ == "__main__":
    unittest.main()
