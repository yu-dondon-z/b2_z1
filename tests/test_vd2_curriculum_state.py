"""CPU-only tests for VD2 curriculum checkpoint state."""

import copy
import contextlib
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/b2_z1/b2_z1/rsl_rl/utils/curriculum_state.py"
)
spec = importlib.util.spec_from_file_location("vd2_curriculum_state", MODULE)
curriculum_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum_state)


AXES = ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")


def make_env(counter=0):
    ranges = SimpleNamespace(
        pos_x=(-1.03, -1.01),
        pos_y=(-0.02, 0.02),
        pos_z=(-0.31, -0.29),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )
    limits = SimpleNamespace(
        pos_x=(-1.08, -0.96),
        pos_y=(-0.15, 0.15),
        pos_z=(-0.34, -0.26),
        roll=(-0.2, 0.2),
        pitch=(-0.2, 0.2),
        yaw=(-0.2, 0.2),
    )
    base = SimpleNamespace(
        cfg=SimpleNamespace(trajectory_curve_offset_scale=0.0, limit_trajectory_curve_offset_scale=0.1),
        _last_curve_curriculum_step=0,
    )
    arm = SimpleNamespace(
        cfg=SimpleNamespace(ranges=ranges, limit_ranges=limits),
        _last_pos_curriculum_step=0,
    )
    commands = {"base_velocity": base, "ee_target": arm}
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=commands.__getitem__),
        common_step_counter=counter,
        max_episode_length=100,
    )
    env.unwrapped = env
    return env, base, arm


class CurriculumStateTests(unittest.TestCase):
    def test_restores_all_ranges_and_relative_curriculum_timers(self):
        source, base, arm = make_env(counter=140)
        base.cfg.trajectory_curve_offset_scale = 0.06
        base._last_curve_curriculum_step = 112
        arm._last_pos_curriculum_step = 98
        expected = {
            "pos_x": (-1.06, -0.98), "pos_y": (-0.08, 0.08),
            "pos_z": (-0.34, -0.26), "roll": (-0.1, 0.1),
            "pitch": (-0.05, 0.05), "yaw": (-0.2, 0.2),
        }
        for name, values in expected.items():
            setattr(arm.cfg.ranges, name, values)

        state = curriculum_state.capture_curriculum_state(source)
        # Mutating live config must not mutate the checkpoint's stored state.
        arm.cfg.ranges.pos_y = (-0.01, 0.01)
        target, target_base, target_arm = make_env(counter=3)
        curriculum_state.restore_curriculum_state(target, state)

        self.assertEqual(target.common_step_counter, 3)
        self.assertAlmostEqual(target_base.cfg.trajectory_curve_offset_scale, 0.06)
        for name, values in expected.items():
            self.assertEqual(tuple(getattr(target_arm.cfg.ranges, name)), values)
        self.assertEqual(target.common_step_counter - target_base._last_curve_curriculum_step, 28)
        self.assertEqual(target.common_step_counter - target_arm._last_pos_curriculum_step, 42)
        self.assertEqual(target.max_episode_length - 28, 72)
        self.assertEqual(target.max_episode_length - 42, 58)

    def test_restore_rejects_missing_or_invalid_state_without_mutation(self):
        target, base, arm = make_env(counter=25)
        before = curriculum_state.capture_curriculum_state(target)
        nonfinite = copy.deepcopy(before)
        nonfinite["curve_level"] = math.nan
        beyond_limit = copy.deepcopy(before)
        beyond_limit["position_ranges"]["pos_y"] = [-0.16, 0.02]
        inverted = copy.deepcopy(before)
        inverted["position_ranges"]["pos_z"] = [-0.29, -0.31]
        negative_timer = copy.deepcopy(before)
        negative_timer["position_elapsed_steps"] = -1
        for bad in (None, {}, {"version": -1}, nonfinite, beyond_limit, inverted, negative_timer):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    curriculum_state.restore_curriculum_state(target, bad)
                self.assertEqual(curriculum_state.capture_curriculum_state(target), before)
        self.assertEqual(base.cfg.trajectory_curve_offset_scale, 0.0)
        self.assertEqual(arm.cfg.ranges.pos_x, (-1.03, -1.01))

    def test_malformed_six_axis_ranges_do_not_partially_restore(self):
        source, base, arm = make_env(counter=73)
        base.cfg.trajectory_curve_offset_scale = 0.06
        arm.cfg.ranges.pos_y = (-0.08, 0.08)
        state = curriculum_state.capture_curriculum_state(source)
        for axis in AXES:
            for values in ([math.nan, 0.0], [1.0, -1.0], [-100.0, 100.0], [0.0], None):
                target, _, _ = make_env(counter=6)
                before = curriculum_state.capture_curriculum_state(target)
                bad = copy.deepcopy(state)
                bad["position_ranges"][axis] = values
                with self.subTest(axis=axis, values=values):
                    with self.assertRaises((ValueError, TypeError)):
                        curriculum_state.restore_curriculum_state(target, bad)
                    self.assertEqual(curriculum_state.capture_curriculum_state(target), before)

    def test_invalid_curves_and_timers_rejected_atomically(self):
        source, base, _ = make_env(counter=50)
        base.cfg.trajectory_curve_offset_scale = 0.06
        state = curriculum_state.capture_curriculum_state(source)
        invalid = [("curve_level", value) for value in (math.nan, math.inf, -math.inf, -0.01, 0.11)]
        invalid += [(name, value) for name in ("curve_elapsed_steps", "position_elapsed_steps")
                    for value in (True, False, -1, 3.5, "4")]
        for key, value in invalid:
            target, _, _ = make_env(counter=9)
            before = curriculum_state.capture_curriculum_state(target)
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises((ValueError, TypeError)):
                    curriculum_state.restore_curriculum_state(target, bad)
                self.assertEqual(curriculum_state.capture_curriculum_state(target), before)

    def test_legacy_approximation_expands_and_clamps_position_ranges(self):
        target, base, arm = make_env(counter=0)
        state = curriculum_state.legacy_curriculum_state(target, 0.06, 5)
        curriculum_state.restore_curriculum_state(target, state)
        self.assertAlmostEqual(base.cfg.trajectory_curve_offset_scale, 0.06)
        for actual, expected in zip(arm.cfg.ranges.pos_x, (-1.08, -0.96)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(arm.cfg.ranges.pos_y, (-0.07, 0.07)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(arm.cfg.ranges.pos_z, (-0.34, -0.26)):
            self.assertAlmostEqual(actual, expected)
        for axis in ("roll", "pitch", "yaw"):
            self.assertEqual(tuple(getattr(arm.cfg.ranges, axis)), (0.0, 0.0))
        self.assertEqual(target.common_step_counter - base._last_curve_curriculum_step, 0)
        self.assertEqual(target.common_step_counter - arm._last_pos_curriculum_step, 0)

    def test_legacy_approximation_rejects_invalid_inputs(self):
        for curve, steps in ((-0.01, 5), (0.11, 5), (math.nan, 5), (0.06, -1)):
            target, _, _ = make_env()
            before = copy.deepcopy(curriculum_state.capture_curriculum_state(target))
            with self.subTest(curve=curve, steps=steps):
                with self.assertRaises((ValueError, TypeError)):
                    curriculum_state.legacy_curriculum_state(target, curve, steps)
                self.assertEqual(curriculum_state.capture_curriculum_state(target), before)


class CurriculumRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from test_amp import AMP_OnPolicyRunner, FakeEnv, runner_config

        cls.torch = torch
        cls.runner_class = AMP_OnPolicyRunner
        cls.fake_env_class = FakeEnv
        cls.runner_config = staticmethod(runner_config)

    def runner(self, counter=0):
        env, _, _ = make_env(counter)
        fake = self.fake_env_class()
        fake.command_manager = env.command_manager
        fake.common_step_counter = counter
        cfg = self.runner_config()
        cfg["experiment_name"] = "b2_z1_vd2_curve_amp"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = self.runner_class(fake, cfg, device="cpu")
        return runner

    def test_checkpoint_save_and_load_restore_curriculum(self):
        runner = self.runner(counter=140)
        base = runner.env.command_manager.get_term("base_velocity")
        arm = runner.env.command_manager.get_term("ee_target")
        base.cfg.trajectory_curve_offset_scale = 0.06
        base._last_curve_curriculum_step = 112
        arm.cfg.ranges.pos_y = (-0.08, 0.08)
        arm._last_pos_curriculum_step = 98
        runner.current_learning_iteration = 600
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            runner.save(path)
            saved = self.torch.load(path, weights_only=False)
            self.assertIn("vd2_curriculum_state", saved)
            target = self.runner(counter=3)
            with contextlib.redirect_stdout(io.StringIO()):
                target.load(path, restore_curriculum=True)
        self.assertEqual(target.current_learning_iteration, 600)
        self.assertEqual(curriculum_state.capture_curriculum_state(target.env), saved["vd2_curriculum_state"])

    def test_legacy_checkpoint_requires_explicit_reconstruction(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "legacy.pt")
            runner.save(path)
            saved = self.torch.load(path, weights_only=False)
            saved.pop("vd2_curriculum_state")
            self.torch.save(saved, path)
            target = self.runner()
            with self.assertRaisesRegex(ValueError, "no VD2 curriculum state"):
                target.load(path, restore_curriculum=True)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                target.load(path, restore_curriculum=True, legacy_curriculum=(0.06, 5))
            self.assertIn("approximate", output.getvalue())
        base = target.env.command_manager.get_term("base_velocity")
        arm = target.env.command_manager.get_term("ee_target")
        self.assertAlmostEqual(base.cfg.trajectory_curve_offset_scale, 0.06)
        self.assertAlmostEqual(arm.cfg.ranges.pos_y[1], 0.07)

    def test_default_load_and_warm_start_leave_curriculum_config_unchanged(self):
        runner = self.runner()
        runner.env.command_manager.get_term("base_velocity").cfg.trajectory_curve_offset_scale = 0.06
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            runner.save(path)
            for load_optimizer in (True, False):
                target = self.runner(counter=5)
                before = curriculum_state.capture_curriculum_state(target.env)
                with self.subTest(load_optimizer=load_optimizer):
                    target.load(path, load_optimizer=load_optimizer)
                    self.assertEqual(curriculum_state.capture_curriculum_state(target.env), before)

    def test_saved_curriculum_rejects_legacy_override(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            runner.save(path)
            target = self.runner()
            before = curriculum_state.capture_curriculum_state(target.env)
            with self.assertRaisesRegex(ValueError, "cannot override"):
                target.load(path, restore_curriculum=True, legacy_curriculum=(0.06, 5))
            self.assertEqual(curriculum_state.capture_curriculum_state(target.env), before)


if __name__ == "__main__":
    unittest.main()
