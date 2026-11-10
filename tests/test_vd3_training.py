"""CPU checks for VD3 warm start, exact resume and bounded checkpoint storage."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


options = load_file("vd3_training_options", "scripts/rsl_rl/training_options.py")
cli = load_file("vd3_training_cli", "scripts/rsl_rl/cli_args.py")
retention = load_file("vd3_checkpoint_retention", "source/b2_z1/b2_z1/rsl_rl/utils/checkpoint_retention.py")


class LaunchOptionsTests(unittest.TestCase):
    def test_experiment_name_is_applied(self):
        cfg = SimpleNamespace(experiment_name="old", logger="tensorboard")
        args = SimpleNamespace(resume=False, load_run=None, checkpoint=None, run_name=None,
                               experiment_name="b2_z1_vd3_curve_amp", logger=None)
        self.assertEqual(cli.update_rsl_rl_cfg(cfg, args).experiment_name, "b2_z1_vd3_curve_amp")

    def test_warm_start_resets_iteration_and_uses_fresh_optimizer_and_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 2050
        options.load_training_checkpoint(runner, env, "b2_z1_vd3_curve_amp", warm_start="old.pt")
        runner.load.assert_called_once_with("old.pt", load_optimizer=False, restore_curriculum=False)
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once_with()

    def test_vd3_and_vd2_resume_restore_curriculum_and_reset_environment(self):
        for experiment in ("b2_z1_vd3_curve_amp", "b2_z1_vd2_curve_amp"):
            runner, env = Mock(), Mock()
            options.load_training_checkpoint(runner, env, experiment, resume_path="resume.pt")
            runner.load.assert_called_once_with("resume.pt", restore_curriculum=True, legacy_curriculum=None)
            env.reset.assert_called_once_with()

    def test_conflicting_load_modes_fail_before_loading(self):
        runner, env = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            options.load_training_checkpoint(runner, env, "b2_z1_vd3_curve_amp", "a.pt", "b.pt")
        runner.load.assert_not_called()

    def test_startup_adaptation_explicitly_keeps_optimizer_iteration_and_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 3250
        options.load_training_checkpoint(runner, env, "b2_z1_vd5_startup_amp", startup_from="model_3250.pt")
        runner.load.assert_called_once_with("model_3250.pt", restore_curriculum=True, migrate_vd5_startup=True)
        self.assertEqual(runner.current_learning_iteration, 3250)
        env.reset.assert_called_once_with()
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner, env, "b2_z1_vd5_startup_amp",
                                             resume_path="new.pt", startup_from="old.pt")

    def test_ram_log_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "persistent"):
            options.persistent_log_root("/dev/shm/vd3_test_logs")
        self.assertEqual(options.persistent_log_root(ROOT / "logs"), ROOT / "logs")

    def test_shell_extra_arguments_and_resume_suppress_default_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "capture_python"
            fake_python.write_text(
                "#!/usr/bin/env python3\nimport json, os, sys\n"
                "print(json.dumps({'args':sys.argv[1:], 'pythonpath':os.environ['PYTHONPATH']}))\n"
            )
            fake_python.chmod(0o755)
            env = dict(os.environ, VD3_PYTHON=str(fake_python), VD3_CACHE_ROOT=str(Path(directory) / "cache"))
            for extra in (["--max_iterations", "3"], ["--resume", "--load_run", "saved_run"]):
                output = subprocess.check_output(["bash", str(ROOT / "scripts/train_vd3.sh"), *extra],
                                                 env=env, text=True)
                captured = json.loads(output)
                self.assertEqual(captured["args"][-len(extra):], extra)
                self.assertTrue(captured["pythonpath"].startswith(str(ROOT / "source/b2_z1")))
                self.assertEqual("--vd3-warm-start" in captured["args"], "--resume" not in extra)

    def test_vd5_launcher_uses_own_task_experiment_and_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "capture_python"
            fake_python.write_text(
                "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            fake_python.chmod(0o755)
            asset_root = Path(directory) / "assets"
            for asset in ("Isaac/Environments/Grid/default_environment.usd",
                          "Isaac/Props/UIElements/frame_prim.usd", "Isaac/Props/UIElements/arrow_x.usd"):
                asset_file = asset_root / asset
                asset_file.parent.mkdir(parents=True, exist_ok=True)
                asset_file.touch()
            environment = dict(os.environ, VD5_PYTHON=str(fake_python),
                               VD5_ASSET_ROOT=str(asset_root),
                               VD5_CACHE_ROOT=str(Path(directory) / "cache"))
            arguments = json.loads(subprocess.check_output(
                ["bash", str(ROOT / "scripts/train_vd5.sh")], env=environment, text=True
            ))
            self.assertEqual(arguments[arguments.index("--task") + 1], "Unitree-b2z1-VD5-Velocity")
            self.assertEqual(arguments[arguments.index("--experiment_name") + 1], "b2_z1_vd5_world_seam_amp")
            parser = argparse.ArgumentParser()
            parser.add_argument("--kit_args")
            parsed, _ = parser.parse_known_args(arguments)
            self.assertEqual(parsed.kit_args, f"--/persistent/isaac/asset_root/cloud={asset_root}")
            self.assertEqual(arguments[arguments.index("--vd5-warm-start") + 1],
                             str(ROOT / "logs/rsl_rl/b2_z1_vd3_curve_amp/2026-11-05_09-57-36_vd3_leg_mirror_from_10500/model_2000.pt"))
            resumed = json.loads(subprocess.check_output(
                ["bash", str(ROOT / "scripts/train_vd5.sh"), "--resume", "--load_run", "vd5_saved"],
                env=environment, text=True,
            ))
            self.assertNotIn("--vd5-warm-start", resumed)
            self.assertEqual(resumed[-3:], ["--resume", "--load_run", "vd5_saved"])

    def test_vd5_launcher_rejects_missing_local_assets_before_starting_isaac(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, VD5_PYTHON="/must/not/start/python",
                               VD5_ASSET_ROOT=directory)
            result = subprocess.run(["bash", str(ROOT / "scripts/train_vd5.sh")],
                                    env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Missing VD5 scene asset", result.stderr)

    def test_mirror_launcher_selects_10500_profile_and_preserves_resume_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "capture_python"
            fake_python.write_text(
                "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            fake_python.chmod(0o755)
            environment = dict(os.environ, VD3_PYTHON=str(fake_python),
                               VD3_CACHE_ROOT=str(Path(directory) / "cache"))
            environment.pop("VD3_WARM_START", None)
            script = str(ROOT / "scripts/train_vd3_symmetry.sh")
            arguments = json.loads(subprocess.check_output(["bash", script], env=environment, text=True))
            self.assertEqual(arguments[arguments.index("--agent") + 1], "rsl_rl_symmetry_cfg_entry_point")
            self.assertEqual(arguments[arguments.index("--vd3-warm-start") + 1],
                             "/home/dong/rl_sar_b2_z1/outputs/VD3_SUPPORT_REVIEW_20261104/model_10500.pt")
            iteration_values = [arguments[index + 1] for index, value in enumerate(arguments)
                                if value == "--max_iterations"]
            self.assertEqual(iteration_values[-1], "3000")
            extra = ["--resume", "--load_run", "mirror_saved_run", "--checkpoint", "model_250.pt"]
            resumed = json.loads(subprocess.check_output(["bash", script, *extra], env=environment, text=True))
            self.assertNotIn("--vd3-warm-start", resumed)
            self.assertEqual(resumed[-len(extra):], extra)


class RetentionTests(unittest.TestCase):
    def test_only_successfully_registered_new_files_in_current_run_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, previous = root / "new_run", root / "old_run"
            current.mkdir()
            previous.mkdir()
            old_run_model = previous / "model_1.pt"
            preexisting = current / "model_2.pt"
            old_run_model.write_text("old run")
            preexisting.write_text("pre-existing")
            policy = retention.CheckpointRetention(current, 2)
            policy.saved(old_run_model)
            policy.saved(preexisting, existed_before=True)
            for index in range(3, 8):
                model = current / f"model_{index}.pt"
                model.write_text("new")
                policy.saved(model)
            self.assertTrue(old_run_model.exists())
            self.assertTrue(preexisting.exists())
            self.assertEqual(sorted(p.name for p in current.glob("model_*.pt")),
                             ["model_2.pt", "model_6.pt", "model_7.pt"])

    def test_symlinks_and_noncheckpoint_files_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "protected.txt"
            target.write_text("protected")
            link = root / "model_0.pt"
            link.symlink_to(target)
            policy = retention.CheckpointRetention(root, 1)
            policy.saved(link)
            policy.saved(target)
            for index in (1, 2):
                model = root / f"model_{index}.pt"
                model.write_text("new")
                policy.saved(model)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(), "protected")
            self.assertFalse((root / "model_1.pt").exists())


class VD3RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from test_amp import AMP_OnPolicyRunner, FakeEnv, runner_config, capture_vd5_curriculum_state
        from test_vd2_curriculum_state import make_env, curriculum_state
        cls.torch = torch
        cls.runner_class = AMP_OnPolicyRunner
        cls.fake_env_class = FakeEnv
        cls.runner_config = staticmethod(runner_config)
        cls.make_env = staticmethod(make_env)
        cls.curriculum_state = curriculum_state
        cls.capture_vd5_curriculum_state = staticmethod(capture_vd5_curriculum_state)

    def runner(self, log_dir=None, keep_last=0, experiment="b2_z1_vd3_curve_amp"):
        env, _, arm = self.make_env()
        arm._vd5_ground_blend = 0.0
        arm._vd5_last_ground_curriculum_step = 0
        arm.cfg.vd5_ground_transition_steps = 100_000
        arm.cfg.vd5_seam_trailing_distance = 0.70
        fake = self.fake_env_class()
        fake.command_manager = env.command_manager
        fake.common_step_counter = 0
        cfg = self.runner_config()
        cfg.update(experiment_name=experiment, keep_last_checkpoints=keep_last)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.runner_class(fake, cfg, log_dir=log_dir, device="cpu")

    def test_vd3_resume_restores_optimizer_iteration_and_own_curriculum(self):
        source = self.runner()
        source.current_learning_iteration = 123
        source.alg.optimizer.param_groups[0]["lr"] = 2e-5
        source.env.command_manager.get_term("base_velocity").cfg.trajectory_curve_offset_scale = 0.06
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model_123.pt")
            source.save(checkpoint)
            saved = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("vd3_curriculum_state", saved)
            self.assertNotIn("vd2_curriculum_state", saved)
            target = self.runner()
            with contextlib.redirect_stdout(io.StringIO()):
                target.load(checkpoint, restore_curriculum=True)
            self.assertEqual(target.current_learning_iteration, 123)
            self.assertEqual(target.alg.learning_rate, 2e-5)
            self.assertEqual(self.curriculum_state.capture_curriculum_state(target.env),
                             saved["vd3_curriculum_state"])

    def test_vd5_startup_migration_and_exact_resume_have_distinct_profiles(self):
        source = self.runner(experiment="b2_z1_vd5_world_seam_amp")
        source.current_learning_iteration = 3250
        source.env.command_manager.get_term("ee_target")._vd5_ground_blend = .78048
        source.alg.optimizer.param_groups[0]["lr"] = 5e-5
        def startup_runner():
            runner = self.runner(experiment="b2_z1_vd5_startup_amp")
            robot = SimpleNamespace(init_state=SimpleNamespace(pos=(0., 0., .50)))
            event = SimpleNamespace(params={"velocity_range": (0., 0.)})
            runner.env.cfg = SimpleNamespace(reference_state_initialization_prob=.5,
                vd5_nominal_base_height_m=.50,
                scene=SimpleNamespace(robot=robot), events=SimpleNamespace(reset_robot_joints=event))
            runner.env.command_manager.get_term("base_velocity").cfg.vd5_startup_duration_s = 2.
            runner.env.command_manager.get_term("ee_target").cfg.vd5_startup_duration_s = 2.
            return runner
        with tempfile.TemporaryDirectory() as directory:
            original = str(Path(directory) / "original.pt")
            source.save(original)
            target = startup_runner()
            with self.assertRaisesRegex(ValueError, "VD5 resume requires"):
                target.load(original, restore_curriculum=True)
            with contextlib.redirect_stdout(io.StringIO()):
                target.load(original, restore_curriculum=True, migrate_vd5_startup=True)
            self.assertEqual(target.current_learning_iteration, 3250)
            self.assertEqual(target.alg.learning_rate, 5e-5)
            adapted = str(Path(directory) / "startup.pt")
            target.save(adapted)
            saved = self.torch.load(adapted, weights_only=False)
            self.assertEqual(saved["training_experiment_name"], "b2_z1_vd5_startup_amp")
            self.assertEqual(saved["vd5_curriculum_state"]["ground_blend"], .78048)
            self.assertEqual(saved["vd5_curriculum_state"]["startup_profile"]["duration_s"], 2.)
            resumed = startup_runner()
            with contextlib.redirect_stdout(io.StringIO()):
                resumed.load(adapted, restore_curriculum=True)
            self.assertEqual(resumed.current_learning_iteration, 3250)
            with self.assertRaisesRegex(ValueError, "VD5 resume requires"):
                source.load(adapted, restore_curriculum=True)

    def test_vd4_has_separate_resume_state_and_rejects_vd3_checkpoints(self):
        source = self.runner(experiment="b2_z1_vd4_front_calibrated_amp")
        source.current_learning_iteration = 12
        source.env.command_manager.get_term("base_velocity").cfg.trajectory_curve_offset_scale = 0.03
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "vd4_model.pt")
            source.save(checkpoint)
            saved = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("vd4_curriculum_state", saved)
            self.assertNotIn("vd3_curriculum_state", saved)
            self.assertEqual(saved["training_experiment_name"], "b2_z1_vd4_front_calibrated_amp")
            target = self.runner(experiment="b2_z1_vd4_front_calibrated_amp")
            with contextlib.redirect_stdout(io.StringIO()):
                target.load(checkpoint, restore_curriculum=True)
            self.assertEqual(target.current_learning_iteration, 12)
            self.assertEqual(self.curriculum_state.capture_curriculum_state(target.env),
                             saved["vd4_curriculum_state"])
            vd3 = self.runner(experiment="b2_z1_vd3_curve_amp")
            vd3_path = str(Path(directory) / "vd3_model.pt")
            vd3.save(vd3_path)
            with self.assertRaisesRegex(ValueError, "VD4 resume requires"):
                target.load(vd3_path, restore_curriculum=True)

    def test_vd5_has_independent_resume_state_and_rejects_vd4_checkpoints(self):
        experiment = "b2_z1_vd5_world_seam_amp"
        source = self.runner(experiment=experiment)
        source.current_learning_iteration = 7
        source.env.command_manager.get_term("ee_target")._vd5_ground_blend = 0.43
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "vd5_model.pt")
            source.save(checkpoint)
            saved = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("vd5_curriculum_state", saved)
            self.assertNotIn("vd4_curriculum_state", saved)
            self.assertEqual(saved["training_experiment_name"], experiment)
            target = self.runner(experiment=experiment)
            with contextlib.redirect_stdout(io.StringIO()):
                target.load(checkpoint, restore_curriculum=True)
            self.assertEqual(target.current_learning_iteration, 7)
            self.assertEqual(self.capture_vd5_curriculum_state(target.env),
                             saved["vd5_curriculum_state"])
            self.assertAlmostEqual(target.env.command_manager.get_term("ee_target")._vd5_ground_blend, 0.43)
            vd4 = self.runner(experiment="b2_z1_vd4_front_calibrated_amp")
            vd4_path = str(Path(directory) / "vd4_model.pt")
            vd4.save(vd4_path)
            with self.assertRaisesRegex(ValueError, "VD5 resume requires"):
                target.load(vd4_path, restore_curriculum=True)

    def test_vd2_can_warm_start_vd3_but_cannot_resume_it(self):
        source = self.runner(experiment="b2_z1_vd2_curve_amp")
        source.current_learning_iteration = 2050
        source.env.command_manager.get_term("base_velocity").cfg.trajectory_curve_offset_scale = 0.1
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model_2050.pt")
            source.save(checkpoint)
            target = self.runner()
            initial = self.curriculum_state.capture_curriculum_state(target.env)
            with self.assertRaisesRegex(ValueError, "requires a VD3 checkpoint"):
                target.load(checkpoint, restore_curriculum=True)
            fresh_lr = target.alg.learning_rate
            target.env.reset = Mock()
            options.load_training_checkpoint(target, target.env, "b2_z1_vd3_curve_amp", warm_start=checkpoint)
            self.assertEqual(target.current_learning_iteration, 0)
            self.assertEqual(target.alg.learning_rate, fresh_lr)
            self.assertEqual(self.curriculum_state.capture_curriculum_state(target.env), initial)
            self.torch.testing.assert_close(next(target.alg.discriminator.parameters()),
                                            next(source.alg.discriminator.parameters()))

    def test_failed_checkpoint_write_preserves_existing_files_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(log_dir=directory, keep_last=1)
            first, second = Path(directory) / "model_1.pt", Path(directory) / "model_2.pt"
            runner.save(str(first))
            with patch.object(self.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    runner.save(str(second))
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())
            self.assertFalse(list(Path(directory).glob("*.partial")))
            runner.save(str(second))
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())


if __name__ == "__main__":
    unittest.main()
