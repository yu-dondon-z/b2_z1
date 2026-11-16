"""CPU-only VD12 launch/checkpoint contracts; no Isaac import or physics."""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "source/b2_z1/b2_z1/rsl_rl/runners/amp_on_policy_runner.py"
VD12 = "b2_z1_vd12_geometry_guard_amp"
VD11 = "b2_z1_vd11_motion_quality_amp"


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


options = load_file("vd12_training_options_cpu", "scripts/rsl_rl/training_options.py")
cli_args = load_file("vd12_training_cli_cpu", "scripts/rsl_rl/cli_args.py")


def geometry_profile():
    return {
        "version": 1, "urdf_sha256": "a" * 64,
        "geometry_method": "vd12_segment_obb_sphere_capsule_lower_bound_v1",
        "soft_margin_m": .03, "hard_threshold_m": 0., "cost_cap": 9.,
        "physics_dt_s": .005, "decimation": 4, "termination_period_s": .02,
        "whole_parent_body_excluded": "link06",
        "scope": "URDF authored robot primitives, fixed-child folding; discrete 200Hz exclusion",
        "clearance_reward_weight": -1., "failure_reward_weight": -100.,
    }


def checkpoint(experiment=VD12):
    result = {
        "training_experiment_name": experiment,
        "model_state_dict": {"actor.weight": "policy"},
        "optimizer_state_dict": {"state": "optimizer"},
        "discriminator_state_dict": {"weight": "discriminator"},
        "amp_normalizer_state_dict": {"mean": "mean", "var": "var", "count": 101},
        "iter": 5999, "infos": {"source": experiment},
        "vd5_curriculum_state": {"version": 1, "ground_blend": .8},
    }
    if experiment == VD12:
        result["vd12_geometry_profile"] = geometry_profile()
    return result


def runner_method(name, namespace):
    tree = ast.parse(RUNNER.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "AMP_OnPolicyRunner")
    method = next(node for node in runner.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


def fake_runner(experiment=VD12):
    policy = Mock()
    policy.load_state_dict.return_value = True
    policy.state_dict.return_value = {"actor.weight": "policy"}
    discriminator = Mock()
    discriminator.state_dict.return_value = {"weight": "discriminator"}
    optimizer = Mock()
    optimizer.param_groups = [{"lr": 2e-5}]
    optimizer.state_dict.return_value = {"state": "optimizer"}
    algorithm = SimpleNamespace(
        policy=policy, discriminator=discriminator, optimizer=optimizer,
        amp_normalizer=SimpleNamespace(mean="new mean", var="new var", count=0),
        rnd=None, learning_rate=5e-5,
    )
    env = SimpleNamespace(cfg=SimpleNamespace(vd12_geometry_profile=geometry_profile()))
    env.unwrapped = env
    return SimpleNamespace(
        cfg={"experiment_name": experiment}, env=env, alg=algorithm,
        current_learning_iteration=777, device="cpu", empirical_normalization=False,
        logger_type="tensorboard", disable_logs=False, checkpoint_retention=Mock(),
    )


def load_method(saved):
    namespace = {
        "torch": SimpleNamespace(load=Mock(return_value=saved)),
        "restore_vd5_curriculum_state": Mock(), "restore_curriculum_state": Mock(),
        "legacy_curriculum_state": Mock(),
    }
    return runner_method("load", namespace), namespace


def parse_training_args(arguments):
    """Execute the real argument checks, stopping before AppLauncher is called."""
    class CPUParser(argparse.ArgumentParser):
        def parse_known_args(self, args=None, namespace=None):
            return super().parse_known_args(arguments if args is None else args, namespace)

    path = ROOT / "scripts/rsl_rl/train.py"
    tree = ast.parse(path.read_text())
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "STARTUP_EXPERIMENTS"
                     for target in node.targets))
    stop = next(index for index, node in enumerate(tree.body)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "app_launcher"
                    for target in node.targets))
    namespace = {
        "argparse": SimpleNamespace(ArgumentParser=CPUParser),
        "sys": SimpleNamespace(argv=["train.py", *arguments]),
        "cli_args": cli_args,
        "AppLauncher": SimpleNamespace(add_app_launcher_args=lambda parser: None),
        "persistent_log_root": options.persistent_log_root,
    }
    exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class VD12LaunchTests(unittest.TestCase):
    def test_migration_keeps_actor_amp_but_not_optimizer_iteration_or_ground_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 5999
        options.load_training_checkpoint(runner, env, VD12, vd12_from="vd11.pt")
        runner.load.assert_called_once_with(
            "vd11.pt", load_optimizer=False, restore_curriculum=False, migrate_vd12=True
        )
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once_with()

    def test_resume_uses_full_same_version_path_and_does_not_reset_iteration(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 1400
        options.load_training_checkpoint(runner, env, VD12, resume_path="vd12.pt")
        runner.load.assert_called_once_with("vd12.pt", restore_curriculum=True, legacy_curriculum=None)
        self.assertEqual(runner.current_learning_iteration, 1400)
        env.reset.assert_called_once_with()

    def test_fresh_empty_value_never_loads_or_changes_initial_curriculum(self):
        for value in (None, ""):
            with self.subTest(value=value):
                runner, env = Mock(), Mock()
                options.load_training_checkpoint(runner, env, VD12, vd12_from=value)
                runner.load.assert_not_called()
                env.reset.assert_not_called()

    def test_failed_load_cannot_reset_environment_or_iteration(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 5999
        runner.load.side_effect = ValueError("wrong source experiment")
        with self.assertRaisesRegex(ValueError, "wrong source"):
            options.load_training_checkpoint(runner, env, VD12, vd12_from="vd5.pt")
        self.assertEqual(runner.current_learning_iteration, 5999)
        env.reset.assert_not_called()

    def test_mutually_exclusive_modes_are_rejected_before_any_loading(self):
        for value in ("vd11.pt", ""):
            for other in ("resume_path", "warm_start", "startup_from", "vd6_from", "vd7_from",
                          "vd8_from", "vd9_from", "vd9_warm_start", "vd10_from", "vd11_from"):
                with self.subTest(value=value, other=other):
                    runner, env = Mock(), Mock()
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        options.load_training_checkpoint(runner, env, VD12, vd12_from=value, **{other: "other.pt"})
                    runner.load.assert_not_called()
                    env.reset.assert_not_called()

    def test_legacy_loading_modes_cannot_bypass_vd12_migration_contract(self):
        for other in ("warm_start", "startup_from", "vd6_from", "vd7_from", "vd8_from", "vd9_from",
                      "vd9_warm_start", "vd10_from", "vd11_from", "legacy_curriculum"):
            with self.subTest(other=other):
                runner, env = Mock(), Mock()
                with self.assertRaisesRegex(ValueError, "VD12 accepts only"):
                    options.load_training_checkpoint(runner, env, VD12, **{other: "other.pt"})
                runner.load.assert_not_called()

    def test_vd12_flag_including_fresh_requires_its_own_experiment(self):
        for value in ("vd11.pt", ""):
            runner, env = Mock(), Mock()
            with self.assertRaisesRegex(ValueError, "VD12 warm start requires"):
                options.load_training_checkpoint(runner, env, VD11, vd12_from=value)
            runner.load.assert_not_called()

    def test_real_cli_rejects_conflicts_before_simulator_launch(self):
        for value in ("vd11.pt", ""):
            for other in ("--resume", "--vd2_warm_start", "--vd3-warm-start", "--vd4-warm-start",
                          "--vd5-warm-start", "--vd5-startup-from", "--vd6-from", "--vd7-from",
                          "--vd8-from", "--vd9-from", "--vd9-warm-start", "--vd10-from", "--vd11-from"):
                arguments = [f"--vd12-from={value}", other]
                if other != "--resume":
                    arguments.append("other.pt")
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_training_args(arguments)
                    self.assertEqual(raised.exception.code, 2)

    def test_real_cli_preserves_vd12_startup_clock_and_accepts_fresh_mode(self):
        parsed = parse_training_args(["--vd12-from="])
        self.assertEqual(parsed["args_cli"].vd12_from, "")
        self.assertIn(VD12, parsed["STARTUP_EXPERIMENTS"])
        self.assertIn(VD12, parsed["TUBE_EXPERIMENTS"])

    def test_shell_defaults_and_user_overrides_preserve_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "capture_python"
            fake.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "print(json.dumps({'args':sys.argv[1:],'asset_cache':os.environ['VD10_ASSET_CACHE']}))\n"
            )
            fake.chmod(0o755)
            environment = dict(os.environ, VD12_PYTHON=str(fake), VD12_CACHE_ROOT=directory)
            for extra in ([], ["--resume", "--load_run", "saved", "--checkpoint", "model_1200.pt"],
                          ["--vd12-from="], ["--vd12-from", "specific.pt"],
                          ["--max_iterations", "2"]):
                with self.subTest(extra=extra):
                    captured = json.loads(subprocess.check_output(
                        ["bash", str(ROOT / "scripts/train_vd12.sh"), *extra],
                        env=environment, text=True,
                    ))
                    arguments = captured["args"]
                    self.assertEqual(arguments[arguments.index("--task") + 1], "Unitree-b2z1-VD12-Velocity")
                    self.assertEqual(arguments[arguments.index("--experiment_name") + 1], VD12)
                    self.assertEqual(arguments[arguments.index("--num_envs") + 1], "256")
                    self.assertEqual(arguments[arguments.index("--max_iterations") + 1], "6000")
                    self.assertEqual(arguments[arguments.index("--keep-last-checkpoints") + 1], "0")
                    self.assertEqual(arguments[arguments.index("--log-root") + 1], str(ROOT / "logs/rsl_rl"))
                    self.assertTrue(captured["asset_cache"].startswith(directory))
                    if extra:
                        self.assertEqual(arguments[-len(extra):], extra)
                    suppressed = any(arg == "--resume" or arg.startswith("--vd12-from") for arg in extra)
                    source = str(ROOT / "logs/rsl_rl/b2_z1_vd11_motion_quality_amp/"
                                 "2026-11-13_17-56-49_vd11_motion_quality_long_20261113/model_5999.pt")
                    self.assertEqual(source in arguments, not suppressed)
                    for obsolete in ("--vd5-warm-start", "--vd10-from", "--vd11-from"):
                        self.assertNotIn(obsolete, arguments)


class VD12CheckpointTests(unittest.TestCase):
    def assert_no_load_mutation(self, runner, namespace):
        runner.alg.policy.load_state_dict.assert_not_called()
        runner.alg.discriminator.load_state_dict.assert_not_called()
        runner.alg.optimizer.load_state_dict.assert_not_called()
        namespace["restore_vd5_curriculum_state"].assert_not_called()
        self.assertEqual(runner.current_learning_iteration, 777)
        self.assertEqual(runner.alg.amp_normalizer.count, 0)

    def test_vd11_migration_loads_complete_amp_and_resets_iteration_without_restoring_old_curriculum(self):
        saved = checkpoint(VD11)
        saved.pop("vd5_curriculum_state")
        runner = fake_runner()
        load, namespace = load_method(saved)
        infos = load(runner, "vd11.pt", load_optimizer=False, restore_curriculum=False, migrate_vd12=True)
        runner.alg.policy.load_state_dict.assert_called_once_with(saved["model_state_dict"])
        runner.alg.discriminator.load_state_dict.assert_called_once_with(saved["discriminator_state_dict"])
        self.assertEqual(runner.alg.amp_normalizer.mean, "mean")
        self.assertEqual(runner.alg.amp_normalizer.var, "var")
        self.assertEqual(runner.alg.amp_normalizer.count, 101)
        runner.alg.optimizer.load_state_dict.assert_not_called()
        namespace["restore_vd5_curriculum_state"].assert_not_called()
        self.assertEqual(runner.alg.learning_rate, 5e-5)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(infos, saved["infos"])

    def test_same_version_resume_restores_profile_checked_curriculum_optimizer_amp_and_iteration(self):
        saved = checkpoint()
        runner = fake_runner()
        load, namespace = load_method(saved)
        with redirect_stdout(io.StringIO()):
            load(runner, "vd12.pt", restore_curriculum=True)
        namespace["restore_vd5_curriculum_state"].assert_called_once_with(
            runner.env, saved["vd5_curriculum_state"], allow_startup_migration=False
        )
        runner.alg.policy.load_state_dict.assert_called_once_with(saved["model_state_dict"])
        runner.alg.discriminator.load_state_dict.assert_called_once_with(saved["discriminator_state_dict"])
        runner.alg.optimizer.load_state_dict.assert_called_once_with(saved["optimizer_state_dict"])
        self.assertEqual(runner.alg.amp_normalizer.count, 101)
        self.assertEqual(runner.alg.learning_rate, 2e-5)
        self.assertEqual(runner.current_learning_iteration, 5999)

    def test_vd11_cannot_be_resumed_as_vd12_even_with_copied_geometry_profile(self):
        saved = checkpoint(VD11)
        saved["vd12_geometry_profile"] = geometry_profile()
        runner = fake_runner()
        load, namespace = load_method(saved)
        with self.assertRaisesRegex(ValueError, "VD12 load requires b2_z1_vd12"):
            load(runner, "renamed_vd11.pt", restore_curriculum=True)
        self.assert_no_load_mutation(runner, namespace)

    def test_migration_refuses_vd5_vd10_vd12_and_missing_experiment(self):
        for experiment in ("b2_z1_vd5_world_seam_amp", "b2_z1_vd10_hollow_gun_amp", VD12, None):
            with self.subTest(experiment=experiment):
                runner = fake_runner()
                load, namespace = load_method(checkpoint(experiment))
                with self.assertRaisesRegex(ValueError, "requires b2_z1_vd11"):
                    load(runner, "other.pt", load_optimizer=False, migrate_vd12=True)
                self.assert_no_load_mutation(runner, namespace)

    def test_migration_rejects_optimizer_curriculum_legacy_or_other_migration_flags(self):
        variants = ({"load_optimizer": True}, {"restore_curriculum": True},
                    {"legacy_curriculum": (1., 1)}, {"migrate_vd11": True}, {"migrate_vd9": True},
                    {"migrate_vd8": True}, {"migrate_vd7": True}, {"migrate_vd6": True},
                    {"migrate_vd5_startup": True})
        for changed in variants:
            with self.subTest(changed=changed):
                runner = fake_runner()
                load, namespace = load_method(checkpoint(VD11))
                arguments = dict(load_optimizer=False, restore_curriculum=False, migrate_vd12=True)
                arguments.update(changed)
                with self.assertRaisesRegex(ValueError, "VD12 migration requires"):
                    load(runner, "vd11.pt", **arguments)
                self.assert_no_load_mutation(runner, namespace)
        runner = fake_runner(experiment=VD11)
        load, namespace = load_method(checkpoint(VD11))
        with self.assertRaisesRegex(ValueError, "VD12 migration requires"):
            load(runner, "vd11.pt", load_optimizer=False, migrate_vd12=True)
        self.assert_no_load_mutation(runner, namespace)

    def test_resume_requires_entire_geometry_profile_not_just_version(self):
        for key in geometry_profile():
            with self.subTest(key=key):
                saved = checkpoint()
                saved["vd12_geometry_profile"].pop(key)
                runner = fake_runner()
                load, namespace = load_method(saved)
                with self.assertRaisesRegex(ValueError, "geometry profile differs"):
                    load(runner, "different.pt", restore_curriculum=True)
                self.assert_no_load_mutation(runner, namespace)
        for changed in (None, {}, dict(geometry_profile(), extra="new rule"),
                        dict(geometry_profile(), soft_margin_m=.04)):
            saved = checkpoint()
            saved["vd12_geometry_profile"] = changed
            runner = fake_runner()
            load, namespace = load_method(saved)
            with self.assertRaisesRegex(ValueError, "geometry profile differs"):
                load(runner, "different.pt", restore_curriculum=True)
            self.assert_no_load_mutation(runner, namespace)

    def test_generic_load_cannot_bypass_full_same_version_curriculum_restore(self):
        for optimizer, curriculum in ((True, False), (False, False), (False, True)):
            runner = fake_runner()
            load, namespace = load_method(checkpoint())
            with self.assertRaisesRegex(ValueError, "full curriculum restoration"):
                load(runner, "vd12.pt", load_optimizer=optimizer, restore_curriculum=curriculum)
            self.assert_no_load_mutation(runner, namespace)

    def test_incomplete_amp_and_optimizer_fail_before_any_state_changes(self):
        for source, arguments in ((VD11, {"load_optimizer": False, "migrate_vd12": True}),
                                  (VD12, {"restore_curriculum": True})):
            for key in ("model_state_dict", "discriminator_state_dict", "amp_normalizer_state_dict", "iter", "infos"):
                with self.subTest(source=source, missing=key):
                    saved = checkpoint(source)
                    saved.pop(key)
                    runner = fake_runner()
                    load, namespace = load_method(saved)
                    with self.assertRaisesRegex(ValueError, "complete policy and AMP"):
                        load(runner, "incomplete.pt", **arguments)
                    self.assert_no_load_mutation(runner, namespace)
            for key in ("mean", "var", "count"):
                with self.subTest(source=source, amp_field=key):
                    saved = checkpoint(source)
                    saved["amp_normalizer_state_dict"].pop(key)
                    runner = fake_runner()
                    load, namespace = load_method(saved)
                    with self.assertRaisesRegex(ValueError, "complete policy and AMP"):
                        load(runner, "incomplete.pt", **arguments)
                    self.assert_no_load_mutation(runner, namespace)
        saved = checkpoint()
        saved.pop("optimizer_state_dict")
        runner = fake_runner()
        load, namespace = load_method(saved)
        with self.assertRaisesRegex(ValueError, "plus optimizer for resume"):
            load(runner, "incomplete.pt", restore_curriculum=True)
        self.assert_no_load_mutation(runner, namespace)

    def test_resume_refuses_invalid_iteration_and_missing_curriculum_before_policy_mutation(self):
        for iteration in (-1, True, None, 2.5):
            with self.subTest(iteration=iteration):
                saved = checkpoint()
                saved["iter"] = iteration
                runner = fake_runner()
                load, namespace = load_method(saved)
                with self.assertRaisesRegex(ValueError, "invalid learning iteration"):
                    load(runner, "invalid.pt", restore_curriculum=True)
                self.assert_no_load_mutation(runner, namespace)
        saved = checkpoint()
        saved.pop("vd5_curriculum_state")
        runner = fake_runner()
        load, namespace = load_method(saved)
        with self.assertRaisesRegex(ValueError, "no VD5 curriculum state"):
            load(runner, "incomplete.pt", restore_curriculum=True)
        self.assert_no_load_mutation(runner, namespace)

    def test_missing_current_geometry_profile_fails_before_loading(self):
        for current in (None, {}, {"soft_margin_m": .03}):
            runner = fake_runner()
            runner.env.cfg.vd12_geometry_profile = current
            load, namespace = load_method(checkpoint())
            with self.assertRaisesRegex(ValueError, "declared geometry profile"):
                load(runner, "vd12.pt", restore_curriculum=True)
            self.assert_no_load_mutation(runner, namespace)

    def test_save_publishes_independent_profile_and_curriculum_atomically(self):
        runner = fake_runner()
        runner.current_learning_iteration = 1400
        saved = []

        def fake_torch_save(payload, path):
            saved.append(payload)
            Path(path).write_bytes(b"complete checkpoint")

        namespace = {
            "torch": SimpleNamespace(save=fake_torch_save), "deepcopy": deepcopy,
            "capture_vd5_curriculum_state": Mock(return_value={"version": 1, "ground_blend": .4}),
            "os": os, "tempfile": tempfile,
        }
        save = runner_method("save", namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model_1400.pt")
            save(runner, path, infos={"diagnostic": True})
            self.assertEqual(Path(path).read_bytes(), b"complete checkpoint")
            self.assertFalse(list(Path(directory).glob("*.partial")))
            runner.checkpoint_retention.saved.assert_called_once_with(path, existed_before=False)
        self.assertEqual(saved[0]["training_experiment_name"], VD12)
        self.assertEqual(saved[0]["vd12_geometry_profile"], geometry_profile())
        self.assertEqual(saved[0]["vd5_curriculum_state"], {"version": 1, "ground_blend": .4})
        self.assertEqual(saved[0]["iter"], 1400)
        runner.env.cfg.vd12_geometry_profile["soft_margin_m"] = .5
        self.assertEqual(saved[0]["vd12_geometry_profile"]["soft_margin_m"], .03)

    def test_save_without_geometry_profile_cannot_publish_or_touch_retention(self):
        runner = fake_runner()
        runner.env.cfg.vd12_geometry_profile = {}
        namespace = {
            "torch": SimpleNamespace(save=Mock()), "deepcopy": deepcopy,
            "capture_vd5_curriculum_state": Mock(return_value={"version": 1}),
        }
        save = runner_method("save", namespace)
        with self.assertRaisesRegex(ValueError, "declared geometry profile"):
            save(runner, "must_not_be_written.pt")
        namespace["torch"].save.assert_not_called()
        runner.checkpoint_retention.saved.assert_not_called()


if __name__ == "__main__":
    unittest.main()
