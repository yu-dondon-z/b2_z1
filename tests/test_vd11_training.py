"""Simulator-independent contracts for isolated VD11 objective and migration."""

import ast
import copy
import importlib.util
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
TASK = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1"
spec = importlib.util.spec_from_file_location("vd11_training_options", ROOT / "scripts/rsl_rl/training_options.py")
options = importlib.util.module_from_spec(spec)
spec.loader.exec_module(options)


def extract_node(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class VD11TrainingTests(unittest.TestCase):
    def test_migration_keeps_geometry_and_curriculum_not_optimizer_or_iteration(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 5998
        options.load_training_checkpoint(runner, env, "b2_z1_vd11_motion_quality_amp", vd11_from="vd10.pt")
        runner.load.assert_called_once_with(
            "vd10.pt", load_optimizer=False, restore_curriculum=True, migrate_vd11=True
        )
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once_with()

    def test_migration_fails_without_resetting_environment_or_iteration(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 5998
        runner.load.side_effect = ValueError("profile differs")
        with self.assertRaisesRegex(ValueError, "profile differs"):
            options.load_training_checkpoint(runner, env, "b2_z1_vd11_motion_quality_amp", vd11_from="old.pt")
        self.assertEqual(runner.current_learning_iteration, 5998)
        env.reset.assert_not_called()

    def test_wrong_experiment_and_conflicting_modes_fail_before_loading(self):
        for extra in ({"resume_path": "resume.pt"}, {"vd10_from": "vd5.pt"},
                      {"vd9_from": "vd5.pt"}, {"warm_start": "other.pt"}):
            runner, env = Mock(), Mock()
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                options.load_training_checkpoint(
                    runner, env, "b2_z1_vd11_motion_quality_amp", vd11_from="vd10.pt", **extra
                )
            runner.load.assert_not_called()
            env.reset.assert_not_called()
        runner, env = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "VD11 warm start requires"):
            options.load_training_checkpoint(runner, env, "b2_z1_vd10_hollow_gun_amp", vd11_from="old.pt")
        runner.load.assert_not_called()

    def test_exact_resume_restores_optimizer_iteration_and_curriculum(self):
        runner, env = Mock(), Mock()
        runner.current_learning_iteration = 1200
        options.load_training_checkpoint(runner, env, "b2_z1_vd11_motion_quality_amp", resume_path="vd11.pt")
        runner.load.assert_called_once_with("vd11.pt", restore_curriculum=True, legacy_curriculum=None)
        self.assertEqual(runner.current_learning_iteration, 1200)
        env.reset.assert_called_once_with()

    def test_empty_warm_start_keeps_original_from_scratch_curriculum(self):
        runner, env = Mock(), Mock()
        options.load_training_checkpoint(runner, env, "b2_z1_vd11_motion_quality_amp", vd11_from="")
        runner.load.assert_not_called()
        env.reset.assert_not_called()

    def test_config_only_changes_the_two_objectives(self):
        original_descent = SimpleNamespace(weight=-2.0, params={"speed_limit": .35})
        tracking = object()
        cfg = SimpleNamespace(
            scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="plane")),
            sim=SimpleNamespace(dt=.005), decimation=4,
            rewards=SimpleNamespace(
                vd9_tcp_highpass_200hz=SimpleNamespace(weight=-.1, params={"cutoff": 3.0}),
                vd9_foot_descent_200hz=original_descent, tracking=tracking,
            ),
        )
        highpass, impact = object(), object()
        configure = extract_node(TASK / "b2_z1_vd11_env_cfg.py", "_configure_vd11", {
            "RewTerm": SimpleNamespace,
            "SceneEntityCfg": lambda name, **kw: SimpleNamespace(name=name, **kw),
            "PhysicsRateWorldTCPHighpass": highpass, "PhysicsRateFootImpact": impact,
        })
        configure(cfg)
        self.assertIs(cfg.rewards.tracking, tracking)
        self.assertIs(cfg.rewards.vd9_foot_descent_200hz, original_descent)
        self.assertEqual(cfg.rewards.vd9_foot_descent_200hz.weight, -2.0)
        self.assertIsNone(cfg.rewards.vd9_tcp_highpass_200hz)
        world = cfg.rewards.vd11_world_tcp_highpass_200hz
        self.assertIs(world.func, highpass)
        self.assertEqual(world.weight, -.3)
        self.assertEqual({k: v for k, v in world.params.items() if k != "asset_cfg"}, {
            "base_name": "base_link", "cutoff": 1.0,
            "deadzone": .001, "scale": .005, "lateral_weight": 2.0, "active_after": 2.0,
        })
        foot = cfg.rewards.vd11_foot_impact_200hz
        self.assertIs(foot.func, impact)
        self.assertEqual(foot.weight, -.5)
        self.assertEqual({k: v for k, v in foot.params.items() if k not in {"asset_cfg", "sensor_cfg"}}, {
            "contact_threshold": 10.0, "airborne_duration": .02, "window_s": .05,
            "force_limit": 300.0, "force_scale": 200.0, "active_after": 2.0,
        })
        for key in ("asset_cfg", "sensor_cfg"):
            self.assertEqual(foot.params[key].body_names, ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
            self.assertTrue(foot.params[key].preserve_order)
        for dt, decimation, terrain in ((.01, 4, "plane"), (.005, 2, "plane"), (.005, 4, "generator")):
            cfg.sim.dt, cfg.decimation, cfg.scene.terrain.terrain_type = dt, decimation, terrain
            with self.assertRaisesRegex(ValueError, "200 Hz"):
                configure(cfg)

    def test_runner_retains_shape_symmetry_and_is_conservative(self):
        class Algorithm(SimpleNamespace):
            def replace(self, **changes):
                result = copy.copy(self)
                result.__dict__.update(changes)
                return result

        symmetry = object()
        original = Algorithm(learning_rate=5e-5, schedule="fixed", entropy_coef=.01, symmetry_cfg=symmetry)
        base = type("VD10PPORunnerCfg", (), {"algorithm": original, "policy": object(), "save_interval": 100})
        runner_cls = extract_node(TASK / "agents/rsl_rl_ppo_cfg.py", "VD11PPORunnerCfg", {
            "VD10PPORunnerCfg": base, "configclass": lambda value: value,
        })
        self.assertEqual(runner_cls.experiment_name, "b2_z1_vd11_motion_quality_amp")
        self.assertEqual(runner_cls.algorithm.learning_rate, 2e-5)
        self.assertEqual(runner_cls.algorithm.schedule, "fixed")
        self.assertEqual(runner_cls.algorithm.entropy_coef, .002)
        self.assertIs(runner_cls.algorithm.symmetry_cfg, symmetry)
        self.assertIs(runner_cls.policy, base.policy)
        self.assertEqual(original.entropy_coef, .01)

    def test_registration_keeps_vd10_reset_environment_and_config_inheritance(self):
        source = (TASK / "__init__.py").read_text()
        self.assertIn('id=f"Unitree-b2z1-VD11-Velocity{_suffix}"', source)
        tree = ast.parse((TASK / "b2_z1_vd11_env_cfg.py").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for name, base in (("B2Z1VD11EnvCfg", "B2Z1VD10EnvCfg"),
                           ("B2Z1VD11PlayEnvCfg", "B2Z1VD10PlayEnvCfg")):
            self.assertEqual(ast.unparse(classes[name].bases[0]), base)
            self.assertIn("super().__post_init__()", ast.unparse(classes[name]))

    def test_shell_launch_defaults_and_explicit_override_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "capture_python"
            fake.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "print(json.dumps({'args':sys.argv[1:],'asset_cache':os.environ['VD10_ASSET_CACHE']}))\n"
            )
            fake.chmod(0o755)
            env = dict(os.environ, VD11_PYTHON=str(fake), VD11_CACHE_ROOT=directory)
            for extra in ([], ["--resume", "--load_run", "saved"], ["--vd11-from="],
                          ["--vd11-from", "specific.pt"], ["--max_iterations", "2"]):
                result = subprocess.check_output(
                    ["bash", str(ROOT / "scripts/train_vd11.sh"), *extra], env=env, text=True
                )
                captured = json.loads(result)
                args = captured["args"]
                self.assertIn("Unitree-b2z1-VD11-Velocity", args)
                self.assertIn("b2_z1_vd11_motion_quality_amp", args)
                self.assertIn("--keep-last-checkpoints", args)
                self.assertEqual(args[args.index("--keep-last-checkpoints") + 1], "0")
                self.assertTrue(captured["asset_cache"].startswith(directory))
                if extra:
                    self.assertEqual(args[-len(extra):], extra)
                suppressed = any(arg == "--resume" or arg.startswith("--vd11-from") for arg in extra)
                self.assertEqual(any(arg.endswith("model_5998.pt") for arg in args), not suppressed)
                self.assertNotIn("--vd10-from", args)


if __name__ == "__main__":
    unittest.main()
