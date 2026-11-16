"""CPU checks for posture geometry, isolated config, migration and resume."""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

import test_vd12_training as runner_cpu

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "source/b2_z1/b2_z1/assets"
MDP = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package("vd14_assets_cpu", ASSETS)
contract = load("vd14_assets_cpu.vd14_contract", ASSETS / "vd14_contract.py")
mount = contract.vd13_contract
geometry = load("vd14_assets_cpu.vd12_contract", ASSETS / "vd12_contract.py")
package("vd14_mdp_cpu", MDP)
rewards = load("vd14_mdp_cpu.vd14_rewards", MDP / "vd14_rewards.py")
tilt = load("vd14_mdp_cpu.vd3_support", MDP / "vd3_support.py")
options = runner_cpu.options


def checkpoint(experiment=contract.EXPERIMENT):
    saved = dict(
        model_state_dict={}, discriminator_state_dict={}, optimizer_state_dict={},
        amp_normalizer_state_dict=dict(mean=0., var=1., count=100),
        iter=6199, infos=None, vd5_curriculum_state={"version": 2},
        training_experiment_name=experiment,
        vd13_mount_profile=mount.mount_profile(),
        vd12_geometry_profile=geometry.geometry_profile(mount.ASSET),
    )
    if experiment == contract.EXPERIMENT:
        saved["vd14_posture_profile"] = contract.posture_profile()
    return saved


class PostureRewards(unittest.TestCase):
    def test_cost_is_zero_at_safe_width_and_cannot_reward_extra_splay(self):
        values = torch.tensor([.42, .52, .9])
        torch.testing.assert_close(rewards.front_width_deficit(values), torch.zeros(3))

    def test_narrow_and_crossed_feet_cost_more(self):
        values = torch.tensor([.42, .32, .22, -.52])
        costs = rewards.front_width_deficit(values)
        self.assertTrue(bool(torch.all(costs[1:] > costs[:-1])))
        self.assertAlmostEqual(float(costs[1]), 1., places=5)

    def test_narrow_foot_gradient_points_outwards(self):
        width = torch.tensor([.21], requires_grad=True)
        (-2. * rewards.front_width_deficit(width)).sum().backward()
        self.assertGreater(float(width.grad), 0.)

    def test_parameters_reject_invalid_scales(self):
        for key in ("minimum_width", "scale"):
            for value in (0., -1., float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    rewards.front_width_deficit(torch.tensor([.2]), **{key: value})

    def test_full_base_frame_not_world_y_even_with_roll_pitch_yaw(self):
        # Arbitrary normalized rotation, translated base, two asymmetric feet.
        quat = torch.tensor([[.7, .3, -.4, .5]])
        quat /= quat.norm(dim=-1, keepdim=True)
        root = torch.tensor([[10., -8., 1.5]])
        feet_b = torch.tensor([[[.4, .12, -.5], [.4, -.09, -.49]]])
        q2 = quat[:, None].expand(-1, 2, -1)
        frame = sys.modules["vd14_mdp_cpu.tcp_frame"]
        data = NS(root_pos_w=root, root_quat_w=quat,
                  body_pos_w=frame.quat_apply(q2, feet_b) + root[:, None],
                  projected_gravity_b=torch.tensor([[0., 0., -1.]]))
        robot = NS(data=data, body_names=["FL_foot", "FR_foot"])
        goal = NS(metrics={})
        env = NS(scene={"robot": robot}, command_manager=NS(get_term=lambda _: goal))
        selector = NS(name="robot", body_ids=[0, 1])
        cost = rewards.front_stance_width_penalty(env, selector)
        torch.testing.assert_close(cost, torch.tensor([4.41]), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(goal.metrics["vd14_front_width_m"], torch.tensor([.21]),
                                   atol=1e-6, rtol=1e-5)
        self.assertEqual(float(goal.metrics["vd14_front_width_below_min"]), 1.)
        with self.assertRaises(ValueError):
            rewards.front_stance_width_penalty(env, NS(name="robot", body_ids=[1, 0]))

    def test_new_tilt_penalty_is_small_in_reference_range_and_stronger_at_sag(self):
        angles = torch.tensor([0., math.radians(3.), math.radians(10.6)])
        gravity = torch.stack((angles.sin(), torch.zeros(3), -angles.cos()), dim=1)
        p = contract.posture_profile()
        new = -p["base_tilt_weight"] * tilt.inclination_deadzone_penalty(
            gravity, p["base_tilt_deadzone_rad"], p["base_tilt_scale_rad"])
        old = 3. * tilt.inclination_deadzone_penalty(gravity, .1, .2)
        torch.testing.assert_close(new[:2], torch.zeros(2))
        self.assertGreater(float(new[2]), 8. * float(old[2]))


class PostureCheckpoint(unittest.TestCase):
    def validate(self, saved, **kwargs):
        contract.validate_checkpoint(saved, contract.posture_profile(), mount.mount_profile(),
                                     geometry.geometry_profile(mount.ASSET), **kwargs)

    def test_migration_retains_curriculum_but_refuses_old_optimizer(self):
        saved = checkpoint(mount.EXPERIMENT)
        self.validate(saved, migration=True, load_optimizer=False, restore_curriculum=True)
        for args in (dict(migration=True, load_optimizer=True, restore_curriculum=True),
                     dict(migration=True, load_optimizer=False, restore_curriculum=False)):
            with self.assertRaises(ValueError):
                self.validate(saved, **args)

    def test_resume_rejects_changed_rewards_mount_or_incomplete_state(self):
        args = dict(migration=False, load_optimizer=True, restore_curriculum=True)
        self.validate(checkpoint(), **args)
        for key in ("vd14_posture_profile", "vd13_mount_profile", "vd12_geometry_profile",
                    "vd5_curriculum_state", "optimizer_state_dict", "discriminator_state_dict"):
            saved = checkpoint()
            saved.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(saved, **args)
        saved = checkpoint()
        saved["vd14_posture_profile"]["front_min_width_m"] = .2
        with self.assertRaises(ValueError):
            self.validate(saved, **args)

    def test_wrong_source_cannot_be_renamed_by_loader(self):
        saved = checkpoint("b2_z1_vd5_world_seam_amp")
        with self.assertRaises(ValueError):
            self.validate(saved, migration=True, load_optimizer=False, restore_curriculum=True)
        self.assertEqual(saved["training_experiment_name"], "b2_z1_vd5_world_seam_amp")

    def test_environment_reward_overrides_cannot_leave_stale_profile(self):
        p = contract.posture_profile()
        cfg = NS(
            scene=NS(robot=NS(spawn=NS(asset_path=mount.ASSET))),
            commands=NS(ee_target=NS(vd5_seam_trailing_distance=.95)), sim=NS(dt=.005), decimation=4,
            vd13_mount_profile=mount.mount_profile(),
            vd12_geometry_profile=geometry.geometry_profile(mount.ASSET), vd14_posture_profile=p,
            rewards=NS(base_tilt=NS(func=tilt.base_tilt_penalty, weight=-8.,
                                    params=dict(deadzone=.07, scale=.15)),
                       vd14_front_width=NS(func=rewards.front_stance_width_penalty, weight=-2.,
                                           params=dict(minimum_width=.42, scale=.10,
                                                       asset_cfg=NS(body_names=["FL_foot", "FR_foot"],
                                                                    preserve_order=True)))),
        )
        contract.validate_environment_profile(cfg)
        cfg.rewards.base_tilt.weight = -3.
        with self.assertRaises(ValueError):
            contract.validate_environment_profile(cfg)

    def test_saved_checkpoint_contains_independent_posture_mount_and_curriculum(self):
        runner = runner_cpu.fake_runner(contract.EXPERIMENT)
        runner.env.cfg.vd12_geometry_profile = geometry.geometry_profile(mount.ASSET)
        runner.env.cfg.vd13_mount_profile = mount.mount_profile()
        runner.env.cfg.vd14_posture_profile = contract.posture_profile()
        saved_curriculum = dict(version=1, ground_blend=1.)
        ns = dict(torch=torch, deepcopy=deepcopy, os=os, tempfile=tempfile,
                  capture_vd5_curriculum_state=Mock(return_value=saved_curriculum))
        save = runner_cpu.runner_method("save", ns)
        with tempfile.TemporaryDirectory(prefix="vd14_checkpoint_") as directory:
            path = str(Path(directory) / "model_777.pt")
            save(runner, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.validate(saved, migration=False, load_optimizer=True, restore_curriculum=True)
            self.assertEqual(saved["vd5_curriculum_state"], saved_curriculum)
            self.assertEqual(saved["iter"], 777)
            self.assertFalse(list(Path(directory).glob("*.partial")))
            runner.checkpoint_retention.saved.assert_called_once_with(path, existed_before=False)
        runner.env.cfg.vd14_posture_profile["front_min_width_m"] = .2
        self.assertEqual(saved["vd14_posture_profile"]["front_min_width_m"], .42)

    def test_options_migration_resume_fresh_and_conflicting_modes(self):
        runner, env = Mock(), Mock()
        options.load_training_checkpoint(runner, env, contract.EXPERIMENT, vd14_from="vd13.pt")
        runner.load.assert_called_once_with("vd13.pt", load_optimizer=False,
                                           restore_curriculum=True, migrate_vd14=True)
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once()
        runner.reset_mock()
        options.load_training_checkpoint(runner, env, contract.EXPERIMENT, resume_path="vd14.pt")
        runner.load.assert_called_once_with("vd14.pt", restore_curriculum=True, legacy_curriculum=None)
        for source in ("vd13.pt", ""):
            for key in ("resume_path", "warm_start", "vd12_from", "vd13_from"):
                with self.subTest(source=source, key=key), self.assertRaises(ValueError):
                    options.load_training_checkpoint(Mock(), Mock(), contract.EXPERIMENT,
                                                     vd14_from=source, **{key: "other.pt"})
        runner = Mock()
        options.load_training_checkpoint(runner, Mock(), contract.EXPERIMENT, vd14_from="")
        runner.load.assert_not_called()

    def test_real_runner_migration_and_resume_semantics(self):
        for migration in (True, False):
            source = mount.EXPERIMENT if migration else contract.EXPERIMENT
            saved = checkpoint(source)
            method, namespace = runner_cpu.load_method(saved)
            runner = runner_cpu.fake_runner(contract.EXPERIMENT)
            runner.env.cfg.vd12_geometry_profile = saved["vd12_geometry_profile"]
            runner.env.cfg.vd13_mount_profile = mount.mount_profile()
            runner.env.cfg.vd14_posture_profile = contract.posture_profile()
            with patch.dict(sys.modules, {"b2_z1.assets.vd14_contract": contract}):
                method(runner, "saved.pt", load_optimizer=not migration,
                       restore_curriculum=True, migrate_vd14=migration)
            namespace["restore_vd5_curriculum_state"].assert_called_once()
            runner.alg.policy.load_state_dict.assert_called_once_with(saved["model_state_dict"])
            self.assertEqual(runner.alg.optimizer.load_state_dict.call_count, int(not migration))
            self.assertEqual(runner.current_learning_iteration, 0 if migration else 6199)

    def test_real_runner_rejects_bad_profile_before_mutating_any_state(self):
        saved = checkpoint()
        saved["vd14_posture_profile"]["front_min_width_m"] = .2
        method, namespace = runner_cpu.load_method(saved)
        runner = runner_cpu.fake_runner(contract.EXPERIMENT)
        runner.env.cfg.vd12_geometry_profile = saved["vd12_geometry_profile"]
        runner.env.cfg.vd13_mount_profile = mount.mount_profile()
        runner.env.cfg.vd14_posture_profile = contract.posture_profile()
        with patch.dict(sys.modules, {"b2_z1.assets.vd14_contract": contract}), self.assertRaises(ValueError):
            method(runner, "bad.pt", restore_curriculum=True)
        namespace["restore_vd5_curriculum_state"].assert_not_called()
        runner.alg.policy.load_state_dict.assert_not_called()
        runner.alg.optimizer.load_state_dict.assert_not_called()


class PostureConfigAndLaunch(unittest.TestCase):
    def test_launcher_default_resume_and_fresh_are_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="vd14_launch_") as directory:
            environment = dict(os.environ, VD14_PYTHON="/usr/bin/echo", VD14_CACHE_ROOT=directory)
            for extra in ([], ["--resume", "--load_run", "saved"], ["--vd14-from="]):
                command = subprocess.check_output(["bash", str(ROOT / "scripts/train_vd14.sh"), *extra],
                                                  env=environment, text=True)
                self.assertEqual("--vd14-from " in command, not extra)
                self.assertIn("--task Unitree-b2z1-VD14-Velocity", command)
                self.assertIn("--keep-last-checkpoints 0", command)
                self.assertIn("--logger tensorboard", command)

    def test_cli_conflicts_rejected_before_app_launch(self):
        for extra in (["--resume"], ["--vd13-from="], ["--vd12-from=x"], ["--vd5-warm-start=x"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner_cpu.parse_training_args(["--vd14-from=x", *extra])

    def test_nominal_probe_override_occurs_after_checkpoint_validation(self):
        tree = ast.parse((ROOT / "scripts/rsl_rl/train.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        statements = main.body
        load_index = next(i for i, node in enumerate(statements) if isinstance(node, ast.Expr)
                          and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name)
                          and node.value.func.id == "load_training_checkpoint")
        assignments = [i for i, node in enumerate(statements) for item in ast.walk(node)
                       if isinstance(item, ast.Assign) and any(
                           isinstance(target, ast.Attribute)
                           and target.attr == "reference_state_initialization_prob"
                           for target in item.targets)]
        self.assertTrue(assignments)
        self.assertTrue(all(i > load_index for i in assignments))

    def test_config_edits_only_tilt_and_front_width(self):
        @dataclass
        class Term:
            weight: float
            params: dict

            def replace(self, **kwargs):
                return replace(self, **kwargs)

        path = MDP.parent / "b2_z1_vd14_env_cfg.py"
        tree = ast.parse(path.read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_vd14")
        ns = dict(posture_profile=contract.posture_profile, RewTerm=lambda **kw: NS(**kw),
                  SceneEntityCfg=lambda *args, **kw: NS(name=args[0], **kw),
                  front_stance_width_penalty=rewards.front_stance_width_penalty)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        other = object()
        cfg = NS(rewards=NS(base_tilt=Term(-3., dict(asset_cfg="same", deadzone=.1, scale=.2)),
                            unchanged=other), scene=other, commands=other, observations=other)
        ns["_configure_vd14"](cfg)
        self.assertIs(cfg.scene, other)
        self.assertIs(cfg.commands, other)
        self.assertIs(cfg.observations, other)
        self.assertIs(cfg.rewards.unchanged, other)
        self.assertEqual(cfg.rewards.base_tilt.weight, -8.)
        self.assertEqual(cfg.rewards.base_tilt.params["asset_cfg"], "same")
        self.assertEqual(cfg.rewards.vd14_front_width.params["asset_cfg"].body_names, ["FL_foot", "FR_foot"])


if __name__ == "__main__":
    unittest.main()
