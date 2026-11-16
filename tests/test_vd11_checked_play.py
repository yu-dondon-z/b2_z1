"""CPU-only checks for the saved VD11 Play contracts, without Isaac imports."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnostics/play_vd11_checked.py"
spec = importlib.util.spec_from_file_location("vd11_checked_play_cpu", SCRIPT)
play = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play)


def actor_contract():
    return (
        {"training_experiment_name": "b2_z1_vd11_motion_quality_amp", "iter": 5999,
         "vd5_curriculum_state": {"ground_blend": 1.0}},
        {"empirical_normalization": False, "clip_actions": None,
         "policy": {"activation": "elu", "actor_hidden_dims": [512, 256, 128]}},
    )


def configuration():
    configuration = {
        "sim": {"dt": .005}, "decimation": 4, "episode_length_s": 30.,
        "reference_state_initialization": True,
        "reference_state_initialization_prob": .8,
        "observations": {"policy": {"enable_corruption": False}},
        "scene": {"num_envs": 256, "robot": {
            "spawn": {"articulation_props": {"enabled_self_collisions": False}},
            "init_state": {"joint_pos": {"joint1": 0., "joint2": .66647356},
                           "pos": (0., 0., .5)},
            "actuators": {"arm": {"joint_names_expr": ["joint1", "joint2"],
                "stiffness": 160., "damping": 5., "effort_limit_sim": 120.,
                "velocity_limit_sim": 3.1415}},
        }},
        "actions": {"JointPositionAction": {
            "scale": .25, "use_default_offset": True, "joint_names": [".*"],
            "preserve_order": False, "clip": {".*": (-100., 100.)}, "offset": 0.,
        }},
        "viewer": {"resolution": (1280, 720)},
    }
    for term in play.POLICY_TERMS:
        configuration["observations"]["policy"][term] = {
            "scale": .2 if term == "base_ang_vel" else .05 if term == "joint_vel_rel" else None,
            "clip": (-100., 100.),
            "params": {"command_name": "base_velocity"} if term == "velocity_commands"
                else {"command_name": "ee_target"} if term == "target_commands" else {},
        }
    return configuration


class VD11CheckedPlayTests(unittest.TestCase):
    def test_module_import_does_not_import_isaac_or_torch(self):
        source = (
            "import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('cpu_play',{str(SCRIPT)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert not any(k.startswith(('isaaclab','omni','pxr','torch')) for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-B", "-c", source], check=True, timeout=30)

    def test_accepts_saved_actor_contract(self):
        checkpoint, agent = actor_contract()
        self.assertIsNone(play.validate_actor_contract(checkpoint, agent))

    def test_rejects_wrong_experiment(self):
        for name in (None, "b2_z1_vd5_startup_amp", "b2_z1_vd10_hollow_gun_amp"):
            checkpoint, agent = actor_contract()
            checkpoint["training_experiment_name"] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Only the VD11"):
                play.validate_actor_contract(checkpoint, agent)

    def test_rejects_wrong_iteration_including_boolean(self):
        for iteration in (None, True, False, 5998, "5999", 6000):
            checkpoint, agent = actor_contract()
            checkpoint["iter"] = iteration
            with self.subTest(iteration=iteration), self.assertRaisesRegex(ValueError, "model_5999"):
                play.validate_actor_contract(checkpoint, agent)

    def test_rejects_missing_or_enabled_observation_normalization(self):
        for value in (None, True, 0, "false"):
            checkpoint, agent = actor_contract()
            agent["empirical_normalization"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "normalization"):
                play.validate_actor_contract(checkpoint, agent)
        checkpoint, agent = actor_contract()
        checkpoint["obs_norm_state_dict"] = {}
        with self.assertRaisesRegex(ValueError, "normalization"):
            play.validate_actor_contract(checkpoint, agent)

    def test_rejects_missing_or_changed_wrapper_action_clipping(self):
        for value in (1., 100., False, "null"):
            checkpoint, agent = actor_contract()
            agent["clip_actions"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unclipped"):
                play.validate_actor_contract(checkpoint, agent)
        checkpoint, agent = actor_contract()
        del agent["clip_actions"]
        with self.assertRaisesRegex(ValueError, "unclipped"):
            play.validate_actor_contract(checkpoint, agent)

    def test_rejects_wrong_actor_architecture(self):
        for key, value in (("activation", "relu"), ("actor_hidden_dims", [256, 128]),
                           ("actor_hidden_dims", [512, 256, 128, 64])):
            checkpoint, agent = actor_contract()
            agent["policy"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "architecture"):
                play.validate_actor_contract(checkpoint, agent)

    def test_rejects_unfinished_ground_curriculum(self):
        for value in (None, 0., .999, "1.0"):
            checkpoint, agent = actor_contract()
            checkpoint["vd5_curriculum_state"]["ground_blend"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "ground curriculum"):
                play.validate_actor_contract(checkpoint, agent)

    def test_runtime_preserves_interleaved_arm_joint_order(self):
        observations = np.zeros((4, 70))
        self.assertEqual(play.JOINT_NAMES[8], "joint1")
        self.assertIsNone(play.validate_runtime_contract(play.JOINT_NAMES, play.POLICY_TERMS, observations))
        reordered = list(play.JOINT_NAMES)
        reordered[8], reordered[9] = reordered[9], reordered[8]
        with self.assertRaisesRegex(ValueError, "joint order"):
            play.validate_runtime_contract(reordered, play.POLICY_TERMS, observations)

    def test_runtime_rejects_changed_observation_order_and_dimensions(self):
        reordered = list(play.POLICY_TERMS)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, "terms/order"):
            play.validate_runtime_contract(play.JOINT_NAMES, reordered, np.zeros((1, 70)))
        for shape in ((70,), (1, 69), (1, 71), (1, 1, 70)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "70 actor"):
                play.validate_runtime_contract(play.JOINT_NAMES, play.POLICY_TERMS, np.zeros(shape))

    def test_joint_selection_accepts_full_slices_lists_tuples_and_arrays(self):
        for ids in (slice(None), slice(0, 18, 1), list(range(18)),
                    tuple(range(18)), np.arange(18)):
            with self.subTest(ids=repr(ids)):
                self.assertIsNone(play.validate_joint_selection_contract(play.JOINT_NAMES, ids, ids, ids))

    def test_joint_selection_rejects_wrong_action_names(self):
        names = list(play.JOINT_NAMES)
        names[8], names[9] = names[9], names[8]
        with self.assertRaisesRegex(ValueError, "action joint names"):
            play.validate_joint_selection_contract(names, slice(None), slice(None), slice(None))

    def test_joint_selection_rejects_wrong_ids_for_each_action_or_observation(self):
        reordered = list(range(18))
        reordered[8], reordered[9] = reordered[9], reordered[8]
        bad_ids = (reordered, list(range(12)), list(range(1, 19)),
                   [0] * 18, slice(None, None, -1), slice(0, 18, 2), np.arange(18)[::-1])
        for index, label in enumerate(("action", "position observation", "velocity observation")):
            for values in bad_ids:
                selections = [slice(None), slice(None), slice(None)]
                selections[index] = values
                with self.subTest(label=label, values=repr(values)), self.assertRaisesRegex(
                    ValueError, "Resolved " + label + " joint IDs"
                ):
                    play.validate_joint_selection_contract(play.JOINT_NAMES, *selections)

    def test_side_view_follows_yaw_without_mutating_pose_inputs(self):
        base = np.array((2., -3., .5))
        original_base = base.copy()
        identity = np.array((1., 0., 0., 0.))
        original_identity = identity.copy()
        eye, target = play.side_view(base, identity)
        np.testing.assert_allclose(eye, base + (-1.4, -3., 1.1))
        np.testing.assert_allclose(target, base + (-.35, 0., .05))
        quarter_turn = np.array((np.sqrt(.5), 0., 0., np.sqrt(.5)))
        original_turn = quarter_turn.copy()
        turned_eye, turned_target = play.side_view(base, quarter_turn)
        np.testing.assert_allclose(turned_eye, base + (3., -1.4, 1.1), atol=1e-12)
        np.testing.assert_allclose(turned_target, base + (0., -.35, .05), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(turned_eye - turned_target), np.linalg.norm(eye - target))
        np.testing.assert_array_equal(base, original_base)
        np.testing.assert_array_equal(identity, original_identity)
        np.testing.assert_array_equal(quarter_turn, original_turn)

    def test_settings_normalize_yaml_tuples_without_mutating_config(self):
        current = configuration()
        before = copy.deepcopy(current)
        saved = json.loads(json.dumps(current))
        checks = play.compare_training_settings(SimpleNamespace(to_dict=lambda: current), saved)
        self.assertEqual(checks["scene.robot.init_state.pos"], [0., 0., .5])
        self.assertEqual(current, before)

    def test_settings_allow_only_rendering_and_environment_count_overrides(self):
        current, saved = configuration(), configuration()
        current["scene"]["num_envs"] = 4
        current["viewer"]["resolution"] = (960, 540)
        self.assertIsInstance(play.compare_training_settings(SimpleNamespace(to_dict=lambda: current), saved), dict)

    def test_settings_reject_physics_drive_reset_or_action_drift(self):
        changes = (("sim.dt", .01), ("decimation", 2),
                   ("reference_state_initialization_prob", 0.),
                   ("observations.policy.enable_corruption", True),
                   ("scene.robot.spawn.articulation_props.enabled_self_collisions", True),
                   ("scene.robot.actuators.arm.stiffness", 80.),
                   ("scene.robot.actuators.arm.effort_limit_sim", 30.),
                   ("actions.JointPositionAction.scale", .5),
                   ("actions.JointPositionAction.use_default_offset", False),
                   ("actions.JointPositionAction.joint_names", ["joint.*"]),
                   ("actions.JointPositionAction.preserve_order", True),
                   ("actions.JointPositionAction.clip", {".*": (-1., 1.)}),
                   ("actions.JointPositionAction.offset", .2),
                   ("observations.policy.base_ang_vel.scale", 1.),
                   ("observations.policy.joint_vel_rel.scale", 1.),
                   ("observations.policy.joint_pos_rel.clip", (-1., 1.)),
                   ("observations.policy.target_commands.params", {"command_name": "base_velocity"}))
        for path, value in changes:
            current, saved = configuration(), configuration()
            target = current
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "differs from saved"):
                play.compare_training_settings(SimpleNamespace(to_dict=lambda: current), saved)

    def test_yaml_python_tags_are_data_not_constructed_objects(self):
        with tempfile.TemporaryDirectory(prefix="vd11_yaml_test_") as directory:
            path = Path(directory) / "snapshot.yaml"
            path.write_text("pos: !!python/tuple [0, 0, 0.5]\nfunc: !!python/name:untrusted.function ''\n")
            loaded = play.load_saved_env(path)
            self.assertEqual(loaded["pos"], [0, 0, .5])
            self.assertIsInstance(loaded["func"], str)

    def test_wrapper_preserves_marker_results_and_rejects_silent_kit_exit(self):
        """Exercise the real shell wrapper, but never launch Kit or a simulator."""
        wrapper = ROOT / "scripts/play_vd11_checked.sh"
        with tempfile.TemporaryDirectory(prefix="vd11_wrapper_test_") as directory:
            temporary = Path(directory)
            executables = temporary / "bin"
            executables.mkdir()
            interpreter = executables / "fake_vd11_python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json,os,pathlib,sys\n"
                "with open(os.environ['VD11_FAKE_CALLS'], 'a') as calls:\n"
                "    calls.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1] == '-c':\n"
                "    os.execv(sys.executable, [sys.executable]+sys.argv[1:])\n"
                "assert sys.argv[1] == 'scripts/diagnostics/play_vd11_checked.py'\n"
                "assert not any(k.startswith(('isaaclab','omni','pxr','torch')) for k in sys.modules)\n"
                "marker_exit = os.environ['VD11_FAKE_MARKER_EXIT']\n"
                "if marker_exit != 'missing':\n"
                "    pathlib.Path(os.environ['VD11_PLAY_RESULT_FILE']).write_text(json.dumps(\n"
                "        {'exit_code':int(marker_exit),'status':'invalid' if marker_exit=='2' else 'safe_sampled_geometry'}))\n"
                "sys.exit(int(os.environ['VD11_FAKE_CHILD_EXIT']))\n"
            )
            interpreter.chmod(0o755)
            # Keep even the wrapper's hard-coded /dev/shm mktemp request inside
            # this automatically cleaned test directory, not the user's cache.
            mktemp = executables / "mktemp"
            mktemp.write_text(
                f"#!{sys.executable}\n"
                "import os,tempfile\n"
                "print(tempfile.mkdtemp(prefix='playcache_',dir=os.environ['VD11_FAKE_CACHE_ROOT']))\n"
            )
            mktemp.chmod(0o755)
            for label, marker_exit, child_exit, expected, expected_calls in (
                ("invalid", "2", 0, 2, 2),
                ("safe", "0", 0, 0, 2),
                ("missing", "missing", 0, 1, 1),
                ("child_error", "2", 7, 7, 1),
            ):
                with self.subTest(result=label):
                    calls = temporary / (label + "_calls.jsonl")
                    environment = dict(os.environ,
                        VD11_PYTHON=str(interpreter),
                        VD11_FAKE_CACHE_ROOT=str(temporary),
                        VD11_FAKE_CALLS=str(calls),
                        VD11_FAKE_MARKER_EXIT=marker_exit,
                        VD11_FAKE_CHILD_EXIT=str(child_exit),
                        PATH=str(executables) + os.pathsep + os.environ["PATH"])
                    result = subprocess.run(["bash", str(wrapper), "--duration", "0.1"],
                        env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, expected, msg=result.stdout + result.stderr)
                    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
                    self.assertEqual(len(recorded), expected_calls)
                    self.assertEqual(recorded[0][0], "scripts/diagnostics/play_vd11_checked.py")
                    if expected_calls == 2:
                        self.assertEqual(recorded[1][0], "-c")
                    if label == "missing":
                        self.assertIn("did not finish a result", result.stderr)


if __name__ == "__main__":
    unittest.main()
