"""CPU contracts for read-only VD12 Play; never imports or starts Isaac."""

import ast
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnostics/play_vd12_checked.py"
VD12 = "b2_z1_vd12_geometry_guard_amp"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


play = load_file("vd12_checked_play_cpu", SCRIPT)
common = sys.modules["play_vd11_checked"]
contract = load_file("vd12_checked_profile_cpu", ROOT / "source/b2_z1/b2_z1/assets/vd12_contract.py")
URDF = ROOT / "source/b2_z1/b2_z1/assets/b2_z1/b2_z1_vd10_hollow_gun.urdf"


def saved_contract():
    profile = contract.geometry_profile(URDF)
    return (
        {"training_experiment_name": VD12, "iter": 2000, "vd12_geometry_profile": deepcopy(profile),
         "vd5_curriculum_state": {"ground_blend": 1.0}},
        {"experiment_name": VD12, "empirical_normalization": False, "clip_actions": None,
         "policy": {"activation": "elu", "actor_hidden_dims": [512, 256, 128]}},
        profile,
        {"vd12_geometry_profile": deepcopy(profile)},
    )


class TensorStub:
    """Only CPU array operations used by the real evaluation control loop."""

    def __init__(self, values):
        self.array = np.asarray(values)

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.array.tolist()

    def any(self):
        return bool(self.array.any())

    def all(self):
        return bool(self.array.all())

    def nonzero(self, as_tuple=False):
        if not as_tuple:
            return TensorStub(np.transpose(np.nonzero(self.array)))
        return tuple(TensorStub(indices) for indices in np.nonzero(self.array))

    def __or__(self, other):
        return TensorStub(self.array | other.array)

    def __and__(self, other):
        return TensorStub(self.array & other.array)


class VD12PlayContractTests(unittest.TestCase):
    def test_module_import_does_not_import_isaac_or_torch(self):
        source = (
            "import importlib.util,sys; "
            f"sys.path.insert(0,{str(SCRIPT.parent)!r}); "
            f"s=importlib.util.spec_from_file_location('cpu_play',{str(SCRIPT)!r}); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "assert not any(k.startswith(('isaaclab','omni','pxr','torch')) for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-B", "-c", source], check=True, timeout=30)

    def test_actual_vd12_contract_accepts_any_nonnegative_training_iteration(self):
        for iteration in (0, 1, 2000, 5999, 9000):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["iter"] = iteration
            with self.subTest(iteration=iteration):
                self.assertIsNone(play.validate_checkpoint(checkpoint, agent, profile, saved_env))
        self.assertEqual(play.TASK, "Unitree-b2z1-VD12-Velocity")
        self.assertIn("b2_z1_vd12_geometry_guard_amp", str(play.DEFAULT_CHECKPOINT))

    def test_rejects_vd11_vd5_and_missing_source_even_with_copied_profile(self):
        for experiment in (None, "b2_z1_vd5_world_seam_amp", "b2_z1_vd5_startup_amp",
                           "b2_z1_vd10_hollow_gun_amp", "b2_z1_vd11_motion_quality_amp"):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["training_experiment_name"] = experiment
            with self.subTest(experiment=experiment), self.assertRaisesRegex(ValueError, "actual VD12 checkpoint"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_saved_agent_must_independently_identify_vd12(self):
        for experiment in (None, "b2_z1_vd11_motion_quality_amp", "b2_z1_vd5_world_seam_amp"):
            checkpoint, agent, profile, saved_env = saved_contract()
            agent["experiment_name"] = experiment
            with self.subTest(experiment=experiment), self.assertRaisesRegex(ValueError, "agent experiment"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        checkpoint, agent, profile, saved_env = saved_contract()
        agent.pop("experiment_name")
        with self.assertRaisesRegex(ValueError, "agent experiment"):
            play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_rejects_missing_negative_boolean_string_or_fractional_iteration(self):
        for iteration in (None, -1, True, False, "2000", 2000., float("nan")):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["iter"] = iteration
            with self.subTest(iteration=iteration), self.assertRaisesRegex(ValueError, "iteration"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_every_profile_field_must_match_both_checkpoint_and_saved_environment(self):
        for source in ("checkpoint", "saved_env", "current"):
            for key in contract.geometry_profile(URDF):
                checkpoint, agent, profile, saved_env = saved_contract()
                target = (checkpoint["vd12_geometry_profile"] if source == "checkpoint" else
                          saved_env["vd12_geometry_profile"] if source == "saved_env" else profile)
                del target[key]
                with self.subTest(source=source, key=key), self.assertRaisesRegex(ValueError, "profiles differ"):
                    play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        for source in ("checkpoint", "saved_env"):
            checkpoint, agent, profile, saved_env = saved_contract()
            target = checkpoint if source == "checkpoint" else saved_env
            target.pop("vd12_geometry_profile")
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "profiles differ"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_matching_but_empty_missing_or_unsupported_profiles_are_not_a_contract(self):
        for bad_profile in (None, {}, {"version": 0}, {"version": 2}, {"version": "1"}):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["vd12_geometry_profile"] = deepcopy(bad_profile)
            saved_env["vd12_geometry_profile"] = deepcopy(bad_profile)
            with self.subTest(profile=bad_profile), self.assertRaisesRegex(ValueError, "profiles differ"):
                play.validate_checkpoint(checkpoint, agent, deepcopy(bad_profile), saved_env)

    def test_rejects_normalizer_state_and_missing_or_enabled_normalization(self):
        for enabled in (None, True, 0, "false"):
            checkpoint, agent, profile, saved_env = saved_contract()
            agent["empirical_normalization"] = enabled
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, "normalization"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        checkpoint, agent, profile, saved_env = saved_contract()
        checkpoint["obs_norm_state_dict"] = {}
        with self.assertRaisesRegex(ValueError, "normalization"):
            play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_rejects_missing_or_changed_wrapper_action_clipping(self):
        for value in (False, 0., 1., 100., "null"):
            checkpoint, agent, profile, saved_env = saved_contract()
            agent["clip_actions"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unclipped"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        checkpoint, agent, profile, saved_env = saved_contract()
        agent.pop("clip_actions")
        with self.assertRaisesRegex(ValueError, "unclipped"):
            play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_rejects_changed_actor_architecture(self):
        for key, value in (("activation", "relu"), ("actor_hidden_dims", [256, 128]),
                           ("actor_hidden_dims", [512, 256, 128, 64])):
            checkpoint, agent, profile, saved_env = saved_contract()
            agent["policy"][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "architecture"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_ground_curriculum_is_actual_saved_level_not_forced_final_or_initial(self):
        for blend in (0., .25, .8, 1.):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["vd5_curriculum_state"]["ground_blend"] = blend
            with self.subTest(blend=blend):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        for blend in (None, True, False, -.1, 1.1, float("inf"), float("nan"), "1"):
            checkpoint, agent, profile, saved_env = saved_contract()
            checkpoint["vd5_curriculum_state"]["ground_blend"] = blend
            with self.subTest(blend=blend), self.assertRaisesRegex(ValueError, "ground curriculum"):
                play.validate_checkpoint(checkpoint, agent, profile, saved_env)
        checkpoint, agent, profile, saved_env = saved_contract()
        checkpoint.pop("vd5_curriculum_state")
        with self.assertRaisesRegex(ValueError, "ground curriculum"):
            play.validate_checkpoint(checkpoint, agent, profile, saved_env)

    def test_reuses_common_ordering_validation_without_modifying_it(self):
        self.assertIs(play.validate_runtime_contract, common.validate_runtime_contract)
        self.assertIs(play.validate_joint_selection_contract, common.validate_joint_selection_contract)
        play.validate_runtime_contract(common.JOINT_NAMES, play.POLICY_TERMS, np.zeros((4, 70)))
        play.validate_joint_selection_contract(common.JOINT_NAMES, slice(None), list(range(18)), np.arange(18))
        wrong_names = list(common.JOINT_NAMES)
        wrong_names[8], wrong_names[9] = wrong_names[9], wrong_names[8]
        with self.assertRaisesRegex(ValueError, "joint order"):
            play.validate_runtime_contract(wrong_names, play.POLICY_TERMS, np.zeros((1, 70)))
        wrong_ids = list(range(18))
        wrong_ids[8], wrong_ids[9] = wrong_ids[9], wrong_ids[8]
        for index in range(3):
            selected = [slice(None)] * 3
            selected[index] = wrong_ids
            with self.subTest(selection=index), self.assertRaisesRegex(ValueError, "joint IDs"):
                play.validate_joint_selection_contract(common.JOINT_NAMES, *selected)
        for shape in ((70,), (1, 69), (1, 71), (1, 1, 70)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "70 actor"):
                play.validate_runtime_contract(common.JOINT_NAMES, play.POLICY_TERMS, np.zeros(shape))

    def test_native_episode_end_stops_loop_before_another_action_or_reset_episode_sample(self):
        """Run the real while-loop AST using deterministic stubs, no simulator."""
        tree = ast.parse(SCRIPT.read_text())
        run = next(node for node in tree.body if getattr(node, "name", None) == "run")
        loop = next(node for node in ast.walk(run) if isinstance(node, ast.While))
        calls = []
        state = {"physics_steps": 0, "actor_steps": 0, "terminal": None}
        done, timeout = TensorStub([True]), TensorStub([False])

        def env_step(actions):
            calls.append("native_step")
            state["physics_steps"] += 4
            # Returned observations already represent an autoreset. A second
            # actor call would mix episodes; the loop must stop immediately.
            return {"policy": TensorStub(np.full((1, 70), 999.))}, None, done, timeout, {}

        env = SimpleNamespace(step=env_step)
        raw = SimpleNamespace(
            num_envs=1, physics_dt=.005, step_dt=.02,
            termination_manager=SimpleNamespace(active_terms=["tool_support", "time_out"],
                get_term=lambda name: done if name == "tool_support" else timeout),
        )
        actor = Mock(return_value=TensorStub(np.zeros((1, 18))))
        namespace = dict(
            state,
            args=SimpleNamespace(duration=30., real_time=False),
            simulation_app=SimpleNamespace(is_running=lambda: True),
            torch=SimpleNamespace(inference_mode=nullcontext,
                isfinite=lambda value: TensorStub(np.isfinite(value.array))),
            observations={"policy": TensorStub(np.zeros((1, 70)))},
            raw=raw, env=env, actor=actor, actor_file=io.StringIO(), json=json, time=__import__("time"),
        )
        # env_step updates the execution namespace as a real callback does.
        state = namespace
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        self.assertEqual(calls, ["native_step"])
        self.assertEqual(actor.call_count, 1)
        self.assertEqual(namespace["physics_steps"], 4)
        self.assertEqual(namespace["actor_steps"], 1)
        self.assertEqual(namespace["terminal"]["reason"], "native_episode_end")
        self.assertEqual(namespace["terminal"]["causes"], {"tool_support": [0], "time_out": []})
        self.assertIn("only pre-reset physics samples", namespace["terminal"]["note"])
        recorded = [json.loads(line) for line in namespace["actor_file"].getvalue().splitlines()]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["observations"], [[0.] * 70])

    def test_entry_restores_curriculum_before_nominal_override_and_reset(self):
        tree = ast.parse(SCRIPT.read_text())
        run = next(node for node in tree.body if getattr(node, "name", None) == "run")
        calls = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(run)
                       if isinstance(node, ast.Call))

        def first(name):
            return next(line for line, function in calls if function == name)

        self.assertLess(first("validate_checkpoint"), first("gym.make"))
        self.assertLess(first("compare_training_settings"), first("gym.make"))
        self.assertLess(first("restore_vd5_curriculum_state"), first("env.reset"))
        overrides = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Assign)
                     and any(ast.unparse(target) == "raw.cfg.reference_state_initialization_prob" for target in node.targets)]
        self.assertEqual(len(overrides), 1)
        self.assertLess(first("restore_vd5_curriculum_state"), overrides[0])
        self.assertLess(overrides[0], first("env.reset"))
        for node in ast.walk(run):
            if isinstance(node, ast.Assign):
                self.assertFalse(any("_vd5_ground_blend" in ast.unparse(target)
                                     or "_vd5_ground_transition_steps" in ast.unparse(target)
                                     for target in node.targets), "Play must not force or freeze ground curriculum")
        self.assertIn("validate_joint_selection_contract", [function for _, function in calls])

    def test_no_render_frame_returns_before_touching_camera_viewport_or_video(self):
        tree = ast.parse(SCRIPT.read_text())
        run = next(node for node in tree.body if getattr(node, "name", None) == "run")
        frame = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "frame")
        # Preserve the original nested function, including its nonlocal. A
        # minimal outer scope makes every rendering dependency deliberately
        # unavailable: this must still return for --no-render.
        builder = ast.FunctionDef(name="build_frame", args=ast.arguments(posonlyargs=[], args=[],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=[
                ast.Assign(targets=[ast.Name(id="last_rgb", ctx=ast.Store())], value=ast.Constant(None)),
                frame, ast.Return(value=ast.Name(id="frame", ctx=ast.Load()))], decorator_list=[])
        namespace = {"args": SimpleNamespace(no_render=True)}
        module = ast.fix_missing_locations(ast.Module(body=[builder], type_ignores=[]))
        exec(compile(module, str(SCRIPT), "exec"), namespace)
        no_render_frame = namespace["build_frame"]()
        self.assertIsNone(no_render_frame("reset", 0., append_video=True))
        self.assertIsNone(no_render_frame("exclusion", .005, 1))

    def test_video_clock_is_fixed_25hz_and_not_snapshot_or_viewer_render_rate(self):
        tree = ast.parse(SCRIPT.read_text())
        run = next(node for node in tree.body if getattr(node, "name", None) == "run")
        frame = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "frame")
        record = next(node for node in ast.walk(run)
                      if isinstance(node, ast.FunctionDef) and node.name == "record_physics")
        append_guards = [node for node in ast.walk(frame) if isinstance(node, ast.If)
                         and any(isinstance(child, ast.Call) and ast.unparse(child.func) == "writer.append_data"
                                 for statement in node.body for child in ast.walk(statement))]
        self.assertEqual(len(append_guards), 1)
        self.assertEqual(ast.unparse(append_guards[0].test), "writer is not None and append_video")
        calls = [node for node in ast.walk(record) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "frame"]
        self.assertEqual(len(calls), 2)  # terminal snapshot and normal periodic render
        periodic = next(node for node in calls if any(keyword.arg == "append_video" for keyword in node.keywords))
        cadence = next(keyword.value for keyword in periodic.keywords if keyword.arg == "append_video")
        self.assertEqual(ast.unparse(cadence), "physics_steps % 8 == 0")
        terminal = next(node for node in calls if node is not periodic)
        self.assertFalse(any(keyword.arg == "append_video" for keyword in terminal.keywords))
        main = next(node for node in tree.body if getattr(node, "name", None) == "main")
        camera_assign = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                             and any(ast.unparse(target) == "args.enable_cameras" for target in node.targets))
        self.assertEqual(ast.unparse(camera_assign.value), "not args.no_render")

    def test_wrapper_preserves_rejection_interruption_missing_marker_and_child_error(self):
        with tempfile.TemporaryDirectory(prefix="vd12_play_wrapper_") as directory:
            root = Path(directory)
            executables = root / "bin"
            executables.mkdir()
            interpreter = executables / "fake_python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json,os,pathlib,sys\n"
                "with open(os.environ['VD12_FAKE_CALLS'],'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1]=='-c':\n"
                "    os.execv(sys.executable,[sys.executable]+sys.argv[1:])\n"
                "assert sys.argv[1]=='scripts/diagnostics/play_vd12_checked.py'\n"
                "assert not any(k.startswith(('isaaclab','omni','pxr','torch')) for k in sys.modules)\n"
                "marker=os.environ['VD12_FAKE_MARKER']\n"
                "if marker!='missing':\n"
                "    pathlib.Path(os.environ['VD12_PLAY_RESULT_FILE']).write_text(json.dumps({'exit_code':int(marker)}))\n"
                "sys.exit(int(os.environ['VD12_FAKE_CHILD']))\n",
                encoding="utf-8",
            )
            interpreter.chmod(0o755)
            mktemp = executables / "mktemp"
            mktemp.write_text(
                f"#!{sys.executable}\nimport os,tempfile\n"
                "print(tempfile.mkdtemp(prefix='cache_',dir=os.environ['VD12_FAKE_CACHE_ROOT']))\n",
                encoding="utf-8",
            )
            mktemp.chmod(0o755)
            for label, marker, child, expected, count in (
                    ("rejected", "2", 0, 2, 2), ("completed", "0", 0, 0, 2),
                    ("interrupted", "3", 0, 3, 2), ("missing", "missing", 0, 1, 1),
                    ("child_error", "2", 7, 7, 1)):
                with self.subTest(label=label):
                    calls = root / (label + ".jsonl")
                    environment = dict(os.environ, VD12_PYTHON=str(interpreter), VD12_FAKE_CALLS=str(calls),
                        VD12_FAKE_MARKER=marker, VD12_FAKE_CHILD=str(child), VD12_FAKE_CACHE_ROOT=str(root),
                        PATH=str(executables) + os.pathsep + os.environ["PATH"])
                    result = subprocess.run(["bash", str(ROOT / "scripts/play_vd12_checked.sh"), "--duration", ".1"],
                                            env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, expected, msg=result.stdout + result.stderr)
                    captured = [json.loads(line) for line in calls.read_text().splitlines()]
                    self.assertEqual(len(captured), count)
                    self.assertEqual(captured[0][-2:], ["--duration", ".1"])
                    if label == "missing":
                        self.assertIn("does not imply a pass", result.stderr)


if __name__ == "__main__":
    unittest.main()
