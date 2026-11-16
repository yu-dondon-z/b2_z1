"""Stubbed CPU integration of VD12 guard managers; never starts physics."""

import ast
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/b2_z1/b2_z1"
ASSETS = PACKAGE / "assets"
MDP = PACKAGE / "tasks/manager_based/b2_z1/mdp/vd12_rewards.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state = load_module("vd12_runtime_guard_state_cpu", ASSETS / "vd12_guard_state.py")
contract = load_module("vd12_runtime_contract_cpu", ASSETS / "vd12_contract.py")


class ManagerTermStub:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


class GeometryStub:
    """Predetermined measured-pose result, not fake physical contact."""

    created = []
    source_hash_override = None

    def __init__(self, urdf_path, body_names, **options):
        self.constructor = (urdf_path, body_names, options)
        digest = hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()
        self.metadata = {"source_urdf_sha256": self.source_hash_override or digest}
        self.minimum, self.valid = None, None
        self.calls = []
        type(self).created.append(self)

    def __call__(self, positions, quaternions):
        self.calls.append((positions, quaternions))
        minimum = self.minimum.clone()
        return {
            "minimum_clearance_lower_bound_m": minimum,
            "minimum_pair_index": torch.zeros_like(minimum, dtype=torch.long),
            "input_valid": self.valid.clone(),
        }


def load_runtime():
    """Execute actual manager implementations with no Isaac import side effect."""
    namespace = {
        "torch": torch, "ManagerTermBase": ManagerTermStub,
        "GunRobotClearance": GeometryStub, "GeometryWindow": state.GeometryWindow,
    }
    tree = ast.parse(MDP.read_text())
    nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MDP), "exec"), namespace)
    return SimpleNamespace(**namespace)


class VD12RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="vd12_runtime_test_")
        self.addCleanup(directory.cleanup)
        self.urdf = Path(directory.name) / "measured_gun.urdf"
        self.urdf.write_text('<robot name="stub"><link name="link06"/></robot>', encoding="utf-8")
        GeometryStub.created = []
        GeometryStub.source_hash_override = None
        self.runtime = load_runtime()
        self.env = self.make_env()

    def make_env(self, count=3):
        positions = torch.zeros(count, 2, 3)
        quaternions = torch.zeros(count, 2, 4)
        quaternions[..., 0] = 1.
        robot = SimpleNamespace(
            body_names=["base_link", "link06"],
            data=SimpleNamespace(body_pos_w=positions, body_quat_w=quaternions),
        )
        cfg = SimpleNamespace(
            decimation=4, sim=SimpleNamespace(dt=.005),
            vd12_geometry_profile=contract.geometry_profile(self.urdf),
            scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(self.urdf)))),
        )
        env = SimpleNamespace(
            physics_dt=.005, step_dt=.02, cfg=cfg, num_envs=count, device="cpu",
            scene={"robot": robot}, _physics_step_callbacks=[],
            common_step_counter=0, _sim_step_counter=0,
        )
        env.unwrapped = env
        return env

    def terms(self, reward_first=False):
        if reward_first:
            reward = self.runtime.GunClearancePenalty(SimpleNamespace(), self.env)
            termination = self.runtime.GunClearanceTermination(SimpleNamespace(), self.env)
        else:
            termination = self.runtime.GunClearanceTermination(SimpleNamespace(), self.env)
            reward = self.runtime.GunClearancePenalty(SimpleNamespace(), self.env)
        return termination, reward

    def sample(self, minima, valid=None):
        geometry = self.env._vd12_gun_guard.geometry
        geometry.minimum = torch.tensor(minima, dtype=torch.float32)
        geometry.valid = torch.ones(self.env.num_envs, dtype=torch.bool) if valid is None else torch.tensor(valid)
        self.env._sim_step_counter += 1
        for callback in tuple(self.env._physics_step_callbacks):
            callback()

    def actor_window(self, rows):
        for row in rows:
            self.sample(row)
        # This increment occurs after all four callbacks and before managers.
        self.env.common_step_counter += 1

    def test_term_first_registers_one_shared_guard_and_one_physics_callback(self):
        termination, reward = self.terms()
        self.assertIs(termination.guard, reward.guard)
        self.assertIs(termination.guard, self.runtime.geometry_guard(self.env))
        self.assertEqual(len(self.env._physics_step_callbacks), 1)
        self.assertEqual(len(GeometryStub.created), 1)
        options = GeometryStub.created[0].constructor[2]
        self.assertEqual(options["numerical_guard_m"], self.env.cfg.vd12_geometry_profile["numerical_guard_m"])
        self.assertEqual(options["minimum_quaternion_norm"], self.env.cfg.vd12_geometry_profile["minimum_quaternion_norm"])

    def test_reward_first_also_reuses_guard_without_manager_construction_dependency(self):
        termination, reward = self.terms(reward_first=True)
        self.assertIs(termination.guard, reward.guard)
        self.assertEqual(len(self.env._physics_step_callbacks), 1)
        self.assertEqual(len(GeometryStub.created), 1)

    def test_actual_link_pose_buffers_are_sampled_after_physics_not_targets_or_com(self):
        self.terms()
        self.sample([.05, .05, .05])
        positions, quaternions = GeometryStub.created[0].calls[0]
        self.assertIs(positions, self.env.scene["robot"].data.body_pos_w)
        self.assertIs(quaternions, self.env.scene["robot"].data.body_quat_w)
        self.assertEqual(self.env._vd12_gun_guard.window.last_physics_step, 1)

    def test_every_substep_latches_but_manager_rejects_incomplete_20ms_window(self):
        termination, reward = self.terms()
        for minima in ([.05, .05, .05], [-.001, .05, .05], [.05, .05, .05]):
            self.sample(minima)
            for reader in (termination, reward, self.runtime.gun_exclusion_failure):
                with self.subTest(step=self.env._sim_step_counter, reader=reader):
                    with self.assertRaisesRegex(RuntimeError, "four current 200 Hz"):
                        reader(self.env)
        self.sample([.05, .05, .05])
        self.env.common_step_counter += 1
        guard = self.env._vd12_gun_guard
        self.assertEqual(termination(self.env).tolist(), [True, False, False])
        self.assertEqual(guard.window.first_bad_physics_step.tolist(), [2, 0, 0])
        self.assertAlmostEqual(guard.window.minimum_m[0].item(), -.001, places=7)
        torch.testing.assert_close(guard.window.last_m, torch.full((3,), .05))
        self.assertGreater(reward(self.env)[0].item(), 0.)

    def test_termination_then_reward_then_failure_reads_same_unconsumed_window(self):
        termination, reward = self.terms()
        self.actor_window([[.03, .03, .03], [0., .015, -.001], [.03, .03, .03], [0., .015, .03]])
        expected = torch.tensor([.5, .125, ((.03 + .001) / .03) ** 2 / 4])
        first = termination(self.env)
        self.assertEqual(first.tolist(), [False, False, True])
        torch.testing.assert_close(reward(self.env), expected)
        self.assertEqual(self.runtime.gun_exclusion_failure(self.env).tolist(), [0., 0., 1.])
        # Readers can be reordered/repeated without consuming another's state.
        self.assertEqual(self.runtime.gun_exclusion_failure(self.env).tolist(), [0., 0., 1.])
        torch.testing.assert_close(reward(self.env), expected)
        self.assertEqual(termination(self.env).tolist(), first.tolist())
        first.fill_(False)
        self.assertEqual(termination(self.env).tolist(), [False, False, True])

    def test_partial_reset_is_idempotent_and_keeps_other_envs_and_pre_reset_dones(self):
        termination, reward = self.terms()
        self.actor_window([[-.001, -.002, .05]] * 4)
        captured = termination(self.env)
        before_cost = reward(self.env).clone()
        window = self.env._vd12_gun_guard.window
        ids = torch.tensor([0])
        # Real manager reset invokes both manager terms on selected env IDs.
        reward.reset(ids)
        termination.reset(ids)
        self.assertEqual(captured.tolist(), [True, True, False])
        self.assertEqual(window.violated.tolist(), [False, True, False])
        self.assertEqual(window.first_bad_physics_step.tolist(), [0, 1, 0])
        self.assertEqual(window.sample_count, 4)
        self.assertEqual(window.last_physics_step, 4)
        self.assertEqual(window.total_cost[0].item(), 0.)
        self.assertEqual(window.total_cost[1].item(), (before_cost[1] * 4).item())
        # The next global actor window clears old evidence for all envs only
        # after termination/reward have already observed the previous one.
        self.actor_window([[.05, .05, .05]] * 4)
        self.assertEqual(termination(self.env).tolist(), [False] * 3)
        self.assertEqual(reward(self.env).tolist(), [0.] * 3)
        self.assertEqual(window.first_bad_physics_step.tolist(), [0] * 3)

    def test_new_window_tracks_first_bad_in_current_episode_not_old_reset(self):
        termination, reward = self.terms()
        self.actor_window([[-.001, .05, .05]] * 4)
        self.assertEqual(termination(self.env).tolist(), [True, False, False])
        termination.reset(torch.tensor([0]))
        reward.reset(torch.tensor([0]))
        self.actor_window([[.05, .05, .05], [.05, -.001, .05], [.05, .05, .05], [.05, .05, .05]])
        self.assertEqual(termination(self.env).tolist(), [False, True, False])
        self.assertEqual(self.env._vd12_gun_guard.window.first_bad_physics_step.tolist(), [0, 6, 0])

    def test_invalid_geometry_input_fails_closed_without_nonfinite_training_reward(self):
        termination, reward = self.terms()
        self.sample([.05, .05, .05], valid=[False, True, True])
        for _ in range(3):
            self.sample([.05, .05, .05])
        self.env.common_step_counter += 1
        self.assertEqual(termination(self.env).tolist(), [True, False, False])
        self.assertTrue(torch.isfinite(reward(self.env)).all())
        self.assertAlmostEqual(reward(self.env)[0].item(), 9. / 4)
        self.assertEqual(self.env._vd12_gun_guard.window.first_bad_physics_step.tolist(), [1, 0, 0])

    def test_stale_physics_window_cannot_be_reused_as_current_reward_or_termination(self):
        termination, reward = self.terms()
        self.actor_window([[.05, .05, .05]] * 4)
        self.env._sim_step_counter += 1  # A physical step without the mandatory callback.
        for reader in (termination, reward, self.runtime.gun_exclusion_failure):
            with self.subTest(reader=reader), self.assertRaisesRegex(RuntimeError, "four current 200 Hz"):
                reader(self.env)

    def test_real_environment_step_orders_four_samples_then_term_reward_and_amp_reset(self):
        termination, reward = self.terms()
        env = self.env
        guard = env._vd12_gun_guard
        events, snapshots = [], []
        rows = [[.05, .05, .05], [-.001, .05, .05], [.05, .05, .05], [.05, .05, .05]]

        class SceneStub(dict):
            def write_data_to_sim(self):
                pass

            def update(self, dt):
                self_test.assertEqual(dt, .005)
                guard.geometry.minimum = torch.tensor(rows[env._sim_step_counter - 1])
                guard.geometry.valid = torch.ones(env.num_envs, dtype=torch.bool)
                events.append("physics_update")

        self_test = self
        env.scene = SceneStub(env.scene)
        env.cfg.sim.render_interval = 4
        env.cfg.rerender_on_reset = False
        env.sim = SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: False,
                                  step=Mock(), forward=Mock())
        env.action_manager = SimpleNamespace(prev_action=torch.zeros(env.num_envs, 18),
                                             process_action=Mock(), apply_action=Mock())
        env.recorder_manager = SimpleNamespace(active_terms=[], record_pre_step=Mock(),
                                              record_pre_reset=Mock(), record_post_reset=Mock())
        env.termination_manager = SimpleNamespace(terminated=None, time_outs=torch.zeros(env.num_envs, dtype=torch.bool))

        def term_compute():
            events.append("termination")
            self.assertEqual(env.common_step_counter, 1)
            self.assertEqual(guard.window.sample_count, 4)
            env.termination_manager.terminated = termination(env)
            return env.termination_manager.terminated.clone()

        def reward_compute(dt):
            events.append("reward")
            snapshots.append((guard.window.minimum_m.clone(), guard.window.first_bad_physics_step.clone()))
            return -reward(env) * dt - 100 * self.runtime.gun_exclusion_failure(env) * dt

        terminal_amp = torch.full((env.num_envs, 43), 7.)

        def reset_idx(ids):
            events.append("reset")
            reward.reset(ids)
            termination.reset(ids)
            env.episode_length_buf[ids] = 0
            terminal_amp[ids] = -3.  # Detect terminal-state capture after reset.

        env.termination_manager.compute = term_compute
        env.reward_manager = SimpleNamespace(compute=reward_compute)
        env._reset_idx = reset_idx
        env.command_manager = SimpleNamespace(compute=Mock())
        env.event_manager = SimpleNamespace(available_modes=[])
        env.observation_manager = SimpleNamespace(
            group_obs_dim={"amp_obs": 43}, compute_group=lambda group: terminal_amp,
            compute=lambda **kwargs: {"policy": torch.zeros(env.num_envs, 70)},
        )
        env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
        env.extras = {}
        path = PACKAGE / "envs/b2z1_rl_env.py"
        tree = ast.parse(path.read_text())
        env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == "B2Z1ManagerBasedRLEnv")
        method = next(node for node in env_class.body if getattr(node, "name", None) == "step")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        _, earned_reward, dones, _, extras = namespace["step"](env, torch.zeros(env.num_envs, 18))
        self.assertEqual(events, ["physics_update"] * 4 + ["termination", "reward", "reset"])
        self.assertEqual(len(guard.geometry.calls), 4)
        self.assertEqual(dones.tolist(), [True, False, False])
        self.assertEqual(guard.window.violated.tolist(), [False, False, False])
        self.assertEqual(snapshots[0][1].tolist(), [2, 0, 0])
        self.assertLess(snapshots[0][0][0].item(), 0.)
        self.assertLess(earned_reward[0].item(), -2.)
        self.assertEqual(extras["reset_env_ids"].tolist(), [0])
        torch.testing.assert_close(extras["terminal_amp_states"], torch.full((1, 43), 7.))

    def test_contract_factory_pins_actual_urdf_and_geometry_math_source(self):
        profile = contract.geometry_profile(self.urdf)
        self.assertEqual(profile["urdf_sha256"], hashlib.sha256(self.urdf.read_bytes()).hexdigest())
        self.assertEqual(profile["geometry_source_sha256"],
                         hashlib.sha256((ASSETS / "vd12_geometry.py").read_bytes()).hexdigest())
        profile["soft_margin_m"] = .99
        self.assertEqual(contract.geometry_profile(self.urdf)["soft_margin_m"], .03)

    def test_runtime_physics_mismatch_rejected_before_registering_callback(self):
        for dt, decimation in ((.01, 4), (.005, 2)):
            with self.subTest(dt=dt, decimation=decimation):
                env = self.make_env()
                env.physics_dt, env.cfg.decimation = dt, decimation
                with self.assertRaisesRegex(ValueError, "200 Hz"):
                    self.runtime.geometry_guard(env)
                self.assertFalse(env._physics_step_callbacks)

    def test_urdf_profile_mismatch_cannot_register_a_guard(self):
        GeometryStub.source_hash_override = "b" * 64
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            self.runtime.geometry_guard(self.env)
        self.assertFalse(self.env._physics_step_callbacks)
        self.assertFalse(hasattr(self.env, "_vd12_gun_guard"))

    def test_any_profile_drift_is_rejected_on_shared_guard_access(self):
        guard = self.runtime.geometry_guard(self.env)
        original = deepcopy(self.env.cfg.vd12_geometry_profile)
        for key in original:
            with self.subTest(key=key):
                self.env.cfg.vd12_geometry_profile = dict(original)
                self.env.cfg.vd12_geometry_profile[key] = "drift"
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    self.runtime.geometry_guard(self.env)
        self.env.cfg.vd12_geometry_profile = original
        self.assertIs(self.runtime.geometry_guard(self.env), guard)
        self.assertEqual(len(self.env._physics_step_callbacks), 1)

    def test_active_failure_reward_detects_profile_drift_in_actual_20ms_compute_path(self):
        self.terms()
        self.actor_window([[.05, .05, .05]] * 4)
        self.env.cfg.vd12_geometry_profile["soft_margin_m"] = .04
        with self.assertRaisesRegex(ValueError, "contract changed"):
            self.runtime.gun_exclusion_failure(self.env)

    def test_configuration_adds_only_explicit_geometry_terms_and_uses_profile_weights(self):
        path = PACKAGE / "tasks/manager_based/b2_z1/b2_z1_vd12_env_cfg.py"
        tree = ast.parse(path.read_text())
        configure_node = next(node for node in tree.body if getattr(node, "name", None) == "_configure_vd12")
        namespace = {
            "geometry_profile": contract.geometry_profile, "RewTerm": SimpleNamespace, "DoneTerm": SimpleNamespace,
            "GunClearancePenalty": self.runtime.GunClearancePenalty,
            "GunClearanceTermination": self.runtime.GunClearanceTermination,
            "gun_exclusion_failure": self.runtime.gun_exclusion_failure,
        }
        exec(compile(ast.Module(body=[configure_node], type_ignores=[]), str(path), "exec"), namespace)
        original_reward, original_termination = object(), object()
        cfg = self.env.cfg
        cfg.rewards, cfg.terminations = (SimpleNamespace(tracking=original_reward),
                                         SimpleNamespace(time_out=original_termination))
        namespace["_configure_vd12"](cfg)
        self.assertIs(cfg.rewards.tracking, original_reward)
        self.assertIs(cfg.terminations.time_out, original_termination)
        self.assertIs(cfg.rewards.vd12_gun_clearance.func, self.runtime.GunClearancePenalty)
        self.assertIs(cfg.rewards.vd12_geometry_failure.func, self.runtime.gun_exclusion_failure)
        self.assertIs(cfg.terminations.vd12_gun_exclusion.func, self.runtime.GunClearanceTermination)
        self.assertEqual(cfg.rewards.vd12_gun_clearance.weight, cfg.vd12_geometry_profile["clearance_reward_weight"])
        self.assertEqual(cfg.rewards.vd12_geometry_failure.weight, cfg.vd12_geometry_profile["failure_reward_weight"])
        self.assertEqual(cfg.vd12_geometry_profile, contract.geometry_profile(self.urdf))


if __name__ == "__main__":
    unittest.main()
