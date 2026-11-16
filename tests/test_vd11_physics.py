"""CPU checks of measured world jitter and real first-contact event rewards."""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = ModuleType("vd11_physics_cpu")
package.__path__ = [str(MDP)]
sys.modules[package.__name__] = package
physics = load("vd11_physics_cpu.vd11_physics", MDP / "vd11_physics.py")
tcp = load("vd11_physics_cpu.tcp_frame", MDP / "tcp_frame.py")


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


with patch.dict(sys.modules, {"isaaclab.managers": NS(ManagerTermBase=ManagerTermBase)}):
    rewards = load("vd11_physics_cpu.vd11_rewards", MDP / "vd11_rewards.py")


def quaternion(yaw, dtype=torch.float64):
    return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=dtype)


def filter_signal(signal, *, active_at=0, yaw=0.0, deadzone=0.0, lateral_weight=2.0, cutoff=3.0):
    lowpass = torch.zeros(1, 3, dtype=signal.dtype)
    second = torch.zeros_like(lowpass)
    valid = torch.zeros(1, dtype=torch.bool)
    penalties, residuals = [], []
    for tick, sample in enumerate(signal):
        penalty, lowpass, second, valid, residual = physics.world_tcp_highpass_penalty(
            sample.unsqueeze(0), lowpass, second, valid,
            torch.tensor([tick >= active_at]), quaternion(yaw, signal.dtype),
            .005, cutoff=cutoff, deadzone=deadzone, lateral_weight=lateral_weight,
        )
        penalties.append(penalty)
        residuals.append(residual)
    return torch.cat(penalties), torch.cat(residuals)


def impact_signal(values, *, dt=.005, force_limit=804.0, force_scale=804.0):
    force = torch.as_tensor(values, dtype=torch.float64)
    if force.ndim == 1:
        force = force[:, None, None]
    shape = force.shape[1:]
    airborne = torch.zeros(shape, dtype=torch.long)
    window = torch.zeros_like(airborne)
    peak = torch.zeros(shape, dtype=torch.float64)
    increments, onsets = [], []
    for sample in force:
        increment, airborne, window, peak, onset = physics.foot_impact_step(
            sample, airborne, window, peak, dt,
            force_limit=force_limit, force_scale=force_scale,
        )
        increments.append(increment)
        onsets.append(onset)
    return torch.stack(increments), torch.stack(onsets), (airborne, window, peak)


class VD11WorldFilterTests(unittest.TestCase):
    def test_candidate_one_hz_filter_separates_slow_curve_and_gait_band(self):
        time = torch.arange(2000, dtype=torch.float64) * .005
        energies = []
        for frequency in (.1, 2.0):
            signal = torch.zeros(2000, 3, dtype=torch.float64)
            signal[:, 1] = .01 * torch.sin(2 * math.pi * frequency * time)
            penalty, _ = filter_signal(signal, active_at=400, cutoff=1.0, deadzone=.001)
            energies.append(penalty[800:].mean().item())
        self.assertLess(energies[0], 1e-9)
        self.assertGreater(energies[1], .5)

    def test_constant_position_and_uniform_travel_have_zero_steady_cost(self):
        time = torch.arange(1000, dtype=torch.float64) * .005
        for velocity in (0.0, .2, -.4):
            signal = torch.zeros(1000, 3, dtype=torch.float64)
            signal[:, 0] = 50.0 + velocity * time
            signal[:, 1:] = torch.tensor([17., .8])
            penalty, residual = filter_signal(signal)
            self.assertLess(penalty[400:].max().item(), 1e-15)
            self.assertLess(residual[400:].abs().max().item(), 1e-9)

    def test_same_amplitude_ten_hz_jitter_is_not_slow_path_curvature(self):
        time = torch.arange(1000, dtype=torch.float64) * .005
        energies = []
        for frequency in (.5, 10.):
            signal = torch.zeros(1000, 3, dtype=torch.float64)
            signal[:, 1] = .01 * torch.sin(2 * math.pi * frequency * time)
            penalty, _ = filter_signal(signal)
            energies.append(penalty[400:].mean().item())
        self.assertGreater(energies[1], energies[0] * 100)
        self.assertGreater(energies[1], 1.0)

    def test_rotating_base_does_not_create_jitter_at_stationary_world_tcp(self):
        position = torch.tensor([[12., -4., .6]], dtype=torch.float64)
        lowpass, second = torch.zeros_like(position), torch.zeros_like(position)
        valid = torch.zeros(1, dtype=torch.bool)
        for tick in range(600):
            yaw = math.sin(tick * .005 * 2 * math.pi * 10.)
            penalty, lowpass, second, valid, residual = physics.world_tcp_highpass_penalty(
                position, lowpass, second, valid, torch.ones(1, dtype=torch.bool),
                quaternion(yaw), .005,
            )
            torch.testing.assert_close(penalty, torch.zeros_like(penalty))
            torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_yaw_projection_and_lateral_weight_use_same_frame_after_filter(self):
        vector = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        projected = physics.yaw_frame_components(vector, quaternion(math.pi / 2))
        torch.testing.assert_close(projected, torch.tensor([[2., -1., 3.]], dtype=torch.float64))
        position = torch.tensor([[.02, 0., 0.]], dtype=torch.float64)
        lowpass, second = torch.zeros_like(position), torch.zeros_like(position)
        valid = active = torch.ones(1, dtype=torch.bool)
        costs = []
        residuals = []
        for yaw in (0., math.pi / 2):
            cost, _, _, _, residual = physics.world_tcp_highpass_penalty(
                position, lowpass, second, valid, active, quaternion(yaw), .005, deadzone=0.,
            )
            costs.append(cost)
            residuals.append(residual)
        torch.testing.assert_close(costs[1], costs[0] * 2)
        torch.testing.assert_close(residuals[1], residuals[0])

    def test_two_second_preheat_does_not_restart_uniform_travel_filter(self):
        time = torch.arange(600, dtype=torch.float64) * .005
        signal = torch.zeros(600, 3, dtype=torch.float64)
        signal[:, 0] = .3 * time
        penalty, residual = filter_signal(signal, active_at=400)
        torch.testing.assert_close(penalty[:400], torch.zeros_like(penalty[:400]))
        self.assertLess(penalty[400].item(), 1e-15)
        self.assertLess(residual[400].abs().max().item(), 1e-9)

    def test_reset_seeds_real_teleport_without_creating_a_filter_spike(self):
        position = torch.tensor([[20., -1., 4.]], dtype=torch.float64)
        penalty, first, second, valid, residual = physics.world_tcp_highpass_penalty(
            position, torch.full_like(position, .5), torch.full_like(position, .2),
            torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), quaternion(.7), .005,
        )
        torch.testing.assert_close(first, position)
        torch.testing.assert_close(second, torch.zeros_like(second))
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        self.assertTrue(valid.all())
        self.assertEqual(penalty.item(), 0.0)


class VD11ImpactMathTests(unittest.TestCase):
    def test_real_onset_first_frame_peak_is_included(self):
        increments, onsets, _ = impact_signal([0.] * 4 + [1608.] + [200.] * 15)
        self.assertTrue(onsets[4, 0, 0])
        self.assertEqual(onsets.sum().item(), 1)
        self.assertEqual(increments[4, 0, 0].item(), 1.0)
        self.assertEqual(increments.sum().item(), 1.0)

    def test_no_contact_and_initial_supported_foot_are_not_fictional_impacts(self):
        for samples in ([0.] * 100, [2000.] * 100, [0.] * 3 + [2000.] * 97):
            increments, onsets, _ = impact_signal(samples)
            self.assertEqual(increments.sum().item(), 0.0)
            self.assertFalse(onsets.any())

    def test_support_force_and_chatter_are_not_repeatedly_penalized(self):
        values = [0.] * 4 + [1608.] * 20 + [0.] * 3 + [1608.] * 20
        increments, onsets, _ = impact_signal(values)
        self.assertEqual(onsets.sum().item(), 1)
        self.assertEqual(increments.sum().item(), 1.0)
        values += [0.] * 4 + [1608.] * 20
        increments, onsets, _ = impact_signal(values)
        self.assertEqual(onsets.sum().item(), 2)
        self.assertEqual(increments.sum().item(), 2.0)

    def test_fifty_ms_window_includes_onset_plus_ten_following_samples(self):
        self.assertEqual(physics.landing_sample_counts(.005), (4, 11))
        for offset, expected in ((0, 1.), (10, 1.), (11, 0.)):
            values = [0.] * 4 + [200.] * 20
            values[4 + offset] = 1608.
            increments, _, _ = impact_signal(values)
            self.assertEqual(increments.sum().item(), expected)

    def test_only_new_event_peak_increments_are_integrated_once(self):
        values = [0.] * 4 + [1206., 1608., 1407., 2412.] + [200.] * 20
        increments, onsets, _ = impact_signal(values)
        torch.testing.assert_close(increments[4:8, 0, 0], torch.tensor([.25, .75, 0., 3.], dtype=torch.float64))
        self.assertEqual(increments.sum().item(), 4.0)
        self.assertEqual(onsets.sum().item(), 1)

    def test_event_cost_is_independent_of_actor_boundary_and_sample_rate(self):
        for dt in (.005, .0025):
            airborne = round(.02 / dt)
            for padding in range(4):
                increments, _, _ = impact_signal([0.] * (airborne + padding) + [1608.] * 40, dt=dt)
                # Summing event increments, not averaging over physics samples,
                # makes this cost independent of both onset substep and dt.
                self.assertEqual(increments.sum().item(), 1.0)

    def test_vectorized_feet_and_envs_do_not_share_rearming(self):
        values = torch.full((20, 2, 4), 2000., dtype=torch.float64)
        values[:4, 0, 1] = 0.
        values[:4, 1, 3] = 0.
        increments, onsets, _ = impact_signal(values)
        expected = torch.zeros(2, 4, dtype=torch.float64)
        expected[0, 1] = expected[1, 3] = ((2000. - 804.) / 804.) ** 2
        torch.testing.assert_close(increments.sum(0), expected)
        self.assertEqual(onsets.sum().item(), 2)

    def test_invalid_parameters_are_rejected(self):
        for dt, airborne, window in ((0., .02, .05), (.005, 0., .05), (.005, .02, -1.),
                                     (.005, math.nan, .05)):
            with self.assertRaises(ValueError):
                physics.landing_sample_counts(dt, airborne, window)


class Scene(dict):
    env_origins = torch.zeros(2, 3)


def environment():
    names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    asset = NS(body_names=names, find_bodies=lambda name: ([0], [name]))
    sensor = NS(body_names=names, cfg=NS(update_period=.005),
                data=NS(net_forces_w=torch.zeros(2, 4, 3)))
    env = NS(num_envs=2, device="cpu", physics_dt=.005, step_dt=.02,
             cfg=NS(decimation=4, scene=NS(terrain=NS(terrain_type="plane"))),
             episode_length_buf=torch.full((2,), 100),
             scene=Scene(robot=asset, contact_forces=sensor), _physics_step_callbacks=[])
    return env


def foot_config(**extra):
    return NS(params={"asset_cfg": NS(name="robot", body_ids=[0, 1, 2, 3]),
                      "sensor_cfg": NS(name="contact_forces", body_ids=[0, 1, 2, 3]), **extra})


class VD11RewardIntegrationTests(unittest.TestCase):
    def test_candidate_300n_target_includes_moderate_first_contact_peak(self):
        env = environment()
        cfg = foot_config(force_limit=300.0, force_scale=200.0)
        term = rewards.PhysicsRateFootImpact(cfg, env)
        for _ in range(4):
            term._sample()
        term(env, **cfg.params)
        env.scene["contact_forces"].data.net_forces_w[:, 2, 2] = 500.0
        for _ in range(4):
            term._sample()
        # (500-300)^2/200^2, averaged over four feet, integrated once.
        torch.testing.assert_close(term(env, **cfg.params) * env.step_dt, torch.full((2,), .25))

    def test_drive_torque_is_never_used_as_contact_force(self):
        env = environment()
        # A large actuator output is not evidence of a foot contact or impact.
        env.scene["robot"].data = NS(applied_torque=torch.full((2, 18), 10000.0))
        cfg = foot_config()
        term = rewards.PhysicsRateFootImpact(cfg, env)
        for _ in range(4):
            term._sample()
        torch.testing.assert_close(term(env, **cfg.params), torch.zeros(2))
        self.assertFalse(term.last_onset.any())
        # Only changing the measured foot sensor creates a landing event.
        env.scene["contact_forces"].data.net_forces_w[:, 0, 2] = 1608.
        for _ in range(4):
            term._sample()
        torch.testing.assert_close(term(env, **cfg.params) * env.step_dt, torch.full((2,), .25))

    def test_callbacks_observe_actual_contact_and_integrate_dt_exactly_once(self):
        env = environment()
        cfg = foot_config()
        term = rewards.PhysicsRateFootImpact(cfg, env)
        self.assertEqual(len(env._physics_step_callbacks), 1)
        costs = []
        for _ in range(4):
            term._sample()
        costs.append(term(env, **cfg.params) * env.step_dt)
        # First sample of the next actor interval is the genuine force peak.
        env.scene["contact_forces"].data.net_forces_w[:, 0, 2] = 1608.
        for _ in range(4):
            term._sample()
        costs.append(term(env, **cfg.params) * env.step_dt)
        for _ in range(8):
            term._sample()
            if term.sample_count == 4:
                costs.append(term(env, **cfg.params) * env.step_dt)
        # Four-foot mean: exactly 1/4 for the one landing foot, not 1/16.
        torch.testing.assert_close(torch.stack(costs).sum(0), torch.full((2,), .25))

    def test_episode_reset_clears_only_requested_env_without_phantom_onset(self):
        env = environment()
        term = rewards.PhysicsRateFootImpact(foot_config(), env)
        term.airborne_samples[:] = 4
        term.window_remaining[:] = 5
        term.peak_cost[:] = 1.
        term.total[:] = 2.
        term.reset(torch.tensor([0]))
        self.assertEqual(term.airborne_samples[0].sum().item(), 0)
        self.assertEqual(term.window_remaining[0].sum().item(), 0)
        self.assertEqual(term.peak_cost[0].sum().item(), 0)
        self.assertEqual(term.total[0].item(), 0)
        self.assertEqual(term.total[1].item(), 2)
        env.scene["contact_forces"].data.net_forces_w[:, :, 2] = 1608.
        term._sample()
        self.assertFalse(term.last_onset[0].any())
        self.assertTrue(term.last_onset[1].all())

    def test_world_tcp_callback_ignores_commands_and_preheats_inactive_filter(self):
        env = environment()
        env.command_manager = None  # A held 50 Hz goal is deliberately not read.
        cfg = NS(params={"asset_cfg": NS(name="robot"), "active_after": 2.0})
        term = rewards.PhysicsRateWorldTCPHighpass(cfg, env)
        identity = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
        for tick in range(500):
            env.episode_length_buf[:] = tick // 4
            position = torch.tensor([[tick * .005 * .2, 0., .5]]).repeat(2, 1)
            with patch.object(rewards, "tcp_pose_w", return_value=(position, identity)), \
                 patch.object(rewards, "link_pose_w", return_value=(torch.zeros(2, 3), identity)):
                term._sample()
            if term.sample_count == 4:
                result = term(env, **cfg.params)
                torch.testing.assert_close(result, torch.zeros_like(result))
        term.reset(torch.tensor([0]))
        self.assertFalse(term.valid[0])
        self.assertTrue(term.valid[1])
        self.assertEqual(term.lowpass_highpass[0].abs().sum().item(), 0.)

    def test_bad_rate_terrain_sensor_period_and_foot_order_are_rejected(self):
        for key in ("rate", "terrain", "sensor", "order"):
            env = environment()
            cfg = foot_config()
            if key == "rate":
                env.physics_dt = .01
            elif key == "terrain":
                env.cfg.scene.terrain.terrain_type = "generator"
            elif key == "sensor":
                env.scene["contact_forces"].cfg.update_period = .02
            else:
                cfg.params["sensor_cfg"].body_ids = [1, 0, 2, 3]
            with self.assertRaises(ValueError):
                rewards.PhysicsRateFootImpact(cfg, env)

    def test_missing_physics_samples_are_not_silently_accepted(self):
        env = environment()
        term = rewards.PhysicsRateFootImpact(foot_config(), env)
        term._sample()
        with self.assertRaisesRegex(RuntimeError, "expected 4 physics samples"):
            term(env, **foot_config().params)


if __name__ == "__main__":
    unittest.main()
