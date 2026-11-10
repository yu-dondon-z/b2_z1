"""CPU checks for tool support detection, reset boundaries and base inclination."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


MODULE = Path(__file__).resolve().parents[1] / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vd3_support.py"
spec = importlib.util.spec_from_file_location("vd3_support_cpu", MODULE)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def force_history(load, steps=20):
    force = torch.zeros(1, steps, 3, dtype=torch.float64)
    force[..., 2] = load
    return force


def make_env(load=0.0, steps=20, update_period=0.005):
    # The other body deliberately carries huge load to catch wrong-body reads.
    history = torch.zeros(1, steps, 2, 3, dtype=torch.float64)
    history[:, :, 0, 2] = 1000.0
    history[:, :, 1, 2] = load
    sensor = SimpleNamespace(
        body_names=["FL_foot", "link06"],
        cfg=SimpleNamespace(update_period=update_period),
        data=SimpleNamespace(net_forces_w_history=history),
    )
    scene = {"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.]])))}

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    env = SimpleNamespace(scene=Scene(scene), cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.005)))
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[1])
    return env, sensor_cfg, sensor


class UpwardSupportTests(unittest.TestCase):
    def test_small_working_contacts_are_permitted(self):
        for force in (0.0, 1.0, 5.0):
            history = force_history(force)
            self.assertFalse(support.upward_support_diagnostics(history)["tool_support"].item())
            self.assertEqual(support.upward_support_force_penalty(history).item(), 0.0)

    def test_sustained_loading_requires_every_physics_interval(self):
        history = force_history(21.0)
        self.assertTrue(support.upward_support_diagnostics(history)["sustained_support"].item())
        history[:, 7, 2] = 0.0
        result = support.upward_support_diagnostics(history)
        self.assertFalse(result["tool_support"].item())
        self.assertEqual(result["min_upward_force"].item(), 0.0)
        self.assertAlmostEqual(result["support_sample_fraction"].item(), 19.0 / 20.0)

    def test_impact_is_detected_before_sustained_window_is_filled(self):
        history = force_history(0.0)
        history[:, 0, 2] = 151.0
        result = support.upward_support_diagnostics(history)
        self.assertFalse(result["sustained_support"].item())
        self.assertTrue(result["peak_support"].item())
        self.assertTrue(result["tool_support"].item())

    def test_thresholds_are_strict(self):
        self.assertFalse(support.upward_support_diagnostics(force_history(20.0))["sustained_support"].item())
        history = force_history(0.0)
        history[:, 0, 2] = 150.0
        self.assertFalse(support.upward_support_diagnostics(history)["peak_support"].item())

    def test_horizontal_and_downward_forces_do_not_support_weight(self):
        history = force_history(-500.0)
        history[..., :2] = 1000.0
        result = support.upward_support_diagnostics(history)
        self.assertFalse(result["tool_support"].item())
        self.assertEqual(result["peak_upward_force"].item(), 0.0)
        self.assertEqual(support.upward_support_force_penalty(history).item(), 0.0)

    def test_force_cost_keeps_increasing_at_large_loads(self):
        self.assertAlmostEqual(support.upward_support_force_penalty(force_history(45.0)).item(), 0.4)
        self.assertAlmostEqual(support.upward_support_force_penalty(force_history(1005.0)).item(), 10.0)

    def test_multiple_environments_are_independent(self):
        history = torch.cat([force_history(5.0), force_history(21.0), force_history(0.0)])
        history[2, 0, 2] = 151.0
        result = support.upward_support_diagnostics(history)
        self.assertEqual(result["tool_support"].tolist(), [False, True, True])
        self.assertEqual(result["sustained_support"].tolist(), [False, True, False])

    def test_invalid_thresholds_and_shapes_are_rejected(self):
        history = force_history(1.0)
        for kwargs in ({"sustained_force": 0.0}, {"peak_force": float("nan")}, {"peak_force": 10.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                support.upward_support_diagnostics(history, **kwargs)
        for kwargs in ({"deadzone": -1.0}, {"force_scale": 0.0}, {"force_scale": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                support.upward_support_force_penalty(history, **kwargs)
        for invalid in (torch.zeros(20, 3), torch.zeros(1, 0, 3), torch.zeros(1, 20, 2)):
            with self.assertRaises(ValueError):
                support.upward_support_diagnostics(invalid)


class EnvironmentIntegrationTests(unittest.TestCase):
    def test_twenty_physics_samples_form_the_point_one_second_window(self):
        env, cfg, sensor = make_env(21.0)
        self.assertTrue(support.tool_support_termination(env, cfg).item())
        self.assertAlmostEqual(support.tool_support_force(env, cfg).item(), 0.16)
        sensor.data.net_forces_w_history = sensor.data.net_forces_w_history[:, :19]
        with self.assertRaisesRegex(ValueError, "20 physics samples"):
            support.tool_support_termination(env, cfg)

    def test_reset_zero_history_cannot_terminate_for_sustained_support(self):
        env, cfg, sensor = make_env(21.0)
        history = sensor.data.net_forces_w_history
        history[:, 19, 1] = 0.0  # One reset-filled interval remains.
        self.assertFalse(support.tool_support_termination(env, cfg).item())
        history[:, :, 1] = 0.0
        self.assertFalse(support.tool_support_termination(env, cfg).item())
        self.assertEqual(support.tool_support_force(env, cfg).item(), 0.0)

    def test_history_uses_the_newest_window_and_correct_body(self):
        env, cfg, sensor = make_env(0.0, steps=24)
        sensor.data.net_forces_w_history[:, 20:, 1, 2] = 1000.0
        self.assertFalse(support.tool_support_termination(env, cfg).item())
        self.assertEqual(support.tool_support_diagnostics(env, cfg)["peak_upward_force"].item(), 0.0)
        cfg.body_ids = [0]
        with self.assertRaisesRegex(ValueError, "link06"):
            support.tool_support_termination(env, cfg)
        cfg.body_ids = slice(None)
        with self.assertRaisesRegex(ValueError, "exactly"):
            support.tool_support_termination(env, cfg)

    def test_noninteger_duration_rounds_up_to_complete_physics_intervals(self):
        env, cfg, _ = make_env(21.0)
        with self.assertRaisesRegex(ValueError, "21 physics samples"):
            support.tool_support_termination(env, cfg, duration_s=0.101)

    def test_sensor_must_sample_at_physics_rate(self):
        for period in (0.0, 0.005):
            env, cfg, _ = make_env(21.0, update_period=period)
            self.assertTrue(support.tool_support_termination(env, cfg).item())
        env, cfg, _ = make_env(21.0, update_period=0.02)
        with self.assertRaisesRegex(ValueError, "every physics step"):
            support.tool_support_termination(env, cfg)

    def test_invalid_durations_and_physics_cadence_are_rejected(self):
        env, cfg, sensor = make_env()
        for duration in (0.0, -0.1, float("nan"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                support.tool_support_termination(env, cfg, duration_s=duration)
        env.cfg.sim.dt = 0.0
        with self.assertRaises(ValueError):
            support.tool_support_termination(env, cfg)
        env.cfg.sim.dt = 0.005
        sensor.cfg.update_period = -0.1
        with self.assertRaises(ValueError):
            support.tool_support_termination(env, cfg)

    def test_force_reward_records_threshold_aligned_diagnostics(self):
        env, cfg, sensor = make_env(33.635)
        command = SimpleNamespace(metrics={"existing_metric": torch.tensor([7.0])})
        env.command_manager = SimpleNamespace(get_term=lambda name: command if name == "ee_target" else None)
        support.tool_support_force(env, cfg)
        metrics = command.metrics
        self.assertAlmostEqual(metrics["tool_upward_force_peak"].item(), 33.635)
        self.assertAlmostEqual(metrics["tool_upward_force_mean"].item(), 33.635)
        self.assertEqual(metrics["tool_support_active"].item(), 1.0)
        self.assertTrue(metrics["tool_support_active"].is_floating_point())
        self.assertTrue(support.tool_support_termination(env, cfg).item())
        self.assertEqual(metrics["existing_metric"].item(), 7.0)
        sensor.data.net_forces_w_history[:, 19, 1, 2] = 0.0
        support.tool_support_force(env, cfg)
        self.assertEqual(metrics["tool_support_active"].item(), 0.0)
        self.assertAlmostEqual(metrics["tool_upward_force_mean"].item(), 33.635 * 19 / 20)

    def test_force_reward_records_peak_support_after_reset(self):
        env, cfg, sensor = make_env()
        command = SimpleNamespace(metrics={})
        env.command_manager = SimpleNamespace(get_term=lambda name: command)
        sensor.data.net_forces_w_history[:, 0, 1, 2] = 151.0
        support.tool_support_force(env, cfg)
        self.assertEqual(command.metrics["tool_support_active"].item(), 1.0)
        sensor.data.net_forces_w_history[:, :, 1] = 0.0
        support.tool_support_force(env, cfg)
        self.assertEqual(command.metrics["tool_support_active"].item(), 0.0)
        self.assertEqual(command.metrics["tool_upward_force_peak"].item(), 0.0)


class BaseTiltTests(unittest.TestCase):
    def test_deadzone_and_normalized_radian_squared_cost(self):
        angles = torch.tensor([0.0, 0.05, 0.1, 0.3, math.pi], dtype=torch.float64)
        gravity = torch.stack([angles.sin(), torch.zeros_like(angles), -angles.cos()], dim=-1)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, ((math.pi - 0.1) / 0.2) ** 2], dtype=torch.float64)
        torch.testing.assert_close(support.inclination_deadzone_penalty(gravity), expected)

    def test_scale_controls_strength_without_saturating(self):
        angle = math.radians(8.3)
        gravity = torch.tensor([[math.sin(angle), 0.0, -math.cos(angle)]], dtype=torch.float64)
        default = support.inclination_deadzone_penalty(gravity)
        self.assertAlmostEqual(default.item(), ((angle - 0.1) / 0.2) ** 2)
        self.assertAlmostEqual(support.inclination_deadzone_penalty(gravity, scale=0.1).item(), default.item() * 4)
        gravity[:, 2] = 1.0
        self.assertGreater(support.inclination_deadzone_penalty(gravity).item(), 200.0)

    def test_roundoff_is_clamped_before_acos(self):
        gravity = torch.tensor([[0.0, 0.0, -1.00000001], [0.0, 0.0, 1.00000001]], dtype=torch.float64)
        result = support.inclination_deadzone_penalty(gravity)
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(result[0].item(), 0.0)

    def test_environment_adapter_reads_projected_gravity(self):
        env, _, _ = make_env()
        self.assertEqual(support.base_tilt_penalty(env).item(), 0.0)
        env.scene["robot"].data.projected_gravity_b = torch.tensor(
            [[0.0, math.sin(0.3), -math.cos(0.3)]], dtype=torch.float64,
        )
        self.assertAlmostEqual(support.base_tilt_penalty(env, SimpleNamespace(name="robot")).item(), 1.0, places=6)
        self.assertAlmostEqual(support.base_tilt_penalty(env, scale=0.4).item(), 0.25, places=6)

    def test_environment_adapter_records_raw_inclination_in_radians(self):
        env, _, _ = make_env()
        command = SimpleNamespace(metrics={})
        env.command_manager = SimpleNamespace(get_term=lambda name: command)
        env.scene["robot"].data.projected_gravity_b = torch.tensor(
            [[0.0, math.sin(0.3), -math.cos(0.3)]], dtype=torch.float64,
        )
        self.assertAlmostEqual(support.base_tilt_penalty(env).item(), 1.0, places=6)
        self.assertAlmostEqual(command.metrics["base_inclination_rad"].item(), 0.3, places=6)

    def test_invalid_parameters_and_shape_are_rejected(self):
        for deadzone in (-0.1, float("nan"), math.pi):
            with self.subTest(deadzone=deadzone), self.assertRaises(ValueError):
                support.inclination_deadzone_penalty(torch.zeros(1, 3), deadzone)
        for scale in (0.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                support.inclination_deadzone_penalty(torch.zeros(1, 3), scale=scale)
        with self.assertRaises(ValueError):
            support.inclination_deadzone_penalty(torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
