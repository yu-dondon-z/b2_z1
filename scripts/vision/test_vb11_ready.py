"""CPU regression tests for final-height gating, load quotas and contact debt."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "source/b2_z1/b2_z1"
spec = importlib.util.spec_from_file_location("vb11", SRC / "tasks/manager_based/b2_z1/mdp/vb11_tool_control.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture():
    return dict(error=torch.zeros(3, 3), target_error=torch.zeros(3, 3),
                arm_speed=torch.zeros(3), upright=torch.ones(3), height=torch.ones(3) * .52,
                force=torch.zeros(3), clearance=torch.ones(3) * .03)


class VB11Tests(unittest.TestCase):
    def test_final_height_not_interpolated_waypoint(self):
        data = fixture()
        data["error"][0, 2] = .10
        self.assertFalse(tool.ready_mask(**data)[0])
        data["error"].zero_()
        data["target_error"][0, 2] = .10
        self.assertFalse(tool.ready_mask(**data)[0])

    def test_independent_ready_environments(self):
        data = fixture()
        data["force"][1] = 20.
        torch.testing.assert_close(tool.ready_mask(**data), torch.tensor([True, False, True]))

    def test_force_clearance_tilt_height_speed_xy_and_nan_block(self):
        for key, value in [("force", 6.), ("clearance", 0.), ("upright", .9),
                           ("height", .4), ("arm_speed", 1.)]:
            data = fixture()
            data[key][0] = value
            self.assertFalse(tool.ready_mask(**data)[0], key)
        data = fixture()
        data["error"][0, 0] = .09
        self.assertFalse(tool.ready_mask(**data)[0])
        data["error"][0, 0] = math.nan
        self.assertFalse(tool.ready_mask(**data)[0])

    def test_settle_reset_and_one_shot_latch(self):
        elapsed = torch.zeros(2)
        latched = torch.zeros(2, dtype=torch.bool)
        for _ in range(4):
            elapsed, latched = tool.settle_latch(torch.tensor([True, False]), elapsed, latched, .02, .06)
        torch.testing.assert_close(latched, torch.tensor([True, False]))
        elapsed, latched = tool.settle_latch(torch.zeros(2, dtype=torch.bool), elapsed, latched, .02, .06)
        self.assertTrue(latched[0])
        self.assertEqual(elapsed.sum().item(), 0.)

    def test_intermittent_contact_accumulates_and_no_contact_recovers(self):
        debt = torch.zeros(2)
        age = torch.ones(2) * 10
        for i in range(100):
            force = torch.tensor([20. if i % 2 == 0 else 0., 0.])
            debt = tool.contact_debt_step(force, age, debt, .02)
        self.assertGreater(debt[0], .4)
        self.assertEqual(debt[1].item(), 0.)
        for _ in range(200):
            debt = tool.contact_debt_step(torch.zeros(2), age, debt, .02)
        self.assertEqual(debt.sum().item(), 0.)

    def test_grace_and_per_environment_reset_clear_debt(self):
        debt = tool.contact_debt_step(torch.ones(2) * 20, torch.tensor([0., 10.]),
                                      torch.ones(2), .02)
        self.assertEqual(debt[0].item(), 0.)
        self.assertGreater(debt[1].item(), 1.)

    def test_bottom_geometry_is_flat_cylinder_not_capsule(self):
        asset = NS(data=NS(body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
                           body_link_pos_w=torch.tensor([[[0., 0., .63]]])))
        self.assertAlmostEqual(tool.gun_bottom_z(asset, 0).item(), .03, places=6)

    def test_height_cost_is_signed_and_precision_bounded(self):
        robot = NS(data=NS(body_pos_w=torch.tensor([[[0., 0., .04]], [[0., 0., -.02]], [[0., 0., .01]]])))
        term = NS(robot=robot, body_idx=0, tracking_pose_w=torch.tensor([[0., 0., .01]]).expand(3, 3),
                  road_height_w=lambda: torch.zeros(3))
        env = NS(command_manager=NS(get_term=lambda _: term))
        cost = tool.tcp_height_shortfall(env)
        self.assertEqual(cost[0].item(), 0.)
        self.assertGreater(cost[1], 0.)
        self.assertEqual(cost[2].item(), 0.)
        self.assertGreater(tool.tcp_below_road(env)[1], 0.)
        self.assertEqual(tool.tcp_height_precision(env)[2].item(), 1.)

    def test_startup_timeout_uses_command_age_not_random_episode_buffer(self):
        term = NS(cfg=NS(mode="synthetic"), _startup_wait_age_s=torch.tensor([1., 13., 13.]),
                  _startup_ready=torch.tensor([False, False, True]))
        env = NS(command_manager=NS(get_term=lambda _: term), episode_length_buf=torch.ones(3) * 2000)
        torch.testing.assert_close(tool.startup_not_ready(env), torch.tensor([False, True, False]))

    def test_nominal_hose_quota_overrides_zero_sampling_only_those_domains(self):
        tree = ast.parse((SRC / "envs/b2z1_ready_tool_env.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_hose_parameters")
        # Resolve real production override and its super() call with a minimal parent.
        parent = type("Parent", (), {"_sample_hose_parameters": lambda self, ids: None})
        cls_node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                                keywords=[], body=[method], decorator_list=[])
        scope = {"torch": torch, "Parent": parent}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls_node], type_ignores=[])), "production", "exec"), scope)
        obj = scope["Production"]()
        obj._vb11_nominal_load_mask = torch.tensor([True, False, True])
        obj.cfg = NS(vb11_nominal_hose_mass_kg=3.)
        obj._hose_mass_kg = torch.tensor([0., 1., 0.])
        obj._hose_drag = torch.zeros(3)
        obj._sample_hose_parameters(torch.tensor([0, 1]))
        torch.testing.assert_close(obj._hose_mass_kg, torch.tensor([3., 1., 0.]))
        torch.testing.assert_close(obj._hose_drag, torch.tensor([4., 0., 0.]))

    def test_staged_command_releases_without_target_jump_or_reclosing(self):
        tree = ast.parse((SRC / "tasks/manager_based/b2_z1/mdp/commands/vb11_ready_command.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VB11ReadyTrajectoryCommand")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_command")
        def parent_update(obj):
            obj._command_transition_elapsed += .02
            obj.pose_command[:, :3] = torch.tensor([[-.8, .1, -.51]])
            obj.tracking_pose_w[:, :3] = obj.pose_command[:, :3]
            obj.base_velocity_command_b[:, 0] = obj._command_transition_elapsed / 4
        parent = type("Parent", (), {"_update_command": parent_update})
        capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_after_reset_physics")
        node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                            keywords=[], body=[method, capture], decorator_list=[])
        scope = {"torch": torch, "Parent": parent, "settle_latch": tool.settle_latch,
                 "quat_apply_inverse": lambda q, v: v}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "production", "exec"), scope)
        obj = scope["Production"]()
        obj.env = NS(step_dt=.02, common_step_counter=1)
        obj.cfg = NS(mode="synthetic", file_standby_settle_s=.06)
        obj._episode_age_s = torch.zeros(2)
        obj._startup_wait_age_s = torch.zeros(2)
        obj._capture_pending = torch.zeros(2, dtype=torch.bool)
        obj._reset_step = torch.zeros(2, dtype=torch.long)
        obj._hold_until_s = torch.tensor([.10, .10])
        obj._standby_only = torch.tensor([False, True])
        obj._startup_ready = torch.zeros(2, dtype=torch.bool)
        obj._settle_s = torch.zeros(2)
        obj._command_transition_elapsed = torch.zeros(2)
        obj._command_transition_start_position_b = torch.zeros(2, 3)
        obj._command_transition_start_velocity_b = torch.zeros(2, 3)
        obj._filtered_velocity_command_b = torch.zeros(2, 3)
        obj.base_velocity_command_b = torch.zeros(2, 3)
        obj.pose_command = torch.zeros(2, 7)
        obj.tracking_pose_w = torch.zeros(2, 7)
        obj.base_idx = 0
        obj.robot = NS(data=NS(body_pos_w=torch.zeros(2, 1, 3), body_state_w=torch.zeros(2, 1, 7)))
        standby = torch.tensor([[-.775, 0., .01]]).expand(2, 3)
        obj.standby_position_w = lambda: standby
        obj._measured_ready = lambda: torch.ones(2, dtype=torch.bool)
        def smooth(desired):
            obj.pose_command[:, :3] = desired
            obj.tracking_pose_w[:, :3] = desired
        obj._set_file_pose_command_b_smooth = smooth
        for _ in range(4):
            obj._update_command()
            self.assertEqual(obj.base_velocity_command_b.sum().item(), 0.)
            self.assertFalse(obj._startup_ready.any())
        for _ in range(5):
            obj._update_command()
        self.assertTrue(obj._startup_ready.all())
        self.assertGreater(obj.base_velocity_command_b[0, 0], 0.)
        self.assertEqual(obj.base_velocity_command_b[1].sum().item(), 0.)
        torch.testing.assert_close(obj._command_transition_start_position_b[0], standby[0])
        obj._measured_ready = lambda: torch.zeros(2, dtype=torch.bool)
        previous_speed = obj.base_velocity_command_b[0, 0].item()
        obj._update_command()
        self.assertGreater(obj.base_velocity_command_b[0, 0], previous_speed)
        self.assertEqual(obj.base_velocity_command_b[1].sum().item(), 0.)

    def test_reset_clears_only_selected_environments(self):
        tree = ast.parse((SRC / "tasks/manager_based/b2_z1/mdp/commands/vb11_ready_command.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VB11ReadyTrajectoryCommand")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_resample_command")
        parent = type("Parent", (), {"_resample_command": lambda obj, ids: None})
        node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
        scope = {"torch": torch, "Parent": parent}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "production", "exec"), scope)
        obj = scope["Production"]()
        obj.device = "cpu"
        obj.env = NS(common_step_counter=10)
        obj.cfg = NS(mode="synthetic", vb11_standby_fraction=1., vb11_hold_range_s=(4., 4.))
        obj._episode_age_s = torch.ones(3) * 20
        obj._startup_wait_age_s = torch.ones(3) * 20
        obj._capture_pending = torch.zeros(3, dtype=torch.bool)
        obj._reset_step = torch.zeros(3, dtype=torch.long)
        obj._settle_s = torch.ones(3)
        obj._startup_ready = torch.ones(3, dtype=torch.bool)
        obj._standby_only = torch.zeros(3, dtype=torch.bool)
        obj._hold_until_s = torch.ones(3) * 7
        obj._resample_command(torch.tensor([1]))
        torch.testing.assert_close(obj._episode_age_s, torch.tensor([20., 0., 20.]))
        torch.testing.assert_close(obj._startup_ready, torch.tensor([True, False, True]))
        self.assertEqual(obj._hold_until_s[1].item(), 4.)
        self.assertTrue(obj._standby_only[1])

    def test_exact_mass_quotas_and_inertia_scaling(self):
        tree = ast.parse((SRC / "tasks/manager_based/b2_z1/b2_z1_vb11_env_cfg.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assign_payload_domains")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "production", "exec"), scope)
        mass = torch.ones(20, 1) * 3.259047
        inertia = torch.ones(20, 1, 9) * .1
        view = NS(get_masses=lambda: mass, get_inertias=lambda: inertia,
                  set_masses=lambda m, ids: None, set_inertias=lambda m, ids: None)
        robot = NS(find_bodies=lambda _: ([0], ["link_gun"]), root_physx_view=view,
                   data=NS(default_mass=mass.clone(), default_inertia=inertia.clone()))
        env = NS(num_envs=20, device="cpu", scene={"robot": robot})
        scope["assign_payload_domains"](env, None, NS(name="robot"))
        self.assertEqual(env._vb11_nominal_load_mask.sum().item(), 10)
        self.assertEqual(env._vb7_light_payload_mask.sum().item(), 2)
        self.assertTrue(torch.all(mass[env._vb11_nominal_load_mask] == 3.259047))
        torch.testing.assert_close(inertia[:, 0, 0], mass[:, 0] / 3.259047 * .1)

    def test_capture_waits_for_physics_and_does_not_track_actual_every_step(self):
        tree = ast.parse((SRC / "tasks/manager_based/b2_z1/mdp/commands/vb11_ready_command.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VB11ReadyTrajectoryCommand")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_after_reset_physics")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "production", "exec"), scope)
        obj = NS(env=NS(common_step_counter=10), _capture_pending=torch.tensor([True, False]),
                 _reset_step=torch.tensor([10, 0]), pose_command=torch.zeros(2, 7),
                 _command_transition_start_position_b=torch.zeros(2, 3),
                 _file_hold_position_b=torch.zeros(2, 3), _command_transition_elapsed=torch.ones(2))
        def capture(ids):
            obj.pose_command[ids, :3] = torch.tensor([1., 2., 3.])
        obj._hold_current_tool_pose = capture
        scope["_capture_after_reset_physics"](obj)
        self.assertEqual(obj.pose_command.sum().item(), 0.)
        obj.env.common_step_counter = 11
        scope["_capture_after_reset_physics"](obj)
        torch.testing.assert_close(obj._file_hold_position_b[0], torch.tensor([1., 2., 3.]))
        obj._hold_current_tool_pose = lambda ids: self.fail("must not repeatedly re-anchor the live target")
        scope["_capture_after_reset_physics"](obj)

    def test_ema_reset_is_coherent_with_actual_clear_pose_without_gain_change(self):
        tree = ast.parse((SRC / "envs/b2z1_ready_tool_env.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx")
        parent = type("Parent", (), {"_reset_idx": lambda obj, ids: None})
        node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
        scope = {"torch": torch, "Parent": parent}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "production", "exec"), scope)
        obj = scope["Production"]()
        q = torch.zeros(2, 18)
        q[1, 8] = .5
        term = NS(_alpha=.20, _scale=.25, _joint_names=list(range(18)), _offset=torch.zeros(18),
                  _raw_actions=torch.zeros(2, 18), _processed_actions=torch.zeros(2, 18))
        obj.action_manager = NS(get_term=lambda _: term, _action=torch.zeros(2, 18), _prev_action=torch.zeros(2, 18))
        obj.scene = {"robot": NS(joint_names=list(range(18)), data=NS(joint_pos=q))}
        obj._reset_idx(torch.tensor([1]))
        self.assertEqual(term._raw_actions[1, 8].item(), 2.)
        self.assertEqual(term._processed_actions[1, 8].item(), .5)
        self.assertEqual(obj.action_manager._prev_action[1, 8].item(), 2.)
        self.assertEqual(term._raw_actions[0].sum().item(), 0.)
        self.assertEqual(term._alpha, .20)

    def test_visual_full_stop_rearms_deadline_once_not_every_stale_frame(self):
        tree = ast.parse((SRC / "tasks/manager_based/b2_z1/mdp/commands/vb11_ready_command.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VB11ReadyTrajectoryCommand")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_file_command_loss")
        parent = type("Parent", (), {"_handle_file_command_loss": lambda obj: setattr(obj, "_file_motion_ready", False)})
        node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
        scope = {"torch": torch, "Parent": parent}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "production", "exec"), scope)
        obj = scope["Production"]()
        obj._file_motion_ready = True
        obj._startup_wait_age_s = torch.tensor([50.])
        obj._handle_file_command_loss()
        self.assertEqual(obj._startup_wait_age_s.item(), 0.)
        obj._startup_wait_age_s.fill_(6.)
        obj._handle_file_command_loss()
        self.assertEqual(obj._startup_wait_age_s.item(), 6.)


if __name__ == "__main__":
    unittest.main()
