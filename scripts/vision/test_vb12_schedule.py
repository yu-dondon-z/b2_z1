"""CPU tests of production VB12 methods, with only simulator dependencies stubbed."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1"
spec = importlib.util.spec_from_file_location("control", SRC / "mdp/vb11_tool_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
guard_spec = importlib.util.spec_from_file_location("guard", ROOT / "scripts/rsl_rl/train_vb12_guarded.py")
guard = importlib.util.module_from_spec(guard_spec)
guard_spec.loader.exec_module(guard)


def rotate(q, v):
    uv = torch.cross(q[:, 1:], v, dim=-1)
    return v + 2 * (q[:, :1] * uv + torch.cross(q[:, 1:], uv, dim=-1))


def inverse(q, v):
    conjugate = q.clone()
    conjugate[:, 1:] *= -1
    return rotate(conjugate, v)


def load_methods(names, parent=None, scope=None):
    tree = ast.parse((SRC / "mdp/commands/vb12_schedule_command.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VB12ScheduleTrajectoryCommand")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "scheduled_standby")
    parent = parent or type("Parent", (), {})
    node = ast.ClassDef(name="Production", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                        keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": torch, "math": math, "Parent": parent,
                 "quat_apply": rotate, "quat_apply_inverse": inverse,
                 "settle_latch": control.settle_latch, **(scope or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper, node], type_ignores=[])),
                 "production", "exec"), namespace)
    return namespace["Production"]


def fixture():
    def curve(obj):
        obj._command_transition_elapsed += .02
        obj.pose_command[:, :3] = torch.tensor([-.8, .1, -.51])
        obj.tracking_pose_w[:, :3] = obj.pose_command[:, :3]
        obj.base_velocity_command_b[:, 0] = obj._command_transition_elapsed / 4
    parent = type("Parent", (), {"_update_command": lambda obj: setattr(obj, "strict_called", True)})
    cls = load_methods(["_update_command"], parent, {"RearVisionTrajectoryCommand": NS(_update_command=curve)})
    obj = cls()
    obj.env = NS(step_dt=.02)
    obj.cfg = NS(vb12_training_schedule=True, vb12_ready_deadline_after_hold_s=.10,
                 file_standby_settle_s=.06)
    for name in ["_episode_age_s", "_startup_wait_age_s", "_settle_s", "_command_transition_elapsed"]:
        setattr(obj, name, torch.zeros(2))
    for name in ["_phase_moving", "_startup_ready", "_success_event", "_failure_event", "_deadline_checked"]:
        setattr(obj, name, torch.zeros(2, dtype=torch.bool))
    obj._hold_until_s = torch.tensor([.10, .10])
    obj._standby_only = torch.tensor([False, True])
    for name in ["_command_transition_start_position_b", "_command_transition_start_velocity_b",
                 "_filtered_velocity_command_b", "base_velocity_command_b"]:
        setattr(obj, name, torch.zeros(2, 3))
    obj.pose_command = torch.zeros(2, 7)
    obj.tracking_pose_w = torch.zeros(2, 7)
    obj.base_idx = 0
    states = torch.zeros(2, 1, 7)
    states[:, :, 3] = 1.
    obj.robot = NS(data=NS(body_pos_w=torch.zeros(2, 1, 3), body_state_w=states))
    obj._capture_after_reset_physics = lambda: None
    obj.standby_position_w = lambda: torch.tensor([[-.775, 0., .01]]).expand(2, 3)
    obj._measured_ready = lambda: torch.zeros(2, dtype=torch.bool)
    def smooth(desired):
        obj.pose_command[:, :3] = desired
        obj.tracking_pose_w[:, :3] = desired
    obj._set_file_pose_command_b_smooth = smooth
    return obj


class VB12Tests(unittest.TestCase):
    def test_not_ready_cannot_withhold_training_motion(self):
        obj = fixture()
        for _ in range(10):
            obj._update_command()
        self.assertFalse(obj._startup_ready.any())
        self.assertGreater(obj.base_velocity_command_b[0, 0], 0.)
        self.assertEqual(obj.base_velocity_command_b[1].sum().item(), 0.)

    def test_release_target_continuity_and_no_reclose(self):
        obj = fixture()
        while obj._episode_age_s[0] < .09:
            obj._update_command()
        previous = obj.pose_command.clone()
        while not obj._phase_moving[0]:
            obj._update_command()
        torch.testing.assert_close(obj.pose_command[0], previous[0])
        torch.testing.assert_close(obj._command_transition_start_position_b[0], previous[0, :3])
        self.assertEqual(obj.base_velocity_command_b[0].sum().item(), 0.)
        obj._update_command()
        self.assertGreater(obj.base_velocity_command_b[0, 0], 0.)

    def test_success_once_and_failure_once(self):
        obj = fixture()
        obj._measured_ready = lambda: torch.tensor([True, False])
        success = torch.zeros(2)
        failure = torch.zeros(2)
        for _ in range(30):
            obj._update_command()
            success += obj._success_event.float()
            failure += obj._failure_event.float()
        torch.testing.assert_close(success, torch.tensor([1., 0.]))
        torch.testing.assert_close(failure, torch.tensor([0., 1.]))
        self.assertAlmostEqual(success[0].item() * 150 * .02, 3.)
        self.assertAlmostEqual(failure[1].item() * -250 * .02, -5.)

    def test_evaluation_still_calls_strict_gate(self):
        obj = fixture()
        obj.cfg.vb12_training_schedule = False
        obj._update_command()
        self.assertTrue(obj.strict_called)
        self.assertEqual(obj.base_velocity_command_b.sum().item(), 0.)

    def test_reset_only_selected_events(self):
        cls = load_methods(["_resample_command"], type("Parent", (), {"_resample_command": lambda obj, ids: None}))
        obj = cls()
        obj.device = "cpu"
        obj.cfg = NS(vb12_training_schedule=False)
        for name in ["_success_event", "_failure_event", "_deadline_checked", "_phase_moving"]:
            setattr(obj, name, torch.ones(3, dtype=torch.bool))
        obj._resample_command(torch.tensor([1]))
        for name in ["_success_event", "_failure_event", "_deadline_checked", "_phase_moving"]:
            torch.testing.assert_close(getattr(obj, name), torch.tensor([True, False, True]))

    def test_cache_hold_world_z_with_heave_and_pitch(self):
        parent = type("Parent", (), {"_hold_file_tool_pose": lambda obj: setattr(obj, "strict_called", True)})
        obj = load_methods(["_hold_file_tool_pose"], parent)()
        obj._file_motion_ready = True
        obj.base_idx = 0
        obj.cfg = NS(tool_height_m=.01)
        obj.road_height_w = lambda: torch.zeros(2)
        pitch = torch.tensor([.02, -.02])
        q = torch.zeros(2, 4)
        q[:, 0], q[:, 2] = torch.cos(pitch / 2), torch.sin(pitch / 2)
        base = torch.tensor([[0., 0., .51], [0., 0., .54]])
        state = torch.cat([base, q], dim=-1).unsqueeze(1)
        obj.robot = NS(data=NS(body_pos_w=base.unsqueeze(1), body_state_w=state))
        obj._file_hold_position_b = torch.tensor([[-.775, .01, -.50]]).expand(2, 3).clone()
        desired = []
        obj._set_file_pose_command_b_smooth = lambda v: desired.append(v.clone())
        obj._hold_file_tool_pose()
        world = base + rotate(q, desired[0])
        torch.testing.assert_close(world[:, 2], torch.ones(2) * .01, atol=1e-6, rtol=0)
        original_world = base + rotate(q, obj._file_hold_position_b)
        torch.testing.assert_close(world[:, :2], original_world[:, :2])
        obj._file_motion_ready = False
        obj._hold_file_tool_pose()
        self.assertTrue(obj.strict_called)

    def test_actual_world_height_filter_does_not_follow_body_heave(self):
        tree = ast.parse((SRC / "mdp/commands/vision_trajectory_command.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RearVisionTrajectoryCommand")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_set_file_pose_command_b_smooth")
        namespace = {"torch": torch, "math": math, "quat_apply": rotate, "quat_apply_inverse": inverse}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "production", "exec"), namespace)
        q = torch.tensor([[1., 0., 0., 0.]])
        base = torch.tensor([[0., 0., .52]])
        obj = NS(env=NS(step_dt=.02), cfg=NS(file_target_filter_time_constant_s=.5,
                 file_target_max_speed_mps=.1, file_target_height_filter_in_world_frame=True, use_capture_pose=True),
                 pose_command=torch.tensor([[-.775, 0., -.51, 1., 0., 0., 0.]]),
                 tracking_pose_w=torch.tensor([[-.775, 0., .01, 1., 0., 0., 0.]]), base_idx=0,
                 _file_hold_position_b=torch.zeros(1, 3),
                 robot=NS(data=NS(body_pos_w=base.unsqueeze(1), body_state_w=torch.cat([base, q], dim=-1).unsqueeze(1))))
        def set_pose(v):
            obj.pose_command[:, :3] = v
            obj.tracking_pose_w[:, :3] = base + rotate(q, v)
        obj._set_pose_command_b = set_pose
        for height in [.51, .54, .53, .52]:
            base[:, 2] = height
            namespace[method.name](obj, torch.tensor([[-.775, 0., .01 - height]]))
            self.assertAlmostEqual(obj.tracking_pose_w[0, 2].item(), .01, places=6)

    def test_dense_error_and_physical_failure_not_timeout(self):
        tree = ast.parse((SRC / "mdp/vb12_task_signals.py").read_text())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        namespace = {"torch": torch, "huber": control.huber}
        exec(compile(ast.Module(body=fns, type_ignores=[]), "production", "exec"), namespace)
        term = NS(robot=NS(data=NS(body_pos_w=torch.zeros(2, 1, 3))), body_idx=0,
                  tracking_pose_w=torch.tensor([[0., 0., 0.], [.1, 0., 0.]]),
                  _success_event=torch.tensor([True, False]), _failure_event=torch.tensor([False, True]))
        env = NS(command_manager=NS(get_term=lambda _: term),
                 termination_manager=NS(terminated=torch.tensor([True, False]), time_outs=torch.tensor([False, True])))
        self.assertEqual(namespace["dense_tcp_error"](env)[0].item(), 0.)
        self.assertGreater(namespace["dense_tcp_error"](env)[1].item(), 0.)
        torch.testing.assert_close(namespace["physical_failure"](env), torch.tensor([1., 0.]))
        torch.testing.assert_close(namespace["preparation_success"](env), torch.tensor([1., 0.]))
        torch.testing.assert_close(namespace["preparation_failure"](env), torch.tensor([0., 1.]))

    def test_guard_normal_warmup_starvation_and_nonfinite(self):
        def data(request=.5, move=.7):
            return {tag: [(i, value) for i in range(220)] for tag, value in {
                "VB12/scheduled_motion_fraction": .8,
                "VB12/requested_motion_fraction": request,
                "VB12/movement_given_request": move}.items()}
        self.assertIsNone(guard.guard_reason(data()))
        self.assertIn("commands disappeared", guard.guard_reason(data(request=0.)))
        self.assertIn("collapsed", guard.guard_reason(data(move=0.)))
        short = {k: v[:100] for k, v in data(move=0.).items()}
        self.assertIsNone(guard.guard_reason(short))
        short["reward"] = [(1, math.nan)]
        self.assertIn("Nonfinite", guard.guard_reason(short))

    def test_schedule_config_keeps_validation_gate_and_contact_checks(self):
        tree = ast.parse((SRC / "b2_z1_vb12_env_cfg.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_schedule")
        import copy
        namespace = {"copy": copy, "VB12ScheduleTrajectoryCommandCfg": NS,
                     "VB12ScheduleTrajectoryCommand": object}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "production", "exec"), namespace)
        for training in [False, True]:
            cfg = NS(commands=NS(ee_target=NS(mode="synthetic", synthetic_world_fixed_path=True)),
                     terminations=NS(startup_not_ready="strict", gun_contact_budget="preserved", base_contact="preserved"))
            namespace[fn.name](cfg, training)
            self.assertEqual(cfg.commands.ee_target.vb12_training_schedule, training)
            self.assertEqual(cfg.terminations.startup_not_ready, None if training else "strict")
            self.assertEqual(cfg.terminations.gun_contact_budget, "preserved")
            self.assertEqual(cfg.terminations.base_contact, "preserved")

    def test_full_load_guard_not_hidden_by_light_domain_motion(self):
        series = {tag: [(i, value) for i in range(220)] for tag, value in {
            "VB12/scheduled_motion_fraction": .8,
            "VB12/requested_motion_fraction": .5,
            "VB12/movement_given_request": .9,
            "VB12/full_load_scheduled_motion_fraction": .7,
            "VB12/full_load_requested_motion_fraction": .3,
            "VB12/full_load_actual_commanded_motion_fraction": .01}.items()}
        self.assertIn("Nominal full-load actual", guard.guard_reason(series))
        series["VB12/full_load_requested_motion_fraction"] = [(i, .01) for i in range(220)]
        self.assertIn("commands disappeared", guard.guard_reason(series))
        series.pop("VB12/full_load_scheduled_motion_fraction")
        self.assertIsNone(guard.guard_reason(series))
        series["Episode_Termination/gun_contact_budget"] = [(i, .99) for i in range(220)]
        self.assertIn("gun-contact", guard.guard_reason(series))


if __name__ == "__main__":
    unittest.main()
