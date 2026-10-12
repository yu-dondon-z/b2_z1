"""CPU tests for VB8 costs and gravity-aligned command, without starting Kit."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp"
spec = importlib.util.spec_from_file_location("costs", MDP / "loaded_tool_stability.py")
costs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(costs)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_conjugate(q):
    return q * torch.tensor([1., -1., -1., -1.])


def from_rpy(r, p, y):
    cr, sr = torch.cos(r/2), torch.sin(r/2)
    cp, sp = torch.cos(p/2), torch.sin(p/2)
    cy, sy = torch.cos(y/2), torch.sin(y/2)
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)


# Execute the exact production method with independent quaternion math.
tree = ast.parse((MDP / "commands/vision_trajectory_command.py").read_text())
command_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RearVisionTrajectoryCommand")
method = next(n for n in command_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fixed_orientation_b")
scope = dict(torch=torch, quat_from_euler_xyz=from_rpy, quat_mul=quat_mul, quat_conjugate=quat_conjugate)
exec(compile(ast.Module(body=[method], type_ignores=[]), "production_orientation", "exec"), scope)


def fixture():
    data = NS(body_link_lin_vel_w=torch.zeros(2, 1, 3),
              body_link_ang_vel_w=torch.zeros(2, 1, 3),
              body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]]).repeat(2, 1, 1),
              joint_pos=torch.zeros(2, 2),
              joint_pos_limits=torch.tensor([[[-1., 1.], [-1., 1.]]]).repeat(2, 1, 1))
    term = NS(_joint_ids=slice(None), processed_actions=torch.zeros(2, 2))
    env = NS(scene={"robot": NS(data=data)},
             action_manager=NS(get_term=lambda _: term))
    return env, NS(name="robot", body_ids=[0], joint_ids=[0, 1]), term


class StabilityTests(unittest.TestCase):
    def test_heave_ignores_forward_motion(self):
        env, cfg, _ = fixture()
        env.scene['robot'].data.body_link_lin_vel_w[:, 0, 0] = 1.
        torch.testing.assert_close(costs.base_world_heave_cost(env, cfg), torch.zeros(2))

    def test_heave_sign_and_magnitude(self):
        env, cfg, _ = fixture()
        env.scene['robot'].data.body_link_lin_vel_w[:, 0, 2] = torch.tensor([.1, -.1])
        result = costs.base_world_heave_cost(env, cfg)
        torch.testing.assert_close(result, torch.full((2,), math.sqrt(2)-1))

    def test_gun_yaw_is_free_but_pitch_is_not(self):
        env, cfg, _ = fixture()
        vel = env.scene['robot'].data.body_link_ang_vel_w
        vel[0, 0, 2] = .5
        vel[1, 0, 1] = .5
        result = costs.gun_world_tilt_rate_cost(env, cfg)
        self.assertEqual(result[0].item(), 0.)
        self.assertGreater(result[1].item(), 0.)

    def test_limit_buffer_interior_is_free(self):
        env, cfg, _ = fixture()
        torch.testing.assert_close(costs.arm_limit_buffer_cost(env, cfg), torch.zeros(2))

    def test_vertical_gun_axis_is_yaw_free_and_pitch_penalized(self):
        env, cfg, _ = fixture()
        q = from_rpy(torch.zeros(2), torch.tensor([0., math.pi/2]), torch.tensor([1., -1.]))
        env.scene['robot'].data.body_link_quat_w[:, 0] = q
        torch.testing.assert_close(costs.gun_vertical_axis_cost(env, cfg), torch.tensor([0., 1.]))

    def test_actual_and_out_of_range_targets_are_penalized(self):
        env, cfg, term = fixture()
        env.scene['robot'].data.joint_pos[0, 0] = -.95
        term.processed_actions[1, 0] = -1.5
        result = costs.arm_limit_buffer_cost(env, cfg)
        self.assertGreater(result[0].item(), 0.)
        self.assertGreater(result[1].item(), result[0].item())

    def test_reordered_actions_are_refused(self):
        env, cfg, term = fixture()
        term._joint_ids = [1, 0]
        with self.assertRaisesRegex(ValueError, 'articulation order'):
            costs.arm_limit_buffer_cost(env, cfg)

    def test_invalid_scale_is_refused(self):
        env, cfg, _ = fixture()
        for invalid in (0., -1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                costs.base_world_heave_cost(env, cfg, speed_scale=invalid)
            with self.assertRaises(ValueError):
                costs.arm_limit_buffer_cost(env, cfg, margin_rad=invalid)

    def test_gravity_alignment_cancels_roll_pitch_and_keeps_yaw(self):
        r = torch.tensor([.2, -.25]); p = torch.tensor([-.15, .1]); y = torch.tensor([.7, -1.])
        base = from_rpy(r, p, y)
        state = torch.zeros(2, 1, 13); state[:, 0, 3:7] = base
        obj = NS(num_envs=2, device='cpu', base_idx=0,
                 cfg=NS(tool_orientation_rpy=(0., math.pi/2, math.pi), tool_orientation_gravity_aligned=True),
                 robot=NS(data=NS(body_state_w=state, heading_w=y)))
        aligned_b = scope['_fixed_orientation_b'](obj)
        nominal = from_rpy(torch.zeros(2), torch.full((2,), math.pi/2), torch.full((2,), math.pi))
        expected_w = quat_mul(from_rpy(torch.zeros(2), torch.zeros(2), y), nominal)
        torch.testing.assert_close(quat_mul(base, aligned_b), expected_w)
        obj.cfg.tool_orientation_gravity_aligned = False
        torch.testing.assert_close(scope['_fixed_orientation_b'](obj), nominal)


if __name__ == '__main__':
    unittest.main()
