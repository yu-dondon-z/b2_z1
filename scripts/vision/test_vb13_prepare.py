"""CPU checks: URDF gravity sign, reference bounds, stage safety and graduation."""
import ast
import csv
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'source/b2_z1/b2_z1'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
pose = load('pose', SRC/'rsl_rl/modules/pose_reference_teacher.py')
tree = ast.parse((SRC/'rsl_rl/modules/gravity_pose_reference_teacher.py').read_text())
scope = {'torch': torch, 'math': math, 'PoseReferenceTeacher': pose.PoseReferenceTeacher, 'rotation': pose.rotation}
nodes = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'gravity_reference', 'exec'), scope)
Reference = scope['GravityPoseReferenceTeacher']
stages = load('stages', SRC/'tasks/manager_based/b2_z1/mdp/vb13_curriculum.py')
sys.path.insert(0, str(ROOT/'scripts/rsl_rl'))
guard = load('vb13guard', ROOT/'scripts/rsl_rl/train_vb13_curriculum.py')


def teacher(hose=3.):
    return Reference(torch.nn.Linear(70,18), SRC/'assets/b2_z1/b2_z1_gun.urdf',
                     [8,13,14,15,16,17], [0.,2.55,-1.6,-1.4,0.,0.],
                     arm_stiffness=[60.]*6, hose_mass_kg=hose)


def rows(moving=False):
    return [dict(time_s=i*.02, reset=0, actual_x=0.,actual_y=0.,actual_z=.01,
                 target_x=0.,target_y=0.,target_z=.01,body_tilt_deg=0., base_z=.52,
                 gun_contact_force_n=0.,gun_bottom_clearance_m=.03,
                 command_vx_b=.2 if moving else 0.,base_x=i*.004 if moving else 0.,
                 path_lateral_error_b=0.,path_heading_error_b=0.,
                 **{f'joint{j}': 0. for j in range(1,7)}) for i in range(1500)]


class VB13Tests(unittest.TestCase):
    def test_gravity_torque_matches_potential_gradient_sign(self):
        g = teacher().double()
        q = torch.tensor([[.02,2.76,-2.46,-.31,.02,0.]], dtype=torch.float64)
        up = torch.tensor([[0.,0.,1.]], dtype=torch.float64)
        tau, _ = g.gravity_torque(q, up)
        for j in range(6):
            plus, minus = q.clone(), q.clone()
            plus[:,j] += 1.e-5
            minus[:,j] -= 1.e-5
            gradient = (g.gravity_torque(plus,up)[1]-g.gravity_torque(minus,up)[1])/2.e-5
            torch.testing.assert_close(tau[:,j], gradient, atol=2.e-6, rtol=2.e-6)

    def test_load_changes_support_reference_without_changing_fk(self):
        zero, loaded = teacher(0.), teacher(3.)
        q = torch.tensor([[0.,2.7673,-2.4606,-.3067,0.,0.]])
        up = torch.tensor([[0.,0.,1.]])
        a,_ = zero.gravity_torque(q,up)
        b,_ = loaded.gravity_torque(q,up)
        self.assertGreater((b-a).norm().item(), 1.)
        torch.testing.assert_close(zero.fk(q)[0], loaded.fk(q)[0])

    def test_reference_keeps_legs_and_bounds_all_arm_targets(self):
        g = teacher(); g.set_iteration(200)
        obs = torch.zeros(2,70)
        obs[:,3:6] = torch.tensor([[-.775,0.,-.51],[-.775,0.,-.45]])
        obs[:,13:16] = torch.tensor([[0.,0.,-1.]])
        before = g.actor(obs)
        out = g(obs)
        legs = [i for i in range(18) if i not in g.arm_ids]
        torch.testing.assert_close(out[:,legs], before[:,legs])
        q = out[:,g.arm_ids]*.25+g.default_arm
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(g.last_mask.all())
        self.assertTrue((q >= g.lower-1e-6).all() and (q <= g.upper+1e-6).all())
        self.assertLessEqual(g.last_bias.abs().max().item(), .450001)

    def test_invalid_guide_has_no_teacher_override(self):
        g = teacher()
        obs = torch.zeros(2,70)
        torch.testing.assert_close(g(obs), g.actor(obs))
        self.assertFalse(g.last_mask.any())

    def test_full_reference_loss_can_train_tcp_direction(self):
        g = teacher()
        delta = torch.ones(2,6,requires_grad=True)
        g.action_error_cost(delta).sum().backward()
        self.assertTrue((delta.grad != 0).all())

    def test_stage_progression_and_final_height(self):
        specs = [stages.stage_spec(i) for i in range(5)]
        self.assertEqual([s['height_m'] for s in specs], [.07,.05,.03,.01,.01])
        self.assertEqual([s['hose_kg'] for s in specs], [0.,1.,3.,3.,3.])
        self.assertEqual([s['moving'] for s in specs], [False]*4+[True])
        with self.assertRaises(ValueError): stages.stage_spec(-1)
        with self.assertRaises(ValueError): stages.stage_spec(True)

    def test_no_reset_strict_validation_accepts_hold_and_motion(self):
        self.assertTrue(stages.trace_acceptance(rows())[0])
        self.assertTrue(stages.trace_acceptance(rows(True), moving=True)[0])
        self.assertFalse(stages.trace_acceptance(rows(), moving=True)[0])

    def test_validation_cannot_pass_floor_contact_or_wrong_height(self):
        for key,value in [('gun_contact_force_n',20.),('actual_z',-.02),
                          ('gun_bottom_clearance_m',0.),('body_tilt_deg',15.),('base_z',.4)]:
            r = rows()
            for x in r: x[key]=value
            self.assertFalse(stages.trace_acceptance(r)[0], key)
        r = rows(); r[900]['reset']=1
        self.assertFalse(stages.trace_acceptance(r)[0])

    def test_motion_requires_centerline_and_heading(self):
        for key,value in [('path_lateral_error_b',.2),('path_heading_error_b',.3)]:
            r=rows(True)
            for x in r: x[key]=value
            self.assertFalse(stages.trace_acceptance(r,moving=True)[0])

    def test_finite_and_timestamp_validation(self):
        r=rows(); r[10]['target_z']=math.nan
        self.assertFalse(stages.trace_acceptance(r)[0])
        r=rows(); r[-1]['time_s']=1.
        self.assertFalse(stages.trace_acceptance(r)[0])
        self.assertFalse(stages.trace_acceptance(rows()[:100])[0])

    def test_stationary_tcp_cannot_hide_arm_twitching(self):
        r=rows()
        for i,x in enumerate(r):x['joint1']=.05 if i % 2 else -.05
        self.assertFalse(stages.trace_acceptance(r)[0])

    def test_real_vb12_failures_are_rejected(self):
        path=ROOT/'outputs/vb12_review_20261014/model_12799/3kg.csv'
        if path.is_file():
            with path.open() as stream:
                self.assertFalse(stages.trace_acceptance(list(csv.DictReader(stream)))[0])

    def test_hold_guard_does_not_require_motion_and_catches_nan(self):
        series = {'VB13/hold_ready_fraction': [(i,.5) for i in range(220)],
                  'VB12/requested_motion_fraction': [(i,0.) for i in range(220)],
                  'VB11/startup_ready_fraction': [(i,0.) for i in range(220)]}
        self.assertIsNone(guard.guard_reason(series,0))
        self.assertIsNotNone(guard.guard_reason(series,4))
        series['reward']=[(1, math.nan)]
        self.assertIsNotNone(guard.guard_reason(series,0))

    def test_previous_stage_certificate_checks_hash_and_stage(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as directory:
            ck=Path(directory)/'model_1.pt';ck.write_bytes(b'test fixture, not a model')
            certificate=Path(directory)/'acceptance.json'
            data={'protocol_version':1,'stage':0,'passed':True,'checkpoint':str(ck),'sha256':guard.checksum(ck)}
            certificate.write_text(json.dumps(data))
            guard.check_previous(certificate,ck,1)
            with self.assertRaises(ValueError):guard.check_previous(certificate,ck,2)
            ck.write_bytes(b'changed fixture')
            with self.assertRaises(ValueError):guard.check_previous(certificate,ck,1)


if __name__=='__main__':
    unittest.main()
