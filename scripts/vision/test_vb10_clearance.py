"""CPU tests for VB10 asset identity, contact timers and startup gate."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import xml.etree.ElementTree as ET
import torch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'source/b2_z1/b2_z1'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
assets=load('assets',SRC/'assets/vb10_gun_collision.py')
contact=load('contact',SRC/'tasks/manager_based/b2_z1/mdp/gun_clearance.py')
tree=ast.parse((SRC/'tasks/manager_based/b2_z1/mdp/commands/vision_trajectory_command.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RearVisionTrajectoryCommand')
def quat_apply(q,v):
    t=2*torch.cross(q[:,1:],v,dim=-1)
    return v+q[:,:1]*t+torch.cross(q[:,1:],t,dim=-1)
def quat_inverse_apply(q,v):
    return quat_apply(q*torch.tensor([1.,-1.,-1.,-1.]),v)
scope=dict(torch=torch,math=math,quat_apply=quat_apply,quat_apply_inverse=quat_inverse_apply)
for name in ('_velocity_start_blend','_gate_file_startup','_hold_file_tool_pose'):
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'production_gate','exec'),scope)

def gate_fixture():
    body=torch.tensor([[[0.,0.,.5],[0.,0.,.01]]])
    obj=NS(cfg=NS(file_standby_before_motion=True,file_standby_position_tolerance_m=.08,
                  file_standby_height_tolerance_m=.015,file_standby_settle_s=.06),
           _file_motion_ready=False,_file_settle_elapsed=0.,base_idx=0,body_idx=1,
           _file_arm_joint_ids=list(range(6)),_command_transition_elapsed=torch.ones(1)*20.,
           tracking_pose_w=torch.tensor([[0.,0.,.01,1.,0.,0.,0.]]),
           _hold_file_tool_pose=lambda:None,
           robot=NS(data=NS(body_pos_w=body,joint_vel=torch.zeros(1,6),
                           projected_gravity_b=torch.tensor([[0.,0.,-1.]]))),
           env=NS(step_dt=.02,scene=NS(env_origins=torch.zeros(1,3)),
                  gun_contact_force_w=torch.zeros(1,3)))
    return obj


class VB10Tests(unittest.TestCase):
    def test_only_gun_collision_changes(self):
        source=SRC/'assets/b2_z1/b2_z1_gun.urdf'
        old=ET.parse(source).getroot(); new=ET.parse(assets.flat_gun_urdf(source)).getroot()
        for a,b in zip(old.findall('joint'),new.findall('joint')):
            self.assertEqual(ET.tostring(a),ET.tostring(b))
        for a,b in zip(old.findall('link'),new.findall('link')):
            self.assertEqual(a.get('name'),b.get('name'))
            self.assertEqual(ET.tostring(a.find('inertial')) if a.find('inertial') is not None else None,
                             ET.tostring(b.find('inertial')) if b.find('inertial') is not None else None)
            if a.get('name')!='link_gun':
                self.assertEqual([ET.tostring(x) for x in a.findall('collision')],
                                 [ET.tostring(x) for x in b.findall('collision')])
        gun=new.find("./link[@name='link_gun']")
        mesh=gun.find('./collision/geometry/mesh')
        self.assertTrue(Path(mesh.get('filename')).is_file())
        self.assertEqual(gun.find('./collision/origin').get('xyz'),'0.003 -0.032 -0.05')

    def test_mesh_is_closed_and_not_smaller_than_cylinder(self):
        lines=assets.cylinder_obj().splitlines()
        v=torch.tensor([[float(x) for x in s.split()[1:]] for s in lines if s.startswith('v ')])
        faces=[[int(x)-1 for x in s.split()[1:]] for s in lines if s.startswith('f ')]
        self.assertAlmostEqual(float(v[:,2].max()-v[:,2].min()),1.1,places=6)
        n=(len(v)-2)//2
        for i in range(n):
            midpoint=(v[i,:2]+v[(i+1)%n,:2])/2
            self.assertGreaterEqual(float(midpoint.norm()),.028-1e-8)
        edges={}
        for face in faces:
            for a,b in zip(face,face[1:]+face[:1]):
                key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
        self.assertTrue(all(count==2 for count in edges.values()))
        # Old capsule bottom -0.628; correct cylinder bottom -0.600; TCP -0.620.
        self.assertAlmostEqual(-.05+float(v[:,2].min())-(-.62),.02,places=6)

    def test_cache_is_stable_and_original_not_modified(self):
        source=SRC/'assets/b2_z1/b2_z1_gun.urdf'; before=source.read_bytes()
        self.assertEqual(assets.flat_gun_urdf(source),assets.flat_gun_urdf(source))
        self.assertEqual(source.read_bytes(),before)

    def fixture(self):
        force=torch.tensor([[[0.,0.,65.]],[[0.,0.,0.]]])
        return NS(scene={'contact_forces':NS(data=NS(net_forces_w=force))},
                  episode_length_buf=torch.ones(2)*100,step_dt=.02),NS(name='contact_forces',body_ids=[0])

    def test_force_cost_is_zero_without_contact(self):
        env,cfg=self.fixture();cost=contact.gun_support_force_cost(env,cfg)
        self.assertGreater(cost[0].item(),0.);self.assertEqual(cost[1].item(),0.)

    def test_contact_duration_and_independent_environments(self):
        env,cfg=self.fixture()
        for _ in range(3):
            self.assertFalse(contact.sustained_gun_contact(env,cfg,duration_s=.10,grace_s=0.).any())
        for _ in range(3): result=contact.sustained_gun_contact(env,cfg,duration_s=.10,grace_s=0.)
        self.assertTrue(result[0]);self.assertFalse(result[1])

    def test_grace_reset_and_spike_clear_timer(self):
        env,cfg=self.fixture();env.episode_length_buf.zero_()
        for _ in range(20): self.assertFalse(contact.sustained_gun_contact(env,cfg,duration_s=.10,grace_s=.1).any())
        env.episode_length_buf.fill_(20)
        contact.sustained_gun_contact(env,cfg,duration_s=.10,grace_s=.1)
        env.scene['contact_forces'].data.net_forces_w.zero_()
        contact.sustained_gun_contact(env,cfg,duration_s=.10,grace_s=.1)
        self.assertEqual(env._vb10_gun_contact_elapsed.sum().item(),0.)

    def test_delay_is_zero_speed_during_arm_preparation(self):
        obj=NS(cfg=NS(base_motion_delay_s=1.5,command_transition_time_s=4.),env=NS(step_dt=.02),
               _command_transition_elapsed=torch.tensor([0.,1.,1.5,2.,5.5]))
        value=scope['_velocity_start_blend'](obj)
        torch.testing.assert_close(value[:3],torch.zeros(3))
        self.assertGreater(value[3].item(),0.);self.assertEqual(value[-1].item(),1.)

    def test_old_tasks_keep_original_velocity_blend(self):
        obj=NS(cfg=NS(base_motion_delay_s=0.,command_transition_time_s=4.),env=NS(step_dt=.02),
               _command_transition_elapsed=torch.tensor([0.,1.,2.,4.]))
        x=obj._command_transition_elapsed/4
        torch.testing.assert_close(scope['_velocity_start_blend'](obj),x*x*(3-2*x))

    def test_gate_requires_settling_then_latches(self):
        obj=gate_fixture();self.assertFalse(scope['_gate_file_startup'](obj))
        for _ in range(4): ready=scope['_gate_file_startup'](obj)
        self.assertTrue(ready);self.assertEqual(obj._command_transition_elapsed.item(),0.)
        obj.env.gun_contact_force_w[:,2]=65.
        self.assertTrue(scope['_gate_file_startup'](obj))

    def test_contact_tilt_low_base_and_vertical_error_block_start(self):
        for failure in ('contact','tilt','height','error','speed'):
            obj=gate_fixture()
            if failure=='contact':obj.env.gun_contact_force_w[:,2]=65.
            if failure=='tilt':obj.robot.data.projected_gravity_b[:,2]=-.9
            if failure=='height':obj.robot.data.body_pos_w[:,0,2]=.4
            if failure=='error':obj.tracking_pose_w[:,2]+=.05
            if failure=='speed':obj.robot.data.joint_vel[:]=2.
            for _ in range(10):self.assertFalse(scope['_gate_file_startup'](obj),failure)

    def test_old_file_tasks_have_no_new_gate(self):
        obj=gate_fixture();obj.cfg.file_standby_before_motion=False
        obj.env.gun_contact_force_w[:,2]=65.
        self.assertTrue(scope['_gate_file_startup'](obj))

    def test_standby_preserves_world_height_with_pitch(self):
        for pitch in (0.,-.08,.08):
            obj=gate_fixture();obj._file_hold_position_b=torch.zeros(1,3)
            obj.cfg.rear_work_x_range_m=(-1.,-.55)
            obj.cfg.file_road_surface_z_w_m=0.
            obj.cfg.tool_height_m=.01
            obj.robot.data.body_state_w=torch.zeros(1,2,13)
            q=torch.tensor([[math.cos(pitch/2),0.,math.sin(pitch/2),0.]])
            obj.robot.data.body_state_w[:,0,3:7]=q
            captured=[];obj._set_file_pose_command_b_smooth=lambda value:captured.append(value)
            scope['_hold_file_tool_pose'](obj)
            world=obj.robot.data.body_pos_w[:,0]+quat_apply(q,captured[0])
            self.assertAlmostEqual(world[0,2].item(),.01,places=6)

    def test_old_hold_keeps_captured_target(self):
        obj=gate_fixture();obj.cfg.file_standby_before_motion=False
        obj._file_hold_position_b=torch.tensor([[-.9,.1,-.4]])
        captured=[];obj._set_pose_command_b=lambda value:captured.append(value.clone())
        scope['_hold_file_tool_pose'](obj)
        torch.testing.assert_close(captured[0],obj._file_hold_position_b)


if __name__=='__main__':unittest.main()
