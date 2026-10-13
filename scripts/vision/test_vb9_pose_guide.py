"""CPU checks for train-only guidance, Jacobians and physical connector costs."""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT/'source/b2_z1/b2_z1/assets/b2_z1/b2_z1_gun.urdf'

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

guide = load('guide','source/b2_z1/b2_z1/rsl_rl/modules/pose_reference_teacher.py')
costs = load('costs','source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/loaded_tool_stability.py')


class GuideTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.ids=[8,13,14,15,16,17]
        self.actor=torch.nn.Sequential(torch.nn.Linear(70,18))
        self.teacher=guide.PoseReferenceTeacher(self.actor,URDF,self.ids,[0.,2.55,-1.6,-1.4,0.,0.])
        self.obs=torch.zeros(8,70);self.obs[:,3:6]=torch.tensor([-.775,0.,-.51]);self.obs[:,15]=-1.

    def test_clear_nominal_pose(self):
        action=self.teacher(self.obs)
        q=action[:,self.ids]*.25+self.teacher.default_arm
        pos,axis,_=self.teacher.fk(q)
        torch.testing.assert_close(pos,self.obs[:,3:6],atol=1.e-5,rtol=1.e-5)
        torch.testing.assert_close(axis,torch.tensor([0.,0.,1.]).expand(8,3),atol=1.e-5,rtol=1.e-5)
        self.assertTrue(torch.all(q[:,3]>-1.))
        self.assertTrue(torch.all(torch.abs(action[:,self.ids])<5))

    def test_valid_references_preserve_leg_actions(self):
        before=self.actor(self.obs);after=self.teacher(self.obs)
        legs=[i for i in range(18) if i not in self.ids]
        torch.testing.assert_close(before[:,legs],after[:,legs])
        self.assertFalse(after.requires_grad)
        self.assertFalse(any(p.requires_grad for p in self.teacher.parameters()))

    def test_unreachable_target_has_no_guidance(self):
        self.obs[:,3]=5.
        after=self.teacher(self.obs)
        self.assertEqual(self.teacher.guidance_mask(self.obs).sum().item(),0.)
        torch.testing.assert_close(after,self.actor(self.obs))

    def test_nonfinite_input_rejected_and_outputs_finite(self):
        self.obs[0,3]=float('nan');self.obs[1,13]=float('inf')
        after=self.teacher(self.obs)
        self.assertTrue(torch.isfinite(after).all())
        self.assertEqual(self.teacher.guidance_mask(self.obs)[:2].sum().item(),0.)

    def test_schedule_does_not_change_reference_geometry(self):
        a=self.teacher(self.obs);first=self.teacher.guidance_mask(self.obs).clone()
        self.teacher.set_iteration(200);b=self.teacher(self.obs)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(first,torch.full((8,),.1))
        torch.testing.assert_close(self.teacher.guidance_mask(self.obs),torch.ones(8))

    def test_wrong_contract_refused(self):
        with self.assertRaises(ValueError):self.teacher(torch.zeros(8,69))

    def test_nullspace_guidance_leaves_task_motion_mostly_free(self):
        self.obs[:,16:34][:,self.ids]=self.teacher.seed-self.teacher.default_arm
        self.teacher(self.obs)
        _,_,J=self.teacher.fk(self.teacher.seed.unsqueeze(0))
        Jp=J[0,:3]
        task=Jp.T@torch.tensor([1.,0.,0.]);task/=torch.linalg.vector_norm(task)
        _,_,vh=torch.linalg.svd(Jp,full_matrices=True)
        null=vh[-1]
        task_cost=self.teacher.action_error_cost(task.expand(8,6)).mean()
        null_cost=self.teacher.action_error_cost(null.expand(8,6)).mean()
        self.assertLess(task_cost.item(),.15*null_cost.item())
        delta=torch.ones(8,6,requires_grad=True)
        self.teacher.action_error_cost(delta).mean().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_jacobian_matches_finite_difference(self):
        q=torch.tensor([[.02,2.7,-2.35,-.35,.01,.02]],dtype=torch.float64)
        self.teacher=self.teacher.double()
        _,_,J=self.teacher.fk(q)
        for i in range(6):
            dq=torch.zeros_like(q);dq[:,i]=1.e-6
            p0,z0,_=self.teacher.fk(q-dq);p1,z1,_=self.teacher.fk(q+dq)
            numeric=torch.cat((p1-p0,z1-z0),dim=1)/2.e-6
            torch.testing.assert_close(J[:,:,i],numeric,atol=2.e-8,rtol=2.e-6)

    def test_gravity_roll_pitch_reference(self):
        self.obs[:,13:16]=torch.tensor([.03,-.02,-.99935])
        a=self.teacher(self.obs);q=a[:,self.ids]*.25+self.teacher.default_arm
        p,z,_=self.teacher.fk(q)
        self.assertTrue((self.teacher.guidance_mask(self.obs)>0).all())
        self.assertLess(torch.linalg.vector_norm(p-self.obs[:,3:6],dim=1).max().item(),.015)
        up=-self.obs[:,13:16]/torch.linalg.vector_norm(self.obs[:,13:16],dim=1,keepdim=True)
        self.assertLess(torch.linalg.vector_norm(z-up,dim=1).max().item(),.08)

    def test_connector_rotation_visible_even_without_com_velocity(self):
        data=NS(body_link_quat_w=torch.tensor([[[1.,0.,0.,0.]]]),
                body_link_pos_w=torch.zeros(1,1,3),body_com_pos_w=torch.zeros(1,1,3),
                body_com_lin_vel_w=torch.zeros(1,1,3),body_com_ang_vel_w=torch.tensor([[[1.,0.,0.]]]),
                heading_w=torch.zeros(1))
        env=NS(scene={'robot':NS(data=data)},command_manager=NS(get_term=lambda _:NS(_path_heading_error_b=torch.zeros(1))))
        result=costs.gun_connector_normal_velocity_cost(env,NS(name='robot',body_ids=[0]))
        self.assertGreater(result.item(),0.)

    def test_connector_forward_progress_is_free(self):
        data=NS(body_link_quat_w=torch.tensor([[[1.,0.,0.,0.]]]),
                body_link_pos_w=torch.zeros(1,1,3),body_com_pos_w=torch.zeros(1,1,3),
                body_com_lin_vel_w=torch.tensor([[[.22,0.,0.]]]),body_com_ang_vel_w=torch.zeros(1,1,3),
                heading_w=torch.zeros(1))
        env=NS(scene={'robot':NS(data=data)},command_manager=NS(get_term=lambda _:NS(_path_heading_error_b=torch.zeros(1))))
        self.assertEqual(costs.gun_connector_normal_velocity_cost(env,NS(name='robot',body_ids=[0])).item(),0.)

    def test_reset_mixture_changes_only_selected_arm_state(self):
        source=ROOT/'source/b2_z1/b2_z1/tasks/manager_based/b2_z1/b2_z1_vb9_env_cfg.py'
        tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reset_clear_arm_mixture')
        scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'reset_test','exec'),scope)
        recorded=[]
        robot=NS(write_joint_state_to_sim=lambda q,dq,**kw:recorded.append((q.clone(),dq.clone(),kw)))
        env=NS(scene={'robot':robot},device='cpu',num_envs=4)
        cfg=NS(name='robot',joint_ids=self.ids)
        scope['reset_clear_arm_mixture'](env,None,cfg,fraction=1.)
        q,dq,kw=recorded[0]
        self.assertEqual(kw['joint_ids'],self.ids)
        torch.testing.assert_close(kw['env_ids'],torch.arange(4))
        torch.testing.assert_close(q,self.teacher.seed.expand(4,6))
        torch.testing.assert_close(dq,torch.zeros_like(dq))


if __name__=='__main__':unittest.main()
