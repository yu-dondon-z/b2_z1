"""Physical cases for the new rewards, without starting Isaac Sim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load('vd6_reward_tests', ROOT/'source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vd6_rewards.py')
options = load('vd6_load_tests', ROOT/'scripts/rsl_rl/training_options.py')


class VD6RewardsTests(unittest.TestCase):
    def test_only_near_ground_descending_flight_is_penalized(self):
        height = torch.tensor([[.04,.04,.20,.04]])
        vel = torch.tensor([[-1.,1.,-1.,-1.]])
        contact = torch.tensor([[False,False,False,True]])
        torch.testing.assert_close(r.descending_foot_penalty(height,vel,contact),
                                   torch.tensor([(.65**2)/4]))

    def test_physics_interval_peak_includes_subsided_impact(self):
        history = torch.zeros(1,20,4,3)
        history[0,2,0,2] = 550.
        history[0,8,1,2] = 9999.  # old actor interval must not be charged again
        env = NS(step_dt=.02, physics_dt=.005,
            scene={'contact': NS(data=NS(net_forces_w_history=history))})
        torch.testing.assert_close(r.feet_impact_excess(env, NS(name='contact',body_ids=[0,1,2,3])),
                                   torch.tensor([.25]))

    def test_arm_penalty_excludes_leg_action_and_uses_radians(self):
        action = torch.zeros(1,18); previous = action.clone()
        action[0,9] = 100.  # calf: must be ignored
        action[0,13] = .28  # joint2: 0.07 rad -0.02 deadzone = 0.05 rad
        env = NS(action_manager=NS(action=action,prev_action=previous), pre_pre_action=previous.clone(),
                 scene={'robot': NS(joint_names=list(range(18)))})
        cfg = NS(name='robot', joint_ids=[8,13,14,15,16,17])
        torch.testing.assert_close(r.arm_action_continuity(env,cfg), torch.tensor([1./6]))
        env.pre_pre_action = -action
        torch.testing.assert_close(r.arm_action_continuity(env,cfg,second_order=True), torch.zeros(1))

    def test_vd6_migration_and_resume_use_distinct_modes(self):
        runner, env = Mock(), Mock()
        options.load_training_checkpoint(runner,env,'b2_z1_vd6_smooth_amp',vd6_from='vd5.pt')
        runner.load.assert_called_once_with('vd5.pt',restore_curriculum=True,migrate_vd6=True)
        env.reset.assert_called_once()
        runner.reset_mock(); env.reset_mock()
        options.load_training_checkpoint(runner,env,'b2_z1_vd6_smooth_amp',resume_path='vd6.pt')
        runner.load.assert_called_once_with('vd6.pt',restore_curriculum=True,legacy_curriculum=None)
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner,env,'b2_z1_vd5_startup_amp',vd6_from='vd5.pt')
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner,env,'b2_z1_vd6_smooth_amp',vd6_from='vd5.pt',resume_path='vd6.pt')


if __name__ == '__main__':
    unittest.main()
