import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import torch

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


r=load('vd7_filter_test',ROOT/'source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vd7_filter.py')
options=load('vd7_options_test',ROOT/'scripts/rsl_rl/training_options.py')


class VD7RewardTests(unittest.TestCase):
    def test_constant_error_and_reset_teleport_do_not_create_jitter_penalty(self):
        error=torch.tensor([[.2,-.1,.05],[0.,0.,0.]])
        low=torch.zeros_like(error);valid=torch.zeros(2,dtype=torch.bool)
        active=torch.ones(2,dtype=torch.bool)
        for _ in range(10):
            p,low,valid=r.filter_error(error,low,valid,active,.02)
            torch.testing.assert_close(p,torch.zeros(2))
        error[0]=torch.tensor([2.,3.,4.]);valid[0]=False
        p,low,valid=r.filter_error(error,low,valid,active,.02)
        torch.testing.assert_close(p,torch.zeros(2))

    def test_all_axes_penalized_after_startup_but_inactive_history_is_seeded(self):
        error=torch.full((3,3),.1);low=torch.zeros_like(error)
        p,low,valid=r.filter_error(error,low,torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),.02)
        torch.testing.assert_close(p,torch.zeros(3))
        error=error+torch.eye(3)*.02
        p,_,_=r.filter_error(error,low,valid,torch.ones(3,dtype=torch.bool),.02)
        self.assertTrue(torch.all(p>0));torch.testing.assert_close(p,p[0].expand(3))

    def test_higher_frequency_same_amplitude_gets_larger_penalty(self):
        energies=[]
        for hz in (0.5,8.):
            low=torch.zeros(1,3);valid=torch.zeros(1,dtype=torch.bool);values=[]
            for t in torch.arange(500)*.02:
                error=torch.zeros(1,3);error[0,1]=.01*torch.sin(2*torch.pi*hz*t)
                p,low,valid=r.filter_error(error,low,valid,torch.ones(1,dtype=torch.bool),.02)
                values.append(p.item())
            energies.append(sum(values[100:])/400)
        self.assertGreater(energies[1],energies[0]*10)

    def test_migration_resume_and_conflicts(self):
        runner,env=Mock(),Mock()
        options.load_training_checkpoint(runner,env,'b2_z1_vd7_tcp_amp',vd7_from='vd5.pt')
        runner.load.assert_called_once_with('vd5.pt',restore_curriculum=True,migrate_vd7=True)
        env.reset.assert_called_once()
        runner.reset_mock()
        options.load_training_checkpoint(runner,env,'b2_z1_vd7_tcp_amp',resume_path='vd7.pt')
        runner.load.assert_called_once_with('vd7.pt',restore_curriculum=True,legacy_curriculum=None)
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner,env,'b2_z1_vd7_tcp_amp',vd7_from='vd5.pt',vd6_from='vd5.pt')
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner,env,'b2_z1_vd6_smooth_amp',vd7_from='vd5.pt')


if __name__=='__main__':unittest.main()
