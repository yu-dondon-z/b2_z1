"""Independent tube inertia, TCP and launch/resume checks; no Isaac import."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'source/b2_z1/b2_z1/assets'
package = types.ModuleType('vd13_cpu')
package.__path__ = [str(ASSETS)]
sys.modules['vd13_cpu'] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load('vd13_cpu.vd13_contract', ASSETS/'vd13_contract.py')
geometry = load('vd13_cpu.vd12_contract', ASSETS/'vd12_contract.py')
asset = load('vd13_cpu.asset', ASSETS/'vd13_mount75_asset.py')
options = load('vd13_cpu.training_options', ROOT/'scripts/rsl_rl/training_options.py')


def tensor(element):
    return np.array([[float(element.get('i'+''.join(sorted(a+b)))) for b in 'xyz'] for a in 'xyz'])


class Mount75Asset(unittest.TestCase):
    def test_generation_is_deterministic_and_old_assets_untouched(self):
        old = ASSETS/'b2_z1/b2_z1_vd10_hollow_gun.urdf'
        saved_hash = json.loads(old.with_suffix('.manifest.json').read_text())['urdf_sha256']
        xml, manifest = asset.payload()
        self.assertEqual(xml, contract.ASSET.read_bytes())
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), saved_hash)
        self.assertEqual(hashlib.sha256(xml).hexdigest(), manifest['urdf_sha256'])

    def test_clamp_is_relative_to_flange_not_link06_origin(self):
        source = ET.parse(ASSETS/'b2_z1/b2_z1_gun.urdf').getroot()
        flange = np.fromstring(source.find("joint[@name='joint_mounter']/origin").get('xyz'), sep=' ')
        np.testing.assert_allclose(flange, [.075, 0, 0], atol=1e-10)
        clamp = np.array(asset.CLAMP_IN_LINK06_M)
        np.testing.assert_allclose(clamp-flange, [0, 0, .1])
        np.testing.assert_allclose(np.array(contract.TCP_POSITION)-clamp, [.75, 0, 0])
        self.assertAlmostEqual(asset.TIP_FROM_CLAMP_M+asset.REAR_FROM_CLAMP_M, 1.1)

    def test_inertia_against_independent_parallel_axis_calculation(self):
        source = ET.parse(ASSETS/'b2_z1/b2_z1_vd3_zero_gun.urdf').getroot()
        original = source.find("link[@name='link06']/inertial")
        m0, m1 = float(original.find('mass').get('value')), 2.93
        p0 = np.fromstring(original.find('origin').get('xyz'), sep=' ')
        p1 = np.array([.275, 0., .1])
        com = (m0*p0+m1*p1)/(m0+m1)
        axial = m1*(.028**2+.026**2)/2
        transverse = m1*(3*(.028**2+.026**2)+1.1**2)/12
        expected = tensor(original.find('inertia'))+np.diag([axial, transverse, transverse])
        for mass, position in ((m0, p0), (m1, p1)):
            offset = position-com
            expected += mass*(np.eye(3)*np.dot(offset, offset)-np.outer(offset, offset))
        actual = ET.parse(contract.ASSET).getroot().find("link[@name='link06']/inertial")
        np.testing.assert_allclose(tensor(actual.find('inertia')), expected, atol=1e-13)
        np.testing.assert_allclose(np.fromstring(actual.find('origin').get('xyz'), sep=' '), com, atol=1e-13)
        self.assertTrue((np.linalg.eigvalsh(expected) > 0).all())

    def test_robot_except_link06_preserved(self):
        old = ET.parse(ASSETS/'b2_z1/b2_z1_vd10_hollow_gun.urdf').getroot()
        new = ET.parse(contract.ASSET).getroot()
        old.remove(old.find("link[@name='link06']"))
        new.remove(new.find("link[@name='link06']"))
        self.assertEqual(ET.tostring(old), ET.tostring(new))

    def test_actual_tcp_selector(self):
        tcp = load('vd13_tcp', ASSETS.parent/'tasks/manager_based/b2_z1/mdp/tcp_frame.py')
        robot = types.SimpleNamespace(cfg=types.SimpleNamespace(spawn=types.SimpleNamespace(asset_path=contract.ASSET)))
        self.assertEqual(tcp.tcp_transform_for_asset(robot), (contract.TCP_POSITION, (1., 0., 0., 0.)))


class Mount75Checkpoint(unittest.TestCase):
    def setUp(self):
        self.profile = contract.mount_profile()
        self.geometry = geometry.geometry_profile(contract.ASSET)
        self.checkpoint = dict(model_state_dict={}, discriminator_state_dict={},
                               amp_normalizer_state_dict=dict(mean=0, var=1, count=1),
                               optimizer_state_dict={}, vd5_curriculum_state={'version': 2}, iter=100, infos=None,
                               training_experiment_name=contract.EXPERIMENT,
                               vd13_mount_profile=self.profile, vd12_geometry_profile=self.geometry)

    def validate(self, checkpoint, **kwargs):
        contract.validate_checkpoint(checkpoint, self.profile, self.geometry, **kwargs)

    def test_live_task_override_cannot_leave_stale_metadata(self):
        ns = types.SimpleNamespace
        cfg = ns(scene=ns(robot=ns(spawn=ns(asset_path=contract.ASSET))),
                 commands=ns(ee_target=ns(vd5_seam_trailing_distance=.95)),
                 sim=ns(dt=.005), decimation=4,
                 vd13_mount_profile=self.profile, vd12_geometry_profile=self.geometry)
        contract.validate_environment_profile(cfg)
        cfg.commands.ee_target.vd5_seam_trailing_distance = .8
        with self.assertRaises(ValueError):
            contract.validate_environment_profile(cfg)

    def test_same_version_resume_and_missing_state_rejection(self):
        args = dict(migration=False, load_optimizer=True, restore_curriculum=True)
        self.validate(self.checkpoint, **args)
        for key in ('optimizer_state_dict', 'vd5_curriculum_state', 'vd13_mount_profile', 'vd12_geometry_profile'):
            broken = copy.deepcopy(self.checkpoint)
            del broken[key]
            with self.assertRaises(ValueError):
                self.validate(broken, **args)

    def test_migration_accepts_only_vd12_and_resets_optimizer(self):
        old = copy.deepcopy(self.checkpoint)
        old['training_experiment_name'] = 'b2_z1_vd12_geometry_guard_amp'
        old['vd12_geometry_profile'] = geometry.geometry_profile(contract.ASSET.with_name('b2_z1_vd10_hollow_gun.urdf'))
        args = dict(migration=True, load_optimizer=False, restore_curriculum=False)
        self.validate(old, **args)
        with self.assertRaises(ValueError):
            self.validate(old, **dict(args, load_optimizer=True))
        old['vd12_geometry_profile']['urdf_sha256'] = self.geometry['urdf_sha256']
        with self.assertRaises(ValueError):
            self.validate(old, **args)

    def test_load_dispatch(self):
        runner, env = Mock(), Mock()
        options.load_training_checkpoint(runner, env, contract.EXPERIMENT, vd13_from='old.pt')
        runner.load.assert_called_once_with('old.pt', load_optimizer=False, restore_curriculum=False, migrate_vd13=True)
        self.assertEqual(runner.current_learning_iteration, 0)
        env.reset.assert_called_once()
        runner.reset_mock()
        options.load_training_checkpoint(runner, env, contract.EXPERIMENT, resume_path='new.pt')
        runner.load.assert_called_once_with('new.pt', restore_curriculum=True, legacy_curriculum=None)
        with self.assertRaises(ValueError):
            options.load_training_checkpoint(runner, env, contract.EXPERIMENT, resume_path='new.pt', vd13_from='old.pt')

    def test_launcher_default_and_resume_are_exclusive(self):
        with tempfile.TemporaryDirectory(prefix='vd13_launch_') as directory:
            environment = dict(os.environ, VD13_PYTHON='/usr/bin/echo', VD13_CACHE_ROOT=directory)
            for extra in ([], ['--resume', '--load_run', 'saved_run'], ['--vd13-from=']):
                command = subprocess.check_output(['bash', str(ROOT/'scripts/train_vd13.sh'), *extra],
                                                  env=environment, text=True)
                self.assertEqual('--vd13-from ' in command, not extra)
                self.assertIn('--task Unitree-b2z1-VD13-Velocity', command)
                self.assertIn('--logger tensorboard', command)
                self.assertIn('--keep-last-checkpoints 0', command)


if __name__ == '__main__':
    unittest.main()
