"""Read-only identity of the completed VD14 posture candidate, not an export."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'source/b2_z1/b2_z1/assets'
TASK = 'Unitree-b2z1-VD14-Velocity'
EXPERIMENT = 'b2_z1_vd14_posture_amp'
DEFAULT_CHECKPOINT = ROOT / 'logs/rsl_rl/b2_z1_vd14_posture_amp/2026-11-16_10-41-45_vd14_posture_long_20261116/model_5999.pt'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mount_evaluation = load_source('vd14_mount_evaluation', ROOT / 'scripts/diagnostics/vd13_play_contract.py')
package = types.ModuleType('vd14_eval_assets')
package.__path__ = [str(ASSETS)]
sys.modules[package.__name__] = package
posture = load_source('vd14_eval_assets.vd14_contract', ASSETS / 'vd14_contract.py')
mount, geometry = mount_evaluation.mount, mount_evaluation.geometry
URDF, TCP_POSITION = mount_evaluation.URDF, mount_evaluation.TCP_POSITION
STARTUP_PROFILE, WORK_DISTANCE_M = mount_evaluation.STARTUP_PROFILE, mount_evaluation.WORK_DISTANCE_M


def validate_posture_checkpoint(checkpoint, agent, saved, profile):
    state = mount_evaluation.validate_mount_checkpoint(
        checkpoint, agent, saved, profile, expected_experiment=EXPERIMENT)
    expected = posture.posture_profile()
    if (checkpoint.get('vd14_posture_profile') != expected
            or saved.get('vd14_posture_profile') != expected):
        raise ValueError('VD14 checkpoint/current/saved posture profiles differ')
    rewards = saved.get('rewards', {})
    tilt, width = rewards.get('base_tilt', {}), rewards.get('vd14_front_width', {})
    tp, wp = tilt.get('params', {}), width.get('params', {})
    selector = wp.get('asset_cfg', {})
    if (tilt.get('weight') != expected['base_tilt_weight']
            or tp.get('deadzone') != expected['base_tilt_deadzone_rad']
            or tp.get('scale') != expected['base_tilt_scale_rad']
            or tilt.get('func') != 'b2_z1.tasks.manager_based.b2_z1.mdp.vd3_support:base_tilt_penalty'
            or width.get('weight') != expected['front_width_weight']
            or wp.get('minimum_width') != expected['front_min_width_m']
            or wp.get('scale') != expected['front_width_scale_m']
            or selector.get('name') != 'robot'
            or selector.get('body_names') != expected['front_body_names']
            or selector.get('preserve_order') is not True
            or width.get('func') != 'b2_z1.tasks.manager_based.b2_z1.mdp.vd14_rewards:front_stance_width_penalty'):
        raise ValueError('VD14 saved actual posture rewards differ from the declared objective')
    return state
