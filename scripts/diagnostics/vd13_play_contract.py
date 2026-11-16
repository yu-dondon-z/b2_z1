"""Pure-Python evaluation identity for the completed VD13 mount75 run.

No Isaac imports, actor updates, exports, or simulator state changes.
"""
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'source/b2_z1/b2_z1/assets'
TASK = 'Unitree-b2z1-VD13-Velocity'
EXPERIMENT = 'b2_z1_vd13_mount75_amp'
DEFAULT_CHECKPOINT = ROOT/'logs/rsl_rl/b2_z1_vd13_mount75_amp/2026-11-15_23-51-49_vd13_mount75_resume/model_6199.pt'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mount = load_source('vd13_eval_mount', ASSETS/'vd13_contract.py')
geometry = load_source('vd13_eval_geometry', ASSETS/'vd12_contract.py')
URDF = mount.ASSET
WORK_DISTANCE_M = mount.WORK_DISTANCE_M
TCP_POSITION = mount.TCP_POSITION
STARTUP_PROFILE = dict(
    version=1, duration_s=2., reference_initialization_prob=.8,
    nominal_base_height_m=.5, nominal_joint_velocity_range=[0., 0.],
    vd10=dict(version=2, safe_pitch_rad=.3, seam_pitch_rad=math.pi/2-.45,
              working_surface_clearance_m=.005, tube_length_m=1.1,
              tube_outer_radius_m=.028, reset_commands_after_forward=True,
              mount_version=13, tcp_in_link06_m=list(TCP_POSITION), clamp_to_tip_m=.75,
              arm_joint_reset_rad=list(mount.ARM_RESET)))


def validate_mount_checkpoint(checkpoint, agent, saved, profile, *, expected_experiment=EXPERIMENT):
    """Reject stale mount, TCP, source, curriculum or scene before simulation."""
    if (checkpoint.get('training_experiment_name') != expected_experiment
            or agent.get('experiment_name') != expected_experiment):
        raise ValueError(f'Mount75 Play requires an actual {expected_experiment} checkpoint and agent')
    expected = mount.mount_profile()
    if (checkpoint.get('vd13_mount_profile') != expected
            or saved.get('vd13_mount_profile') != expected
            or profile != geometry.geometry_profile(URDF)
            or checkpoint.get('vd12_geometry_profile') != profile
            or saved.get('vd12_geometry_profile') != profile):
        raise ValueError('VD13 saved/current mount, TCP or geometry profile differs')
    scene = saved.get('scene', {}).get('robot', {}).get('spawn', {})
    if Path(scene.get('asset_path', '')).resolve() != URDF.resolve():
        raise ValueError('VD13 saved scene does not use the mount75 URDF')
    ee = saved.get('commands', {}).get('ee_target', {})
    if (ee.get('vd5_seam_trailing_distance') != WORK_DISTANCE_M
            or ee.get('class_type') != 'b2_z1.tasks.manager_based.b2_z1.b2_z1_vd13_env_cfg:VD13SafeSeamCommand'):
        raise ValueError('VD13 saved command implementation/work distance differs')
    state = checkpoint.get('vd5_curriculum_state', {})
    if (state.get('version') != 1 or state.get('startup_profile') != STARTUP_PROFILE
            or state.get('ground_transition_steps') != 20000
            or state.get('ground_blend') != 1.
            or state.get('seam_trailing_distance_m') != WORK_DISTANCE_M):
        raise ValueError('VD13 evaluation requires the completed .95m ground/startup curriculum')
    return state
