"""Simulator-independent constants for the isolated 75cm/.95m candidate."""
from pathlib import Path
import hashlib
import json

ASSET = Path(__file__).resolve().parent/'b2_z1/b2_z1_vd13_mount75.urdf'
TCP_POSITION = (.825, 0., .1)
TCP_QUATERNION = (1., 0., 0., 0.)
WORK_DISTANCE_M = .95
EXPERIMENT = 'b2_z1_vd13_mount75_amp'
ARM_RESET = (0., .66647356, -1.25178768, .88531411, 0., 0.)


def mount_profile():
    manifest = json.loads(ASSET.with_suffix('.manifest.json').read_text())
    actual = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if actual != manifest['urdf_sha256'] or manifest['tcp_in_link06_m'] != list(TCP_POSITION):
        raise ValueError('VD13 asset and TCP manifest disagree')
    return dict(version=1, urdf_sha256=actual, tcp_position_m=list(TCP_POSITION),
                clamp_to_tip_m=.75, clamp_to_rear_m=.35, clamp_in_link06_m=[.075, 0., .1],
                work_distance_m=WORK_DISTANCE_M, gun_mass_kg=2.93,
                tube_length_m=1.1, tube_outer_radius_m=.028,
                arm_reset_rad=list(ARM_RESET), command_source='world seam teacher; NOT camera recognition')


def validate_environment_profile(cfg):
    """Reject stale profile metadata after Hydra overrides actual task fields."""
    from .vd12_contract import geometry_profile
    if (Path(cfg.scene.robot.spawn.asset_path).resolve() != ASSET.resolve()
            or cfg.vd13_mount_profile != mount_profile()
            or cfg.vd12_geometry_profile != geometry_profile(ASSET)
            or cfg.commands.ee_target.vd5_seam_trailing_distance != WORK_DISTANCE_M
            or cfg.sim.dt != .005 or cfg.decimation != 4):
        raise ValueError('VD13 actual spawn/TCP/work distance/clock differs from its saved profile')


def validate_checkpoint(checkpoint, profile, geometry, *, migration, load_optimizer, restore_curriculum):
    """Validate provenance before any policy/optimizer/curriculum is mutated."""
    if not profile or profile != mount_profile() or not geometry or geometry.get('urdf_sha256') != profile['urdf_sha256']:
        raise ValueError('VD13 active mount and geometry profiles must match the candidate asset')
    required = {'model_state_dict', 'discriminator_state_dict', 'amp_normalizer_state_dict', 'iter', 'infos'}
    if not migration:
        required.add('optimizer_state_dict')
        required.add('vd5_curriculum_state')
    if required.difference(checkpoint) or any(checkpoint[k] is None for k in required-{'infos'}):
        raise ValueError('VD13 needs a complete policy/AMP checkpoint (optimizer for resume)')
    amp = checkpoint['amp_normalizer_state_dict']
    if not isinstance(amp, dict) or any(amp.get(k) is None for k in ('mean', 'var', 'count')):
        raise ValueError('VD13 requires complete AMP normalization state')
    iteration = checkpoint['iter']
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError('VD13 invalid checkpoint iteration')
    if migration:
        if load_optimizer or restore_curriculum or checkpoint.get('training_experiment_name') != 'b2_z1_vd12_geometry_guard_amp':
            raise ValueError('VD13 warm start accepts VD12 weights/AMP only; reset optimizer and curricula')
        from .vd12_contract import geometry_profile
        if checkpoint.get('vd12_geometry_profile') != geometry_profile(ASSET.with_name('b2_z1_vd10_hollow_gun.urdf')):
            raise ValueError('VD13 warm start source is not the preserved VD12 measured-tube geometry')
    elif (not load_optimizer or not restore_curriculum
          or checkpoint.get('training_experiment_name') != EXPERIMENT
          or checkpoint.get('vd13_mount_profile') != profile
          or checkpoint.get('vd12_geometry_profile') != geometry):
        raise ValueError('VD13 resume requires the same mount/geometry, optimizer and complete curriculum')
