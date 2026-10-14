"""VB13 load-bearing preparation curriculum. Playback safety is not relaxed."""
import copy
import os
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, EventTermCfg, SceneEntityCfg
from .b2_z1_vb11_env_cfg import assign_payload_domains
from .b2_z1_vb12_env_cfg import (B2Z1VB12ScheduleAMPEnvCfg, VB12ScheduleRewardsCfg,
    B2Z1VB12ScheduleAMPEnvCfg_Play, B2Z1VB12ScheduleVisionAMPEnvCfg_Play)
from .mdp.commands.vb13_prepare_command import VB13PrepareTrajectoryCommandCfg, VB13PrepareTrajectoryCommand
from .mdp.vb13_curriculum import stage_spec
from .mdp import vb13_prepare_signals as signals


@configclass
class VB13PrepareRewardsCfg(VB12ScheduleRewardsCfg):
    final_pose_error = RewTerm(func=signals.final_pose_error, weight=-6.)
    clear_hold = RewTerm(func=signals.clear_hold, weight=4.)


def configure_preparation(cfg, training=False):
    cfg.vb13_stage = int(os.environ.get('B2_Z1_VB13_STAGE', str(cfg.vb13_stage)))
    spec = stage_spec(cfg.vb13_stage)
    if cfg.commands.ee_target.mode == 'file' and not spec['moving']:
        raise ValueError('VB13 full visual motion requires stage 4; stages 0-3 are standby-only checks')
    command = VB13PrepareTrajectoryCommandCfg()
    for key, value in vars(cfg.commands.ee_target).items():
        setattr(command, key, copy.deepcopy(value))
    command.class_type = VB13PrepareTrajectoryCommand
    command.vb12_training_schedule = False
    command.tool_height_m = spec['height_m']
    command.tool_height_range_m = (spec['height_m'], spec['height_m'])
    command.vb11_standby_fraction = 0. if spec['moving'] else 1.
    command.vb11_hold_range_s = (4., 4.)
    cfg.commands.ee_target = command
    # All domains keep the real URDF gun, not a near-massless proxy. Each
    # stage has a known hose load; widening randomization is a later task.
    cfg.vb11_nominal_hose_mass_kg = spec['hose_kg']
    cfg.hose_effective_mass_kg = spec['hose_kg']
    cfg.hose_mass_range_kg = None
    cfg.hose_zero_load_probability = 0.
    cfg.events.gun_payload = EventTermCfg(func=assign_payload_domains, mode='startup',
        params={'asset_cfg': SceneEntityCfg('robot', body_names='link_gun'),
                'nominal_fraction': 1., 'light_fraction': 0.})
    if not spec['moving']:
        cfg.curriculum.truth_curve_level = None
        # Constant standby and unchanged body pose; not a locomotion claim.
        command.synthetic_initial_lateral_offset_range_m = (0., 0.)
        command.synthetic_initial_heading_offset_range_rad = (0., 0.)
    if training:
        cfg.episode_length_s = 30. if not spec['moving'] else 60.
        cfg.terminations.startup_not_ready = None
        cfg.events.vb9_pose_reset.params['fraction'] = .75 if cfg.vb13_stage == 0 else .50
        cfg.rewards.gun_support_force.weight = -4.
        cfg.rewards.tcp_height_shortfall.weight = -4.
        cfg.rewards.tcp_below_road.weight = -6.
        if not spec['moving']:
            cfg.rewards.track_lin_vel_xy = None
            cfg.rewards.track_ang_vel_z = None
            cfg.rewards.along_path_speed_ratio = None
        else:
            cfg.rewards.final_pose_error = None
            cfg.rewards.clear_hold = None
            cfg.vb9_arm_guide_weight = .12
            cfg.vb13_leg_std_initial = .15
            cfg.vb13_leg_std_cap = .35
        # The same measured gate controls motion in train AND play.
        # A failed preparation cannot silently graduate via a high reward.


@configclass
class B2Z1VB13PrepareAMPEnvCfg(B2Z1VB12ScheduleAMPEnvCfg):
    rewards: VB13PrepareRewardsCfg = VB13PrepareRewardsCfg()
    vb13_stage: int = 0
    vb13_gravity_reference: bool = True
    vb9_arm_guide_weight: float = .60
    vb9_guide_warmup_iterations: int = 50
    vb13_arm_std_initial: float = .025
    vb13_arm_std_cap: float = .04
    vb13_arm_std_min: float = .005
    vb13_leg_std_initial: float = .06
    vb13_leg_std_cap: float = .08

    def __post_init__(self):
        super().__post_init__()
        configure_preparation(self, training=True)


@configclass
class B2Z1VB13PrepareAMPEnvCfg_Play(B2Z1VB12ScheduleAMPEnvCfg_Play):
    vb13_stage: int = 0
    vb11_nominal_hose_mass_kg: float = 0.

    def __post_init__(self):
        super().__post_init__()
        configure_preparation(self)
        self.episode_length_s = 120.


@configclass
class B2Z1VB13PrepareVisionAMPEnvCfg_Play(B2Z1VB12ScheduleVisionAMPEnvCfg_Play):
    vb13_stage: int = 0
    vb11_nominal_hose_mass_kg: float = 0.

    def __post_init__(self):
        super().__post_init__()
        configure_preparation(self)
        self.episode_length_s = 120.
